// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utility/paged_graph_index_hash.h"
#include <filesystem>
#include <future>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec::core {
namespace {
class HashSegment : public IndexStorage::Segment {
 public:
  explicit HashSegment(size_t bytes, bool direct)
      : bytes_(bytes, 0xa5), direct_(direct) {}
  size_t data_size() const override {
    return bytes_.size();
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return 0;
  }
  size_t capacity() const override {
    return bytes_.size();
  }
  size_t fetch(size_t offset, void *out, size_t bytes) const override {
    if (fail_read) return 0;
    ++reads;
    max_read = std::max(max_read, bytes);
    std::memcpy(out, bytes_.data() + offset, bytes);
    return bytes;
  }
  size_t read(size_t offset, const void **out, size_t bytes) override {
    scratch.resize(bytes);
    const size_t n = fetch(offset, scratch.data(), bytes);
    *out = scratch.data();
    return n;
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &out,
              size_t bytes) override {
    const void *data = nullptr;
    const size_t n = read(offset, &data, bytes);
    out.reset(const_cast<void *>(data));
    return n;
  }
  size_t write(size_t offset, const void *data, size_t bytes) override {
    if (fail_write) return 0;
    std::memcpy(bytes_.data() + offset, data, bytes);
    return bytes;
  }
  size_t resize(size_t bytes) override {
    bytes_.resize(bytes);
    return bytes;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return nullptr;
  }
  const uint8_t *base_data() const override {
    return direct_ ? bytes_.data() : nullptr;
  }
  std::vector<uint8_t> bytes_, scratch;
  bool direct_, fail_read{false}, fail_write{false};
  mutable size_t reads{0}, max_read{0};
};

struct HashBroker {
  using Pointer = std::shared_ptr<HashBroker>;
  static constexpr int CHUNK_TYPE_NEIGHBOR_INDEX = 5;
  std::vector<IndexStorage::Segment::Pointer> slots;
  IndexStorage::Pointer storage;
  bool direct{false};
  size_t get_chunk_cnt(int) const {
    return slots.size();
  }
  IndexStorage::Segment::Pointer get_chunk(int, size_t id) const {
    return slots.at(id);
  }
  std::pair<int, IndexStorage::Segment::Pointer> alloc_chunk(int, size_t id,
                                                             size_t bytes) {
    if (storage) {
      const auto name = "hash-" + std::to_string(id);
      int ret = storage->append(name, bytes);
      if (ret != 0) return {ret, nullptr};
      slots.push_back(storage->get(name));
    } else {
      slots.push_back(std::make_shared<HashSegment>(bytes, direct));
    }
    return {0, slots.back()};
  }
};
using Hash = PagedGraphIndexHash<uint32_t, uint32_t, HashBroker>;

TEST(PagedGraphIndexHashTest,
     CollisionWrapAndValueLifetimeAcrossReadsAndUpdates) {
  for (bool direct : {false, true}) {
    auto broker = std::make_shared<HashBroker>();
    broker->direct = direct;
    Hash hash;
    ASSERT_EQ(0, hash.init(broker, 64, 32, 256, 2.0f));
    // All eight keys collide at the last slot and wrap to the beginning.
    for (uint32_t i = 0; i < 8; ++i)
      ASSERT_TRUE(hash.insert(7 + i * 8, 100 + i));
    auto saved = hash.find(7);
    ASSERT_NE(saved, hash.end());
    EXPECT_FALSE(hash.insert(71, 1000));
    int status = -1;
    EXPECT_EQ(hash.find(71, &status), hash.end());
    EXPECT_EQ(0, status);
    ASSERT_TRUE(hash.insert(7, 999));
    for (uint32_t i = 1; i < 8; ++i)
      ASSERT_EQ(100 + i, hash.find(7 + i * 8)->second);
    EXPECT_EQ(100u, saved->second);
    EXPECT_EQ(999u, hash.find(7)->second);
    auto slot = std::static_pointer_cast<HashSegment>(broker->slots.front());
    EXPECT_EQ(direct ? 0u : sizeof(Hash::Item), slot->max_read);
    ASSERT_EQ(0, hash.cleanup());
    slot->reads = 0;
    ASSERT_EQ(0, hash.init(broker, 64, 32, 256, 2.0f));
    EXPECT_EQ(0u, slot->reads);  // Opening never materializes a slot.
    EXPECT_EQ(999u, hash.find(7)->second);
    ASSERT_EQ(0, hash.cleanup());
    broker.reset();
    EXPECT_EQ(100u, saved->second);
  }
}

TEST(PagedGraphIndexHashTest, IoFailuresDoNotPublishAnUnpersistedShadowValue) {
  auto broker = std::make_shared<HashBroker>();
  Hash hash;
  ASSERT_EQ(0, hash.init(broker, 4096, 16, 8192, 2.0f));
  ASSERT_TRUE(hash.insert(17, 20));
  auto slot = std::static_pointer_cast<HashSegment>(broker->slots.front());
  slot->fail_write = true;
  EXPECT_FALSE(hash.insert(17, 30));
  EXPECT_EQ(20u, hash.find(17)->second);
  slot->fail_write = false;
  slot->fail_read = true;
  int status = 0;
  EXPECT_EQ(hash.find(17, &status), hash.end());
  EXPECT_EQ(IndexError_ReadData, status);
  EXPECT_FALSE(hash.insert(18, 40));
  slot->fail_read = false;
  EXPECT_EQ(hash.find(18), hash.end());
}

TEST(PagedGraphIndexHashTest,
     RealPoolReopenAndParallelQueriesStayWithinBudget) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "paged_graph_index_hash_test.data";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(old));
    std::filesystem::remove(path);
  });
  const size_t budget =
      ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
      64 * 1024;
  ASSERT_EQ(0, pool.init(budget));
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  auto broker = std::make_shared<HashBroker>();
  broker->storage = storage;
  Hash hash;
  ASSERT_EQ(0, hash.init(broker, 4096, 2, 32768, 2.0f));
  for (uint32_t i = 0; i < 32768; ++i) ASSERT_TRUE(hash.insert(i, i + 1));
  auto saved = hash.find(17);
  ASSERT_NE(saved, hash.end());
  EXPECT_LE(pool.used(), budget);
  const size_t slot_count = broker->slots.size();
  ASSERT_EQ(0, hash.cleanup());
  broker->slots.clear();
  ASSERT_EQ(0, storage->flush());
  ASSERT_EQ(0, storage->close());
  ASSERT_EQ(0, storage->open(path, false));
  for (size_t i = 0; i < slot_count; ++i)
    broker->slots.push_back(storage->get("hash-" + std::to_string(i)));
  // The current capacity estimate must not truncate a persisted larger index.
  ASSERT_EQ(0, hash.init(broker, 4096, 2, 4096, 2.0f));
  std::vector<std::future<bool>> workers;
  for (uint32_t worker = 0; worker < 4; ++worker) {
    workers.push_back(std::async(std::launch::async, [&hash, worker] {
      for (uint32_t i = worker; i < 32768; i += 4) {
        auto entry = hash.find(i);
        if (!entry || entry->second != i + 1) return false;
      }
      return true;
    }));
  }
  for (auto &worker : workers) EXPECT_TRUE(worker.get());
  EXPECT_EQ(18u, saved->second);
  EXPECT_LE(pool.used(), budget);
  ASSERT_EQ(0, hash.cleanup());
  broker->slots.clear();
  ASSERT_EQ(0, storage->close());
}
}  // namespace
}  // namespace zvec::core

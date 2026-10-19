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
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "ailego/pattern/defer.h"
#include "hnsw_sparse_index_hash.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
TEST(HnswSparsePagedHashTest,
     ReadsLegacyLayoutAndPersistsUpdatesUnderPressure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const auto old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "sparse_paged_hash_test.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(old));
    std::filesystem::remove(path);
  });
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  IndexStreamer::Stats stats;
  auto broker = std::make_shared<SparseChunkBroker>(stats);
  ASSERT_EQ(0, broker->open(storage, 64 * 1024 * 1024, 2 * 1024 * 1024, false));
  using Hash = HnswSparseIndexHashMap<uint32_t, uint32_t>;
  static constexpr size_t bytes = 2 * 1024 * 1024;
  // Write the old on-disk layout directly, including a wrapped collision.
  auto allocated = broker->alloc_chunk(
      SparseChunkBroker::CHUNK_TYPE_NEIGHBOR_INDEX, 0, bytes);
  ASSERT_EQ(0, allocated.first);
  auto segment = allocated.second;
  ASSERT_EQ(bytes, segment->resize(bytes));
  const std::array<char, 4096> zeros{};
  for (size_t offset = 0; offset < bytes; offset += zeros.size())
    ASSERT_EQ(zeros.size(), segment->write(offset, zeros.data(), zeros.size()));
  const uint32_t slots = bytes / sizeof(Hash::Item);
  const Hash::Item first{slots - 1, 123}, wrapped{2 * slots - 1, 456};
  ASSERT_EQ(sizeof(first),
            segment->write(bytes - sizeof(first), &first, sizeof(first)));
  ASSERT_EQ(sizeof(wrapped), segment->write(0, &wrapped, sizeof(wrapped)));
  Hash hash;
  ASSERT_EQ(0, hash.init(broker, bytes, 4, 4 * slots, 2.0f));
  auto saved = hash.find(slots - 1);
  ASSERT_TRUE(saved.has_value());
  EXPECT_EQ(123u, saved->second);
  ASSERT_TRUE(hash.find(2 * slots - 1).has_value());
  EXPECT_EQ(456u, hash.find(2 * slots - 1)->second);
  ASSERT_TRUE(hash.insert(slots - 1, 789));
  EXPECT_EQ(123u, saved->second);
  ASSERT_EQ(0, hash.cleanup());
  segment.reset();
  allocated.second.reset();
  ASSERT_EQ(0, broker->flush(0));
  ASSERT_EQ(0, broker->close());
  broker.reset();
  ASSERT_EQ(0, storage->close());
  ASSERT_EQ(0, storage->open(path, false));
  broker = std::make_shared<SparseChunkBroker>(stats);
  ASSERT_EQ(0, broker->open(storage, 64 * 1024 * 1024, bytes, false));
  ASSERT_EQ(0, hash.init(broker, bytes, 4, 4 * slots, 2.0f));
  ASSERT_TRUE(hash.find(slots - 1).has_value());
  EXPECT_EQ(789u, hash.find(slots - 1)->second);
  EXPECT_EQ(456u, hash.find(2 * slots - 1)->second);
  EXPECT_LE(pool.used(), pool.capacity());
  ASSERT_EQ(0, hash.cleanup());
  ASSERT_EQ(0, broker->close());
  broker.reset();
  ASSERT_EQ(0, storage->close());
}
}  // namespace zvec::core

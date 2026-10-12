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
#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "flat/flat_builder.h"
#include "flat/flat_streamer.h"
#include "tests/core/algorithm/graph_dump_test_helper.h"

namespace zvec::core {
namespace {
class FlatCapture : public testing::GraphDumpCapture {
 public:
  int append(const std::string &id, size_t size, size_t pad,
             uint32_t crc) override {
    EXPECT_EQ(size + pad, pending.size());
    EXPECT_EQ(std::string(pad, '\0'), pending.substr(size));
    if (crc) EXPECT_EQ(ailego::Crc32c::Hash(pending.data(), size), crc);
    segments[id] = pending.substr(0, size);
    pending.clear();
    return fail_append ? IndexError_WriteData : 0;
  }
  bool fail_append{false};
};

class OnePassRows : public IndexHolder {
 public:
  explicit OnePassRows(size_t n) : n_(n) {}
  size_t count() const override {
    return SIZE_MAX;
  }
  size_t dimension() const override {
    return 8;
  }
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DT_FP32;
  }
  size_t element_size() const override {
    return 8 * sizeof(float);
  }
  bool multipass() const override {
    return false;
  }
  struct Cursor : Iterator {
    explicit Cursor(OnePassRows &owner) : owner_(owner) {}
    const void *data() const override {
      row_.fill(static_cast<float>(position_));
      return owner_.fail_data && position_ == 7 ? nullptr : row_.data();
    }
    bool is_valid() const override {
      return position_ < owner_.n_;
    }
    uint64_t key() const override {
      return (position_ * 17 % 1031) + 1;
    }
    void next() override {
      ++position_;
    }
    int status() const override {
      return owner_.fail_end && position_ == owner_.n_ ? IndexError_ReadData
                                                       : 0;
    }
    OnePassRows &owner_;
    size_t position_{0};
    mutable std::array<float, 8> row_{};
  };
  Iterator::Pointer create_iterator() override {
    ++passes;
    return passes == 1 ? std::make_unique<Cursor>(*this) : nullptr;
  }
  bool fail_data{false}, fail_end{false};
  size_t passes{0};

 private:
  size_t n_;
};

class FlatMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    old_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(0));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_));
  }
  static IndexMeta meta() {
    IndexMeta result(IndexMeta::DT_FP32, 8);
    result.set_metric("SquaredEuclidean", 0, ailego::Params{});
    return result;
  }
  ailego::MemoryLimitPool &pool_ = ailego::MemoryLimitPool::get_instance();
  size_t old_{0};
};

TEST_F(FlatMemoryTest, DumpUnknownCountRowAndColumnMatchWithoutCorpusCopies) {
  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    std::shared_ptr<FlatCapture> reference;
    for (bool buffered : {false, true}) {
      ASSERT_EQ(0, pool_.init(buffered ? 4 * 1024 * 1024 : 0));
      auto metadata = meta();
      metadata.set_major_order(order);
      auto input = std::make_shared<OnePassRows>(70003);
      auto builder = IndexFactory::CreateBuilder("FlatBuilder");
      auto output = std::make_shared<FlatCapture>();
      ASSERT_EQ(0, builder->init(metadata, {}));
      ASSERT_EQ(0, builder->build(nullptr, input));
      ASSERT_EQ(0, builder->dump(output));
      EXPECT_EQ(1u, input->passes);
      EXPECT_EQ(70003u, builder->stats().dumped_count());
      EXPECT_LE(output->max_write, 8192u);
      auto keys = output->records<uint64_t>(FLAT_SEGMENT_KEYS_SEG_ID);
      auto mapping = output->records<uint32_t>(FLAT_SEGMENT_MAPPING_SEG_ID);
      ASSERT_EQ(keys.size(), mapping.size());
      for (size_t i = 1; i < mapping.size(); ++i) {
        EXPECT_LE(keys[mapping[i - 1]], keys[mapping[i]]);
        if (keys[mapping[i - 1]] == keys[mapping[i]])
          EXPECT_LT(mapping[i - 1], mapping[i]);
      }
      if (reference)
        EXPECT_EQ(reference->segments, output->segments);
      else
        reference = output;
    }
  }
}

TEST_F(FlatMemoryTest, EmptyAndSingleRowDumpKeepLegacySegments) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  for (size_t count : {0u, 1u}) {
    auto input = std::make_shared<OnePassRows>(count);
    auto output = std::make_shared<FlatCapture>();
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    ASSERT_EQ(0, builder->init(meta(), {}));
    ASSERT_EQ(0, builder->build(nullptr, input));
    ASSERT_EQ(0, builder->dump(output));
    EXPECT_EQ(count * sizeof(uint64_t),
              output->segments.at(FLAT_SEGMENT_KEYS_SEG_ID).size());
    EXPECT_EQ(count * sizeof(uint32_t),
              output->segments.at(FLAT_SEGMENT_MAPPING_SEG_ID).size());
    EXPECT_EQ(1u, input->passes);
  }
}

TEST_F(FlatMemoryTest, DumpReportsInputAndOutputFailures) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  for (int failure = 0; failure != 4; ++failure) {
    auto input = std::make_shared<OnePassRows>(40);
    input->fail_data = failure == 0;
    input->fail_end = failure == 1;
    auto output = std::make_shared<FlatCapture>();
    output->fail_write = failure == 2;
    output->fail_append = failure == 3;
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    ASSERT_EQ(0, builder->init(meta(), {}));
    ASSERT_EQ(0, builder->build(nullptr, input));
    EXPECT_EQ(failure < 2 ? IndexError_ReadData : IndexError_WriteData,
              builder->dump(output));
  }
}

TEST_F(FlatMemoryTest, ContinuousIdsHolesReopenAndChangedSegmentSize) {
  for (const auto *mode : {"MMapFileStorage", "BufferStorage"}) {
    ASSERT_EQ(0, pool_.init(16 * 1024 * 1024));
    const std::string path = std::string("flat_direct_ids_") + mode;
    ASSERT_FALSE(std::filesystem::exists(path));
    AILEGO_DEFER([&] { std::filesystem::remove(path); });
    for (uint32_t round = 0; round < 2; ++round) {
      auto storage = IndexFactory::CreateStorage(mode);
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(path, true));
      IndexStreamer::Stats stats;
      FlatStreamerEntity entity(stats);
      *entity.mutable_meta() = meta();
      entity.set_use_key_info_map(false);
      entity.set_block_vector_count(32);
      entity.set_linear_list_count(1);
      entity.set_segment_size(round ? 65536 : 4096);
      ASSERT_EQ(0, entity.open(storage, meta()));
      std::array<float, 8> row{};
      const uint32_t begin = round ? 231 : 3;
      const uint32_t end = round ? 631 : 231;
      for (uint32_t id = begin; id < end; ++id) {
        row.fill(static_cast<float>(id));
        ASSERT_EQ(0, entity.add_vector_with_id(id, row.data(), sizeof(row)));
      }
      // A hole is a persisted invalid key, not a synthesized key equal to ID.
      if (!round) EXPECT_EQ(kInvalidKey, entity.key(1));
      row.fill(1.0f);
      ASSERT_EQ(0, entity.add_vector_with_id(1, row.data(), sizeof(row)));
      for (uint32_t id : {1u, 3u, 123u, end - 1}) {
        EXPECT_EQ(id, entity.key(id));
        IndexStorage::MemoryBlock block;
        ASSERT_EQ(0, entity.get_vector_by_key(id, block));
        EXPECT_FLOAT_EQ(static_cast<float>(id),
                        static_cast<const float *>(block.data())[0]);
      }
      EXPECT_EQ(nullptr, entity.get_vector_by_key(uint64_t{1} << 32));
      const uint64_t ids[]{1, 3, 123, end - 1};
      std::vector<IndexStorage::MemoryBlock> blocks;
      ASSERT_EQ(0, entity.get_vectors_by_key(ids, 4, blocks));
      blocks.clear();
      auto iterator = entity.creater_iterator();
      ASSERT_NE(nullptr, iterator);
      // Appending must not extend an already-created iterator.
      row.fill(static_cast<float>(end));
      ASSERT_EQ(0, entity.add_vector_with_id(end, row.data(), sizeof(row)));
      ASSERT_EQ(0, entity.flush(0));
      ASSERT_EQ(0, entity.close());
      size_t visited = 0;
      for (; iterator->is_valid(); iterator->next(), ++visited) {
        EXPECT_FLOAT_EQ(
            visited == 0 || visited == 2 ? 0.0f : static_cast<float>(visited),
            static_cast<const float *>(iterator->data())[0]);
      }
      EXPECT_EQ(end, visited);
      EXPECT_EQ(0, iterator->status());
      iterator.reset();
      ASSERT_EQ(0, storage->close());
    }
    std::filesystem::remove(path);
  }
}

TEST_F(FlatMemoryTest, ArbitraryKeysKeepLookupAndIteratorCompatible) {
  const std::string path = "flat_arbitrary_keys.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  for (int round = 0; round < 2; ++round) {
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, round == 0));
    IndexStreamer::Stats stats;
    FlatStreamerEntity entity(stats);
    *entity.mutable_meta() = meta();
    entity.set_block_vector_count(32);
    entity.set_linear_list_count(1);
    entity.set_segment_size(4096);
    ASSERT_EQ(0, entity.open(storage, meta()));
    std::array<float, 8> row{};
    if (!round)
      for (uint64_t id = 0; id < 199; ++id) {
        row.fill(static_cast<float>(id));
        ASSERT_EQ(
            0, entity.add((uint64_t{1} << 40) + id, row.data(), sizeof(row)));
      }
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, entity.get_vector_by_key((uint64_t{1} << 40) + 198, block));
    EXPECT_FLOAT_EQ(198.0f, static_cast<const float *>(block.data())[0]);
    block.reset(nullptr);
    auto iterator = entity.creater_iterator();
    ASSERT_NE(nullptr, iterator);
    size_t visited = 0;
    for (; iterator->is_valid(); iterator->next(), ++visited)
      EXPECT_EQ((uint64_t{1} << 40) + visited, iterator->key());
    EXPECT_EQ(199u, visited);
    ASSERT_EQ(0, entity.flush(0));
    iterator.reset();
    ASSERT_EQ(0, entity.close());
    ASSERT_EQ(0, storage->close());
  }
  std::filesystem::remove(path);
}
}  // namespace
}  // namespace zvec::core

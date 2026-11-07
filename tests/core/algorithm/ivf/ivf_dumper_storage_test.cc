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

#include <cstring>
#include <map>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "ivf_dumper.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

class RecordingDumper : public IndexDumper {
 public:
  struct Segment {
    std::string bytes;
    size_t size;
    size_t padding;
    uint32_t crc;
    size_t max_write;
  };
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int create(const std::string &) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  int append(const std::string &id, size_t size, size_t padding,
             uint32_t crc) override {
    if (id == fail_append) return IndexError_WriteData;
    EXPECT_EQ(size + padding, pending.size());
    if (crc) EXPECT_EQ(crc, ailego::Crc32c::Hash(pending.data(), size));
    segments.emplace(
        id, Segment{std::move(pending), size, padding, crc, max_write});
    pending.clear();
    max_write = 0;
    return 0;
  }
  size_t write(const void *data, size_t size) override {
    if (segments.size() == fail_write_after_segments) return 0;
    if (size) pending.append(static_cast<const char *>(data), size);
    max_write = std::max(max_write, size);
    return size;
  }
  std::map<std::string, Segment> segments;
  std::string pending;
  std::string fail_append;
  size_t max_write{0};
  size_t fail_write_after_segments{std::numeric_limits<size_t>::max()};
};

class IVFDumperStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    previous_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    budget_ =
        2 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
        64 * 1024;
    ASSERT_EQ(0, pool.init(0));
  }
  void TearDown() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_));
  }
  static uint64_t key(size_t id, size_t count, bool duplicates) {
    return duplicates && id % 4 == 0 ? 42
                                     : (uint64_t{1} << 40) + (count - id) * 19;
  }
  int dump(const std::shared_ptr<RecordingDumper> &output, size_t count,
           bool column, bool blocks, bool duplicates = false) {
    IndexMeta meta(IndexMeta::DT_FP32, 8);
    meta.set_major_order(column ? IndexMeta::MO_COLUMN : IndexMeta::MO_ROW);
    IVFDumper dumper(meta, output, 3, 32);
    int ret =
        dumper.reserve_keys(count + 7);  // Filtering may leave unused capacity.
    if (ret != 0) return ret;
    const size_t split = count > 7 ? count - 7 : count;
    size_t id = 0;
    for (uint32_t list : {0u, 2u}) {
      const size_t end = list == 0 ? split : count;
      while (id < end) {
        if (blocks && end - id >= 32) {
          float rows[32 * 8];
          uint64_t keys[32];
          for (size_t i = 0; i < 32; ++i) {
            keys[i] = key(id + i, count, duplicates);
            for (size_t d = 0; d < 8; ++d)
              rows[column ? d * 32 + i : i * 8 + d] = float(id + i + d);
          }
          ret = dumper.dump_inverted_block(list, keys, rows, 32, column);
          id += 32;
        } else {
          float row[8];
          for (size_t d = 0; d < 8; ++d) row[d] = float(id + d);
          ret = dumper.dump_inverted_vector(list, key(id, count, duplicates),
                                            row);
          ++id;
        }
        if (ret != 0) return ret;
      }
    }
    return dumper.dump_inverted_vector_finished();
  }
  size_t previous_{0};
  size_t budget_{0};
};

TEST_F(IVFDumperStorageTest, PagedKeysAndStreamingOffsetsMatchLegacySegments) {
  for (bool column : {false, true}) {
    for (bool blocks : {false, true}) {
      for (size_t count : {size_t{0}, size_t{35}, size_t{16389}}) {
        SCOPED_TRACE(::testing::Message()
                     << column << ':' << blocks << ':' << count);
        auto &pool = ailego::PoolPayloadBudget::get_instance();
        ASSERT_EQ(0, pool.init(0));
        auto reference = std::make_shared<RecordingDumper>();
        ASSERT_EQ(0, dump(reference, count, column, blocks));
        ASSERT_EQ(0, pool.init(budget_));
        auto actual = std::make_shared<RecordingDumper>();
        ASSERT_EQ(0, dump(actual, count, column, blocks));
        ASSERT_EQ(reference->segments.size(), actual->segments.size());
        for (const auto &entry : reference->segments) {
          SCOPED_TRACE(entry.first);
          ASSERT_EQ(1u, actual->segments.count(entry.first));
          const auto &value = actual->segments.at(entry.first);
          EXPECT_EQ(entry.second.bytes, value.bytes);
          EXPECT_EQ(entry.second.crc, value.crc);
          EXPECT_EQ(entry.second.size, value.size);
          EXPECT_EQ(entry.second.padding, value.padding);
        }
        EXPECT_LE(actual->segments.at(IVF_OFFSETS_SEG_ID).max_write,
                  32u * 1024);
        EXPECT_LE(actual->segments.at(IVF_KEYS_SEG_ID).max_write, 32u * 1024);
        EXPECT_LE(actual->segments.at(IVF_MAPPING_SEG_ID).max_write,
                  16u * 1024);
        // Decode offsets independently, including column-major blocks and
        // row-major tails on both sides of the empty middle list.
        const auto &offsets = actual->segments.at(IVF_OFFSETS_SEG_ID);
        const auto &body = actual->segments.at(IVF_INVERTED_BODY_SEG_ID);
        for (size_t i = 0; i < count; ++i) {
          InvertedVecLocation location(0, false);
          std::memcpy(&location, offsets.bytes.data() + i * sizeof(location),
                      sizeof(location));
          for (size_t d = 0; d < 8; ++d) {
            const size_t offset =
                location.offset +
                d * sizeof(float) * (location.column_major ? 32 : 1);
            ASSERT_LE(offset + sizeof(float), body.size);
            float value;
            std::memcpy(&value, body.bytes.data() + offset, sizeof(value));
            ASSERT_FLOAT_EQ(value, float(i + d));
          }
        }
        EXPECT_EQ(0u, pool.used());
      }
    }
  }
}

TEST_F(IVFDumperStorageTest, DuplicateKeysKeepEveryOriginalOrdinal) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(budget_));
  constexpr size_t count = 8197;
  auto output = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, dump(output, count, true, true, true));
  const auto &keys = output->segments.at(IVF_KEYS_SEG_ID);
  const auto &mapping = output->segments.at(IVF_MAPPING_SEG_ID);
  std::vector<bool> seen(count, false);
  uint64_t previous = 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t id;
    std::memcpy(&id, mapping.bytes.data() + i * sizeof(id), sizeof(id));
    ASSERT_LT(id, count);
    EXPECT_FALSE(seen[id]);
    seen[id] = true;
    uint64_t value;
    std::memcpy(&value, keys.bytes.data() + id * sizeof(value), sizeof(value));
    EXPECT_EQ(key(id, count, true), value);
    EXPECT_GE(value, previous);
    previous = value;
  }
  EXPECT_EQ(0u, pool.used());
}

TEST_F(IVFDumperStorageTest, BudgetAndOutputFailuresDoNotLeakOrFallback) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(1));
  auto output = std::make_shared<RecordingDumper>();
  // Metadata and the first block now require admission in the constructor.
  EXPECT_THROW(dump(output, 8197, false, false), std::bad_alloc);
  EXPECT_TRUE(output->segments.empty());
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(budget_));
  for (size_t fail_after : {3u, 4u, 5u}) {
    output = std::make_shared<RecordingDumper>();
    output->fail_write_after_segments = fail_after;
    EXPECT_EQ(IndexError_WriteData, dump(output, 8197, false, true));
    EXPECT_EQ(0u, pool.used());
  }
  for (const auto &segment :
       {IVF_KEYS_SEG_ID, IVF_MAPPING_SEG_ID, IVF_OFFSETS_SEG_ID}) {
    output = std::make_shared<RecordingDumper>();
    output->fail_append = segment;
    EXPECT_EQ(IndexError_WriteData, dump(output, 8197, false, true));
    EXPECT_EQ(0u, pool.used());
  }
}

}  // namespace
}  // namespace zvec::core

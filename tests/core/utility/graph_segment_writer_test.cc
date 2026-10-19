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

#include "utility/graph_segment_writer.h"
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class CaptureDumper : public IndexDumper {
 public:
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
  size_t write(const void *data, size_t bytes) override {
    max_write = std::max(max_write, bytes);
    if (fail_write) return 0;
    if (bytes) output.append(static_cast<const char *>(data), bytes);
    return bytes;
  }
  int append(const std::string &, size_t bytes, size_t pad,
             uint32_t crc) override {
    if (fail_append) return IndexError_WriteData;
    EXPECT_EQ(bytes + pad, output.size());
    EXPECT_EQ((32 - bytes % 32) % 32, pad);
    EXPECT_EQ(ailego::Crc32c::Hash(output.data(), bytes), crc);
    for (size_t i = bytes; i < output.size(); ++i) EXPECT_EQ(0, output[i]);
    ++segments;
    return 0;
  }
  std::string output;
  size_t max_write{0}, segments{0};
  bool fail_write{false}, fail_append{false};
};

TEST(GraphSegmentWriterTest,
     ChunkBoundariesAndEmptySegmentsPreserveBytesAndCrc) {
  for (size_t count :
       {size_t{0}, size_t{1}, size_t{65535}, size_t{65536}, size_t{196619}}) {
    SCOPED_TRACE(count);
    std::string source(count, 'x');
    for (size_t i = 0; i < count; ++i) source[i] = static_cast<char>(i * 17);
    auto capture = std::make_shared<CaptureDumper>();
    IndexDumper::Pointer dumper = capture;
    GraphSegmentWriter writer(dumper);
    const size_t split = count / 3;
    ASSERT_EQ(0, writer.write(source.data(), split));
    ASSERT_EQ(0, writer.write(source.data() + split, count - split));
    EXPECT_EQ(static_cast<int64_t>((count + 31) / 32 * 32),
              writer.finish("data"));
    EXPECT_EQ(source, capture->output.substr(0, count));
    EXPECT_EQ(1u, capture->segments);
    EXPECT_LE(capture->max_write, 64u * 1024);
  }
}

TEST(GraphSegmentWriterTest,
     MappingMatchesAcrossBudgetAndHandlesDuplicateKeys) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(old));
  });
  const size_t budget =
      2 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
      128 * 1024;
  for (size_t count : {size_t{0}, size_t{1}, size_t{4097}, size_t{32769}}) {
    auto key = [](size_t i) { return static_cast<uint64_t>((i * 313) % 997); };
    std::vector<uint32_t> reference(count);
    std::iota(reference.begin(), reference.end(), 0U);
    std::sort(reference.begin(), reference.end(), [&](uint32_t a, uint32_t b) {
      return key(a) != key(b) ? key(a) < key(b) : a < b;
    });
    for (size_t capacity : {size_t{0}, budget}) {
      SCOPED_TRACE(::testing::Message() << count << ':' << capacity);
      ASSERT_EQ(0, pool.init(capacity));
      auto capture = std::make_shared<CaptureDumper>();
      ASSERT_GE(DumpGraphKeyMapping(capture, "map", count, key), 0);
      EXPECT_EQ(1u, capture->segments);
      ASSERT_GE(capture->output.size(), count * sizeof(uint32_t));
      if (count)
        EXPECT_EQ(0, std::memcmp(reference.data(), capture->output.data(),
                                 count * sizeof(uint32_t)));
      EXPECT_EQ(0u, pool.used());
      EXPECT_LE(capture->max_write, 64u * 1024);
    }
  }
}

TEST(GraphSegmentWriterTest, ReadWriteAndAppendFailuresPropagate) {
  for (bool write_failure : {false, true}) {
    auto capture = std::make_shared<CaptureDumper>();
    capture->fail_write = write_failure;
    capture->fail_append = !write_failure;
    EXPECT_EQ(IndexError_WriteData,
              DumpGraphRecords<uint64_t>(capture, "data", 10000,
                                         [](size_t i, uint64_t *out) {
                                           *out = i;
                                           return 0;
                                         }));
    EXPECT_EQ(0u, capture->segments);
  }
  auto capture = std::make_shared<CaptureDumper>();
  EXPECT_EQ(IndexError_ReadData,
            DumpGraphRecords<uint64_t>(capture, "data", 10,
                                       [](size_t i, uint64_t *out) -> int {
                                         if (i == 3) return IndexError_ReadData;
                                         *out = i;
                                         return 0;
                                       }));
  EXPECT_EQ(0u, capture->segments);
}

TEST(GraphSegmentWriterTest,
     InsufficientMetadataBudgetFailsWithoutHeapFallback) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(old));
  });
  ASSERT_EQ(0, pool.init(4096));
  auto capture = std::make_shared<CaptureDumper>();
  EXPECT_EQ(IndexError_NoMemory,
            DumpGraphKeyMapping(capture, "map", 32769,
                                [](size_t i) { return uint64_t{32769} - i; }));
  EXPECT_EQ(0u, capture->segments);
  EXPECT_EQ(0u, pool.used());
}
}  // namespace
}  // namespace zvec::core

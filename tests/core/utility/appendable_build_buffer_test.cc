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

#include "appendable_build_buffer_test_helper.h"

namespace zvec::core {
namespace {
TEST_F(AppendableBuildBufferTest, GrowsAcrossChunksWithoutLosingEarlierData) {
  for (bool paged : {false, true}) {
    ASSERT_EQ(0, pool_.init(paged ? 4 * 1024 * 1024 : 0));
    AppendableBuildBuffer buffer;
    ASSERT_EQ(0, buffer.init(0));
    static constexpr size_t size = 3 * 1024 * 1024 + 79;
    std::array<char, 4093> row{};
    for (size_t offset = 0; offset < size;) {
      const size_t n = std::min(row.size(), size - offset);
      row.fill(static_cast<char>(offset % 127));
      ASSERT_EQ(0, buffer.resize(offset + n));
      ASSERT_EQ(0, buffer.write(offset, row.data(), n));
      offset += n;
    }
    EXPECT_EQ(paged, buffer.paged());
    for (size_t offset = 0; offset < size;) {
      const size_t n = std::min(row.size(), size - offset);
      ASSERT_EQ(0, buffer.read(offset, row.data(), n));
      const char expected = static_cast<char>(offset % 127);
      ASSERT_TRUE(std::all_of(row.begin(), row.begin() + n,
                              [expected](char byte) { return byte == expected; }))
          << "offset=" << offset;
      offset += n;
    }
    EXPECT_EQ(IndexError_InvalidArgument, buffer.read(size, row.data(), 1));
    EXPECT_EQ(IndexError_InvalidArgument,
              buffer.write(SIZE_MAX, row.data(), 1));
    AppendableBuildBuffer moved(std::move(buffer));
    EXPECT_EQ(size, moved.size());
    EXPECT_EQ(0, moved.read(size - 1, row.data(), 1));
    EXPECT_LE(pool_.used(), pool_.capacity());
  }
}

TEST_F(AppendableBuildBufferTest, FailedAdmissionDoesNotFallBackToHeap) {
  ASSERT_EQ(0, pool_.init(1));
  AppendableBuildBuffer buffer;
  EXPECT_NE(0, buffer.init(1024 * 1024));
  EXPECT_EQ(0u, buffer.size());
  EXPECT_EQ(0u, pool_.used());
}
}  // namespace
}  // namespace zvec::core

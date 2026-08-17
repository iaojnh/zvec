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

#include "diskann_context.h"
#include <gtest/gtest.h>

using namespace zvec::core;

TEST(DiskAnnSearchStatsTest, RecordsBeamAndIoBatchDistribution) {
  SearchStats stats;

  stats.record_query(20);
  stats.record_query(32);
  for (uint32_t batch_size : {1U, 2U, 4U, 5U, 8U, 9U, 16U, 17U, 32U}) {
    stats.record_io_batch(batch_size);
  }
  stats.record_io_batch(0);

  EXPECT_EQ(stats.query_count, 2U);
  EXPECT_EQ(stats.effective_beam_width_sum, 52U);
  EXPECT_EQ(stats.effective_beam_width_max, 32U);
  EXPECT_EQ(stats.io_batch_count, 9U);
  EXPECT_EQ(stats.io_batch_size_sum, 94U);
  EXPECT_EQ(stats.io_batch_size_max, 32U);
  EXPECT_EQ(stats.io_batch_size_histogram[0], 1U);
  EXPECT_EQ(stats.io_batch_size_histogram[1], 2U);
  EXPECT_EQ(stats.io_batch_size_histogram[2], 2U);
  EXPECT_EQ(stats.io_batch_size_histogram[3], 2U);
  EXPECT_EQ(stats.io_batch_size_histogram[4], 2U);
}

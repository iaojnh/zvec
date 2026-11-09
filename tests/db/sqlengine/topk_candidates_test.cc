// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/sqlengine/planner/topk_candidates.h"
#include <numeric>
#include <random>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/initialize.h>
#include <gtest/gtest.h>
#include "db/sqlengine/planner/group_topk_candidates.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::sqlengine {
namespace {
TEST(GroupTopKCandidatesTest, ReenteringGroupKeepsEarlierRunnerUp) {
  ailego::PoolPayloadBudget::get_instance().init(4 * 1024 * 1024);
  for (bool descending : {false, true}) {
    GroupTopKCandidates groups(1, 2, descending);
    const float sign = descending ? -1 : 1;
    groups.offer("a", {0, 0, 0, sign * 100});
    groups.offer("a", {0, 1, 1, sign * 90});
    groups.offer("b", {0, 2, 2, sign * 80});
    groups.offer("a", {0, 3, 3, sign * 1});
    EXPECT_FALSE(groups.paged());
    groups.finish();
    ASSERT_EQ(2u, groups.rows().size());
    EXPECT_EQ(3, groups.rows()[0].row);
    EXPECT_EQ(1, groups.rows()[1].row);
    EXPECT_EQ("a", std::string(groups.group(groups.rows()[0].group)));
  }
}

TEST(TopKCandidatesTest, MatchesArrowSortIncludingTiesNaNsAndInfinities) {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  std::mt19937 random(29);
  std::vector<RankedDoc> input;
  arrow::FloatBuilder scores;
  for (size_t segment = 9; segment > 0; --segment) {
    for (size_t i = 0; i < 199; ++i) {
      float score = static_cast<float>(random() % 23);
      if (i % 13 == 0) score = std::numeric_limits<float>::quiet_NaN();
      if (i % 17 == 0) score = std::numeric_limits<float>::infinity();
      if (i % 19 == 0) score = -std::numeric_limits<float>::infinity();
      if (i % 29 == 0) score = -0.0f;
      input.push_back({segment, i, static_cast<int32_t>(i), score});
      ASSERT_TRUE(scores.Append(score).ok());
    }
  }
  auto array = scores.Finish().ValueOrDie();
  for (bool descending : {false, true}) {
    auto sorted = arrow::compute::SortIndices(
        *array, arrow::compute::ArraySortOptions{
                    descending ? arrow::compute::SortOrder::Descending
                               : arrow::compute::SortOrder::Ascending});
    ASSERT_TRUE(sorted.ok());
    auto indices = std::static_pointer_cast<arrow::UInt64Array>(*sorted);
    for (size_t limit : {0u, 1u, 19u, 513u, 2000u}) {
      TopKCandidates selected(limit, descending);
      auto shuffled = input;
      std::shuffle(shuffled.begin(), shuffled.end(), random);
      for (auto row : shuffled) selected.offer(row);
      selected.finish();
      ASSERT_EQ(std::min(limit, input.size()), selected.rows().size());
      for (size_t i = 0; i < selected.rows().size(); ++i) {
        const auto &expected = input[indices->Value(i)];
        EXPECT_EQ(expected.segment, selected.rows()[i].segment);
        EXPECT_EQ(expected.ordinal, selected.rows()[i].ordinal);
      }
    }
  }
}

TEST(TopKCandidatesTest, BudgetDenialDoesNotLoseEarlierCandidates) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  pool.init(1024 * 1024);
  {
    TopKCandidates selected(32, true);
    for (int i = 0; i < 16; ++i) selected.offer({0, uint64_t(i), i, float(i)});
    const size_t charge = pool.capacity() - pool.used();
    ASSERT_TRUE(pool.try_charge_external(charge));
    EXPECT_THROW(selected.offer({0, 16, 16, 16.0f}), std::bad_alloc);
    pool.release_external(charge);
    EXPECT_EQ(16u, selected.rows().size());
    selected.offer({0, 16, 16, 16.0f});
    selected.finish();
    EXPECT_EQ(16.0f, selected.rows().front().score);
  }
  EXPECT_EQ(0u, pool.external_used());
}
}  // namespace
}  // namespace zvec::sqlengine

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

namespace zvec::sqlengine {
namespace {
TEST(GroupTopKCandidatesTest, MatchesWideSortOnIdenticalSegmentCandidates) {
  ailego::MemoryLimitPool::get_instance().init(4 * 1024 * 1024);
  std::mt19937 random(42);
  std::vector<RankedDoc> input;
  std::vector<std::string> names{""};
  for (size_t g = 1; g < 13; ++g) names.push_back(std::to_string(g));
  for (size_t segment = 0; segment < 31; ++segment)
    for (size_t i = 0; i < 29; ++i)
      input.push_back({segment, i, static_cast<int32_t>(i),
                       static_cast<float>(random() % 101), random() % 13});
  for (bool descending : {false, true}) {
    auto sorted = input;
    TopKCandidates::Better better{descending};
    std::sort(sorted.begin(), sorted.end(), better);
    for (size_t group_count : {1u, 3u, 20u}) {
      for (size_t per_group : {1u, 7u}) {
        std::map<size_t, size_t> counts;
        std::vector<RankedDoc> expected;
        for (auto row : sorted) {
          if (!counts.count(row.group) && counts.size() == group_count)
            continue;
          if (counts[row.group]++ < per_group) expected.push_back(row);
        }
        GroupTopKCandidates groups(group_count, per_group, descending);
        auto shuffled = input;
        std::shuffle(shuffled.begin(), shuffled.end(), random);
        for (auto row : shuffled) groups.offer(names[row.group], row);
        groups.finish();
        ASSERT_EQ(expected.size(), groups.rows().size());
        for (size_t i = 0; i < expected.size(); ++i) {
          const auto &actual = groups.rows()[i];
          EXPECT_EQ(expected[i].segment, actual.segment);
          EXPECT_EQ(expected[i].ordinal, actual.ordinal);
          EXPECT_EQ(names[expected[i].group],
                    std::string(groups.group(actual.group)));
        }
      }
    }
  }
}

TEST(GroupTopKCandidatesTest, ReenteringGroupKeepsEarlierRunnerUp) {
  ailego::MemoryLimitPool::get_instance().init(4 * 1024 * 1024);
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

TEST(GroupTopKCandidatesTest, ManyGroupsSpillReferencesAndChargeNames) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  pool.init(4 * 1024 * 1024);
  {
    GroupTopKCandidates groups(3, 2, false);
    for (int i = 20000; i > 0; --i) {
      const std::string name = std::string(200, 'x') + std::to_string(i);
      groups.offer(name, {0, uint64_t(i), i, float(i)});
      EXPECT_LE(groups.retained_groups(), 3u);
    }
    EXPECT_TRUE(groups.paged());
    groups.finish();
    ASSERT_EQ(3u, groups.rows().size());
    EXPECT_EQ(1, groups.rows()[0].row);
    EXPECT_EQ(3, groups.rows()[2].row);
    EXPECT_LT(pool.external_used(), 64 * 1024u);
  }
  EXPECT_EQ(0u, pool.external_used());
}

TEST(GroupTopKCandidatesTest, BudgetDenialDoesNotLeakWorkspace) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  pool.init(1024 * 1024);
  const size_t charge = pool.capacity() - pool.used();
  ASSERT_TRUE(pool.try_charge_external(charge));
  {
    GroupTopKCandidates groups(1, 1, false);
    EXPECT_THROW(groups.offer("a", {0, 0, 0, 1.0f}), std::bad_alloc);
  }
  EXPECT_EQ(charge, pool.external_used());
  pool.release_external(charge);
  {
    GroupTopKCandidates groups(1, 1, false);
    groups.offer("a", {0, 0, 0, 1.0f});
    groups.finish();
    ASSERT_EQ(1u, groups.rows().size());
  }
  EXPECT_EQ(0u, pool.external_used());
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

TEST(TopKCandidatesTest, MillionCandidatesKeepOnlyBudgetedTopK) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  pool.init(1024 * 1024);
  {
    TopKCandidates selected(100, false);
    for (size_t i = 1000000; i > 0; --i) selected.offer({0, i, 0, float(i)});
    selected.finish();
    ASSERT_EQ(100u, selected.rows().size());
    EXPECT_LE(selected.rows().capacity(), 100u);
    EXPECT_GE(pool.external_used(), 100 * sizeof(RankedDoc));
    EXPECT_LT(pool.external_used(), 8192u);
    EXPECT_EQ(1.0f, selected.rows().front().score);
    EXPECT_EQ(100.0f, selected.rows().back().score);
  }
  EXPECT_EQ(0u, pool.external_used());
}

TEST(TopKCandidatesTest, BudgetDenialDoesNotLoseEarlierCandidates) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
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

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "utility/group_candidate_collector.h"
#include <gtest/gtest.h>
#include "utility/group_topk_heap.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class GroupCandidateCollectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
};

TEST_F(GroupCandidateCollectorTest,
       ReentryRetainsEarlyRunnersUpAndDuplicateKeys) {
  for (size_t budget : {size_t{0}, size_t{4 * 1024 * 1024}}) {
    ASSERT_EQ(0, pool_.init(budget));
    GroupCandidateCollector groups(1, 3);
    ASSERT_EQ(0, groups.add("returning", 1, 10));
    ASSERT_EQ(0, groups.add("returning", 1, 11));
    // More than a staging buffer; high cardinality never expands best_ past 1.
    for (size_t i = 0; i < 30000; ++i) {
      ASSERT_EQ(0, groups.add(std::to_string(i), i + 100, 9));
      ASSERT_LE(groups.retained_groups(), 1u);
    }
    ASSERT_EQ(0, groups.add("returning", 2, 1));
    std::map<std::string, IndexCandidateHeap> result;
    ASSERT_EQ(0, groups.finish_heaps(result, 3));
    ASSERT_EQ(1u, result.size());
    auto &heap = result.at("returning");
    heap.sort();
    ASSERT_EQ(3u, heap.size());
    EXPECT_EQ(2u, heap[0].key());
    EXPECT_EQ(1u, heap[1].key());
    EXPECT_EQ(1u, heap[2].key());
    EXPECT_FLOAT_EQ(11, heap[2].score());
    EXPECT_EQ(budget != 0, groups.paged());
    EXPECT_LE(pool_.used(), pool_.capacity());
  }
}

TEST_F(GroupCandidateCollectorTest, TiesEmptyAndFailureAreExplicit) {
  ASSERT_EQ(0, pool_.init(0));
  {
    GroupCandidateCollector groups(1, 1);
    ASSERT_EQ(0, groups.add("z", 1, 1));
    ASSERT_EQ(0, groups.add("a", 2, 1));
    std::map<std::string, IndexCandidateHeap> result;
    ASSERT_EQ(0, groups.finish_heaps(result, 1));
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ("a", result.begin()->first);
  }
  ASSERT_EQ(0, pool_.init(1));
  {
    GroupCandidateCollector groups(2, 3);
    EXPECT_EQ(IndexError_NoMemory, groups.add("first", 0, 1));
    EXPECT_EQ(0u, pool_.used());
    std::map<std::string, IndexCandidateHeap> result;
    EXPECT_NE(0, groups.finish_heaps(result, 3));
    EXPECT_TRUE(result.empty());
  }
}

TEST(IndexCandidateHeapTest, CompactLazyCapacityAndLargeScratchRelease) {
  EXPECT_EQ(16u, sizeof(IndexCandidate));
  IndexCandidateHeap heap;
  heap.limit(1000000);
  EXPECT_EQ(0u, heap.container().capacity());
  heap.set_threshold(10001);
  for (uint32_t i = 0; i < 10000; ++i) heap.emplace(i, static_cast<float>(i));
  EXPECT_EQ(10000u, heap.size());
  heap.clear();
  EXPECT_EQ(0u, heap.container().capacity());
  heap.limit(3);
  for (uint32_t i = 10; i > 0; --i) heap.emplace(i, static_cast<float>(i));
  heap.sort();
  ASSERT_EQ(3u, heap.size());
  EXPECT_EQ(1u, heap[0].key());
  EXPECT_EQ(3u, heap[2].key());
}

TEST_F(GroupCandidateCollectorTest,
       WinnerMaterializationDoesNotReserveUserTopk) {
  ASSERT_EQ(0, pool_.init(1024 * 1024));
  GroupCandidateCollector collector(3, 1000000);
  for (uint32_t i = 0; i < 3; ++i)
    ASSERT_EQ(0, collector.add(std::to_string(i), i, static_cast<float>(i)));
  GroupTopkMap<uint32_t, float> groups;
  ASSERT_EQ(0, collector.finish_heaps(groups, 1000000));
  ASSERT_EQ(3u, groups.size());
  for (const auto &entry : groups) {
    EXPECT_EQ(1u, entry.second.size());
    EXPECT_LE(entry.second.container().capacity(), 2u);
  }
  EXPECT_GT(pool_.used(), 0u);
  EXPECT_LT(pool_.used(), 4096u);
}

TEST_F(GroupCandidateCollectorTest,
       GraphHeapsAreLazyChargedAndSelectedWithoutCopy) {
  ASSERT_EQ(0, pool_.init(1024 * 1024));
  {
    GroupTopkMap<uint32_t, float> groups;
    for (uint32_t i = 0; i < 1000; ++i) {
      auto &heap = groups[std::to_string(i)];
      heap.limit(1000000);
      EXPECT_EQ(0u, heap.container().capacity());
      heap.emplace(i, static_cast<float>(i));
    }
    EXPECT_LT(pool_.used(), 512u * 1024u);
    const size_t before = pool_.used();
    const auto selected = SelectGroupHeaps(groups, 10);
    EXPECT_GE(pool_.used(), before + selected.capacity() * sizeof(void *));
    EXPECT_LT(pool_.used(), before + 32 * 1024);
    ASSERT_EQ(10u, selected.size());
    for (uint32_t i = 0; i < selected.size(); ++i) {
      EXPECT_EQ(&*groups.find(std::to_string(i)), selected[i]);
      EXPECT_EQ(i, selected[i]->second[0].first);
    }
  }
  EXPECT_EQ(0u, pool_.used());
  ASSERT_EQ(0, pool_.init(1));
  GroupTopkHeap<uint32_t, float> heap;
  heap.limit(1000000);
  EXPECT_THROW(heap.emplace(0, 0.0f), std::bad_alloc);
  EXPECT_TRUE(heap.empty());
  EXPECT_EQ(0u, pool_.used());
}
TEST_F(GroupCandidateCollectorTest,
       RetainedNamesAreChargedButLookupsDoNotCopy) {
  ASSERT_EQ(0, pool_.init(1024 * 1024));
  GroupCandidateCollector collector(1, 1);
  const std::string name(8192, 'g');
  ASSERT_EQ(0, collector.add(name, 1, 10));
  EXPECT_GE(pool_.used(), 2 * name.size());
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  // Updating the same winner only replaces its small rank node. Neither the
  // lookup nor the rank ordering should allocate another copy of the name.
  EXPECT_EQ(0, collector.add(name, 2, 1));
  pressure.retain_at_most(0);
  GroupTopkMap<uint64_t, float> groups;
  ASSERT_EQ(0, collector.finish_heaps(groups, 1));
  ASSERT_EQ(1u, groups.size());
  EXPECT_EQ(2u, groups.begin()->second[0].first);
}

TEST_F(GroupCandidateCollectorTest, GroupResultNameOwnershipSurvivesCopyMove) {
  ASSERT_EQ(0, pool_.init(1024 * 1024));
  const std::string name(8192, 'n');
  GroupIndexDocument source(name, {});
  const auto retained = pool_.used();
  EXPECT_GE(retained, name.size());
  GroupIndexDocument copy(source);
  EXPECT_EQ(2 * retained, pool_.used());
  GroupIndexDocument moved(std::move(copy));
  EXPECT_EQ(2 * retained, pool_.used());
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_THROW(source.set_group_id(std::string(16384, 'x')), std::bad_alloc);
  EXPECT_EQ(name, source.group_id());
  pressure.retain_at_most(0);
  source = GroupIndexDocument();
  EXPECT_EQ(retained, pool_.used());
  EXPECT_EQ(name, moved.group_id());
}
}  // namespace
}  // namespace zvec::core

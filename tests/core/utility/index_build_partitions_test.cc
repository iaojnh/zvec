// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <zvec/core/framework/index_build_partitions.h>

#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

class IndexBuildPartitionsTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(GetParam() ? 16 * 1024 * 1024 : 0));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
};

TEST_P(IndexBuildPartitionsTest, ConcurrentAssignmentsProduceOrderedPartitions) {
  IndexBuildPartitions index;
  constexpr size_t kCount = 8193;
  ASSERT_EQ(0, index.init(4, kCount));
  uint32_t id = 0;
  EXPECT_EQ(IndexError_NoReady, index.read(0, 0, &id));
  std::thread odd([&] {
    for (size_t i = kCount - 2; i < kCount; i -= 2)
      EXPECT_EQ(0, index.assign(i, i % 3));
  });
  for (size_t i = 0; i < kCount; i += 2)
    EXPECT_EQ(0, index.assign(i, i % 3));
  odd.join();
  ASSERT_EQ(0, index.finish());
  EXPECT_EQ(0, index.finish());
  for (size_t partition = 0; partition < 3; ++partition) {
    size_t ordinal = partition;
    for (size_t i = 0; i < index.count(partition); ++i, ordinal += 3) {
      ASSERT_EQ(0, index.read(partition, i, &id));
      EXPECT_EQ(ordinal, id);
    }
    EXPECT_GE(ordinal, kCount);
  }
  EXPECT_EQ(0u, index.count(3));
  EXPECT_EQ(IndexError_OutOfRange, index.read(3, 0, &id));
  EXPECT_EQ(IndexError_OutOfRange, index.read(4, 0, &id));
  EXPECT_EQ(IndexError_NoReady, index.assign(0, 0));
  index.clear();
  EXPECT_EQ(0u, index.size());
}

TEST_P(IndexBuildPartitionsTest, IncompleteAssignmentsCannotBePublished) {
  IndexBuildPartitions index;
  ASSERT_EQ(0, index.init(2, 3));
  EXPECT_EQ(IndexError_OutOfRange, index.assign(3, 0));
  EXPECT_EQ(IndexError_OutOfRange, index.assign(0, 2));
  ASSERT_EQ(0, index.assign(0, 1));
  EXPECT_EQ(IndexError_Mismatch, index.finish());
  uint32_t id = 0;
  EXPECT_EQ(IndexError_NoReady, index.read(1, 0, &id));
  ASSERT_EQ(0, index.assign(1, 0));
  ASSERT_EQ(0, index.assign(2, 1));
  ASSERT_EQ(0, index.finish());
  ASSERT_EQ(0, index.read(1, 1, &id));
  EXPECT_EQ(2u, id);
}

TEST_P(IndexBuildPartitionsTest, RejectedReplacementPreservesReadableState) {
  IndexBuildPartitions index;
  ASSERT_EQ(0, index.init(2, 1));
  ASSERT_EQ(0, index.assign(0, 1));
  ASSERT_EQ(0, index.finish());
  EXPECT_NE(0, index.init(0, 1));
  EXPECT_NE(0, index.init(2, SIZE_MAX));
  if (GetParam()) {
    ailego::MemoryReservation pressure;
    pressure.add(pool_.capacity() - pool_.used());
    EXPECT_EQ(IndexError_NoMemory, index.init(2, 1));
  }
  uint32_t id = 1;
  ASSERT_EQ(0, index.read(1, 0, &id));
  EXPECT_EQ(0u, id);
  ASSERT_EQ(0, index.init(3, 0));
  ASSERT_EQ(0, index.finish());
  EXPECT_EQ(3u, index.size());
  EXPECT_EQ(0u, index.count(0));
}

INSTANTIATE_TEST_SUITE_P(StoragePolicies, IndexBuildPartitionsTest,
                         testing::Bool());

}  // namespace
}  // namespace zvec::core

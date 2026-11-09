// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/process_rss_budget.h>

namespace zvec::ailego {
namespace {
constexpr size_t kMiB = 1UL << 20;
size_t rss = 0;
size_t ReadRss() { return rss; }
class ResidentSamplePublishTest : public testing::Test {
 protected:
  void SetUp() override {
    rss = 64 * kMiB;
    ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss));
    ASSERT_TRUE(budget.try_acquire(256 * kMiB));
    rss = 320 * kMiB;
    budget.refresh(true);
    ASSERT_TRUE(budget.try_acquire(32 * kMiB));  // Untouched owner.
    budget.release(128 * kMiB);
    ASSERT_TRUE(budget.try_acquire(128 * kMiB));
    budget.release(8 * kMiB);
    ASSERT_TRUE(budget.try_acquire(8 * kMiB));
    ASSERT_TRUE(budget.pressure());
  }
  ProcessRssBudget budget;
};
TEST_F(ResidentSamplePublishTest, FreshProofRetainsUntouchedAndInFlightDebits) {
  auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(144 * kMiB, budget.available());
  budget.finish_reservation(8 * kMiB, false);
  token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(161 * kMiB));
}
TEST_F(ResidentSamplePublishTest, RetiredWitnessCannotCoverUntouchedReplacement) {
  const auto token = budget.begin_resident_sample();
  budget.release(64 * kMiB);
  budget.account(64 * kMiB);  // A mandatory, still untouched replacement.
  // Total RSS can remain unchanged after the old owner has gone away.
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(97 * kMiB));
}
TEST_F(ResidentSamplePublishTest, CancelledPendingMappingInvalidatesOldProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.account_pending(64 * kMiB));
  budget.finish_reservation(64 * kMiB, false);
  // An address observed during a scan may have been reused and cancelled.
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
}
TEST_F(ResidentSamplePublishTest, UnsampledMappingRetirementStillInvalidatesProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  const auto stamp = budget.finish_reservation(8 * kMiB, true);
  rss = 328 * kMiB;
  budget.release_unmapped(8 * kMiB, stamp);
  // This mapping was absent from the ordinary RSS epoch, but the concurrent
  // ownership scan could have observed it. Unchanged RSS cannot preserve it.
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 264 * kMiB}));
  EXPECT_EQ(152 * kMiB, budget.available());
}
TEST_F(ResidentSamplePublishTest, ReconfigurationAndReadFailuresRejectOldProof) {
  auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.configure(1024 * kMiB, ReadRss));
  EXPECT_FALSE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  token = budget.begin_resident_sample();
  rss = 0;
  EXPECT_FALSE(budget.publish_resident_sample(token, {320 * kMiB, 256 * kMiB}));
  budget.refresh(true);
  budget.refresh(true);
  EXPECT_EQ(0u, budget.available());
  rss = 320 * kMiB;
  budget.refresh(true);
  EXPECT_GT(budget.available(), 0u);
}
}  // namespace
}  // namespace zvec::ailego

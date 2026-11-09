// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/process_rss_budget.h>

namespace zvec::ailego {
namespace {
constexpr size_t kMiB = 1UL << 20;
std::atomic<size_t> sampled_rss{0};
size_t ReadRss() {
  return sampled_rss.load();
}

size_t sampled_owned_resident = 0;
ProcessRssBudget::ResidentSample ReadResident() {
  return {sampled_rss.load(), sampled_owned_resident};
}

ProcessRssBudget::NativeSample native_sample;
bool native_sample_valid = true;
bool ReadNative(ProcessRssBudget::NativeSample *sample) {
  if (!native_sample_valid) return false;
  *sample = native_sample;
  return true;
}

TEST(ProcessRssBudgetTest, NativeFootprintTracksUntouchedBackingAcrossSamples) {
  ProcessRssBudget budget;
  sampled_rss = 32 * kMiB;
  native_sample = {96 * kMiB, 80 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  EXPECT_EQ(24 * kMiB, budget.available());
  ASSERT_TRUE(budget.try_acquire(16 * kMiB));
  for (int i = 0; i < 3; ++i) budget.refresh(true);
  EXPECT_EQ(8 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(9 * kMiB));
  native_sample = {112 * kMiB, 64 * kMiB};
  budget.refresh(true);
  EXPECT_EQ(8 * kMiB, budget.available());
  budget.release(16 * kMiB);
  native_sample = {96 * kMiB, 80 * kMiB};
  budget.refresh(true);
  EXPECT_EQ(24 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, ResidentWitnessDoesNotCoverUntouchedOwners) {
  ProcessRssBudget budget;
  sampled_rss = 64 * kMiB;
  sampled_owned_resident = 0;
  ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_reserve(128 * kMiB));
  budget.finish_reservation(128 * kMiB, true);
  sampled_rss = 192 * kMiB;
  sampled_owned_resident = 128 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_acquire(256 * kMiB));
  budget.release(128 * kMiB);
  // A caller can keep total RSS unchanged after the managed owner retires.
  // The other 256 MiB remains untouched and needs its full future headroom.
  sampled_owned_resident = 0;
  budget.refresh(true);
  EXPECT_EQ(64 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(65 * kMiB));
  budget.release(256 * kMiB);
}

TEST(ProcessRssBudgetTest, HeapFreeKeepsUnsampledAllocatorRetentionCharged) {
  ProcessRssBudget budget;
  sampled_rss = 480 * kMiB;
  ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_reserve(16 * kMiB));
  const auto stamp = budget.finish_reservation(16 * kMiB, true);
  budget.release_heap(16 * kMiB,
                      ProcessRssBudget::HeapProjection::Full(16 * kMiB, stamp));
  EXPECT_EQ(16 * kMiB, budget.available());
  sampled_rss = 496 * kMiB;  // malloc kept the freed pages resident.
  budget.refresh(true);
  EXPECT_EQ(16 * kMiB, budget.available());
  sampled_rss = 480 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(32 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, NativeLimitCanShrinkAndRecoverBelowConfiguredRss) {
  ProcessRssBudget budget;
  sampled_rss = 32 * kMiB;
  native_sample = {32 * kMiB, 96 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(256 * kMiB, ReadRss, ReadNative));
  ASSERT_TRUE(budget.try_acquire(64 * kMiB));
  native_sample.available = 16 * kMiB;
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_acquire(1));
  budget.release(64 * kMiB);
  native_sample.available = 96 * kMiB;
  budget.refresh(true);
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(88 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, NativeZeroAndFailedSamplesCannotReplenishBudget) {
  ProcessRssBudget budget;
  sampled_rss = 32 * kMiB;
  native_sample = {32 * kMiB, 16 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  ASSERT_TRUE(budget.try_acquire(8 * kMiB));
  native_sample_valid = false;
  sampled_rss = 16 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(2 * kMiB, budget.available());
  EXPECT_FALSE(budget.configure(256 * kMiB, ReadRss, ReadNative));
  EXPECT_EQ(128 * kMiB, budget.limit());
  native_sample_valid = true;
  native_sample.available = 0;
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_acquire(1));
  budget.release(8 * kMiB);
}

TEST(ProcessRssBudgetTest, CompressedFootprintCanBlockDespiteLowRss) {
  ProcessRssBudget budget;
  sampled_rss = 16 * kMiB;
  native_sample = {160 * kMiB, 128 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  EXPECT_TRUE(budget.pressure());
  EXPECT_FALSE(budget.try_acquire(1));
}

TEST(ProcessRssBudgetTest, CriticalRecoveryRequiresBothConfiguredSamples) {
  using Pressure = ProcessRssBudget::PressureLevel;
  ProcessRssBudget budget;
  sampled_rss = 16 * kMiB;
  native_sample = {32 * kMiB, 96 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  budget.set_external_pressure(Pressure::kCritical);
  budget.set_external_pressure(Pressure::kBackground);

  native_sample_valid = false;
  EXPECT_FALSE(budget.materialization_allowed());
  EXPECT_FALSE(budget.try_acquire(1));
  native_sample_valid = true;
  sampled_rss = 0;
  budget.refresh(true);
  EXPECT_FALSE(budget.try_acquire(1));
  EXPECT_EQ(0u, budget.available());

  sampled_rss = 24 * kMiB;
  budget.refresh(true);
  EXPECT_TRUE(budget.materialization_allowed());
  EXPECT_TRUE(budget.pressure());  // Background still blocks cache growth.
  ASSERT_TRUE(budget.try_acquire(kMiB));
  budget.release(kMiB);
}

TEST(ProcessRssBudgetTest, DisabledBudgetPreservesStandaloneAdmission) {
  ProcessRssBudget budget;
  EXPECT_TRUE(budget.try_acquire(std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(budget.account_pending(1));
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(0u, budget.limit());
}

TEST(ProcessRssBudgetTest, UntouchedBackingSurvivesSamplesAndLimitChanges) {
  ProcessRssBudget budget;
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  for (int i = 0; i < 3; ++i) budget.refresh(true);
  EXPECT_EQ(100 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(300 * kMiB));
  ASSERT_TRUE(budget.configure(1001 * kMiB, ReadRss));
  EXPECT_EQ(101 * kMiB, budget.available());
  // Failed allocations and untouched mappings return their admission charge.
  budget.release(300 * kMiB);
  EXPECT_EQ(401 * kMiB, budget.available());
  EXPECT_TRUE(budget.try_acquire(300 * kMiB));
}

TEST(ProcessRssBudgetTest, ResidentGrowthIsNotCountedTwice) {
  ProcessRssBudget budget;
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 900 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(100 * kMiB, budget.available());
  budget.release(300 * kMiB);
  // Releasing a logical charge does not assume malloc immediately returns RSS.
  EXPECT_EQ(100 * kMiB, budget.available());
  sampled_rss = 600 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(400 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, ConcurrentAdmissionsShareUnmaterializedHeadroom) {
  ProcessRssBudget budget;
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(700 * kMiB, ReadRss));
  std::atomic<size_t> acquired{0};
  std::vector<std::thread> workers;
  for (int i = 0; i < 12; ++i) {
    workers.emplace_back([&] {
      budget.refresh(true);
      if (budget.try_acquire(10 * kMiB)) acquired.fetch_add(10 * kMiB);
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(100 * kMiB, acquired.load());
  budget.release(acquired.load());
  EXPECT_EQ(100 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest,
     ReservedMaterializationRequiresMeasuredRoomForBytes) {
  ProcessRssBudget budget;
  sampled_rss = 120 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
  EXPECT_TRUE(budget.materialization_allowed(8 * kMiB));
  EXPECT_FALSE(budget.materialization_allowed(8 * kMiB + 1));
  ASSERT_TRUE(budget.try_reserve(2 * kMiB));
  EXPECT_TRUE(budget.materialization_allowed(6 * kMiB));
  EXPECT_FALSE(budget.materialization_allowed(6 * kMiB + 1));
  budget.finish_reservation(2 * kMiB, false);
  EXPECT_TRUE(budget.materialization_allowed(8 * kMiB));

  sampled_rss = 32 * kMiB;
  native_sample = {100 * kMiB, 28 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  EXPECT_TRUE(budget.materialization_allowed(20 * kMiB));
  EXPECT_FALSE(budget.materialization_allowed(20 * kMiB + 1));
}

TEST(ProcessRssBudgetTest, PrepaidMappingDebitSurvivesAnInFlightSample) {
  constexpr size_t limit = 128 * kMiB;
  constexpr size_t bytes = 6 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    budget.account(bytes);
    sampled_rss = (use_native_sample ? 32 : 120) * kMiB;
    native_sample = {112 * kMiB, 16 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));

    ASSERT_TRUE(budget.try_reserve_materialization(bytes));
    budget.refresh(true);
    EXPECT_EQ(2 * kMiB, budget.available());
    EXPECT_FALSE(budget.try_reserve_materialization(3 * kMiB));
    budget.finish_materialization(bytes, false);
    EXPECT_EQ(8 * kMiB, budget.available());

    ASSERT_TRUE(budget.try_reserve_materialization(bytes));
    auto stamp = budget.finish_materialization(bytes, true);
    EXPECT_EQ(2 * kMiB, budget.available());
    budget.release_materialization(bytes, stamp);
    EXPECT_EQ(8 * kMiB, budget.available());

    ASSERT_TRUE(budget.try_reserve_materialization(bytes));
    stamp = budget.finish_materialization(bytes, true);
    budget.refresh(true);
    ASSERT_TRUE(budget.try_materialize(2 * kMiB));
    budget.release_materialization(bytes, stamp);
    EXPECT_EQ(6 * kMiB, budget.available());
    budget.release(bytes);
  }
}

TEST(ProcessRssBudgetTest, ReuseAndNewBackingShareMeasuredRoomBetweenSamples) {
  using Budget = ProcessRssBudget;
  constexpr size_t limit = 128 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    Budget budget;
    sampled_rss = (use_native_sample ? 32 : 120) * kMiB;
    native_sample = {112 * kMiB, 16 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));
    // Keep the limiting observation unchanged even if a worker is delayed.
    if (use_native_sample) {
      native_sample_valid = false;
    } else {
      sampled_rss = 0;
    }
    std::atomic<int> admitted{0};
    auto reuse = [&] {
      if (budget.try_materialize(6 * kMiB)) ++admitted;
    };
    std::thread first(reuse);
    std::thread second(reuse);
    first.join();
    second.join();
    EXPECT_EQ(1, admitted.load());
    EXPECT_EQ(2 * kMiB, budget.available());
    ASSERT_TRUE(budget.try_reserve(2 * kMiB));
    budget.finish_reservation(2 * kMiB, true);
    EXPECT_EQ(0u, budget.available());
    EXPECT_FALSE(budget.try_materialize(1));
    budget.release(2 * kMiB);

    sampled_rss = (use_native_sample ? 32 : 120) * kMiB;
    native_sample_valid = true;
    budget.refresh(true);
    EXPECT_EQ(8 * kMiB, budget.available());
    EXPECT_TRUE(budget.try_materialize(8 * kMiB));
  }
}
}  // namespace
}  // namespace zvec::ailego

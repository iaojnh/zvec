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

TEST(ProcessRssBudgetTest, HostPressurePersistsUntilExplicitRecovery) {
  using Pressure = ProcessRssBudget::PressureLevel;
  ProcessRssBudget budget;
  sampled_rss = 16 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
  budget.set_external_pressure(Pressure::kBackground);
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  ASSERT_TRUE(budget.try_acquire(kMiB));  // Mandatory work may still progress.
  budget.set_external_pressure(Pressure::kCritical);
  budget.refresh(true);
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_acquire(1));
  budget.release(kMiB);
  budget.set_external_pressure(Pressure::kNormal);
  EXPECT_TRUE(budget.try_acquire(kMiB));
  EXPECT_FALSE(budget.pressure());
  budget.release(kMiB);
}

TEST(ProcessRssBudgetTest, HostRecoverySamplesRssBeforeReadmitting) {
  using Pressure = ProcessRssBudget::PressureLevel;
  ProcessRssBudget budget;
  sampled_rss = 16 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
  budget.set_external_pressure(Pressure::kBackground);
  sampled_rss = 128 * kMiB;
  budget.set_external_pressure(Pressure::kNormal);
  // The lifecycle notification must not sample synchronously or publish
  // spendable headroom from the old observation.
  EXPECT_EQ(16 * kMiB, budget.rss());
  EXPECT_TRUE(budget.pressure());
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_acquire(1));
  EXPECT_EQ(128 * kMiB, budget.rss());
  EXPECT_FALSE(budget.materialization_allowed());

  sampled_rss = 24 * kMiB;
  budget.refresh(true);
  EXPECT_FALSE(budget.pressure());
  ASSERT_TRUE(budget.try_acquire(kMiB));
  budget.release(kMiB);
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
  EXPECT_FALSE(budget.try_acquire(1));
  EXPECT_EQ(0u, budget.available());

  sampled_rss = 24 * kMiB;
  EXPECT_TRUE(budget.materialization_allowed());
  EXPECT_TRUE(budget.pressure());  // Background still blocks cache growth.
  ASSERT_TRUE(budget.try_acquire(kMiB));
  budget.release(kMiB);
}

TEST(ProcessRssBudgetTest, DisabledBudgetPreservesStandaloneAdmission) {
  ProcessRssBudget budget;
  EXPECT_TRUE(budget.try_acquire(std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(0u, budget.limit());
}

TEST(ProcessRssBudgetTest, HostGrowthTriggersPressureWithHysteresis) {
  ProcessRssBudget budget;
  sampled_rss = 32 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
  EXPECT_FALSE(budget.pressure());
  sampled_rss = 121 * kMiB;
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  EXPECT_FALSE(budget.try_acquire(8 * kMiB));
  EXPECT_EQ(7 * kMiB, budget.available());
  // Mandatory work can use the reserve while optional cache growth stops.
  EXPECT_TRUE(budget.try_acquire(kMiB));
  EXPECT_EQ(6 * kMiB, budget.available());
  sampled_rss = 117 * kMiB;
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  sampled_rss = 110 * kMiB;
  budget.refresh(true);
  EXPECT_FALSE(budget.pressure());
  EXPECT_TRUE(budget.try_acquire(8 * kMiB));
}

TEST(ProcessRssBudgetTest, FailedSamplesDoNotRestoreSpentHeadroom) {
  ProcessRssBudget budget;
  sampled_rss = 120 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(6 * kMiB));
  sampled_rss = 0;
  budget.refresh(true);
  EXPECT_EQ(2 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(3 * kMiB));
  EXPECT_FALSE(budget.configure(256 * kMiB, ReadRss));
  EXPECT_EQ(128 * kMiB, budget.limit());
  ASSERT_TRUE(budget.configure(0, nullptr));
  EXPECT_TRUE(budget.try_acquire(256 * kMiB));
}

TEST(ProcessRssBudgetTest, HeadroomScalesWithoutExhaustingSmallBudgets) {
  EXPECT_EQ(0u, ProcessRssBudget::headroom(0));
  EXPECT_EQ(2 * kMiB, ProcessRssBudget::headroom(16 * kMiB));
  EXPECT_EQ(8 * kMiB, ProcessRssBudget::headroom(128 * kMiB));
  EXPECT_EQ(64 * kMiB, ProcessRssBudget::headroom(4UL * 1024 * kMiB));
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

TEST(ProcessRssBudgetTest, ReleasedResidentBackingDoesNotCoverUntouchedOwners) {
  constexpr size_t limit = 1000 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    native_sample_valid = true;
    auto sample = [&](size_t bytes) {
      sampled_rss = use_native_sample ? 32 * kMiB : bytes;
      native_sample = {bytes, limit - bytes};
    };
    sample(300 * kMiB);
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));
    ASSERT_TRUE(budget.try_acquire(300 * kMiB));
    sample(600 * kMiB);
    budget.refresh(true);
    ASSERT_TRUE(budget.try_acquire(200 * kMiB));  // Untouched.
    budget.release(300 * kMiB);

    // The sampled memory can still include the freed owner (e.g. the heap
    // retained it). The other owner still needs 200 MiB to become resident.
    const size_t reserve =
        use_native_sample ? ProcessRssBudget::headroom(limit) : 0;
    EXPECT_EQ(200 * kMiB - reserve, budget.available());
    if (use_native_sample) {
      native_sample_valid = false;
    } else {
      sampled_rss = 0;
    }
    budget.refresh(true);
    EXPECT_EQ(200 * kMiB - reserve, budget.available());
    EXPECT_FALSE(budget.try_acquire(201 * kMiB - reserve));

    native_sample_valid = true;
    sample(600 * kMiB);
    for (int i = 0; i < 3; ++i) {
      budget.refresh(true);
      EXPECT_EQ(200 * kMiB - reserve, budget.available());
      EXPECT_FALSE(budget.try_acquire(201 * kMiB - reserve));
    }
    sample(650 * kMiB);  // At most 50 MiB of new backing can be covered.
    budget.refresh(true);
    EXPECT_EQ(200 * kMiB - reserve, budget.available());
    sample(350 * kMiB);
    budget.refresh(true);
    EXPECT_EQ(500 * kMiB - reserve, budget.available());
    budget.release(200 * kMiB);
  }
}

TEST(ProcessRssBudgetTest, ReleasingResidentOwnerDoesNotLosePendingGrowth) {
  ProcessRssBudget budget;
  sampled_rss = 300 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 600 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  budget.release(300 * kMiB);
  budget.refresh(true);  // Resident release has not reached RSS yet.
  sampled_rss = 300 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(400 * kMiB, budget.available());
  budget.release(300 * kMiB);
  EXPECT_EQ(700 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, ReleasedBackingCannotCoverNewBackingBeforeSample) {
  ProcessRssBudget budget;
  sampled_rss = 300 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(500 * kMiB));
  sampled_rss = 600 * kMiB;
  budget.refresh(true);
  // Only part of this reservation was touched. The allocator may retain
  // those pages even after the entire reservation is released.
  budget.release(500 * kMiB);
  budget.account(100 * kMiB);
  EXPECT_EQ(300 * kMiB, budget.available());
  budget.release(100 * kMiB);
}

TEST(ProcessRssBudgetTest, BaselineCanRecoverAfterStartupMemoryRetires) {
  ProcessRssBudget budget;
  budget.account(100 * kMiB);  // Already present when the guard is enabled.
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  budget.release(300 * kMiB);
  budget.release(100 * kMiB);
  sampled_rss = 200 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(800 * kMiB, budget.available());
  ASSERT_TRUE(budget.try_acquire(700 * kMiB));
  budget.refresh(true);
  EXPECT_EQ(100 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, ReplacingStartupBackingDoesNotRebasePendingOwners) {
  ProcessRssBudget budget;
  budget.account(300 * kMiB);
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));  // Still untouched.
  budget.release(300 * kMiB);  // The older resident owner retires.
  sampled_rss = 300 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(400 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(500 * kMiB));
  budget.release(300 * kMiB);
  budget.refresh(true);
  EXPECT_EQ(700 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, SeparatelyBudgetedBackingStillTriggersPressure) {
  ProcessRssBudget budget;
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  budget.account(500 * kMiB);
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  EXPECT_FALSE(budget.try_acquire(1));
  budget.release(500 * kMiB);
  EXPECT_FALSE(budget.pressure());
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

TEST(ProcessRssBudgetTest, StartupHostMemoryCanRetireWithPermanentBacking) {
  ProcessRssBudget budget;
  budget.account(100 * kMiB);
  sampled_rss = 980 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.pressure());
  sampled_rss = 100 * kMiB;
  for (int i = 0; i < 3; ++i) budget.refresh(true);
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(900 * kMiB, budget.available());
  EXPECT_TRUE(budget.try_acquire(64 * kMiB));
}

TEST(ProcessRssBudgetTest, HostRecoveryKeepsUntouchedBackingReserved) {
  ProcessRssBudget budget;
  budget.account(100 * kMiB);
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 200 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(500 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(501 * kMiB));
  budget.release(300 * kMiB);
  budget.refresh(true);
  EXPECT_EQ(800 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, TemporaryHostGrowthDoesNotConsumePendingBacking) {
  ProcessRssBudget budget;
  sampled_rss = 300 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 700 * kMiB;
  budget.refresh(true);
  sampled_rss = 300 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(400 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(401 * kMiB));
}

TEST(ProcessRssBudgetTest, DelayedReleaseDoesNotRefundAnotherPendingOwner) {
  ProcessRssBudget budget;
  budget.account(100 * kMiB);
  sampled_rss = 300 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 600 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  budget.release(300 * kMiB);
  for (int i = 0; i < 3; ++i) budget.refresh(true);
  sampled_rss = 300 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(400 * kMiB, budget.available());
  budget.release(300 * kMiB);
  budget.refresh(true);
  EXPECT_EQ(700 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, AllocationChurnCannotAccumulateReleaseCredits) {
  ProcessRssBudget budget;
  budget.account(10 * kMiB);
  sampled_rss = 980 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  // These short-lived allocations never contribute to an RSS sample. Their
  // cumulative size must not hide a later release of caller-owned memory.
  for (int i = 0; i < 2000; ++i) {
    ASSERT_TRUE(budget.try_acquire(kMiB));
    budget.release(kMiB);
    budget.refresh(true);
  }
  sampled_rss = 100 * kMiB;
  budget.refresh(true);
  EXPECT_FALSE(budget.pressure());
  EXPECT_GE(budget.available(), 890 * kMiB);
  EXPECT_TRUE(budget.try_acquire(64 * kMiB));
}

TEST(ProcessRssBudgetTest, HostRecoveryAfterResidentRetirementKeepsMetadata) {
  ProcessRssBudget budget;
  budget.account(100 * kMiB);
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 900 * kMiB;
  budget.refresh(true);
  budget.release(300 * kMiB);
  sampled_rss = 600 * kMiB;
  budget.refresh(true);
  sampled_rss = 200 * kMiB;
  budget.refresh(true);
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(800 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, ReservedMaterializationObeysObservedLimits) {
  ProcessRssBudget budget;
  sampled_rss = 32 * kMiB;
  native_sample = {64 * kMiB, 64 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  EXPECT_TRUE(budget.materialization_allowed());
  budget.set_external_pressure(ProcessRssBudget::PressureLevel::kBackground);
  EXPECT_TRUE(budget.materialization_allowed());
  budget.set_external_pressure(ProcessRssBudget::PressureLevel::kCritical);
  EXPECT_FALSE(budget.materialization_allowed());
  budget.set_external_pressure(ProcessRssBudget::PressureLevel::kNormal);
  sampled_rss = 128 * kMiB;
  budget.refresh(true);
  EXPECT_FALSE(budget.materialization_allowed());
  sampled_rss = 32 * kMiB;
  native_sample = {120 * kMiB, 0};
  budget.refresh(true);
  EXPECT_FALSE(budget.materialization_allowed());
  native_sample_valid = false;
  budget.refresh(true);
  EXPECT_FALSE(budget.materialization_allowed());
  native_sample_valid = true;
  native_sample = {64 * kMiB, 64 * kMiB};
  budget.refresh(true);
  EXPECT_TRUE(budget.materialization_allowed());
}
}  // namespace
}  // namespace zvec::ailego

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/process_rss_budget.h>

namespace zvec::ailego {
struct ProcessRssBudgetTestPeer {
  static void make_rss_refresh_due(ProcessRssBudget &budget) {
    std::lock_guard<std::mutex> lock(budget.mutex_);
    budget.sampled_at_ -= std::chrono::seconds(1);
  }
  static void expire_scan(ProcessRssBudget &budget) {
    std::lock_guard<std::mutex> lock(budget.mutex_);
    budget.resident_scan_started_ -= std::chrono::seconds(1);
  }
};
namespace {
constexpr size_t kMiB = 1UL << 20;
std::atomic<size_t> sampled_rss{0};
size_t ReadRss() {
  return sampled_rss.load();
}

size_t sampled_owned_resident{0};
ProcessRssBudget::ResidentSample ReadResident() {
  return {ReadRss(), sampled_owned_resident};
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
  // The footprint increase could be host memory while this owner stays lazy.
  // Without an ownership proof it cannot establish managed overlap.
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_acquire(1));
  budget.release(16 * kMiB);
  native_sample = {96 * kMiB, 80 * kMiB};
  budget.refresh(true);
  EXPECT_EQ(24 * kMiB, budget.available());
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

TEST(ProcessRssBudgetTest, UntouchedBackingSurvivesSampleAndOldHeapRetirement) {
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_rss = 100 * kMiB;
    native_sample = {100 * kMiB, 50 * kMiB};
    native_sample_valid = true;
    // A resident heap owner predates configuration. Its lazy replacement
    // remains untouched across the next ordinary observation.
    budget.account(20 * kMiB);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));
    ASSERT_TRUE(budget.try_reserve(20 * kMiB));
    const auto lazy = budget.finish_reservation(20 * kMiB, true);
    const size_t headroom =
        30 * kMiB - (use_native_sample ? budget.headroom(budget.limit()) : 0);
    ASSERT_EQ(headroom - 20 * kMiB, budget.available());
    budget.refresh(true);
    // Neither startup accounting nor a plain process read establishes
    // residency. The still-live 20 MiB stays in the conservative bound.
    EXPECT_EQ(headroom - 20 * kMiB, budget.available());
    // malloc retains both released halves. Neither release nor unchanged
    // samples may spend the replacement's outstanding residency allowance.
    for (size_t i = 0; i < 2; ++i) {
      budget.release_heap(10 * kMiB, {});
      EXPECT_EQ(headroom - (10 - i * 10) * kMiB, budget.available());
    }
    const bool admitted = budget.try_reserve(40 * kMiB);
    EXPECT_FALSE(admitted);
    if (admitted) budget.finish_reservation(40 * kMiB, false);
    budget.refresh(true);
    EXPECT_EQ(headroom, budget.available());
    // Once the old pages really leave the sample, their headroom recovers.
    sampled_rss = 80 * kMiB;
    native_sample = {80 * kMiB, 70 * kMiB};
    budget.refresh(true);
    EXPECT_EQ(headroom + 20 * kMiB, budget.available());
    budget.release_unmapped(20 * kMiB, lazy);
    EXPECT_EQ(headroom + 40 * kMiB, budget.available());
  }
}

TEST(ProcessRssBudgetTest, ReconfigurationPreservesBackingAcrossDisabledGate) {
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_rss = 100 * kMiB;
    native_sample = {100 * kMiB, 50 * kMiB};
    native_sample_valid = true;
    const auto reader = use_native_sample ? ReadNative : nullptr;
    const size_t room =
        50 * kMiB -
        (use_native_sample ? ProcessRssBudget::headroom(150 * kMiB) : 0);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    ASSERT_TRUE(budget.try_reserve(20 * kMiB));
    const auto lazy = budget.finish_reservation(20 * kMiB, true);
    budget.refresh(true);
    ASSERT_EQ(room - 20 * kMiB, budget.available());
    ASSERT_TRUE(budget.configure(0, nullptr));
    // This allocation is made while sampling is disabled. Neither it nor the
    // existing untouched owner may disappear from the resumed RSS allowance.
    ASSERT_TRUE(budget.try_reserve(10 * kMiB));
    const auto offline = budget.finish_reservation(10 * kMiB, true);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room - 30 * kMiB, budget.available());
    const bool admitted = budget.try_reserve(40 * kMiB);
    EXPECT_FALSE(admitted);
    if (admitted) budget.finish_reservation(40 * kMiB, false);
    ASSERT_TRUE(budget.configure(0, nullptr));
    budget.release_unmapped(10 * kMiB, offline);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room - 20 * kMiB, budget.available());
    budget.release_unmapped(20 * kMiB, lazy);
    EXPECT_EQ(room, budget.available());
    // A current-sample projection can also be released while the gate is off.
    ASSERT_TRUE(budget.try_reserve(10 * kMiB));
    const auto pending_sample = budget.finish_reservation(10 * kMiB, true);
    ASSERT_TRUE(budget.configure(0, nullptr));
    budget.release_unmapped(10 * kMiB, pending_sample);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room, budget.available());
    // A new startup allowance remains charged until a real ownership witness
    // proves residency, including after a previously drained ledger.
    ASSERT_TRUE(budget.configure(0, nullptr));
    budget.account(20 * kMiB);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room - 20 * kMiB, budget.available());
    budget.release_heap(20 * kMiB, {});
  }
}

TEST(ProcessRssBudgetTest, EmptyDisabledLedgerRetainsNewBacking) {
  constexpr size_t limit = 256 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    for (const bool permanent : {false, true}) {
      for (const bool complete_offline : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "native=" << use_native_sample << " permanent="
                     << permanent << " complete_offline=" << complete_offline);
        ProcessRssBudget budget;
        sampled_rss = 128 * kMiB;
        native_sample = {128 * kMiB, 128 * kMiB};
        native_sample_valid = true;
        const auto reader = use_native_sample ? ReadNative : nullptr;
        const size_t room =
            128 * kMiB -
            (use_native_sample ? ProcessRssBudget::headroom(limit) : 0);
        if (permanent) budget.account(16 * kMiB);
        ASSERT_TRUE(budget.configure(limit, ReadRss, reader,
                                     permanent ? 16 * kMiB : 0));
        ASSERT_TRUE(budget.configure(0, nullptr));
        // Adopt an already resident owner after disabling an empty ledger.
        // New OS backing must remain distinct from that startup overlap.
        if (!permanent) budget.account(16 * kMiB);
        ASSERT_TRUE(budget.try_reserve(64 * kMiB));
        ProcessRssBudget::MaterializationStamp lazy;
        if (complete_offline) lazy = budget.finish_reservation(64 * kMiB, true);
        ASSERT_TRUE(budget.configure(limit, ReadRss, reader));
        if (!complete_offline)
          lazy = budget.finish_reservation(64 * kMiB, true);
        EXPECT_EQ(room - (permanent ? 64 : 80) * kMiB, budget.available());
        const bool admitted = budget.try_reserve(96 * kMiB);
        EXPECT_FALSE(admitted);
        if (admitted) budget.finish_reservation(96 * kMiB, false);
        if (!permanent) budget.release_heap(16 * kMiB, {});
        EXPECT_EQ(room - 64 * kMiB, budget.available());
        budget.release_unmapped(64 * kMiB, lazy);
        EXPECT_EQ(room, budget.available());
        // Direct lifetime allowances also represent potentially untouched
        // backing, including before this reader's first observation.
        ASSERT_TRUE(budget.configure(0, nullptr));
        ASSERT_TRUE(budget.try_acquire(32 * kMiB));
        ASSERT_TRUE(budget.configure(limit, ReadRss, reader));
        EXPECT_EQ(room - 32 * kMiB, budget.available());
        budget.release(32 * kMiB);
        EXPECT_EQ(room, budget.available());
      }
    }
  }
}

TEST(ProcessRssBudgetTest, PermanentNativeBackingKeepsManagedResidencyCredit) {
  for (const bool background : {false, true}) {
    for (const bool reenable : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "background=" << background << " reenable=" << reenable);
      ProcessRssBudget budget;
      sampled_rss = 100 * kMiB;
      sampled_owned_resident = 0;
      const auto reader = background ? nullptr : ReadResident;
      // Native images/stacks are outside the provider's registered mappings.
      // Retiring heap backing cannot turn their RSS into a managed witness.
      budget.account(60 * kMiB);
      ASSERT_TRUE(
          budget.configure(150 * kMiB, ReadRss, nullptr, 40 * kMiB, reader));
      if (reenable) {
        ASSERT_TRUE(budget.configure(0, nullptr));
        ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, nullptr, 0, reader));
      }
      budget.release_heap(20 * kMiB, {});
      ASSERT_TRUE(budget.try_reserve(20 * kMiB));
      const auto lazy = budget.finish_reservation(20 * kMiB, true);
      budget.refresh(true);
      EXPECT_EQ(30 * kMiB, budget.available());
      for (const size_t managed : {size_t{0}, 20 * kMiB}) {
        sampled_owned_resident = managed;
        if (background) {
          ASSERT_TRUE(budget.publish_resident_sample(
              budget.begin_resident_sample(), ReadResident()));
        } else {
          budget.refresh(true);
          ASSERT_TRUE(
              budget.configure(150 * kMiB, ReadRss, nullptr, 0, reader));
        }
        // A real managed witness supplies its entire credit, independently
        // of the larger permanent native allowance. No witness stays cold.
        EXPECT_EQ((managed ? 50 : 30) * kMiB, budget.available());
        const bool admitted = budget.try_reserve(40 * kMiB);
        EXPECT_EQ(managed != 0, admitted);
        if (admitted) budget.finish_reservation(40 * kMiB, false);
      }
      budget.release_unmapped(20 * kMiB, lazy);
    }
  }
}

TEST(ProcessRssBudgetTest, PrepaidMappingDebitSurvivesAnInFlightSample) {
  constexpr size_t limit = 128 * kMiB;
  constexpr size_t bytes = 6 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    // A 6 MiB planning allowance and its copy are independent charges. Leave
    // 8 MiB after the allowance, without relying on an old startup witness.
    sampled_rss = (use_native_sample ? 32 : 114) * kMiB;
    native_sample = {106 * kMiB, 22 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));

    ASSERT_TRUE(budget.try_acquire(bytes));

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

TEST(ProcessRssBudgetTest, ReusableMappingSpendsItsOwnLifetimeCharge) {
  for (bool native : {false, true}) {
    SCOPED_TRACE(native);
    ProcessRssBudget budget;
    sampled_rss = sampled_owned_resident = 64 * kMiB;
    native_sample = {64 * kMiB, 192 * kMiB};
    native_sample_valid = true;
    budget.account(64 * kMiB);
    ASSERT_TRUE(budget.configure(256 * kMiB, ReadRss,
                                 native ? ReadNative : nullptr, 0,
                                 native ? nullptr : ReadResident));
    const size_t room = budget.available();
    ASSERT_TRUE(budget.try_reserve(room));
    const auto lazy = budget.finish_reservation(room, true);
    ASSERT_EQ(0u, budget.available());
    // The whole old mapping is already in the lifetime bound, including
    // any future refault. Reuse cannot create a second mapping's footprint.
    ProcessRssBudget::MaterializationStamp old;
    EXPECT_TRUE(budget.try_materialize_reusable(64 * kMiB, old));
    EXPECT_EQ(0u, budget.available());
    // Queue slots extend within the same lifetime-charged mapping. They
    // must obey the same boundary, without changing its allocation age.
    auto extension = ProcessRssBudget::MappingProjection::Full(64 * kMiB, old);
    EXPECT_TRUE(budget.try_extend_mapping(kMiB, 64 * kMiB, extension));
    EXPECT_EQ(old.rss_epoch, extension.rss.epoch);
    EXPECT_EQ(old.native_epoch, extension.native.epoch);
    EXPECT_EQ(0u, budget.available());
    // An independent copy still needs additional headroom. Reuse must not
    // spend the unrelated lazy owner's prepaid fault allowance.
    EXPECT_FALSE(budget.try_reserve_materialization(1));
    budget.release_unmapped(room, lazy);
    budget.release_unmapped(64 * kMiB, old);
  }
}

class ResidentSamplePublishTest : public testing::Test {
 protected:
  void SetUp() override {
    sampled_rss = 64 * kMiB;
    ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss));
    ASSERT_TRUE(budget.try_acquire(256 * kMiB));
    sampled_rss = 320 * kMiB;
    budget.refresh(true);
    ASSERT_TRUE(budget.publish_resident_sample(budget.begin_resident_sample(),
                                               {320 * kMiB, 256 * kMiB}));
    ASSERT_TRUE(budget.try_acquire(32 * kMiB));  // Untouched owner.
    budget.release(128 * kMiB);
    ASSERT_TRUE(budget.try_acquire(128 * kMiB));
    budget.release(8 * kMiB);
    ASSERT_TRUE(budget.try_acquire(8 * kMiB));
    ASSERT_TRUE(budget.pressure());
  }
  ProcessRssBudget budget;
};

TEST_F(ResidentSamplePublishTest, PressureChangesExpireInFlightProofs) {
  using Pressure = ProcessRssBudget::PressureLevel;
  const Pressure levels[] = {Pressure::kNormal, Pressure::kBackground,
                             Pressure::kCritical};
  for (const auto from : levels) {
    for (const auto to : levels) {
      SCOPED_TRACE(static_cast<int>(from));
      SCOPED_TRACE(static_cast<int>(to));
      budget.set_external_pressure(from);
      const auto token = budget.begin_resident_sample();
      budget.set_external_pressure(to);
      const size_t room = budget.available();
      EXPECT_EQ(from == to, budget.publish_resident_sample(
                                token, {sampled_rss.load(), 256 * kMiB}));
      if (from != to) { EXPECT_EQ(room, budget.available()); }
      EXPECT_TRUE(budget.publish_resident_sample(
          budget.begin_resident_sample(), {sampled_rss.load(), 256 * kMiB}));
      if (to == Pressure::kCritical) { EXPECT_EQ(0u, budget.available()); }
    }
  }
  budget.set_external_pressure(Pressure::kNormal);
  const auto token = budget.begin_resident_sample();
  budget.set_external_pressure(Pressure::kCritical);
  budget.set_external_pressure(Pressure::kNormal);
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(0u, budget.available());
  budget.refresh(true);
  EXPECT_TRUE(budget.publish_resident_sample(budget.begin_resident_sample(),
                                             {sampled_rss.load(), 256 * kMiB}));
  EXPECT_GT(budget.available(), 0u);
}

TEST_F(ResidentSamplePublishTest, FreshProofRetainsUntouchedAndInFlightDebits) {
  auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(144 * kMiB, budget.available());
  budget.finish_reservation(8 * kMiB, false);
  token = budget.begin_resident_sample();
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(161 * kMiB));
}

TEST_F(ResidentSamplePublishTest, FailedReadsExpirePreFailureProofs) {
  for (const int failures : {1, 3}) {
    SCOPED_TRACE(failures);
    const auto token = budget.begin_resident_sample();
    sampled_rss = 0;
    for (int i = 0; i < failures; ++i) budget.refresh(true);
    if (failures == 3) { EXPECT_EQ(0u, budget.available()); }
    sampled_rss = 320 * kMiB;
    const auto paused_room = budget.available();
    EXPECT_FALSE(budget.publish_resident_sample(
        token, {sampled_rss.load(), 256 * kMiB}));
    EXPECT_EQ(paused_room, budget.available());
    // A reader failure cannot be recovered by an older ownership scan.
    EXPECT_EQ(0u, budget.begin_resident_sample().configuration);
    budget.refresh(true);
    EXPECT_TRUE(budget.publish_resident_sample(
        budget.begin_resident_sample(), {sampled_rss.load(), 256 * kMiB}));
    EXPECT_EQ(160 * kMiB, budget.available());
  }
  // Failure in the ownership provider must also end its cooperative window,
  // even when the budget's process-only reader is independently healthy.
  const auto token = budget.begin_resident_sample();
  EXPECT_FALSE(budget.publish_resident_sample(token, {}));
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(0u, budget.begin_resident_sample().configuration);
  budget.refresh(true);
  EXPECT_TRUE(budget.publish_resident_sample(budget.begin_resident_sample(),
                                             {sampled_rss.load(), 256 * kMiB}));
}

TEST_F(ResidentSamplePublishTest,
       ScanWindowPreservesProgressAndBoundsStaleness) {
  auto token = budget.begin_resident_sample();
  ProcessRssBudgetTestPeer::make_rss_refresh_due(budget);
  const auto before = budget.available();
  budget.refresh();
  EXPECT_EQ(before, budget.available());
  EXPECT_FALSE(budget.try_reserve(32 * kMiB));
  EXPECT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
  // A forced observation immediately wins over the cooperative scan window.
  token = budget.begin_resident_sample();
  budget.refresh(true);
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  token = budget.begin_resident_sample();
  ProcessRssBudgetTestPeer::expire_scan(budget);
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(0u, budget.available());
  // Abandoned scans cannot renew their own grace period or close the gate.
  token = budget.begin_resident_sample();
  ProcessRssBudgetTestPeer::expire_scan(budget);
  const auto fresh = budget.begin_resident_sample();
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_TRUE(
      budget.publish_resident_sample(fresh, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
}

TEST_F(ResidentSamplePublishTest,
       RetiredWitnessCannotCoverUntouchedReplacement) {
  const auto token = budget.begin_resident_sample();
  budget.release(64 * kMiB);
  budget.account(64 * kMiB);  // A mandatory, still untouched replacement.
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(97 * kMiB));
}

TEST_F(ResidentSamplePublishTest, CancelledPendingMappingInvalidatesOldProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.account_pending(64 * kMiB));
  budget.finish_reservation(64 * kMiB, false);
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
}

TEST_F(ResidentSamplePublishTest,
       UnsampledMappingRetirementStillInvalidatesProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  const auto stamp = budget.finish_reservation(8 * kMiB, true);
  sampled_rss = 328 * kMiB;
  budget.release_unmapped(8 * kMiB, stamp);
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 264 * kMiB}));
  EXPECT_EQ(152 * kMiB, budget.available());
}

TEST_F(ResidentSamplePublishTest,
       ReconfigurationAndReadFailuresRejectOldProof) {
  auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.configure(1024 * kMiB, ReadRss));
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  token = budget.begin_resident_sample();
  sampled_rss = 0;
  EXPECT_FALSE(budget.publish_resident_sample(token, {320 * kMiB, 256 * kMiB}));
  budget.refresh(true);
  budget.refresh(true);
  EXPECT_EQ(0u, budget.available());
  sampled_rss = 320 * kMiB;
  budget.refresh(true);
  EXPECT_GT(budget.available(), 0u);
}
TEST(ProcessRssBudgetTest, ObservedResidentPagesKeepCreditAcrossReplacement) {
  ProcessRssBudget budget;
  sampled_rss = 64 * kMiB;
  sampled_owned_resident = 0;
  ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_reserve(200 * kMiB));
  budget.finish_reservation(200 * kMiB, true);
  sampled_rss = 264 * kMiB;
  sampled_owned_resident = 200 * kMiB;
  budget.refresh(true);
  for (int i = 0; i < 64; ++i) {
    SCOPED_TRACE(i);
    // Old pages were discarded, and their replacements are now resident.
    // Total RSS stays constant while the cache turns over.
    budget.release(4 * kMiB);
    ASSERT_TRUE(budget.try_reserve(4 * kMiB));
    budget.finish_reservation(4 * kMiB, true);
    budget.refresh(true);
    ASSERT_EQ(248 * kMiB, budget.available());
    ASSERT_FALSE(budget.pressure());
  }
  budget.release(200 * kMiB);
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

TEST(ProcessRssBudgetTest, ResidentWitnessRetirementPreservesLazyOwnerCharge) {
  ProcessRssBudget budget;
  sampled_rss = 64 * kMiB;
  sampled_owned_resident = 0;
  ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_reserve(200 * kMiB));
  budget.finish_reservation(200 * kMiB, true);
  sampled_rss = 264 * kMiB;
  sampled_owned_resident = 200 * kMiB;
  budget.refresh(true);
  for (int i = 0; i < 64; ++i) {
    budget.release(4 * kMiB);
    ASSERT_TRUE(budget.try_reserve(4 * kMiB));
    budget.finish_reservation(4 * kMiB, true);
    budget.refresh(true);
  }
  ASSERT_TRUE(budget.try_acquire(100 * kMiB));
  budget.release(200 * kMiB);
  sampled_owned_resident = 0;
  // The old sample may retain freed memory while this owner is untouched.
  budget.refresh(true);
  EXPECT_EQ(148 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(149 * kMiB));
  budget.release(100 * kMiB);
}

TEST(ProcessRssBudgetTest, ResidentWitnessCannotCreditPendingOsAllocations) {
  ProcessRssBudget budget;
  sampled_rss = 64 * kMiB;
  sampled_owned_resident = 0;
  ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_reserve(128 * kMiB));
  budget.finish_reservation(128 * kMiB, true);
  sampled_rss = 192 * kMiB;
  sampled_owned_resident = 128 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_reserve(64 * kMiB));
  budget.refresh(true);
  EXPECT_EQ(256 * kMiB, budget.available());
  budget.finish_reservation(64 * kMiB, false);
  EXPECT_EQ(320 * kMiB, budget.available());
  budget.release(128 * kMiB);
}

TEST(ProcessRssBudgetTest, UnsampledHeapChurnPreservesLiveResidentOverlap) {
  constexpr size_t limit = 512 * kMiB;
  for (const bool use_native : {false, true}) {
    SCOPED_TRACE(use_native);
    ProcessRssBudget budget;
    sampled_owned_resident = 0;
    native_sample_valid = true;
    auto sample = [&](size_t bytes) {
      sampled_rss = use_native ? 32 * kMiB : bytes;
      native_sample = {bytes, limit - bytes};
    };
    sample(64 * kMiB);
    ASSERT_TRUE(
        budget.configure(limit, ReadRss, use_native ? ReadNative : nullptr, 0, use_native ? nullptr : ReadResident));
    ASSERT_TRUE(budget.try_reserve(230 * kMiB));
    budget.finish_reservation(230 * kMiB, true);
    sample(294 * kMiB);
    sampled_owned_resident = 230 * kMiB;
    budget.refresh(true);
    if (use_native) {
      EXPECT_FALSE(budget.try_reserve(4 * kMiB));
      EXPECT_EQ(0u, budget.available());
      budget.release(230 * kMiB);
      continue;
    }
    const size_t available = budget.available();
    for (int i = 0; i < 64; ++i) {
      SCOPED_TRACE(i);
      ASSERT_TRUE(budget.try_reserve(4 * kMiB));
      const auto stamp = budget.finish_reservation(4 * kMiB, true);
      budget.release_heap(
          4 * kMiB, ProcessRssBudget::HeapProjection::Full(4 * kMiB, stamp));
      budget.refresh(true);
      ASSERT_EQ(available, budget.available());
      ASSERT_FALSE(budget.pressure());
    }
    budget.release(230 * kMiB);
  }
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

TEST(ProcessRssBudgetTest, MixedAgeHeapReleaseKeepsUntouchedOwnersCharged) {
  using Projection = ProcessRssBudget::HeapProjection;
  ProcessRssBudget budget;
  sampled_owned_resident = 8 * kMiB;
  budget.account(8 * kMiB);
  sampled_rss = 96 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_reserve(4 * kMiB));
  auto projection =
      Projection::Full(4 * kMiB, budget.finish_reservation(4 * kMiB, true));
  sampled_rss = 100 * kMiB;
  sampled_owned_resident = 12 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_reserve(6 * kMiB));
  projection.append(
      Projection::Full(6 * kMiB, budget.finish_reservation(6 * kMiB, true)));

  budget.release_heap(4 * kMiB, projection.take(10 * kMiB, 4 * kMiB));
  sampled_owned_resident = 8 * kMiB;
  budget.refresh(true);
  // The old 4 MiB can still be resident, and the live 6 MiB is untouched.
  EXPECT_EQ(22 * kMiB, budget.available());
  budget.release_heap(6 * kMiB, projection);
  budget.refresh(true);
  // After another sample, the 6 MiB may itself be retained in the heap.
  // Do not promise its refund while another live owner remains untouched.
  EXPECT_LE(budget.available(), 28 * kMiB);
  budget.release(8 * kMiB);
  sampled_rss = 96 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(32 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, HeapAgeUsesEachReadersSuccessfulSample) {
  ProcessRssBudget budget;
  budget.account(8 * kMiB);
  sampled_rss = 64 * kMiB;
  native_sample = {96 * kMiB, 32 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  ASSERT_EQ(16 * kMiB, budget.available());
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  const auto stamp = budget.finish_reservation(8 * kMiB, true);
  sampled_rss = 72 * kMiB;
  native_sample_valid = false;
  budget.refresh(true);
  budget.release_heap(
      8 * kMiB, ProcessRssBudget::HeapProjection::Full(8 * kMiB, stamp));
  EXPECT_EQ(8 * kMiB, budget.available());
  sampled_rss = 64 * kMiB;
  native_sample_valid = true;
  budget.refresh(true);
  EXPECT_EQ(16 * kMiB, budget.available());
  budget.release(8 * kMiB);
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

TEST(ProcessRssBudgetTest, AdmissionResamplesReleasedCallerInputBeforeDenying) {
  ProcessRssBudget budget;
  sampled_rss = 128 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
  EXPECT_FALSE(budget.try_acquire(kMiB));
  sampled_rss = 32 * kMiB;
  ASSERT_TRUE(budget.try_acquire(kMiB));
  EXPECT_EQ(32 * kMiB, budget.rss());
  budget.release(kMiB);
}

TEST(ProcessRssBudgetTest,
     MaterializationResamplesCompletedCopiesButKeepsPending) {
  for (const bool reserve : {false, true}) {
    SCOPED_TRACE(reserve);
    ProcessRssBudget budget;
    sampled_rss = 32 * kMiB;
    ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
    ASSERT_TRUE(budget.try_reserve_materialization(80 * kMiB));
    EXPECT_FALSE(budget.try_materialize(32 * kMiB));
    budget.finish_materialization(80 * kMiB, true);
    // The completed copies may have reused heap pages or been destroyed.
    // A fresh sample replaces their projection without a timing dependency.
    if (reserve) {
      ASSERT_TRUE(budget.try_reserve_materialization(32 * kMiB));
      budget.finish_materialization(32 * kMiB, false);
    } else {
      ASSERT_TRUE(budget.try_materialize(32 * kMiB));
    }
  }
}

TEST(ProcessRssBudgetTest, RejectedCopyGrowthKeepsOtherOwnersResident) {
  ProcessRssBudget budget;
  sampled_owned_resident = 16 * kMiB;
  budget.account(16 * kMiB);
  sampled_rss = 96 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, nullptr, 0, ReadResident));
  for (int attempt = 0; attempt < 20; ++attempt) {
    EXPECT_FALSE(budget.try_acquire_with_pending_copy(4 * kMiB, 33 * kMiB));
    budget.refresh(true);
    EXPECT_EQ(32 * kMiB, budget.available());
  }
  budget.release(16 * kMiB);
}

TEST(ProcessRssBudgetTest,
     CopyGrowthSharesBackingAndKeepsPendingAcrossSamples) {
  ProcessRssBudget budget;
  sampled_rss = 112 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));
  // A separate copy cannot overlap an unrelated planning allowance.
  EXPECT_FALSE(budget.try_acquire_with_pending_copy(8 * kMiB, 12 * kMiB));
  ASSERT_TRUE(budget.try_acquire_with_pending_copy(8 * kMiB, 8 * kMiB));
  EXPECT_EQ(0u, budget.available());
  budget.refresh(true);
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_reserve_materialization(1));
  budget.finish_materialization(8 * kMiB, false);
  EXPECT_EQ(8 * kMiB, budget.available());
  budget.release(8 * kMiB);
  EXPECT_EQ(16 * kMiB, budget.available());
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

TEST(ProcessRssBudgetTest, DisabledBudgetPreservesStandaloneAdmission) {
  ProcessRssBudget budget;
  EXPECT_TRUE(budget.try_acquire(std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(budget.account_pending(1));
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

TEST(ProcessRssBudgetTest, RepeatedFailurePausesAdmissionUntilReaderRecovers) {
  for (const bool fail_native : {false, true}) {
    SCOPED_TRACE(fail_native);
    ProcessRssBudget budget;
    sampled_rss = 32 * kMiB;
    native_sample = {32 * kMiB, 96 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss,
                                 fail_native ? ReadNative : nullptr));
    ASSERT_TRUE(budget.try_acquire(8 * kMiB));
    if (fail_native)
      native_sample_valid = false;
    else
      sampled_rss = 0;

    budget.refresh(true);
    EXPECT_GT(budget.available(), 0u);
    for (int i = 0; i < 4; ++i) budget.refresh(true);
    EXPECT_EQ(0u, budget.available());
    EXPECT_TRUE(budget.pressure());
    EXPECT_FALSE(budget.try_reserve(1));
    EXPECT_FALSE(budget.try_materialize(1));
    budget.release(8 * kMiB);
    EXPECT_EQ(0u, budget.available());

    sampled_rss = 32 * kMiB;
    native_sample_valid = true;
    budget.refresh(true);
    EXPECT_FALSE(budget.pressure());
    ASSERT_TRUE(budget.try_acquire(kMiB));
    budget.release(kMiB);
  }
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
  sampled_owned_resident = 0 * kMiB;
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 900 * kMiB;
  sampled_owned_resident = 300 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(100 * kMiB, budget.available());
  budget.release(300 * kMiB);
  // Releasing a logical charge does not assume malloc immediately returns RSS.
  EXPECT_EQ(100 * kMiB, budget.available());
  sampled_rss = 600 * kMiB;
  sampled_owned_resident = 0;
  budget.refresh(true);
  EXPECT_EQ(400 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, ReleasedResidentBackingDoesNotCoverUntouchedOwners) {
  constexpr size_t limit = 1000 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_owned_resident = 0;
    native_sample_valid = true;
    auto sample = [&](size_t bytes) {
      sampled_rss = use_native_sample ? 32 * kMiB : bytes;
      native_sample = {bytes, limit - bytes};
    };
    sample(300 * kMiB);
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr, 0, use_native_sample ? nullptr : ReadResident));
    ASSERT_TRUE(budget.try_acquire(300 * kMiB));
    sample(600 * kMiB);
    sampled_owned_resident = 300 * kMiB;
    budget.refresh(true);
    if (use_native_sample) {
      EXPECT_FALSE(budget.try_acquire(200 * kMiB));
      budget.release(300 * kMiB);
      continue;
    }
    ASSERT_TRUE(budget.try_acquire(200 * kMiB));  // Untouched.
    budget.release(300 * kMiB);
    sampled_owned_resident = 0;

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
    sample(650 * kMiB);  // Host growth supplies no new ownership proof.
    budget.refresh(true);
    EXPECT_EQ(150 * kMiB - reserve, budget.available());
    sample(350 * kMiB);
    budget.refresh(true);
    EXPECT_EQ(450 * kMiB - reserve, budget.available());
    budget.release(200 * kMiB);
  }
}

TEST(ProcessRssBudgetTest, ReleasingResidentOwnerDoesNotLosePendingGrowth) {
  ProcessRssBudget budget;
  sampled_owned_resident = 0 * kMiB;
  sampled_rss = 300 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 600 * kMiB;
  sampled_owned_resident = 300 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  budget.release(300 * kMiB);
  sampled_owned_resident = 0;
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

TEST(ProcessRssBudgetTest, FailedAllocationAttemptsKeepResidentOverlap) {
  constexpr size_t limit = 1000 * kMiB;
  constexpr size_t request = 12 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_owned_resident = 0;
    native_sample_valid = true;
    auto sample = [&](size_t bytes) {
      sampled_rss = use_native_sample ? 32 * kMiB : bytes;
      native_sample = {bytes, limit - bytes};
    };
    sample(600 * kMiB);
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr, 0, use_native_sample ? nullptr : ReadResident));
    ASSERT_TRUE(budget.try_acquire(300 * kMiB));
    sample(900 * kMiB);
    sampled_owned_resident = 300 * kMiB;
    budget.refresh(true);
    if (use_native_sample) {
      EXPECT_FALSE(budget.try_reserve(request));
      budget.release(300 * kMiB);
      continue;
    }
    const size_t available =
        100 * kMiB -
        (use_native_sample ? ProcessRssBudget::headroom(limit) : 0);
    for (int i = 0; i < 20; ++i) {
      if (i % 2 == 0) {
        ASSERT_TRUE(budget.try_reserve(request));
      } else {
        // Separately admitted RocksDB block.
        ASSERT_TRUE(budget.account_pending(request));
      }
      budget.refresh(true);
      EXPECT_EQ(available - request, budget.available());
      budget.finish_reservation(request, false);
      EXPECT_EQ(available, budget.available());
    }
    ASSERT_TRUE(budget.try_reserve(request));
    budget.finish_reservation(request, true);
    EXPECT_EQ(available - request, budget.available());
    sample(900 * kMiB + request);
    sampled_owned_resident = 300 * kMiB + request;
    budget.refresh(true);
    budget.release(request);
    sampled_owned_resident = 300 * kMiB;
    sample(900 * kMiB);
    budget.refresh(true);
    EXPECT_EQ(available, budget.available());
    budget.release(300 * kMiB);
  }
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

TEST(ProcessRssBudgetTest, TemporaryFreesKeepPermanentStartupOverlap) {
  constexpr size_t limit = 128 * kMiB;
  constexpr size_t permanent = 56 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    budget.account(permanent);
    native_sample_valid = true;
    auto sample = [&](size_t bytes) {
      sampled_rss = use_native_sample ? 32 * kMiB : bytes;
      native_sample = {bytes, limit - bytes};
      budget.refresh(true);
    };
    sample(64 * kMiB);
    ASSERT_TRUE(budget.configure(
        limit, ReadRss, use_native_sample ? ReadNative : nullptr, permanent));
    ASSERT_TRUE(budget.try_acquire(kMiB));
    sample(96 * kMiB);
    budget.release(kMiB);
    for (size_t i = 0; i < 64; ++i) {
      sample(96 * kMiB);
      ASSERT_TRUE(budget.try_acquire(kMiB));
      budget.release(kMiB);
    }

    const size_t reserve =
        use_native_sample ? ProcessRssBudget::headroom(limit) : 0;
    EXPECT_EQ(32 * kMiB - reserve, budget.available());
    ASSERT_TRUE(budget.try_acquire(16 * kMiB));
    // Until sampled, a live new owner still consumes its full reservation.
    EXPECT_EQ(16 * kMiB - reserve, budget.available());
    budget.release(16 * kMiB);
  }
}

TEST(ProcessRssBudgetTest, ReplacingStartupBackingDoesNotRebasePendingOwners) {
  ProcessRssBudget budget;
  sampled_owned_resident = 300 * kMiB;
  budget.account(300 * kMiB);
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));  // Still untouched.
  budget.release(300 * kMiB);  // The older resident owner retires.
  sampled_rss = 300 * kMiB;
  sampled_owned_resident = 0;
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

TEST(ProcessRssBudgetTest, StartupHostMemoryCanRetireWithPermanentBacking) {
  ProcessRssBudget budget;
  budget.account(100 * kMiB);
  sampled_rss = 980 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss, nullptr, 100 * kMiB));
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
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss, nullptr, 100 * kMiB));
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
  sampled_owned_resident = 0 * kMiB;
  budget.account(100 * kMiB);
  sampled_rss = 300 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss, nullptr, 100 * kMiB, ReadResident));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 600 * kMiB;
  sampled_owned_resident = 300 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  budget.release(300 * kMiB);
  sampled_owned_resident = 0;
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
  sampled_owned_resident = 0 * kMiB;
  budget.account(100 * kMiB);
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(1000 * kMiB, ReadRss, nullptr, 100 * kMiB, ReadResident));
  ASSERT_TRUE(budget.try_acquire(300 * kMiB));
  sampled_rss = 900 * kMiB;
  sampled_owned_resident = 300 * kMiB;
  budget.refresh(true);
  budget.release(300 * kMiB);
  sampled_owned_resident = 0;
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

TEST(ProcessRssBudgetTest, GrowingAllowanceChargesFullMaterialization) {
  constexpr size_t limit = 128 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_owned_resident = 0;
    native_sample_valid = true;
    const size_t reserve =
        use_native_sample ? ProcessRssBudget::headroom(limit) : 0;
    sampled_rss = use_native_sample ? 32 * kMiB : limit - 8 * kMiB;
    native_sample = {limit - 8 * kMiB - reserve, 8 * kMiB + reserve};
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr, 0, use_native_sample ? nullptr : ReadResident));

    ASSERT_TRUE(budget.try_acquire(2 * kMiB, 6 * kMiB));
    // A planning allowance is not an OS birth proving overlap with the copy.
    EXPECT_EQ(0u, budget.available());
    // A denial now refreshes the observation even within the sample interval.
    // Keep the completed materialization resident in that observation.
    sampled_rss += 6 * kMiB;
    sampled_owned_resident = 2 * kMiB;
    native_sample.footprint += 6 * kMiB;
    native_sample.available -= 6 * kMiB;
    budget.refresh(true);
    EXPECT_FALSE(budget.try_acquire(kMiB, 3 * kMiB));
    EXPECT_FALSE(budget.try_materialize(3 * kMiB));
    if (use_native_sample) {
      EXPECT_EQ(0u, budget.available());
      EXPECT_FALSE(budget.try_materialize(2 * kMiB));
    } else {
      EXPECT_EQ(2 * kMiB, budget.available());
      ASSERT_TRUE(budget.try_materialize(2 * kMiB));
    }
    budget.release(2 * kMiB);

    sampled_rss = use_native_sample ? 32 * kMiB : limit - 8 * kMiB;
    native_sample = {limit - 8 * kMiB - reserve, 8 * kMiB + reserve};
    budget.refresh(true);
    EXPECT_EQ(8 * kMiB, budget.available());
  }
}

TEST(ProcessRssBudgetTest, ReusableBackingRefreshesEachReaderIndependently) {
  constexpr size_t bytes = 6 * kMiB;
  ProcessRssBudget budget;
  budget.account(bytes);
  sampled_rss = 112 * kMiB;
  native_sample = {104 * kMiB, 24 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  ProcessRssBudget::MaterializationStamp stamp{};
  ASSERT_TRUE(budget.try_materialize_reusable(bytes, stamp));
  EXPECT_EQ(10 * kMiB, budget.available());
  ASSERT_TRUE(budget.try_materialize_reusable(bytes, stamp));
  EXPECT_EQ(10 * kMiB, budget.available());

  native_sample_valid = false;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_materialize_reusable(bytes, stamp));
  EXPECT_EQ(10 * kMiB, budget.available());
  budget.release_projected_mapping(
      bytes, ProcessRssBudget::MappingProjection::ForReuse(bytes, stamp, {}));
  EXPECT_EQ(16 * kMiB, budget.available());
  native_sample_valid = true;
}

TEST(ProcessRssBudgetTest, CoveredMappingsCanReuseTheLastAvailableByte) {
  using Budget = ProcessRssBudget;
  constexpr size_t limit = 128 * kMiB;
  constexpr size_t bytes = 8 * kMiB;
  for (bool native : {false, true}) {
    SCOPED_TRACE(native);
    Budget budget;
    sampled_rss = (native ? 32 : 120) * kMiB;
    native_sample = {112 * kMiB, 16 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(
        budget.configure(limit, ReadRss, native ? ReadNative : nullptr));
    ASSERT_TRUE(budget.try_reserve(bytes));
    auto stamp = budget.finish_reservation(bytes, true);
    auto projection = Budget::MappingProjection::Full(bytes, stamp);
    EXPECT_EQ(0u, budget.available());
    EXPECT_FALSE(budget.materialization_allowed());
    EXPECT_TRUE(budget.try_materialize_reusable(bytes, stamp));
    EXPECT_TRUE(budget.try_extend_mapping(1, bytes, projection));
    // Reuse is called only for backing owned by the caller. An epoch-zero
    // live mapping can predate configuration; it is not a new allocation.
    EXPECT_FALSE(budget.try_reserve(1));
    EXPECT_FALSE(budget.try_materialize(1));
    budget.set_external_pressure(Budget::PressureLevel::kCritical);
    EXPECT_FALSE(budget.try_materialize_reusable(bytes, stamp));
    budget.release_projected_mapping(bytes, projection);
  }
}

TEST(ProcessRssBudgetTest, PartialMappingRefundsEachReadersProjection) {
  constexpr size_t bytes = 8 * kMiB;
  ProcessRssBudget budget;
  sampled_rss = 112 * kMiB;
  native_sample = {104 * kMiB, 24 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  ASSERT_TRUE(budget.try_reserve(bytes));
  auto projection = ProcessRssBudget::MappingProjection::Full(
      bytes, budget.finish_reservation(bytes, true));
  ASSERT_TRUE(budget.try_materialize(3 * kMiB));
  ASSERT_TRUE(budget.try_extend_mapping(2 * kMiB, bytes, projection));
  EXPECT_EQ(5 * kMiB, budget.available());

  native_sample_valid = false;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_extend_mapping(2 * kMiB, bytes, projection));
  EXPECT_EQ(5 * kMiB, budget.available());
  budget.release_projected_mapping(bytes, projection);
  EXPECT_EQ(13 * kMiB, budget.available());
  native_sample_valid = true;
}

TEST(ProcessRssBudgetTest, ReuseDoesNotHideResidentBackingFromRelease) {
  constexpr size_t bytes = 8 * kMiB;
  ProcessRssBudget budget;
  budget.account(bytes);
  sampled_rss = 120 * kMiB;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss));

  ProcessRssBudget::MaterializationStamp reuse{};
  ASSERT_TRUE(budget.try_materialize_reusable(bytes, reuse));
  budget.release_projected_mapping(
      bytes, ProcessRssBudget::MappingProjection::ForReuse(bytes, reuse, {}));
  ASSERT_TRUE(budget.try_acquire(bytes));
  budget.refresh(true);
  EXPECT_EQ(0u, budget.available());
  sampled_rss = 112 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(bytes, budget.available());
  budget.release(bytes);
}

TEST(ProcessRssBudgetTest, UnmappedBackingRefundsOnlyItsOwnProjection) {
  constexpr size_t kLimit = 128 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    const size_t initial_rss = (use_native_sample ? 32 : 120) * kMiB;
    sampled_rss = initial_rss;
    native_sample = {112 * kMiB, 16 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(kLimit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));
    // Keep the limiting sample stable across a scheduler delay.
    auto fail_limiting_reader = [&] {
      if (use_native_sample) {
        native_sample_valid = false;
      } else {
        sampled_rss = 0;
      }
    };
    fail_limiting_reader();

    ASSERT_TRUE(budget.try_materialize(2 * kMiB));
    ASSERT_TRUE(budget.account_pending(6 * kMiB));
    auto stamp = budget.finish_reservation(6 * kMiB, true);
    EXPECT_FALSE(budget.try_materialize(1));
    budget.release_unmapped(6 * kMiB, stamp);
    EXPECT_EQ(6 * kMiB, budget.available());
    EXPECT_FALSE(budget.try_materialize(7 * kMiB));

    ASSERT_TRUE(budget.account_pending(3 * kMiB));
    stamp = budget.finish_reservation(3 * kMiB, true);
    sampled_rss = initial_rss;
    native_sample_valid = true;
    budget.refresh(true);
    fail_limiting_reader();

    ASSERT_TRUE(budget.try_materialize(5 * kMiB));
    budget.release_unmapped(3 * kMiB, stamp);
    EXPECT_EQ(3 * kMiB, budget.available());
    EXPECT_FALSE(budget.try_materialize(4 * kMiB));
  }
}

TEST(ProcessRssBudgetTest, UnsampledTailsKeepOtherOwnersResident) {
  constexpr size_t kLimit = 128 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_owned_resident = 6 * kMiB;
    budget.account(6 * kMiB);
    sampled_rss = (use_native_sample ? 32 : 120) * kMiB;
    native_sample = {112 * kMiB, 16 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(kLimit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr, 0, use_native_sample ? nullptr : ReadResident));
    if (use_native_sample) {
      EXPECT_EQ(2 * kMiB, budget.available());
      EXPECT_FALSE(budget.try_reserve(6 * kMiB));
      budget.release(6 * kMiB);
      continue;
    } else {
      sampled_rss = 0;
    }

    ASSERT_TRUE(budget.try_reserve(6 * kMiB));
    const auto stamp = budget.finish_reservation(6 * kMiB, true);
    EXPECT_EQ(2 * kMiB, budget.available());
    budget.release_unmapped(4 * kMiB, stamp);
    EXPECT_EQ(6 * kMiB, budget.available());
    budget.release_unmapped(2 * kMiB, stamp);
    EXPECT_EQ(8 * kMiB, budget.available());
    budget.release(6 * kMiB);
  }
}

}  // namespace
}  // namespace zvec::ailego

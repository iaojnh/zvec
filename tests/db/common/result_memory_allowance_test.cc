// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/common/result_memory_allowance.h"
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include "db/common/query_result_memory.h"
#include "db/index/column/fts_column/fts_memory.h"
#include "db/index/column/inverted_column/inverted_search_result.h"
#include "db/index/common/doc_field_converter.h"
#include "../../ailego/buffer/pool_payload_budget.h"
#include "binding/c/c_memory.h"
#include "db/common/metadata_memory.h"

namespace zvec::ailego {

struct MemoryLimitPoolTestPeer {
  static bool configure(MemoryLimitPool &pool, size_t limit, size_t rss) {
    sample_ = rss;
    return pool.process_rss_.configure(limit, +[] { return sample_.load(); });
  }

  static void sample(MemoryLimitPool &pool, size_t rss) {
    sample_ = rss;
    pool.process_rss_.refresh(true);
  }

  static bool workspace_waiting(MemoryLimitPool &pool) {
    return pool.workspace_waiting_size_.load(std::memory_order_relaxed) != 0;
  }

  inline static std::atomic<size_t> sample_{0};
};

}  // namespace zvec::ailego

namespace zvec::detail {
namespace {

constexpr size_t kMiB = 1024 * 1024;

class ResultMemoryAllowanceTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(64 * kMiB));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
};

TEST_F(ResultMemoryAllowanceTest, RetainsAndReusesBackingAfterCallsFinish) {
  ResultMemoryAllowance allowance;
  for (int i = 0; i < 3; ++i) {
    {
      ResultMemoryAllowance::Lease lease(&allowance);
      lease.add(2 * kMiB);
      lease.add(kMiB);
      EXPECT_EQ(3 * kMiB, allowance.active_bytes());
      EXPECT_EQ(6 * kMiB, allowance.reserved_bytes());
    }
    EXPECT_EQ(0u, allowance.active_bytes());
    EXPECT_EQ(6 * kMiB, pool_.external_used());
  }
}

TEST_F(ResultMemoryAllowanceTest, PendingCopyRetainsHeadroomAcrossSamples) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 64 * kMiB, 16 * kMiB));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease warmup(&allowance);
    ASSERT_NO_THROW(warmup.add(8 * kMiB));
  }
  ASSERT_EQ(16 * kMiB, allowance.reserved_bytes());
  Peer::sample(native, 57 * kMiB);
  ResultMemoryAllowance::Lease first(&allowance), next(&allowance);
  {
    ResultMemoryAllowance::PendingCopy pending;
    ASSERT_NO_THROW(first.add_pending(4 * kMiB, pending));
    Peer::sample(native, 57 * kMiB);
    EXPECT_THROW(next.add(4 * kMiB), std::bad_alloc);
    EXPECT_EQ(4 * kMiB, allowance.active_bytes());
    // The producer abandons the copy before touching the destination.
  }
  EXPECT_NO_THROW(next.add(4 * kMiB));
}

TEST_F(ResultMemoryAllowanceTest, RejectedPendingCopyLeavesBackingUnchanged) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  ASSERT_TRUE(
      ailego::MemoryLimitPoolTestPeer::configure(native, 64 * kMiB, 60 * kMiB));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance);
  ResultMemoryAllowance::PendingCopy pending;
  EXPECT_THROW(lease.add_pending(kMiB, 5 * kMiB, pending), std::bad_alloc);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
  EXPECT_NO_THROW(lease.add_pending(kMiB, pending));
}

TEST_F(ResultMemoryAllowanceTest,
       CopyPlanSlackDoesNotBecomeUntouchedRssBacking) {
  ASSERT_EQ(0, pool_.init(128 * kMiB));
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 128 * kMiB, 124 * kMiB));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease copy(&allowance, false, kMiB);
    ResultMemoryAllowance::PendingCopy pending;
    ASSERT_NO_THROW(copy.add_pending(12 * kMiB, kMiB, pending));
    EXPECT_EQ(24 * kMiB, allowance.reserved_bytes());
    Peer::sample(native, 124 * kMiB);
    EXPECT_FALSE(native.try_materialize(4 * kMiB));
    pending.start();
  }
  Peer::sample(native, 125 * kMiB);
  EXPECT_EQ(0u, allowance.reserved_bytes());
  {
    // Work without a scoped copy still reserves its possibly untouched
    // physical backing, independently of the existing logical copy plan.
    ResultMemoryAllowance::Lease unscoped(&allowance, false, kMiB);
    EXPECT_THROW(unscoped.add(12 * kMiB, kMiB), std::bad_alloc);
    EXPECT_EQ(0u, allowance.active_bytes());
    EXPECT_EQ(0u, allowance.reserved_bytes());
  }
  {
    ResultMemoryAllowance::Lease retry(&allowance, false, kMiB);
    ResultMemoryAllowance::PendingCopy pending;
    EXPECT_NO_THROW(retry.add_pending(12 * kMiB, kMiB, pending));
  }
}

TEST_F(ResultMemoryAllowanceTest,
       TinyCopyDoesNotRequireAnUntouchedPythonArena) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 64 * kMiB, 64 * kMiB - kMiB / 2));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease copy(&allowance, false, kMiB);
  ResultMemoryAllowance::PendingCopy pending;
  ASSERT_NO_THROW(copy.add_pending(kMiB, kMiB / 8, pending));
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
  EXPECT_FALSE(native.try_charge_process_backing(kMiB / 2, 0, false));
}

TEST_F(ResultMemoryAllowanceTest, TinyNativeWorkDoesNotOwnArenaSlack) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 64 * kMiB, 64 * kMiB - kMiB / 2));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease work(&allowance, false, kMiB);
  ASSERT_NO_THROW(work.add(kMiB / 8));
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
  EXPECT_FALSE(native.try_charge_process_backing(kMiB / 2, 0, false));
}

TEST_F(ResultMemoryAllowanceTest, CopyPeakAndPayloadHaveSeparateLifetimes) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 64 * kMiB, 60 * kMiB));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease copy(&allowance, false, kMiB);
  ResultMemoryAllowance::PendingCopy pending;
  ASSERT_NO_THROW(copy.add_pending(kMiB, 4 * kMiB, kMiB, pending));
  Peer::sample(native, 60 * kMiB);
  EXPECT_FALSE(native.try_materialize(1));
  pending.start();
  pending.finish();
  Peer::sample(native, 61 * kMiB);
  const size_t extra = 5 * kMiB / 2;
  ASSERT_TRUE(native.try_charge_process_backing(extra, 0, false));
  native.release_process_backing(extra);
}

TEST_F(ResultMemoryAllowanceTest, TrimmingCopyOnlyLeaseReleasesItsProjection) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 64 * kMiB, 63 * kMiB));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease copy(&allowance);
  ResultMemoryAllowance::PendingCopy pending;
  ASSERT_NO_THROW(copy.add_pending(0, kMiB, pending));
  copy.retain_at_most(0);
  pending.finish();
  Peer::sample(native, 64 * kMiB);
  ResultMemoryAllowance::Lease planning(&allowance);
  ResultMemoryAllowance::PendingCopy no_copy;
  EXPECT_NO_THROW(planning.add_pending(kMiB, 0, no_copy));
}

TEST_F(ResultMemoryAllowanceTest, ReclaimsUnusedHighWaterWithLiveSmallLease) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease warmup(&allowance, false, kMiB);
    warmup.add(6 * kMiB);
  }
  ResultMemoryAllowance::Lease live(&allowance, false, kMiB);
  live.add(1);
  ASSERT_EQ(12 * kMiB, allowance.reserved_bytes());
  allowance.reclaim_idle();
  EXPECT_EQ(1u, allowance.active_bytes());
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
  EXPECT_EQ(2 * kMiB, pool_.external_used());
  EXPECT_NO_THROW(live.add(kMiB));
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, PendingCopyWaitsForCompetingRssBacking) {
  ASSERT_EQ(0, pool_.init(128 * kMiB));
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 64 * kMiB, 48 * kMiB));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ASSERT_TRUE(native.try_charge_process_backing(8 * kMiB, 0, false));
  std::promise<void> ready;
  auto started = ready.get_future();
  auto release = std::async(std::launch::async, [&] {
    ready.set_value();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!Peer::workspace_waiting(native) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    native.release_process_backing(8 * kMiB);
  });
  started.wait();
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance, false, kMiB);
  ResultMemoryAllowance::PendingCopy pending;
  EXPECT_NO_THROW(lease.add_pending(12 * kMiB, 12 * kMiB, pending));
  release.get();
  EXPECT_EQ(12 * kMiB, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest,
       ReusedBackingObeysCriticalPressureAndRecovery) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease warmup(&allowance);
    warmup.add(kMiB);
  }
  const auto backing = allowance.reserved_bytes();
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  native.set_memory_pressure(Pressure::kCritical);
  {
    ResultMemoryAllowance::Lease blocked(&allowance);
    EXPECT_THROW(blocked.add(kMiB), std::bad_alloc);
  }
  // Background still permits mandatory work within an existing reservation.
  native.set_memory_pressure(Pressure::kBackground);
  {
    ResultMemoryAllowance::Lease recovered(&allowance);
    EXPECT_NO_THROW(recovered.add(kMiB));
  }
  native.set_memory_pressure(Pressure::kNormal);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(backing, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, PressureRetiresOnlyIdleResultAllowance) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  ASSERT_TRUE(native.configure_process_rss_limit(
      std::numeric_limits<size_t>::max() / 2));
  auto restore = ailego::ScopeGuard::Make([&] {
    native.set_memory_pressure(Pressure::kNormal);
    EXPECT_TRUE(native.configure_process_rss_limit(0));
  });
  ResultMemoryAllowance allowance;
  auto pinned = std::make_unique<ResultMemoryAllowance::Lease>(&allowance);
  pinned->add(kMiB);
  {
    ResultMemoryAllowance::Lease second(&allowance);
    second.add(kMiB);
    native.set_memory_pressure(Pressure::kCritical);
  }
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
  pinned.reset();
  EXPECT_EQ(0u, allowance.reserved_bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, ConcurrentCallsShareOneHighWaterAllowance) {
  ResultMemoryAllowance allowance;
  std::atomic<size_t> ready{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      ResultMemoryAllowance::Lease lease(&allowance);
      lease.add(kMiB);
      ++ready;
      while (!release.load()) std::this_thread::yield();
    });
  }
  while (ready.load() != 4) std::this_thread::yield();
  EXPECT_EQ(4 * kMiB, allowance.active_bytes());
  EXPECT_EQ(8 * kMiB, pool_.external_used());
  release = true;
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(8 * kMiB, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, RoundsEachBackingEquivalentBeforeAdmission) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease lease(&allowance, false, kMiB);
    lease.add(kMiB + 1);
    EXPECT_EQ(kMiB + 1, allowance.active_bytes());
    EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
  }
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
  {
    ResultMemoryAllowance::Lease lease(&allowance, false, kMiB);
    lease.add(1);
    EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
  }
  EXPECT_EQ(4 * kMiB, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, NativeAndQuantizedLeasesShareOneReservation) {
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease native(&allowance);
  native.add(kMiB + 1);
  EXPECT_EQ(2 * (kMiB + 1), allowance.reserved_bytes());
  ResultMemoryAllowance::Lease python(&allowance, false, kMiB);
  python.add(1);
  EXPECT_EQ(kMiB + 2, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
  EXPECT_EQ(4 * kMiB, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, RoundedQuarterLimitRejectsWithoutPoisoning) {
  auto &actual = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, actual.init(66 * kMiB));
  ASSERT_EQ(66 * kMiB, actual.capacity());
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance, false, kMiB);
  // Byte-only accounting fits the 16.5 MiB quarter, but its whole arenas do
  // not.
  EXPECT_THROW(lease.add(8 * kMiB + 1), std::bad_alloc);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
  EXPECT_EQ(0u, pool_.external_used());
  lease.add(8 * kMiB);
  EXPECT_EQ(8 * kMiB, allowance.active_bytes());
  EXPECT_EQ(16 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, QuantumOverflowDoesNotConsumeBudget) {
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease enormous(&allowance, false,
                                        std::numeric_limits<size_t>::max());
  EXPECT_THROW(enormous.add(1), std::bad_alloc);
  ResultMemoryAllowance::Lease invalid(&allowance, false, 0);
  EXPECT_THROW(invalid.add(1), std::bad_alloc);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, ReturnedLeaseDoesNotKeepMaterializationLock) {
  ResultMemoryAllowance allowance;
  auto held = std::make_shared<ResultMemoryAllowance::Lease>(&allowance, true);
  held->add(kMiB);
  held->end_materialization();
  auto next = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease lease(&allowance, true);
    lease.add(kMiB);
    return allowance.active_bytes();
  });
  ASSERT_EQ(std::future_status::ready, next.wait_for(std::chrono::seconds(2)));
  EXPECT_EQ(2 * kMiB, next.get());
  EXPECT_EQ(kMiB, allowance.active_bytes());
  auto release =
      std::async(std::launch::async,
                 [owner = std::move(held)]() mutable { owner.reset(); });
  release.get();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, RejectedGrowthLeavesLeaseAndChargeUnchanged) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease lease(&allowance);
    lease.add(kMiB);
    const size_t pressure = pool_.capacity() - pool_.used();
    ASSERT_TRUE(pool_.try_charge_external(pressure));
    EXPECT_THROW(lease.add(kMiB), std::bad_alloc);
    EXPECT_EQ(kMiB, allowance.active_bytes());
    EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
    pool_.release_external(pressure);
    // Retry succeeds once the competing owner returns real capacity.
    lease.add(kMiB);
    EXPECT_EQ(2 * kMiB, allowance.active_bytes());
  }
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, OverflowDoesNotConsumeBudget) {
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance);
  EXPECT_THROW(lease.add(std::numeric_limits<size_t>::max()), std::bad_alloc);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, AnalyzerExpansionIsAdmittedBeforeAllocation) {
  FtsIndexParams standard;
  FtsIndexParams ngram("ngram", {}, R"({"ngram_min":2048,"ngram_max":2048})");
  const std::string text(32768, 'a');
  const size_t normal = EstimateFtsWorkspace(standard, text);
  const size_t expanded = EstimateFtsWorkspace(ngram, text);
  EXPECT_GT(expanded, 512 * kMiB);
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance);
  lease.add(normal);
  const size_t reserved = allowance.reserved_bytes();
  EXPECT_THROW(lease.add(expanded), std::bad_alloc);
  EXPECT_EQ(reserved, allowance.reserved_bytes());
  EXPECT_EQ(normal, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest, MaterializersWaitBeforeAccumulatingChunks) {
  ResultMemoryAllowance allowance;
  auto first = std::make_unique<ResultMemoryAllowance::Lease>(&allowance, true);
  first->add(kMiB);
  std::promise<void> entered;
  auto waiting = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease second(&allowance, true);
    entered.set_value();
    second.add(kMiB);
    second.add(kMiB);
    EXPECT_EQ(2 * kMiB, allowance.active_bytes());
  });
  entered.get_future().wait();
  EXPECT_EQ(std::future_status::timeout,
            waiting.wait_for(std::chrono::milliseconds(20)));
  // The first call can grow to completion while another call waits with no
  // partial result charge. No sleep or retry is needed for forward progress.
  first->add(2 * kMiB);
  EXPECT_EQ(3 * kMiB, allowance.active_bytes());
  first.reset();
  EXPECT_EQ(std::future_status::ready,
            waiting.wait_for(std::chrono::seconds(5)));
  waiting.get();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(6 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, FailedMaterializerUnblocksNextCall) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease oversized(&allowance, true);
    EXPECT_THROW(oversized.add(pool_.capacity()), std::bad_alloc);
  }
  auto next = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease lease(&allowance, true);
    lease.add(kMiB);
  });
  EXPECT_EQ(std::future_status::ready, next.wait_for(std::chrono::seconds(5)));
  next.get();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, MaterializerWaitsBeforeNativePreparation) {
  ResultMemoryAllowance allowance;
  auto first = std::make_unique<ResultMemoryAllowance::Lease>(&allowance, true);
  first->begin_materialization();
  // Entering the slot alone does not fabricate a byte charge.
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
  std::promise<void> entered;
  std::atomic<bool> prepared{false};
  auto waiting = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease second(&allowance, true);
    entered.set_value();
    second.begin_materialization();
    prepared = true;
    second.add(kMiB);
  });
  entered.get_future().wait();
  EXPECT_EQ(std::future_status::timeout,
            waiting.wait_for(std::chrono::milliseconds(20)));
  EXPECT_FALSE(prepared.load());
  // Exception/empty-result paths must also release a slot with no charge.
  first.reset();
  EXPECT_EQ(std::future_status::ready,
            waiting.wait_for(std::chrono::seconds(5)));
  waiting.get();
  EXPECT_TRUE(prepared.load());
  EXPECT_EQ(0u, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest, DenseVectorCopyKeepsItsOwnAdmission) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  auto doc = MakeQueryDocument(memory);
  auto field = std::make_shared<FieldSchema>("vector", DataType::VECTOR_FP32);
  const std::vector<float> values{1.0f, 2.0f, -3.0f};
  const size_t expected =
      192 + field->name().size() * 2 + values.size() * sizeof(float);
  {
    vector_column_params::VectorDataBuffer source;
    auto &buffer =
        std::get<vector_column_params::DenseVectorBuffer>(source.vector_buffer);
    buffer.reserve(values.size() * sizeof(float));
    buffer.data.assign(reinterpret_cast<const char *>(values.data()),
                       values.size() * sizeof(float));
    ASSERT_TRUE(ConvertVectorDataBufferToDocField(field, source, doc.get(),
                                                  memory.get())
                    .ok());
    EXPECT_EQ(expected, allowance.active_bytes());
  }
  memory.reset();
  EXPECT_EQ(values, doc->get_ref<std::vector<float>>("vector"));
  EXPECT_EQ(expected, allowance.active_bytes());
  auto held = doc;
  doc.reset();
  EXPECT_EQ(expected, allowance.active_bytes());
  held.reset();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(expected * 2, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, SparseCopyRejectsBeforeMutatingAndCanRetry) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  auto doc = MakeQueryDocument(memory);
  auto field =
      std::make_shared<FieldSchema>("vector", DataType::SPARSE_VECTOR_FP32);
  const std::vector<uint32_t> indices{1, 5};
  const std::vector<float> values{2.0f, -3.0f};
  vector_column_params::SparseVectorBuffer buffer;
  buffer.indices.assign(reinterpret_cast<const char *>(indices.data()),
                        indices.size() * sizeof(uint32_t));
  buffer.values.assign(reinterpret_cast<const char *>(values.data()),
                       values.size() * sizeof(float));
  const size_t expected = 192 + field->name().size() * 2 +
                          buffer.indices.size() + buffer.values.size();
  vector_column_params::VectorDataBuffer source{std::move(buffer)};
  const size_t pressure = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(pressure));
  EXPECT_THROW(
      ConvertVectorDataBufferToDocField(field, source, doc.get(), memory.get()),
      std::bad_alloc);
  EXPECT_FALSE(doc->has("vector"));
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
  pool_.release_external(pressure);
  ASSERT_TRUE(
      ConvertVectorDataBufferToDocField(field, source, doc.get(), memory.get())
          .ok());
  using Sparse = std::pair<std::vector<uint32_t>, std::vector<float>>;
  EXPECT_EQ(Sparse(indices, values), doc->get_ref<Sparse>("vector"));
  EXPECT_EQ(expected, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest, OversizedCallDoesNotExhaustFutureCacheSpace) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease lease(&allowance);
    lease.add(kMiB);
    EXPECT_THROW(lease.add(pool_.capacity()), std::bad_alloc);
    EXPECT_EQ(kMiB, allowance.active_bytes());
    EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
  }
  ResultMemoryAllowance::Lease next(&allowance);
  EXPECT_NO_THROW(next.add(kMiB));
  EXPECT_EQ(2 * kMiB, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, DisabledAndUnconfiguredPathsDoNotReserve) {
  ResultMemoryAllowance::Lease disabled(nullptr);
  EXPECT_FALSE(disabled.enabled());
  disabled.add(kMiB);
  EXPECT_EQ(0u, pool_.external_used());
  ASSERT_EQ(0, pool_.init(0));
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance);
  lease.add(kMiB);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, BitmapFilterAndIteratorKeepAdmission) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  memory->add(kMiB);
  auto *bitmap = roaring_bitmap_create();
  ASSERT_NE(nullptr, bitmap);
  roaring_bitmap_add(bitmap, 7);
  auto result = std::make_shared<InvertedSearchResult>(bitmap);
  result->keep_memory(memory);
  auto filter = result->make_filter();
  auto iter = result->create_iterator();
  memory.reset();
  result.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_FALSE(filter->is_filtered(7));
  filter.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_EQ(7u, iter->doc_id());
  iter.reset();
  EXPECT_EQ(0u, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest, QueryDocumentKeepsAdmissionUntilLastOwner) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  memory->add(kMiB);
  auto first = MakeQueryDocument(memory);
  first->set_pk("retained");
  auto second = first;
  memory.reset();
  first.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_EQ("retained", second->pk_ref());
  second.reset();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, MovingDocumentByValueRetainsAdmission) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  memory->add(kMiB);
  auto source = MakeQueryDocument(memory);
  source->set_pk("grouped");
  std::vector<Doc> values;
  values.push_back(std::move(*source));
  memory.reset();
  source.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  {
    Doc extracted = std::move(values.front());
    values.clear();
    EXPECT_EQ("grouped", extracted.pk_ref());
    EXPECT_EQ(kMiB, allowance.active_bytes());
  }
  EXPECT_EQ(0u, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest,
       DisabledDocumentLeaseDoesNotEnablePythonBudget) {
  auto disabled = std::make_shared<ResultMemoryAllowance::Lease>(nullptr);
  EXPECT_FALSE(DocumentMemoryAccess::budgeted(*MakeQueryDocument(disabled)));
}

TEST_F(ResultMemoryAllowanceTest, QueryEstimateCountsOnlySelectedStringSlice) {
  arrow::StringBuilder builder;
  ASSERT_TRUE(builder.Append(std::string(4096, 'x')).ok());
  ASSERT_TRUE(builder.Append("abc").ok());
  std::shared_ptr<arrow::Array> values;
  ASSERT_TRUE(builder.Finish(&values).ok());
  auto schema = arrow::schema({arrow::field("payload", arrow::utf8())});
  auto small = arrow::RecordBatch::Make(schema, 1, {values->Slice(1, 1)});
  auto large = arrow::RecordBatch::Make(schema, 1, {values->Slice(0, 1)});
  EXPECT_EQ(4096u - 3u,
            EstimateQueryResult(*large) - EstimateQueryResult(*small));
}

TEST_F(ResultMemoryAllowanceTest, QueryEstimateCountsNestedStringsAndNulls) {
  auto child = std::make_shared<arrow::StringBuilder>();
  arrow::ListBuilder builder(arrow::default_memory_pool(), child);
  ASSERT_TRUE(builder.Append().ok());
  ASSERT_TRUE(child->Append(std::string(4096, 'a')).ok());
  ASSERT_TRUE(child->Append("b").ok());
  ASSERT_TRUE(builder.AppendNull().ok());
  std::shared_ptr<arrow::Array> values;
  ASSERT_TRUE(builder.Finish(&values).ok());
  QueryResultEstimate estimate;
  estimate.value(*values, 1);
  EXPECT_EQ(0u, estimate.bytes());
  estimate.value(*values, 0);
  EXPECT_GE(estimate.bytes(), 4097u + 2 * 32u);
  EXPECT_THROW(estimate.add(std::numeric_limits<size_t>::max(), 2),
               std::bad_alloc);
}

}  // namespace
}  // namespace zvec::detail

TEST(CMemoryOwnerTest, CopiesKeepAdmissionUntilCrossThreadFree) {
  using namespace zvec::detail;
  constexpr size_t kMiB = 1024 * 1024;
  auto &pool = zvec::ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(64 * kMiB));
  auto &allowance = SharedResultMemoryAllowance();
  const auto initial = allowance.active_bytes();
  auto *copy = AllocateCResult(kMiB);
  ASSERT_NE(nullptr, copy);
  EXPECT_GE(allowance.active_bytes() - initial, kMiB);
  const auto backing = allowance.reserved_bytes();
  std::async(std::launch::async, [copy] { FreeCResult(copy); }).get();
  EXPECT_EQ(initial, allowance.active_bytes());
  EXPECT_EQ(backing, allowance.reserved_bytes());
  copy = AllocateCResult(kMiB);
  EXPECT_EQ(backing, allowance.reserved_bytes());
  FreeCResult(copy);
  EXPECT_EQ(initial, allowance.active_bytes());
  EXPECT_THROW(AllocateCResult(64 * kMiB), std::bad_alloc);
  EXPECT_EQ(initial, allowance.active_bytes());
  EXPECT_THROW(AllocateCResultArray(SIZE_MAX, sizeof(void *)), std::bad_alloc);
  // Existing C allocations remain valid inputs to the public free path.
  FreeCResult(std::malloc(17));
  FreeCResult(nullptr);
}

namespace zvec::detail {

TEST(RetainedMetadataMemoryTest, ReusedAllowanceObeysHostPressure) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  constexpr size_t kMiB = 1024 * 1024;
  ASSERT_EQ(0, pool.init(64 * kMiB));
  RetainedMetadataBudget budget;
  {
    RetainedMetadataBudget::Lease warmup(budget);
    warmup.acquire(kMiB);
  }
  const auto backing = budget.reserved();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  pool.set_memory_pressure(Pressure::kCritical);
  {
    RetainedMetadataBudget::Lease blocked(budget);
    EXPECT_THROW(blocked.acquire(kMiB), std::bad_alloc);
    EXPECT_EQ(0u, blocked.bytes());
  }
  pool.set_memory_pressure(Pressure::kBackground);
  {
    RetainedMetadataBudget::Lease recovered(budget);
    EXPECT_NO_THROW(recovered.acquire(kMiB));
  }
  pool.set_memory_pressure(Pressure::kNormal);
  EXPECT_EQ(backing, budget.reserved());
}

}  // namespace zvec::detail

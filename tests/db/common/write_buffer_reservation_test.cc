// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/common/write_buffer_reservation.h"
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/db/options.h>
#include "memory_budget_test_helper.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static bool configure_rss(MemoryLimitPool &pool, size_t limit, size_t rss) {
    pool.stop_background_evictor();
    sample = rss;
    return pool.process_rss_.configure(limit, +[] { return sample; });
  }

  static void refresh_rss(MemoryLimitPool &pool, size_t rss) {
    sample = rss;
    pool.process_rss_.refresh(true);
  }

 private:
  inline static size_t sample = 0;
};
}  // namespace zvec::ailego

namespace zvec {
namespace {

constexpr size_t kMiB = 1024 * 1024;
// Exercise the same large-buffer pressure on desktop and mobile targets.
constexpr size_t kStressBufferBytes = 64 * kMiB;

class WriteBufferReservationTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    pool_.init(512 * kMiB);
  }
  void TearDown() override {
    EXPECT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    EXPECT_EQ(0u, pool_.external_used());
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
};

TEST_F(WriteBufferReservationTest, CollectionsShareBudgetAndReturnAllowance) {
  pool_.init(528 * kMiB);
  WriteBufferReservation first;
  WriteBufferReservation second;
  WriteBufferReservation third;
  ASSERT_TRUE(first.acquire(kStressBufferBytes).ok());
  ASSERT_TRUE(second.acquire(kStressBufferBytes).ok());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
            third.acquire(kStressBufferBytes).code());
  EXPECT_EQ(0u, third.bytes());
  EXPECT_EQ(528 * kMiB, pool_.external_used());
  first.release();
  ASSERT_TRUE(third.acquire(kStressBufferBytes).ok());
  EXPECT_EQ(528 * kMiB, pool_.external_used());
}

TEST_F(WriteBufferReservationTest, ManagedPlanLeavesRssRoomForArrowBacking) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(pool.configure_process_rss_limit(0)); });
  ASSERT_TRUE(Peer::configure_rss(pool, 512 * kMiB, 128 * kMiB));
  Peer::refresh_rss(pool, 0);

  WriteBufferReservation writer;
  ASSERT_TRUE(writer.acquire_managed(kStressBufferBytes).ok());
  EXPECT_EQ(264 * kMiB, writer.bytes());
  // Arrow's separately charged buffers must fit beside the 8 MiB ordinary
  // workspace, even before another RSS sample. The 264 MiB plan is not backing.
  ASSERT_TRUE(pool.try_charge_workspace(192 * kMiB, 192 * kMiB));
  pool.release_external(192 * kMiB);
  EXPECT_EQ(264 * kMiB, pool_.external_used());
}

TEST_F(WriteBufferReservationTest, ConcurrentAdmissionCannotOverReserve) {
  pool_.init(4 * 264 * kMiB);
  std::atomic<unsigned> admitted{0};
  std::atomic<unsigned> ready{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 12; ++i) {
    threads.emplace_back([&] {
      WriteBufferReservation writer;
      if (writer.acquire(kStressBufferBytes, true).ok()) ++admitted;
      ++ready;
      while (!release.load()) std::this_thread::yield();
    });
  }
  while (ready.load() != threads.size()) std::this_thread::yield();
  EXPECT_EQ(4u, admitted.load());
  EXPECT_EQ(pool_.capacity(), pool_.external_used());
  release = true;
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(0u, detail::SharedWriteBufferMemoryAllowance().active_bytes());
  EXPECT_EQ(4 * 264 * kMiB, pool_.external_used());
}

TEST_F(WriteBufferReservationTest, RejectedManagedWriterDoesNotRetainBacking) {
  WriteBufferReservation first;
  WriteBufferReservation second;
  ASSERT_TRUE(first.acquire_managed(kStressBufferBytes).ok());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
            second.acquire_managed(kStressBufferBytes).code());
  first.release();
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
}

TEST_F(WriteBufferReservationTest,
       RejectedGrowthLeavesRetainedChargeUnchanged) {
  WriteBufferReservation first;
  WriteBufferReservation second;
  auto &allowance = detail::SharedWriteBufferMemoryAllowance();
  ASSERT_TRUE(first.acquire(kStressBufferBytes, true).ok());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
            second.acquire(kStressBufferBytes, true).code());
  EXPECT_EQ(0u, second.bytes());
  EXPECT_EQ(264 * kMiB, allowance.active_bytes());
  EXPECT_EQ(264 * kMiB, allowance.reserved_bytes());
  EXPECT_FALSE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
  first.release();
  ASSERT_TRUE(second.acquire(kStressBufferBytes, true).ok());
  EXPECT_EQ(264 * kMiB, pool_.external_used());
}

}  // namespace
}  // namespace zvec

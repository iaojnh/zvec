// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/common/write_buffer_reservation.h"
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/options.h>
#include "memory_budget_test_helper.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec {
namespace {

constexpr size_t kMiB = 1024 * 1024;

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

TEST_F(WriteBufferReservationTest, DefaultAllowanceIncludesFlushHeadroom) {
  WriteBufferReservation writer;
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(264 * kMiB, writer.bytes());
  EXPECT_EQ(writer.bytes(), pool_.external_used());
  EXPECT_EQ(writer.bytes(), pool_.used());
  // Further writes to the same live buffer must not reserve another allowance.
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(264 * kMiB, pool_.external_used());
  writer.release();
  writer.release();
  EXPECT_EQ(0u, pool_.external_used());
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(264 * kMiB, pool_.external_used());
}

TEST_F(WriteBufferReservationTest, CollectionsShareBudgetAndReturnAllowance) {
  pool_.init(528 * kMiB);
  WriteBufferReservation first;
  WriteBufferReservation second;
  WriteBufferReservation third;
  ASSERT_TRUE(first.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  ASSERT_TRUE(second.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
            third.acquire(DEFAULT_MAX_BUFFER_SIZE).code());
  EXPECT_EQ(0u, third.bytes());
  EXPECT_EQ(528 * kMiB, pool_.external_used());
  first.release();
  ASSERT_TRUE(third.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(528 * kMiB, pool_.external_used());
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
      if (writer.acquire(DEFAULT_MAX_BUFFER_SIZE, true).ok()) ++admitted;
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

TEST_F(WriteBufferReservationTest, ActualWritesRetainAndReuseBacking) {
  auto &allowance = detail::SharedWriteBufferMemoryAllowance();
  for (int i = 0; i < 3; ++i) {
    {
      WriteBufferReservation writer;
      ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE, true).ok());
      EXPECT_EQ(264 * kMiB, allowance.active_bytes());
      EXPECT_EQ(264 * kMiB, pool_.external_used());
      writer.release();
      writer.release();
      EXPECT_EQ(0u, writer.bytes());
    }
    EXPECT_EQ(0u, allowance.active_bytes());
    EXPECT_EQ(264 * kMiB, allowance.reserved_bytes());
    EXPECT_EQ(264 * kMiB, pool_.external_used());
  }
  // An unrelated page/workspace cannot spend the idle writer's backing quota.
  const auto remaining = pool_.capacity() - allowance.reserved_bytes();
  ASSERT_TRUE(pool_.try_charge_external(remaining));
  EXPECT_FALSE(pool_.try_charge_external(1));
  pool_.release_external(remaining);
}

TEST_F(WriteBufferReservationTest, ManagedWritesRetainOnlyOrdinaryWorkspace) {
  auto &allowance = detail::SharedWriteBufferMemoryAllowance();
  for (int i = 0; i < 3; ++i) {
    WriteBufferReservation writer;
    ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
    ASSERT_TRUE(writer.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).ok());
    ASSERT_TRUE(writer.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).ok());
    EXPECT_EQ(264 * kMiB, allowance.active_bytes());
    writer.release();
    EXPECT_EQ(0u, allowance.active_bytes());
    EXPECT_EQ(8 * kMiB, allowance.reserved_bytes());
  }
}

TEST_F(WriteBufferReservationTest, ManagedAndReplayBackingAreCountedPerLease) {
  pool_.init(800 * kMiB);
  WriteBufferReservation first;
  WriteBufferReservation second;
  WriteBufferReservation admission;
  ASSERT_TRUE(first.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).ok());
  ASSERT_TRUE(second.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  ASSERT_TRUE(admission.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  ASSERT_TRUE(second.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).ok());
  auto &allowance = detail::SharedWriteBufferMemoryAllowance();
  admission.release();
  second.release();
  first.release();
  EXPECT_EQ(16 * kMiB, allowance.reserved_bytes());

  ASSERT_TRUE(first.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).ok());
  ASSERT_TRUE(second.acquire(DEFAULT_MAX_BUFFER_SIZE, true).ok());
  // An explicit legacy backing lease must not be downgraded by a later call.
  ASSERT_TRUE(second.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).ok());
  second.release();
  first.release();
  EXPECT_EQ(272 * kMiB, allowance.reserved_bytes());
}

TEST_F(WriteBufferReservationTest, RejectedManagedWriterDoesNotRetainBacking) {
  WriteBufferReservation first;
  WriteBufferReservation second;
  ASSERT_TRUE(first.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
            second.acquire_managed(DEFAULT_MAX_BUFFER_SIZE).code());
  first.release();
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
}

TEST_F(WriteBufferReservationTest, TransientAdmissionOnlyReturnsExcess) {
  auto &allowance = detail::SharedWriteBufferMemoryAllowance();
  WriteBufferReservation writer;
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  writer.release();
  EXPECT_EQ(0u, allowance.reserved_bytes());

  // An explicit backing request promotes an admitted lease without charging
  // twice. A later clean-open check must not increase the retained high water.
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE, true).ok());
  EXPECT_EQ(264 * kMiB, pool_.external_used());
  writer.release();
  ASSERT_TRUE(writer.acquire(96 * kMiB).ok());
  EXPECT_EQ(392 * kMiB, pool_.external_used());
  writer.release();
  EXPECT_EQ(264 * kMiB, pool_.external_used());
}

TEST_F(WriteBufferReservationTest,
       RejectedGrowthLeavesRetainedChargeUnchanged) {
  WriteBufferReservation first;
  WriteBufferReservation second;
  auto &allowance = detail::SharedWriteBufferMemoryAllowance();
  ASSERT_TRUE(first.acquire(DEFAULT_MAX_BUFFER_SIZE, true).ok());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
            second.acquire(DEFAULT_MAX_BUFFER_SIZE, true).code());
  EXPECT_EQ(0u, second.bytes());
  EXPECT_EQ(264 * kMiB, allowance.active_bytes());
  EXPECT_EQ(264 * kMiB, allowance.reserved_bytes());
  EXPECT_FALSE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
  first.release();
  ASSERT_TRUE(second.acquire(DEFAULT_MAX_BUFFER_SIZE, true).ok());
  EXPECT_EQ(264 * kMiB, pool_.external_used());
}

TEST_F(WriteBufferReservationTest, ReservationLeavesOnlyRemainderForCache) {
  WriteBufferReservation writer;
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  const auto remainder = pool_.capacity() - writer.bytes();
  ASSERT_TRUE(pool_.try_charge_external(remainder));
  EXPECT_FALSE(pool_.try_charge_external(1));
  pool_.release_external(remainder);
}

TEST_F(WriteBufferReservationTest, OverflowFailsWithoutConsumingBudget) {
  WriteBufferReservation writer;
  EXPECT_EQ(StatusCode::INVALID_ARGUMENT,
            writer.acquire(std::numeric_limits<size_t>::max()).code());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(WriteBufferReservationTest, UnconfiguredPoolRetainsLegacyBehavior) {
  pool_.init(0);
  WriteBufferReservation writer;
  EXPECT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(0u, writer.bytes());
}

}  // namespace
}  // namespace zvec

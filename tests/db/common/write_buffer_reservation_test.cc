// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/common/write_buffer_reservation.h"
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/options.h>

namespace zvec {
namespace {

constexpr size_t kMiB = 1024 * 1024;

class WriteBufferReservationTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_.init(512 * kMiB);
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
  }
  ailego::MemoryLimitPool &pool_ = ailego::MemoryLimitPool::get_instance();
};

TEST_F(WriteBufferReservationTest, DefaultAllowanceIncludesFlushHeadroom) {
  WriteBufferReservation writer;
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(264 * kMiB, writer.bytes());
  EXPECT_EQ(writer.bytes(), pool_.external_used());
  EXPECT_EQ(writer.bytes(), pool_.used());
  // Reusing the writer after a flush must not reserve a second allowance.
  ASSERT_TRUE(writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok());
  EXPECT_EQ(264 * kMiB, pool_.external_used());
  writer.release();
  writer.release();
  EXPECT_EQ(0u, pool_.external_used());
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
      if (writer.acquire(DEFAULT_MAX_BUFFER_SIZE).ok()) ++admitted;
      ++ready;
      while (!release.load()) std::this_thread::yield();
    });
  }
  while (ready.load() != threads.size()) std::this_thread::yield();
  EXPECT_EQ(4u, admitted.load());
  EXPECT_EQ(pool_.capacity(), pool_.external_used());
  release = true;
  for (auto &thread : threads) thread.join();
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

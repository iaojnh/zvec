// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <type_traits>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include "pool_payload_budget.h"

namespace zvec::ailego {
namespace {
class BudgetedMappedBufferTest : public testing::Test {
 protected:
  void SetUp() override {
    old_capacity_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(16 * page_));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_capacity_));
  }
  PoolPayloadBudget &pool_ = PoolPayloadBudget::get_instance();
  const size_t page_ = PoolPayloadBudget::page_buffer_size();
  size_t old_capacity_{0};
};

TEST_F(BudgetedMappedBufferTest, ChargesRoundedMappingAndReleasesOnClear) {
  BudgetedMappedBuffer buffer;
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0u, buffer.size());
  buffer.resize(page_ + 1);
  EXPECT_EQ(page_ + 1, buffer.size());
  EXPECT_EQ(2 * page_, buffer.mapped_size());
  EXPECT_EQ(2 * page_, pool_.external_used());
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer.data()) % page_);
  std::memset(buffer.data(), 0x35, buffer.size());
  EXPECT_EQ(0x35, buffer.data()[page_]);
  buffer.clear();
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0u, buffer.size());
  EXPECT_EQ(0u, buffer.mapped_size());
  EXPECT_EQ(0u, pool_.used());
  buffer.clear();
}

TEST_F(BudgetedMappedBufferTest, FailedGrowthPreservesOldMappingAndCanRetry) {
  ASSERT_EQ(0, pool_.init(3 * page_));
  BudgetedMappedBuffer buffer(page_);
  buffer.resize(2 * page_);
  char *old = buffer.data();
  std::memset(old, 0x6b, buffer.size());
  EXPECT_THROW(buffer.resize(3 * page_), std::bad_alloc);
  EXPECT_EQ(old, buffer.data());
  EXPECT_EQ(2 * page_, pool_.used());
  EXPECT_EQ(0x6b, buffer.data()[page_]);
  EXPECT_NO_THROW(buffer.resize(2 * page_));
  EXPECT_EQ(old, buffer.data());
  buffer.clear();
  ASSERT_NO_THROW(buffer.resize(3 * page_));
  EXPECT_EQ(3 * page_, pool_.used());
}

TEST_F(BudgetedMappedBufferTest, MovesAndCrossThreadReleaseKeepOwnership) {
  static_assert(std::is_nothrow_move_constructible_v<BudgetedMappedBuffer>);
  static_assert(std::is_nothrow_move_assignable_v<BudgetedMappedBuffer>);
  for (size_t bytes : {page_, 2 * page_}) {
    SCOPED_TRACE(bytes);
    BudgetedMappedBuffer source(bytes);
    BudgetedMappedBuffer destination(page_);
    char *data = source.data();
    data[0] = 'x';
    destination = std::move(source);
    EXPECT_EQ(nullptr, source.data());
    EXPECT_EQ(0u, source.mapped_size());
    EXPECT_EQ(data, destination.data());
    EXPECT_EQ('x', destination.data()[0]);
    EXPECT_EQ(bytes, pool_.used());
    BudgetedMappedBuffer moved(std::move(destination));
    EXPECT_EQ(nullptr, destination.data());
    EXPECT_EQ(data, moved.data());
    EXPECT_EQ(bytes, pool_.used());
    std::thread worker([owned = std::move(moved)]() mutable {
      EXPECT_EQ(owned.mapped_size(),
                PoolPayloadBudget::get_instance().external_used());
      EXPECT_EQ('x', owned.data()[0]);
      owned.clear();
    });
    worker.join();
    EXPECT_EQ(nullptr, moved.data());
    EXPECT_EQ(0u, pool_.used());
  }
}

}  // namespace
}  // namespace zvec::ailego

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <type_traits>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/io/file.h>
#include "pool_payload_budget.h"

namespace zvec::ailego {
namespace {
class BudgetedAlignedBufferTest : public testing::Test {
 protected:
  void SetUp() override {
    old_capacity_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(16UL << 20));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_capacity_));
  }
  PoolPayloadBudget &pool_ = PoolPayloadBudget::get_instance();
  static constexpr size_t kLarge = BudgetedAlignedBuffer::kRetainedThreshold;
  size_t old_capacity_{0};
};

TEST_F(BudgetedAlignedBufferTest, LargeAlignmentChargesPaddingAndWholePages) {
  const size_t alignment = 2 * PoolPayloadBudget::page_buffer_size();
  BudgetedAlignedBuffer buffer(kLarge + 1, alignment);
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer.data()) % alignment);
  EXPECT_EQ(MemoryLimitPool::retained_block_size(kLarge + alignment),
            pool_.external_used());
  auto *bytes = static_cast<char *>(buffer.data());
  bytes[0] = 'a';
  bytes[kLarge] = 'z';
  EXPECT_EQ('a', bytes[0]);
  EXPECT_EQ('z', bytes[kLarge]);
}

TEST_F(BudgetedAlignedBufferTest, RejectedGrowthPreservesOldBuffer) {
  BudgetedAlignedBuffer buffer(kLarge, 64);
  auto *original = static_cast<char *>(buffer.data());
  std::memset(original, 0x6b, kLarge);
  MemoryReservation pressure;
  pressure.add(pool_.available());
  const size_t used = pool_.used();
  EXPECT_THROW(buffer.resize(2 * kLarge, 64), std::bad_alloc);
  EXPECT_EQ(original, buffer.data());
  EXPECT_EQ(kLarge, buffer.size());
  EXPECT_EQ(used, pool_.used());
  EXPECT_EQ(0x6b, original[kLarge - 1]);
  pressure.retain_at_most(0);
  ASSERT_NO_THROW(buffer.resize(2 * kLarge, 64));
  EXPECT_EQ(2 * kLarge, buffer.size());
}

TEST_F(BudgetedAlignedBufferTest, MoveAndCrossThreadReleaseKeepOwnership) {
  static_assert(std::is_nothrow_move_constructible_v<BudgetedAlignedBuffer>);
  static_assert(std::is_nothrow_move_assignable_v<BudgetedAlignedBuffer>);
  BudgetedAlignedBuffer source(kLarge, 64);
  BudgetedAlignedBuffer destination(128, 64);
  void *address = source.data();
  destination = std::move(source);
  EXPECT_EQ(nullptr, source.data());
  EXPECT_EQ(address, destination.data());
  EXPECT_EQ(MemoryLimitPool::retained_block_size(kLarge), pool_.used());
  std::thread worker([owned = std::move(destination)]() mutable {
    owned.resize(32, 64);
    EXPECT_EQ(32u, PoolPayloadBudget::get_instance().used());
  });
  worker.join();
  EXPECT_EQ(nullptr, destination.data());
  EXPECT_EQ(0u, pool_.used());
}

}  // namespace
}  // namespace zvec::ailego

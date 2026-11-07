// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <gtest/gtest.h>
#include "binding/c/c_memory.h"

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

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/common/native_worker_memory.h"
#include <cstdlib>
#include <limits>
#include <gtest/gtest.h>

namespace zvec::detail {
TEST(NativeWorkerMemoryTest, NativePolicyDoesNotChargeUnusedLinuxVirtualStack) {
#if defined(__linux__) || defined(__APPLE__)
  EXPECT_GT(DefaultNativeWorkerStackBytes(), 0u);
  EXPECT_EQ(128UL << 10, NativeWorkerStackAllowance(128UL << 10));
#if defined(__linux__)
  EXPECT_EQ(512UL << 10, NativeWorkerStackAllowance(8UL << 20));
  EXPECT_LE(DefaultNativeWorkerStackBytes(), 512UL << 10);
#else
  EXPECT_EQ(8UL << 20, NativeWorkerStackAllowance(8UL << 20));
#endif
#endif
}

TEST(NativeWorkerMemoryDeathTest, AdmissionIsIdempotentAndOverflowSafe) {
#if GTEST_HAS_DEATH_TEST && (defined(__linux__) || defined(__APPLE__))
  const auto previous = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        if (pool.init(32UL << 20) != 0) std::_Exit(1);
        const auto before = pool.external_used();
        ReserveNativeWorkerStacks(2, 1, 1);
        const size_t expected = (2 + 1 + RocksDbBackgroundThreads(1) + 2) *
                                DefaultNativeWorkerStackBytes();
        if (NativeWorkerStackReservation().bytes() != expected ||
            pool.external_used() != before + expected)
          std::_Exit(2);
        ReserveNativeWorkerStacks(2, 1, 1);
        if (pool.external_used() != before + expected) std::_Exit(3);
        bool overflow = false;
        try {
          ReserveNativeWorkerStacks(std::numeric_limits<size_t>::max(), 1, 1);
        } catch (const std::bad_alloc &) {
          overflow = true;
        }
        if (!overflow || pool.external_used() != before + expected)
          std::_Exit(4);
        bool rejected = false;
        try {
          ReserveNativeWorkerStacks(1024, 1024, 16);
        } catch (const std::bad_alloc &) {
          rejected = true;
        }
        if (!rejected || pool.external_used() != before + expected)
          std::_Exit(5);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous;
#endif
}
}  // namespace zvec::detail

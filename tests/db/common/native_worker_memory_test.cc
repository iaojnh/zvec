// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/common/native_worker_memory.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>

namespace zvec::detail {
#if defined(_WIN32)
TEST(NativeWorkerMemoryTest, WindowsStackSettingsHonorCommitAndPageRounding) {
  std::vector<unsigned char> image(sizeof(IMAGE_DOS_HEADER) +
                                   sizeof(IMAGE_NT_HEADERS));
  IMAGE_DOS_HEADER dos{};
  dos.e_magic = IMAGE_DOS_SIGNATURE;
  dos.e_lfanew = sizeof(dos);
  IMAGE_NT_HEADERS nt{};
  nt.Signature = IMAGE_NT_SIGNATURE;
  nt.FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER);
  nt.OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR_MAGIC;
  const auto publish = [&] {
    std::memcpy(image.data(), &dos, sizeof(dos));
    std::memcpy(image.data() + sizeof(dos), &nt, sizeof(nt));
  };
  size_t bytes = 73;
  nt.OptionalHeader.SizeOfStackReserve = 8UL << 20;
  nt.OptionalHeader.SizeOfStackCommit = 4096;
  publish();
  ASSERT_TRUE(
      WindowsWorkerStackBytes(image.data(), image.size(), 4096, &bytes));
  EXPECT_EQ(512UL << 10, bytes);
  nt.OptionalHeader.SizeOfStackCommit = (1UL << 20) + 1;
  publish();
  ASSERT_TRUE(
      WindowsWorkerStackBytes(image.data(), image.size(), 4096, &bytes));
  EXPECT_EQ((1UL << 20) + 4096, bytes);
  nt.OptionalHeader.SizeOfStackReserve = 64 * 1024;
  nt.OptionalHeader.SizeOfStackCommit = 4096;
  publish();
  ASSERT_TRUE(
      WindowsWorkerStackBytes(image.data(), image.size(), 4096, &bytes));
  EXPECT_EQ(64UL << 10, bytes);
  EXPECT_FALSE(WindowsWorkerStackBytes(nullptr, image.size(), 4096, &bytes));
  EXPECT_FALSE(
      WindowsWorkerStackBytes(image.data(), image.size() - 1, 4096, &bytes));
  EXPECT_FALSE(WindowsWorkerStackBytes(image.data(), image.size(), 0, &bytes));
  dos.e_lfanew = -1;
  publish();
  EXPECT_FALSE(
      WindowsWorkerStackBytes(image.data(), image.size(), 4096, &bytes));
  EXPECT_EQ(64UL << 10, bytes);
}
#endif

TEST(NativeWorkerMemoryTest, NativePolicyDoesNotChargeUnusedVirtualStack) {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  EXPECT_GT(DefaultNativeWorkerStackBytes(), 0u);
  EXPECT_EQ(128UL << 10, NativeWorkerStackAllowance(128UL << 10));
#if defined(__linux__) || defined(_WIN32)
  EXPECT_EQ(512UL << 10, NativeWorkerStackAllowance(8UL << 20));
#if defined(__linux__)
  EXPECT_LE(DefaultNativeWorkerStackBytes(), 512UL << 10);
#endif
#else
  EXPECT_EQ(8UL << 20, NativeWorkerStackAllowance(8UL << 20));
#endif
#endif
}

TEST(NativeWorkerMemoryDeathTest, AdmissionIsIdempotentAndOverflowSafe) {
#if GTEST_HAS_DEATH_TEST && \
    (defined(__linux__) || defined(__APPLE__) || defined(_WIN32))
  const auto previous = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        if (pool.init(32UL << 20) != 0) std::_Exit(1);
        const auto before = pool.external_used();
        ReserveNativeWorkerStacks(2, 1, 1, 3);
        const size_t expected = (2 + 1 + RocksDbBackgroundThreads(1) + 2 + 3) *
                                DefaultNativeWorkerStackBytes();
        if (NativeWorkerStackReservation().bytes() != expected ||
            pool.external_used() != before + expected)
          std::_Exit(2);
        ReserveNativeWorkerStacks(2, 1, 1, 3);
        if (pool.external_used() != before + expected) std::_Exit(3);
        bool overflow = false;
        try {
          ReserveNativeWorkerStacks(std::numeric_limits<size_t>::max(), 1, 1);
        } catch (const std::bad_alloc &) {
          overflow = true;
        }
        if (!overflow || pool.external_used() != before + expected)
          std::_Exit(4);
        overflow = false;
        try {
          ReserveNativeWorkerStacks(2, 1, 1,
                                    std::numeric_limits<size_t>::max());
        } catch (const std::bad_alloc &) {
          overflow = true;
        }
        if (!overflow || pool.external_used() != before + expected)
          std::_Exit(6);
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

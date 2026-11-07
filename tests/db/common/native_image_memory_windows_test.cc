// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <gtest/gtest.h>
#include "db/common/native_image_memory.h"

#if defined(_WIN32)
namespace zvec::detail {
namespace {
class LoadedImage {
 public:
  LoadedImage() : module_(::LoadLibraryA(ZVEC_NATIVE_IMAGE_WINDOWS_DLL)) {}
  ~LoadedImage() {
    if (module_) ::FreeLibrary(module_);
  }
  const void *anchor() const {
    if (!module_) return nullptr;
    auto symbol = reinterpret_cast<const void *(*)()>(
        ::GetProcAddress(module_, "ZvecNativeImageFixtureAnchor"));
    return symbol ? symbol() : nullptr;
  }

 private:
  HMODULE module_;
};

TEST(WindowsNativeImageMemory, CountsUntouchedDllZeroFillNotFileBytes) {
  LoadedImage image;
  ASSERT_NE(nullptr, image.anchor());
  const size_t bytes = NativeImageBytesForAddress(image.anchor());
  SYSTEM_INFO system{};
  ::GetSystemInfo(&system);
  EXPECT_GE(bytes, 4UL << 20);
  EXPECT_EQ(0u, bytes % system.dwPageSize);
  WIN32_FILE_ATTRIBUTE_DATA file{};
  ASSERT_TRUE(::GetFileAttributesExA(ZVEC_NATIVE_IMAGE_WINDOWS_DLL,
                                     GetFileExInfoStandard, &file));
  EXPECT_EQ(0u, file.nFileSizeHigh);
  EXPECT_LT(file.nFileSizeLow, bytes);
}

TEST(WindowsNativeImageMemory, ExcludesCallerExecutableAndRejectsNonImages) {
  static const char caller_anchor = 0;
  EXPECT_EQ(0u, NativeImageBytesForAddress(&caller_anchor));
  EXPECT_THROW(NativeImageBytesForAddress(nullptr), std::bad_alloc);
  const char stack_anchor = 0;
  EXPECT_THROW(NativeImageBytesForAddress(&stack_anchor), std::bad_alloc);
}

TEST(WindowsNativeImageMemoryDeathTest, ReservationKeepsLiveImageCharged) {
#if GTEST_HAS_DEATH_TEST
  const auto previous = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        LoadedImage image;
        if (!image.anchor()) std::_Exit(1);
        auto &pool = ailego::MemoryLimitPool::get_instance();
        if (pool.init(64UL << 20) != 0) std::_Exit(2);
        const size_t bytes = NativeImageBytesForAddress(image.anchor());
        const auto before = pool.stats();
        ReserveNativeImage(image.anchor());
        const auto after = pool.stats();
        if (after.external_used != before.external_used + bytes ||
            after.committed != before.committed + bytes ||
            NativeImageReservation().bytes() != bytes)
          std::_Exit(3);
        ReserveNativeImage(image.anchor());
        if (pool.committed() != after.committed) std::_Exit(4);
        bool rejected = false;
        try {
          NativeImageReservation().add(pool.capacity());
        } catch (const std::bad_alloc &) {
          rejected = true;
        }
        if (!rejected || pool.committed() != after.committed ||
            NativeImageReservation().bytes() != bytes)
          std::_Exit(5);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous;
#else
  GTEST_SKIP() << "Process-isolated tests are not supported on this platform";
#endif
}
}  // namespace
}  // namespace zvec::detail
#endif

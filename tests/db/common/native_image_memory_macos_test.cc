// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include "db/common/native_image_memory.h"

#if defined(__APPLE__) && TARGET_OS_OSX
namespace {
using namespace zvec;

class LoadedImage {
 public:
  explicit LoadedImage(const char *path)
      : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {}
  ~LoadedImage() {
    if (handle_) dlclose(handle_);
  }
  const void *anchor() const {
    if (!handle_) return nullptr;
    auto symbol = reinterpret_cast<const void *(*)()>(
        dlsym(handle_, "ZvecNativeImageFixtureAnchor"));
    return symbol ? symbol() : nullptr;
  }

 private:
  void *handle_;
};

TEST(MachONativeImageMemoryDeathTest,
     ChargesRealPoolOnceAndPreservesLiveOwnershipOnFailure) {
#if GTEST_HAS_DEATH_TEST
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        LoadedImage image(ZVEC_NATIVE_IMAGE_MACOS_DYLIB);
        const void *anchor = image.anchor();
        if (!anchor) std::_Exit(1);
        auto &pool = ailego::MemoryLimitPool::get_instance();
        if (pool.init(128u * 1024 * 1024) != 0) std::_Exit(2);
        const size_t bytes = detail::NativeImageBytesForAddress(anchor);
        if (bytes < 32768 ||
            std::filesystem::file_size(ZVEC_NATIVE_IMAGE_MACOS_DYLIB) >= bytes)
          std::_Exit(6);
        const auto before = pool.stats();
        { detail::ReserveNativeImage(anchor); }
        const auto after = pool.stats();
        if (!bytes || after.external_used != before.external_used + bytes ||
            after.committed != before.committed + bytes ||
            detail::NativeImageReservation().bytes() != bytes)
          std::_Exit(3);
        detail::ReserveNativeImage(anchor);
        if (pool.stats().committed != after.committed) std::_Exit(4);
        bool rejected = false;
        try {
          detail::NativeImageReservation().add(pool.capacity());
        } catch (const std::bad_alloc &) {
          rejected = true;
        }
        if (!rejected || detail::NativeImageReservation().bytes() != bytes ||
            pool.stats().committed != after.committed)
          std::_Exit(5);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
#else
  GTEST_SKIP() << "Process-isolated tests are not supported on this platform";
#endif
}
}  // namespace
#endif

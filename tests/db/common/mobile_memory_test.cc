// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/common/mobile_memory.h"
#include "db/common/native_image_memory.h"
#include <limits>
#include <ailego/utility/mobile_memory.h>
#include <gtest/gtest.h>
#include <zvec/db/config.h>
#include "db/common/cgroup_util.h"

namespace zvec {
#if defined(__linux__)
TEST(MobileMemoryTest, AndroidExecutablePathDoesNotBecomeALibraryCharge) {
  ElfW(Phdr) executable_headers[1]{};
  ElfW(Phdr) library_headers[1]{};
  dl_phdr_info image{};
  image.dlpi_phdr = executable_headers;
  image.dlpi_name = "/data/local/tmp/zvec_test";
  EXPECT_TRUE(detail::IsMainElfImage(image, executable_headers));
  image.dlpi_name = "/data/app/libzvec.so";
  image.dlpi_phdr = library_headers;
  EXPECT_FALSE(detail::IsMainElfImage(image, executable_headers));
  EXPECT_FALSE(detail::IsMainElfImage(image, nullptr));
  image.dlpi_name = "";
  EXPECT_TRUE(detail::IsMainElfImage(image, executable_headers));
}
#endif

TEST(MobileMemoryTest, DefaultDoesNotConsumeMostOfDeviceMemory) {
  constexpr uint64_t mib = uint64_t{1} << 20;
  EXPECT_EQ(128 * mib, detail::MobileDefaultMemoryLimit(8 * 1024 * mib));
  EXPECT_EQ(64 * mib, detail::MobileDefaultMemoryLimit(512 * mib));
  EXPECT_EQ(0u, detail::MobileDefaultMemoryLimit(0));
  EXPECT_EQ(128 * mib, detail::MobileDefaultMemoryLimit(UINT64_MAX));
  if (detail::kMobileMemoryPolicy) {
    GlobalConfig::ConfigData config;
    EXPECT_EQ(detail::MobileDefaultMemoryLimit(CgroupUtil::get_memory_limit()),
              config.memory_limit_bytes);
    config.memory_limit_bytes = detail::MobileDefaultMemoryLimit(512 * mib);
    EXPECT_TRUE(GlobalConfig::Instance().validate(config).ok());
    config.memory_limit_bytes = detail::kMobileMinimumMemoryLimit - 1;
    EXPECT_FALSE(GlobalConfig::Instance().validate(config).ok());
  }
}

TEST(MobileMemoryTest, NativeReaderUsesDeviceApplicationMetrics) {
  auto reader = ailego::NativeApplicationMemoryReader();
#if defined(__APPLE__) && TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR && \
    !TARGET_OS_MACCATALYST
  ASSERT_NE(nullptr, reader);
  ailego::ProcessRssBudget::NativeSample sample;
  ASSERT_TRUE(reader(&sample));
  EXPECT_GT(sample.footprint, 0u);
#else
  EXPECT_EQ(nullptr, reader);
#endif
}
}  // namespace zvec

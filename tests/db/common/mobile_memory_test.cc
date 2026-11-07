// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/common/mobile_memory.h"
#include <limits>
#include <ailego/utility/mobile_memory.h>
#include <gtest/gtest.h>
#include <zvec/db/config.h>
#include <zvec/db/options.h>
#include "db/common/cgroup_util.h"
#include "db/common/global_resource.h"
#include "db/common/native_image_memory.h"
#include "db/common/write_buffer_reservation.h"

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

TEST(MobileMemoryTest, WriterDefaultsFitConservativeProcessBudgets) {
  const uint32_t expected = detail::kMobileMemoryPolicy
                                ? MOBILE_DEFAULT_MAX_BUFFER_SIZE
                                : 64U * 1024 * 1024;
  EXPECT_EQ(expected, CollectionOptions().max_buffer_size_);
  EXPECT_EQ(expected, CollectionOptions(false, false).max_buffer_size_);
  EXPECT_EQ(expected, (SegmentOptions{false, false}).max_buffer_size_);
  EXPECT_EQ(
      64U * 1024 * 1024,
      CollectionOptions(false, false, 64U * 1024 * 1024).max_buffer_size_);

  auto &pool = ailego::MemoryLimitPool::get_instance();
  for (const size_t total : {64UL << 20, 128UL << 20}) {
    ASSERT_EQ(
        0,
        pool.init(GlobalResource::calculate_buffer_pool_memory_budget(total)));
    WriteBufferReservation writer;
    ASSERT_TRUE(writer.acquire(MOBILE_DEFAULT_MAX_BUFFER_SIZE).ok());
    EXPECT_EQ(24UL << 20, writer.bytes());
    // Explicit large requests retain their original rejection semantics.
    WriteBufferReservation large;
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, large.acquire(64UL << 20).code());
  }
}
}  // namespace zvec

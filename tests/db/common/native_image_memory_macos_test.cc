// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "db/common/native_image_memory.h"

#if defined(__APPLE__)
namespace {
using namespace zvec;

class ImageHeader {
 public:
  ImageHeader() : data(sizeof(mach_header_64), 0) {
    header.magic = MH_MAGIC_64;
    header.filetype = MH_DYLIB;
    publish();
  }

  void segment(const char *name, uint64_t address, uint64_t capacity,
               vm_prot_t protection = VM_PROT_READ, uint64_t file_bytes = 0) {
    segment_command_64 command{};
    command.cmd = LC_SEGMENT_64;
    command.cmdsize = sizeof(command);
    std::strncpy(command.segname, name, sizeof(command.segname) - 1);
    command.vmaddr = address;
    command.vmsize = capacity;
    command.filesize = file_bytes;
    command.initprot = command.maxprot = protection;
    append(&command, sizeof(command));
  }

  void other_command() {
    uuid_command command{};
    command.cmd = LC_UUID;
    command.cmdsize = sizeof(command);
    append(&command, sizeof(command));
  }

  void publish() { std::memcpy(data.data(), &header, sizeof(header)); }

  template <typename T>
  void replace_first(const T &command) {
    std::memcpy(data.data() + sizeof(header), &command, sizeof(command));
  }

  mach_header_64 header{};
  std::vector<unsigned char> data;

 private:
  void append(const void *command, size_t size) {
    const size_t offset = data.size();
    data.resize(offset + size);
    std::memcpy(data.data() + offset, command, size);
    ++header.ncmds;
    header.sizeofcmds += static_cast<uint32_t>(size);
    publish();
  }
};

bool image_bytes(const ImageHeader &image, size_t page_size, size_t *bytes) {
  return detail::MachOLoadImageBytes(image.data.data(), image.data.size(),
                                    page_size, bytes);
}

#if TARGET_OS_OSX
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

// Independent sorted-vector oracle for real, loader-validated fixture headers.
size_t mapped_capacity(const void *anchor) {
  Dl_info owner{};
  if (!dladdr(anchor, &owner) || !owner.dli_fbase) return 0;
  const auto *header = static_cast<const mach_header_64 *>(owner.dli_fbase);
  const auto *command = reinterpret_cast<const load_command *>(header + 1);
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (uint32_t i = 0; i < header->ncmds; ++i) {
    if (command->cmd == LC_SEGMENT_64) {
      const auto *segment = reinterpret_cast<const segment_command_64 *>(command);
      if (segment->vmsize && std::strcmp(segment->segname, SEG_PAGEZERO) != 0 &&
          (segment->initprot & (VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE))) {
        const uint64_t end = segment->vmaddr + segment->vmsize;
        ranges.emplace_back(segment->vmaddr / page_size * page_size,
                            (end / page_size + (end % page_size != 0)) * page_size);
      }
    }
    command = reinterpret_cast<const load_command *>(
        reinterpret_cast<const unsigned char *>(command) + command->cmdsize);
  }
  std::sort(ranges.begin(), ranges.end());
  uint64_t covered_end = 0;
  size_t bytes = 0;
  for (const auto &range : ranges) {
    const auto start = std::max(covered_end, range.first);
    if (range.second > start) bytes += range.second - start;
    covered_end = std::max(covered_end, range.second);
  }
  return bytes;
}

#endif

TEST(MachONativeImageMemory, CountsUnsortedPageUnionAndZeroFillNotFileBytes) {
  ImageHeader image;
  image.segment("__DATA", 0x2f00, 0x300, VM_PROT_READ | VM_PROT_WRITE);
  image.other_command();
  image.segment("__TEXT", 0x1003, 0x1800, VM_PROT_READ | VM_PROT_EXECUTE, 16);
  image.segment("__LINKEDIT", 0x8001, 1, VM_PROT_READ, 1);
  size_t bytes = 73;
  ASSERT_TRUE(image_bytes(image, 4096, &bytes));
  EXPECT_EQ(4u * 4096, bytes);
  ASSERT_TRUE(image_bytes(image, 16384, &bytes));
  EXPECT_EQ(2u * 16384, bytes);
}

TEST(MachONativeImageMemory, ExcludesPageZeroInaccessibleAndEmptySegments) {
  ImageHeader image;
  image.segment(SEG_PAGEZERO, 0, uint64_t{1} << 32, VM_PROT_NONE);
  image.segment("__GUARD", 0x100000000, uint64_t{1} << 32, VM_PROT_NONE);
  image.segment(SEG_PAGEZERO, 0, 0x10000000, VM_PROT_READ);
  image.segment("__EMPTY", 0x10000, 0);
  image.segment("__TEXT", 0x1001, 1, VM_PROT_READ, 1);
  size_t bytes = 73;
  ASSERT_TRUE(image_bytes(image, 4096, &bytes));
  EXPECT_EQ(4096u, bytes);
}

TEST(MachONativeImageMemory, HandlesUnalignedHeadersAndDuplicateNestedRanges) {
  ImageHeader image;
  image.segment("__TEXT", 0x1001, 0x2000);
  image.segment("__DATA", 0x1100, 1);
  image.segment("__MORE", 0x2fff, 2);
  std::vector<unsigned char> unaligned(image.data.size() + 1);
  std::memcpy(unaligned.data() + 1, image.data.data(), image.data.size());
  size_t bytes = 73;
  ASSERT_TRUE(detail::MachOLoadImageBytes(unaligned.data() + 1,
                                        image.data.size(), 4096, &bytes));
  EXPECT_EQ(3u * 4096, bytes);
}

TEST(MachONativeImageMemory, RejectsInvalidHeadersWithoutPublishingPartialBytes) {
  ImageHeader image;
  image.segment("__TEXT", 0x1000, 1);
  size_t bytes = 73;
  EXPECT_FALSE(detail::MachOLoadImageBytes(nullptr, 0, 4096, &bytes));
  EXPECT_FALSE(detail::MachOLoadImageBytes(image.data.data(), 1, 4096, &bytes));
  EXPECT_FALSE(image_bytes(image, 0, &bytes));
  EXPECT_FALSE(image_bytes(image, 4096, nullptr));
  image.header.magic = MH_CIGAM_64;
  image.publish();
  EXPECT_FALSE(image_bytes(image, 4096, &bytes));
  image.header.magic = MH_MAGIC_64;
  ++image.header.sizeofcmds;
  image.publish();
  EXPECT_FALSE(image_bytes(image, 4096, &bytes));
  --image.header.sizeofcmds;
  image.header.ncmds = 0;
  image.publish();
  EXPECT_FALSE(image_bytes(image, 4096, &bytes));
  image.header.ncmds = 100;
  image.publish();
  EXPECT_FALSE(image_bytes(image, 4096, &bytes));
  EXPECT_EQ(73u, bytes);
}

TEST(MachONativeImageMemory, RejectsTruncatedMisalignedAndInvalidCommandSizes) {
  for (uint32_t size : {0u, 8u, 71u, 73u, 4096u}) {
    ImageHeader image;
    image.segment("__TEXT", 0x1000, 1);
    load_command command{LC_SEGMENT_64, size};
    image.replace_first(command);
    size_t bytes = 73;
    EXPECT_FALSE(image_bytes(image, 4096, &bytes)) << size;
    EXPECT_EQ(73u, bytes);
  }
  ImageHeader image;
  image.segment("__TEXT", 0x1000, 1);
  segment_command_64 command{};
  std::memcpy(&command, image.data.data() + sizeof(image.header), sizeof(command));
  command.nsects = 1;
  image.replace_first(command);
  size_t bytes = 73;
  EXPECT_FALSE(image_bytes(image, 4096, &bytes));
  EXPECT_EQ(73u, bytes);
}

TEST(MachONativeImageMemory, RejectsOverflowAndImpossibleFileCapacity) {
  for (const auto &range : {std::pair<uint64_t, uint64_t>{UINT64_MAX - 1, 4},
                            {UINT64_MAX - 1, 1}}) {
    ImageHeader image;
    image.segment("__TEXT", range.first, range.second);
    size_t bytes = 73;
    EXPECT_FALSE(image_bytes(image, 4096, &bytes));
    EXPECT_EQ(73u, bytes);
  }
  ImageHeader image;
  image.segment("__TEXT", 0x1000, 1, VM_PROT_READ, 2);
  size_t bytes = 73;
  EXPECT_FALSE(image_bytes(image, 4096, &bytes));
  EXPECT_EQ(73u, bytes);
}

TEST(MachONativeImageMemory, ExcludesMainExecutableAndInvalidAnchors) {
  static const char main_anchor = 0;
  EXPECT_EQ(0u, detail::NativeImageBytesForAddress(&main_anchor));
  EXPECT_THROW(detail::NativeImageBytesForAddress(nullptr), std::bad_alloc);
  const char stack_anchor = 0;
  EXPECT_THROW(detail::NativeImageBytesForAddress(&stack_anchor), std::bad_alloc);
}

#if TARGET_OS_OSX
TEST(MachONativeImageMemory,
     ResolvesDylibAndBundleButExcludesCallerAndInvalidAnchors) {
  for (const char *path :
       {ZVEC_NATIVE_IMAGE_MACOS_DYLIB, ZVEC_NATIVE_IMAGE_MACOS_BUNDLE}) {
    LoadedImage image(path);
    const void *anchor = image.anchor();
    ASSERT_NE(nullptr, anchor) << path;
    const size_t bytes = detail::NativeImageBytesForAddress(anchor);
    EXPECT_GE(bytes, 32768u);
    EXPECT_EQ(mapped_capacity(anchor), bytes);
    EXPECT_EQ(0u, bytes % static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  }
  static const char main_anchor = 0;
  EXPECT_EQ(0u, detail::NativeImageBytesForAddress(&main_anchor));
  EXPECT_THROW(detail::NativeImageBytesForAddress(nullptr), std::bad_alloc);
  const char stack_anchor = 0;
  EXPECT_THROW(detail::NativeImageBytesForAddress(&stack_anchor), std::bad_alloc);
}

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
#endif
}  // namespace
#endif

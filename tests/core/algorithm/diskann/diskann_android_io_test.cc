// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#if defined(__ANDROID__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <linux/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include "diskann_file_reader.h"

namespace {
std::atomic<int> g_open_error{0};
std::atomic<int> g_read_error{0};
std::atomic<int> g_simulated_direct_fd{-1};
std::atomic<bool> g_simulate_direct_open{false};
}

// Target-local linker wrapping makes error coverage independent of whether the
// emulator's particular filesystem implements O_DIRECT. Production binaries
// do not contain these wrappers or fault controls.
extern "C" int __real_open(const char *, int, ...);
extern "C" int __real_open64(const char *, int, ...);
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __real_pread64(int, void *, size_t, off64_t);
#if __ANDROID_API__ >= 23
extern "C" ssize_t __real___pread_chk(int, void *, size_t, off_t, size_t);
extern "C" ssize_t __real___pread64_chk(int, void *, size_t, off64_t, size_t);
#endif

namespace {
int WrappedOpen(const char *path, int flags, mode_t mode, bool large) {
  const bool direct = (flags & O_DIRECT) != 0;
  if (direct && g_open_error.load() != 0) {
    errno = g_open_error.load();
    return -1;
  }
  const bool simulated = direct && g_simulate_direct_open.load();
  if (simulated) flags &= ~O_DIRECT;
  const int fd = large ? __real_open64(path, flags, mode)
                       : __real_open(path, flags, mode);
  if (simulated && fd >= 0) g_simulated_direct_fd.store(fd);
  return fd;
}

bool InjectReadFailure(int fd) {
  if (fd != g_simulated_direct_fd.load()) return false;
  const int error = g_read_error.exchange(0);
  if (error == 0) return false;
  errno = error;
  return true;
}
}  // namespace

extern "C" int __wrap_open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, mode_t);
    va_end(args);
  }
  return WrappedOpen(path, flags, mode, false);
}
extern "C" int __wrap_open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, mode_t);
    va_end(args);
  }
  return WrappedOpen(path, flags, mode, true);
}
extern "C" int __wrap___open_2(const char *path, int flags) {
  return WrappedOpen(path, flags, 0, false);
}
extern "C" int __wrap___open64_2(const char *path, int flags) {
  return WrappedOpen(path, flags, 0, true);
}
extern "C" ssize_t __wrap_pread(int fd, void *buffer, size_t size, off_t offset) {
  if (InjectReadFailure(fd)) return -1;
  return __real_pread(fd, buffer, size, offset);
}
extern "C" ssize_t __wrap_pread64(int fd, void *buffer, size_t size, off64_t offset) {
  if (InjectReadFailure(fd)) return -1;
  return __real_pread64(fd, buffer, size, offset);
}
#if __ANDROID_API__ >= 23
extern "C" ssize_t __wrap___pread_chk(int fd, void *buffer, size_t size,
                                       off_t offset, size_t capacity) {
  if (InjectReadFailure(fd)) return -1;
  return __real___pread_chk(fd, buffer, size, offset, capacity);
}
extern "C" ssize_t __wrap___pread64_chk(int fd, void *buffer, size_t size,
                                         off64_t offset, size_t capacity) {
  if (InjectReadFailure(fd)) return -1;
  return __real___pread64_chk(fd, buffer, size, offset, capacity);
}
#endif

namespace zvec::core {
namespace {
constexpr size_t kBlock = 4096;

class AndroidTestReader : public LinuxAlignedFileReader {
 public:
  explicit AndroidTestReader(int support_error = 0)
      : support_error_(support_error) {}
 protected:
  int android_direct_io_support_error(int) const override {
    return support_error_;
  }
 private:
  int support_error_;
};

class AndroidFile : public ::testing::Test {
 protected:
  void SetUp() override {
    fd_ = ::mkstemp(path_);
    ASSERT_GE(fd_, 0);
    ASSERT_EQ(::posix_memalign(&buffer_, kBlock, kBlock * 2), 0);
    std::memset(buffer_, 0x39, kBlock);
    ASSERT_EQ(::pwrite(fd_, buffer_, kBlock, 0), static_cast<ssize_t>(kBlock));
    ASSERT_EQ(::fsync(fd_), 0);
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);
  }
  void TearDown() override {
    g_open_error.store(0);
    g_read_error.store(0);
    g_simulate_direct_open.store(false);
    g_simulated_direct_fd.store(-1);
    if (fd_ >= 0) ::close(fd_);
    ::unlink(path_);
    std::free(buffer_);
  }
  std::vector<AlignedRead> Requests() {
    return {AlignedRead(0, kBlock, buffer_)};
  }
  void ExpectOriginalBytes() {
    const auto *bytes = static_cast<const uint8_t *>(buffer_);
    for (size_t i = 0; i < kBlock; ++i) ASSERT_EQ(bytes[i], 0x39);
  }
  int fd_{-1};
  char path_[64] = "DiskAnnAndroidIO.XXXXXX";
  void *buffer_{nullptr};
};

TEST_F(AndroidFile, OpenFailureFallsBackWithoutChangingSourceFlags) {
  const int flags = ::fcntl(fd_, F_GETFL);
  g_open_error.store(EOPNOTSUPP);
  AndroidTestReader reader;
  ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
  EXPECT_FALSE(reader.direct_io_requested());
  EXPECT_EQ(reader.direct_io_fallback_errno(), EOPNOTSUPP);
  EXPECT_EQ(::fcntl(fd_, F_GETFL), flags);
  std::memset(buffer_, 0, kBlock);
  auto requests = Requests();
  IOContext context = nullptr;
  ASSERT_EQ(reader.read(requests, context), 0);
  ExpectOriginalBytes();
}

TEST_F(AndroidFile, PathOpenAlsoAttemptsDirectBeforeFallback) {
  g_open_error.store(EINVAL);
  AndroidTestReader reader;
  reader.open(path_);
  EXPECT_FALSE(reader.direct_io_requested());
  EXPECT_EQ(reader.direct_io_fallback_errno(), EINVAL);
  auto requests = Requests();
  IOContext context = nullptr;
  ASSERT_EQ(reader.read(requests, context), 0);
  ExpectOriginalBytes();
}

TEST_F(AndroidFile, ReadFallbackKeepsCapturedInodeAfterPathReplacement) {
  g_simulate_direct_open.store(true);
  AndroidTestReader reader;
  ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
  ASSERT_TRUE(reader.direct_io_requested());
  ASSERT_EQ(::unlink(path_), 0);
  const int replacement = ::open(path_, O_CREAT | O_WRONLY | O_TRUNC, 0600);
  ASSERT_GE(replacement, 0);
  std::memset(buffer_, 0x71, kBlock);
  ASSERT_EQ(::pwrite(replacement, buffer_, kBlock, 0), static_cast<ssize_t>(kBlock));
  ASSERT_EQ(::close(replacement), 0);
  g_read_error.store(EINVAL);
  IOContext context = nullptr;
  auto requests = Requests();
  PendingBatch batch;
  ASSERT_EQ(reader.submit(batch, requests, context), 0);
  std::vector<uint32_t> completed;
  EXPECT_EQ(reader.get_completed(batch, context, 1, completed), 1);
  EXPECT_FALSE(reader.direct_io_requested());
  EXPECT_EQ(reader.direct_io_fallback_errno(), EINVAL);
  ExpectOriginalBytes();
  EXPECT_EQ(::fcntl(fd_, F_GETFL) & O_DIRECT, 0);
}

TEST_F(AndroidFile, StorageErrorsAndEofAreNotMaskedByFallback) {
  g_simulate_direct_open.store(true);
  AndroidTestReader reader;
  ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
  g_read_error.store(EIO);
  IOContext context = nullptr;
  auto requests = Requests();
  EXPECT_NE(reader.read(requests, context), 0);
  EXPECT_TRUE(reader.direct_io_requested());
  requests[0].len = kBlock * 2;
  EXPECT_NE(reader.read(requests, context), 0);
  EXPECT_TRUE(reader.direct_io_requested());
  requests[0].len = kBlock;
  ASSERT_EQ(reader.read(requests, context), 0);
  ExpectOriginalBytes();
}

TEST_F(AndroidFile, ConcurrentReadersSurviveReadFallback) {
  g_simulate_direct_open.store(true);
  AndroidTestReader reader;
  ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
  g_read_error.store(EOPNOTSUPP);
  std::atomic<bool> correct{true};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      void *output = nullptr;
      if (::posix_memalign(&output, kBlock, kBlock) != 0) {
        correct.store(false);
        return;
      }
      IOContext context = nullptr;
      std::vector<AlignedRead> requests{AlignedRead(0, kBlock, output)};
      for (size_t repeat = 0; repeat < 32; ++repeat) {
        if (reader.read(requests, context) != 0) correct.store(false);
        const auto *bytes = static_cast<const uint8_t *>(output);
        for (size_t j = 0; j < kBlock; ++j) {
          if (bytes[j] != 0x39) correct.store(false);
        }
      }
      std::free(output);
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_TRUE(correct.load());
  EXPECT_FALSE(reader.direct_io_requested());
}

TEST_F(AndroidFile, FileCapabilityRejectionAfterSuccessfulOpenIsExplicit) {
  g_simulate_direct_open.store(true);
  AndroidTestReader reader(EOPNOTSUPP);
  ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
  EXPECT_FALSE(reader.direct_io_requested());
  EXPECT_EQ(reader.direct_io_fallback_errno(), EOPNOTSUPP);
  auto requests = Requests();
  IOContext context = nullptr;
  ASSERT_EQ(reader.read(requests, context), 0);
  ExpectOriginalBytes();
}

uint64_t AndroidReadBytes() {
  std::ifstream input("/proc/thread-self/io");
  std::string key;
  uint64_t value;
  while (input >> key >> value) {
    if (key == "read_bytes:") return value;
  }
  throw std::runtime_error("Thread storage counters unavailable");
}

uint64_t ResidentBytes(int fd, size_t length) {
  void *mapping = ::mmap(nullptr, length, PROT_NONE, MAP_SHARED, fd, 0);
  if (mapping == MAP_FAILED) throw std::runtime_error("mmap for mincore failed");
  const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> pages((length + page_size - 1) / page_size);
  const int ret = ::mincore(mapping, length, pages.data());
  ::munmap(mapping, length);
  if (ret != 0) throw std::runtime_error("mincore failed");
  uint64_t resident = 0;
  for (auto page : pages) if (page & 1) resident += page_size;
  return resident;
}

// This opt-in probe uses the production reader, without injected failures.
// It verifies payloads and checks page-cache residency as well as read bytes.
TEST_F(AndroidFile, VerifyRealCacheBehavior) {
  if (std::getenv("ZVEC_ANDROID_IO_PROBE") == nullptr) GTEST_SKIP();
  constexpr size_t kSize = 64 * 1024 * 1024;
  constexpr size_t kReads = 4096;
  for (size_t offset = 0; offset < kSize; offset += kBlock) {
    const uint64_t word = 0x47a91bd39e510000ULL + offset / kBlock;
    for (size_t j = 0; j < kBlock; j += sizeof(word)) {
      std::memcpy(static_cast<char *>(buffer_) + j, &word, sizeof(word));
    }
    ASSERT_EQ(::pwrite(fd_, buffer_, kBlock, offset), static_cast<ssize_t>(kBlock));
  }
  ASSERT_EQ(::fsync(fd_), 0);
  std::ostringstream cases;
  bool first = true;
  for (const std::string state : {"buffered-warm", "production-warm", "production-cold", "buffered-cold"}) {
    const bool production = state.find("production") == 0;
    const bool cold = state.find("cold") != std::string::npos;
    if (cold) ASSERT_EQ(::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED), 0);
    else {
      for (size_t offset = 0; offset < kSize; offset += kBlock) {
        ASSERT_EQ(::pread(fd_, buffer_, kBlock, offset), static_cast<ssize_t>(kBlock));
      }
    }
    LinuxAlignedFileReader reader;
    ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
    const auto resident_before = ResidentBytes(fd_, kSize);
    const auto bytes_before = AndroidReadBytes();
    const auto start = std::chrono::steady_clock::now();
    IOContext context = nullptr;
    for (size_t i = 0; i < kReads; ++i) {
      const size_t block = (i * 1009) % (kSize / kBlock);
      const size_t offset = block * kBlock;
      if (production) {
        std::vector<AlignedRead> requests{AlignedRead(offset, kBlock, buffer_)};
        ASSERT_EQ(reader.read(requests, context), 0);
      } else {
        ASSERT_EQ(::pread(fd_, buffer_, kBlock, offset), static_cast<ssize_t>(kBlock));
      }
      const uint64_t expected = 0x47a91bd39e510000ULL + block;
      for (size_t j = 0; j < kBlock; j += sizeof(expected)) {
        uint64_t actual;
        std::memcpy(&actual, static_cast<char *>(buffer_) + j, sizeof(actual));
        ASSERT_EQ(actual, expected);
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto read_bytes = AndroidReadBytes() - bytes_before;
    const auto resident_after = ResidentBytes(fd_, kSize);
    if (!first) cases << ',';
    first = false;
    cases << "{\"state\":\"" << state << "\",\"direct_requested\":"
          << (production && reader.direct_io_requested() ? "true" : "false")
          << ",\"fallback_errno\":" << reader.direct_io_fallback_errno()
          << ",\"operations\":" << kReads << ",\"requested_bytes\":" << kReads * kBlock
          << ",\"read_bytes\":" << read_bytes << ",\"resident_before\":" << resident_before
          << ",\"resident_after\":" << resident_after << ",\"seconds\":" << seconds << '}';
  }
  int file_flags = 0;
  const bool has_file_flags = ::ioctl(fd_, FS_IOC_GETFLAGS, &file_flags) == 0;
  std::ostringstream capabilities;
  capabilities << "{\"file_flags\":" << (has_file_flags ? std::to_string(file_flags) : "null");
#if defined(STATX_DIOALIGN) && defined(__NR_statx) && defined(AT_EMPTY_PATH)
  struct statx attributes {};
  const long statx_result = ::syscall(__NR_statx, fd_, "", AT_EMPTY_PATH, STATX_DIOALIGN, &attributes);
  capabilities << ",\"statx_headers\":true,\"statx_result\":" << statx_result
               << ",\"statx_mask\":" << attributes.stx_mask
               << ",\"dio_mem_align\":" << attributes.stx_dio_mem_align
               << ",\"dio_offset_align\":" << attributes.stx_dio_offset_align;
#else
  capabilities << ",\"statx_headers\":false";
#endif
  capabilities << '}';
  std::cout << "ZVEC_ANDROID_IO_RESULT {\"fixture_bytes\":" << kSize
            << ",\"file_capabilities\":" << capabilities.str()
            << ",\"payload_validated\":true,\"cases\":[" << cases.str() << "]}" << std::endl;
}

}  // namespace
}  // namespace zvec::core
#endif

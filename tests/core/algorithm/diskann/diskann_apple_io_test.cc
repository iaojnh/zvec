// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#if !TARGET_OS_IPHONE
#include <libproc.h>
#endif
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include "diskann_file_reader.h"
#include "../../../apple_io_phase_gate.h"

namespace zvec::core {
namespace {
constexpr size_t kBlock = 4096;

class ProbeReader : public LinuxAlignedFileReader {
 public:
  int Evict(uint64_t bytes) const {
    return invalidate_apple_cache_range(AlignedRead(0, bytes, nullptr));
  }
};

class FailingInvalidator : public LinuxAlignedFileReader {
 protected:
  int invalidate_apple_cache_range(const AlignedRead &) const override {
    return ENOTSUP;
  }
};

class AppleFile : public ::testing::Test {
 protected:
  void SetUp() override {
    fd_ = ::mkstemp(path_);
    ASSERT_GE(fd_, 0);
    ASSERT_EQ(::posix_memalign(&buffer_, static_cast<size_t>(::sysconf(_SC_PAGESIZE)), 8192), 0);
    std::memset(buffer_, 0x39, kBlock);
    ASSERT_EQ(::pwrite(fd_, buffer_, kBlock, 0), static_cast<ssize_t>(kBlock));
    ASSERT_EQ(::fsync(fd_), 0);
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);
  }
  void TearDown() override {
    if (fd_ >= 0) ::close(fd_);
    ::unlink(path_);
    std::free(buffer_);
  }
  void ExpectBytes(uint8_t expected) {
    for (size_t i = 0; i < kBlock; ++i) {
      ASSERT_EQ(static_cast<uint8_t *>(buffer_)[i], expected);
    }
  }
  int fd_{-1};
  char path_[64] = "DiskAnnAppleIO.XXXXXX";
  void *buffer_{nullptr};
};

TEST_F(AppleFile, SourceFlagsAndCapturedInodeSurviveReplacement) {
  const int original_flags = ::fcntl(fd_, F_GETFL);
  ASSERT_EQ(::unlink(path_), 0);
  const int replacement = ::open(path_, O_CREAT | O_WRONLY | O_TRUNC, 0600);
  ASSERT_GE(replacement, 0);
  std::memset(buffer_, 0x71, kBlock);
  ASSERT_EQ(::pwrite(replacement, buffer_, kBlock, 0), static_cast<ssize_t>(kBlock));
  ASSERT_EQ(::close(replacement), 0);
  LinuxAlignedFileReader reader;
  ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
  ASSERT_TRUE(reader.cache_invalidation_requested());
  IOContext context = nullptr;
  std::vector<AlignedRead> requests{AlignedRead(0, kBlock, buffer_)};
  ASSERT_EQ(reader.read(requests, context), 0);
  ExpectBytes(0x39);
  EXPECT_EQ(::fcntl(fd_, F_GETFL), original_flags);
}

TEST_F(AppleFile, InvalidationFailureIsExplicitWithoutLosingData) {
  FailingInvalidator reader;
  reader.open(path_);
  ASSERT_TRUE(reader.cache_invalidation_requested());
  IOContext context = nullptr;
  std::vector<AlignedRead> requests{AlignedRead(0, kBlock, buffer_)};
  ASSERT_EQ(reader.read(requests, context), 0);
  EXPECT_FALSE(reader.cache_invalidation_requested());
  EXPECT_EQ(reader.cache_fallback_errno(), ENOTSUP);
  ExpectBytes(0x39);
  ASSERT_EQ(reader.read(requests, context), 0);
}

TEST_F(AppleFile, DirtyDataIsPreservedAndEofRemainsAnError) {
  std::memset(buffer_, 0x52, kBlock);
  ASSERT_EQ(::pwrite(fd_, buffer_, kBlock, 0), static_cast<ssize_t>(kBlock));
  // Intentionally no fsync here: invalidation must preserve dirty file data.
  LinuxAlignedFileReader reader;
  reader.open(path_);
  IOContext context = nullptr;
  std::vector<AlignedRead> requests{AlignedRead(0, kBlock, buffer_)};
  ASSERT_EQ(reader.read(requests, context), 0);
  ExpectBytes(0x52);
  requests[0].len = 8192;
  EXPECT_NE(reader.read(requests, context), 0);
  requests[0].len = kBlock;
  ASSERT_EQ(reader.read(requests, context), 0);
  ExpectBytes(0x52);
  std::memset(buffer_, 0, kBlock);
  ASSERT_EQ(::pread(fd_, buffer_, kBlock, 0), static_cast<ssize_t>(kBlock));
  ExpectBytes(0x52);
}

TEST_F(AppleFile, ConcurrentReadsKeepTheirPayloads) {
  LinuxAlignedFileReader reader;
  ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
  std::atomic<bool> correct{true};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      void *out = nullptr;
      if (::posix_memalign(&out, kBlock, kBlock) != 0) {
        correct.store(false);
        return;
      }
      IOContext context = nullptr;
      std::vector<AlignedRead> requests{AlignedRead(0, kBlock, out)};
      for (size_t repeat = 0; repeat < 16; ++repeat) {
        if (reader.read(requests, context) != 0) correct.store(false);
        for (size_t j = 0; j < kBlock; ++j) {
          if (static_cast<uint8_t *>(out)[j] != 0x39) correct.store(false);
        }
      }
      std::free(out);
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_TRUE(correct.load());
}

uint64_t ResidentBytes(int fd, size_t size, bool requested_only = false) {
  void *mapping = ::mmap(nullptr, size, PROT_NONE, MAP_SHARED, fd, 0);
  if (mapping == MAP_FAILED) throw std::runtime_error("mincore mapping failed");
  const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  std::vector<char> pages((size + page - 1) / page);
  const int ret = ::mincore(mapping, size, pages.data());
  ::munmap(mapping, size);
  if (ret != 0) throw std::runtime_error("mincore failed");
  std::vector<bool> requested(pages.size(), !requested_only);
  if (requested_only) {
    for (size_t i = 0; i < 4096; ++i) {
      requested[((i * 1009) % (size / kBlock)) * kBlock / page] = true;
    }
  }
  uint64_t bytes = 0;
  for (size_t i = 0; i < pages.size(); ++i) {
    if (requested[i] && (pages[i] & 1)) bytes += page;
  }
  return bytes;
}

void DiskBytes(std::ostream &out) {
#if !TARGET_OS_IPHONE
  rusage_info_v4 info{};
  if (::proc_pid_rusage(::getpid(), RUSAGE_INFO_V4,
                       reinterpret_cast<rusage_info_t *>(&info)) != 0) {
    throw std::runtime_error("proc_pid_rusage failed");
  }
  out << info.ri_diskio_bytesread;
#else
  out << "null";  // Simulator host measures the same live process at gates.
#endif
}

TEST_F(AppleFile, VerifyRealCacheBehavior) {
  if (std::getenv("ZVEC_APPLE_IO_PROBE") == nullptr) GTEST_SKIP();
  constexpr size_t kSize = 64 * 1048576;
  constexpr size_t kReads = 4096;
  for (size_t offset = 0; offset < kSize; offset += kBlock) {
    const uint64_t word = 0x47a91bd39e510000ULL + offset / kBlock;
    for (size_t j = 0; j < kBlock; j += 8) {
      std::memcpy(static_cast<char *>(buffer_) + j, &word, 8);
    }
    ASSERT_EQ(::pwrite(fd_, buffer_, kBlock, offset), static_cast<ssize_t>(kBlock));
  }
  ASSERT_EQ(::fsync(fd_), 0);
  ProbeReader control;
  ASSERT_EQ(control.open_from_handle(path_, fd_), 0);
  std::ostringstream cases;
  bool first = true;
  for (const char *label : {"buffered-warm", "hint-warm", "production-path-warm",
                            "production-handle-warm", "production-cold", "production-repeat", "buffered-cold"}) {
    const std::string state(label);
    const bool production = state.find("production") == 0;
    const bool cold = state.find("cold") != std::string::npos;
    if (cold) ASSERT_EQ(control.Evict(kSize), 0);
    else if (state != "production-repeat") {
      for (size_t offset = 0; offset < kSize; offset += kBlock) {
        ASSERT_EQ(::pread(fd_, buffer_, kBlock, offset), static_cast<ssize_t>(kBlock));
      }
    }
    LinuxAlignedFileReader reader;
    if (state == "production-path-warm") reader.open(path_);
    else ASSERT_EQ(reader.open_from_handle(path_, fd_), 0);
    const int raw = ::open(path_, O_RDONLY);
    ASSERT_GE(raw, 0);
    if (state == "hint-warm") {
      ASSERT_EQ(::fcntl(raw, F_NOCACHE, 1), 0);
      ASSERT_EQ(::fcntl(raw, F_RDAHEAD, 0), 0);
    }
    const auto resident_before = ResidentBytes(fd_, kSize);
    testing_util::AppleIOGate("ZVEC_APPLE_IO_CONTROL", state, "before");
    std::ostringstream bytes_before;
    DiskBytes(bytes_before);
    const auto start = std::chrono::steady_clock::now();
    IOContext context = nullptr;
    for (size_t i = 0; i < kReads; ++i) {
      const size_t block = (i * 1009) % (kSize / kBlock);
      if (production) {
        std::vector<AlignedRead> requests{AlignedRead(block * kBlock, kBlock, buffer_)};
        ASSERT_EQ(reader.read(requests, context), 0);
      } else {
        ASSERT_EQ(::pread(raw, buffer_, kBlock, block * kBlock), static_cast<ssize_t>(kBlock));
      }
      const uint64_t expected = 0x47a91bd39e510000ULL + block;
      for (size_t j = 0; j < kBlock; j += 8) {
        uint64_t actual;
        std::memcpy(&actual, static_cast<char *>(buffer_) + j, 8);
        ASSERT_EQ(actual, expected);
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ostringstream bytes_after;
    DiskBytes(bytes_after);
    testing_util::AppleIOGate("ZVEC_APPLE_IO_CONTROL", state, "after");
    const auto resident_after = ResidentBytes(fd_, kSize);
    const auto requested_resident_after = ResidentBytes(fd_, kSize, true);
    ASSERT_EQ(::close(raw), 0);
    if (!first) cases << ',';
    first = false;
    cases << "{\"state\":\"" << state << "\",\"operations\":" << kReads
          << ",\"requested_bytes\":" << kReads * kBlock
          << ",\"io_before\":" << bytes_before.str() << ",\"io_after\":" << bytes_after.str()
          << ",\"resident_before\":" << resident_before << ",\"resident_after\":" << resident_after
          << ",\"requested_resident_after\":" << requested_resident_after
          << ",\"invalidation_requested\":" << (production && reader.cache_invalidation_requested() ? "true" : "false")
          << ",\"nocache_requested\":" << (reader.nocache_requested() ? "true" : "false")
          << ",\"fallback_errno\":" << reader.cache_fallback_errno()
          << ",\"seconds\":" << seconds << '}';
  }
  std::ostringstream record;
  record << "ZVEC_APPLE_IO_RESULT {\"fixture_bytes\":" << kSize
         << ",\"page_size\":" << ::sysconf(_SC_PAGESIZE)
         << ",\"payload_validated\":true,\"cases\":[" << cases.str() << "]}\n";
  const char *path = std::getenv("ZVEC_APPLE_IO_RESULT_PATH");
  if (path != nullptr) {
    std::ofstream out(path);
    out << record.str();
    out.close();
    ASSERT_TRUE(out.good());
  }
  std::cout << record.str() << std::flush;
}
}  // namespace
}  // namespace zvec::core
#endif

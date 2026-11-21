// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if (defined(__linux) || defined(__linux__)) && !defined(__ANDROID__)

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <ailego/io/iouring_loader.h>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include "../buffer/pool_payload_budget.h"

namespace zvec {
namespace ailego {

// Stack-owned control fields exercise staging on hosts without io_uring.
// Submissions still use real positioned reads into the ring-owned mapping.
class IoUringStagingTestPeer {
 public:
  void attach(IoUringRing &ring, int fd) {
    active_ = this;
    ring.ring_fd_ = ::dup(fd);
    ring.sq_head_ = &submitted_;
    ring.sq_tail_ = &tail_;
    ring.sq_ring_mask_ = &mask_;
    ring.sq_array_ = indices_.data();
    ring.sqes_ = requests_.data();
    ring.cq_head_ = &completed_;
    ring.cq_tail_ = &available_;
    ring.cq_ring_mask_ = &mask_;
    ring.cqes_ = events_.data();
    ring.sq_entries_ = ring.cq_entries_ = indices_.size();
    ring.enter_override_ = enter;
    for (size_t i = 0; i < indices_.size(); ++i) indices_[i] = i;
  }

  void stall_after_first_submission() {
    stall_submission_ = true;
  }

  void resume_submissions() {
    stall_submission_ = false;
  }

 private:
  static int enter(int, unsigned count, unsigned, unsigned) {
    auto &peer = *active_;
    if (peer.stall_submission_) {
      if (count == 0) {
        peer.available_ = peer.submitted_;
        return 0;
      }
      if (peer.submitted_ != 0) {
        errno = EAGAIN;
        return -1;
      }
      count = 1;
    }
    for (unsigned i = 0; i < count; ++i) {
      const unsigned slot = peer.submitted_++ & peer.mask_;
      const auto &request = peer.requests_[slot];
      auto &event = peer.events_[slot];
      event.user_data = request.user_data;
      event.res = ::pread(request.fd, reinterpret_cast<void *>(request.addr),
                          request.len, request.off);
    }
    if (!peer.stall_submission_) peer.available_ = peer.submitted_;
    return static_cast<int>(count);
  }
  inline static thread_local IoUringStagingTestPeer *active_{nullptr};
  bool stall_submission_{false};
  unsigned submitted_{0}, tail_{0}, completed_{0}, available_{0}, mask_{7};
  std::array<unsigned, 8> indices_{};
  std::array<io_uring_sqe, 8> requests_{};
  std::array<io_uring_cqe, 8> events_{};
};

namespace {

constexpr size_t kBlockSize = 512;
constexpr size_t kBlockCount = 4;

class IoUringLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    old_capacity_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(16 * PoolPayloadBudget::page_buffer_size()));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_capacity_));
  }
  PoolPayloadBudget &pool_ = PoolPayloadBudget::get_instance();
  size_t old_capacity_{0};
};

class TemporaryFile {
 public:
  TemporaryFile() : fd_(::mkstemp(path_)) {}

  ~TemporaryFile() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    ::unlink(path_);
  }

  int fd() const {
    return fd_;
  }

 private:
  char path_[64] = "IoUringLoaderTest.XXXXXX";
  int fd_;
};

void *allocate_aligned(size_t size) {
  void *buffer = nullptr;
  if (::posix_memalign(&buffer, kBlockSize, size) != 0) {
    return nullptr;
  }
  std::memset(buffer, 0, size);
  return buffer;
}

TEST_F(IoUringLoaderTest, ReadsBatchAndZeroPadsExpectedShortRead) {
  IoUringRing ring;
  if (!ring.setup(/*entries=*/8)) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  constexpr size_t kTailSize = 123;
  std::vector<uint8_t> source(kBlockSize + kTailSize);
  std::fill(source.begin(), source.begin() + kBlockSize, 0x3c);
  std::fill(source.begin() + kBlockSize, source.end(), 0x7d);
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));

  void *backing = allocate_aligned(3 * kBlockSize);
  ASSERT_NE(backing, nullptr);
  for (size_t bias : {size_t{0}, size_t{1}}) {
    auto *output = static_cast<uint8_t *>(backing) + bias;
    std::vector<IoUringRead> requests;
    requests.emplace_back(/*offset=*/0, kBlockSize, output);
    requests.emplace_back(kBlockSize, kBlockSize, output + kBlockSize,
                          kTailSize);
    for (bool asynchronous : {false, true}) {
      SCOPED_TRACE(asynchronous);
      std::memset(output, 0xac, 2 * kBlockSize);
      if (!asynchronous) {
        ASSERT_EQ(0, ring.execute(file.fd(), requests));
      } else {
        ASSERT_EQ(
            0, ring.submit_reads(file.fd(), requests.data(), requests.size()));
        std::array<bool, 2> seen{};
        size_t total = 0;
        while (total < requests.size()) {
          std::array<uint32_t, 2> completed{};
          const int count =
              ring.get_completed_reads(1, completed.data(), completed.size());
          ASSERT_GT(count, 0);
          for (int i = 0; i < count; ++i) {
            const size_t id = completed[i];
            ASSERT_LT(id, requests.size());
            ASSERT_FALSE(seen[id]);
            seen[id] = true;
            const size_t bytes = id == 0 ? kBlockSize : kTailSize;
            EXPECT_EQ(
                0, std::memcmp(static_cast<uint8_t *>(output) + id * kBlockSize,
                               source.data() + id * kBlockSize, bytes));
          }
          total += count;
        }
      }
      EXPECT_EQ(0u, pool_.used());
      EXPECT_EQ(std::memcmp(output, source.data(), kBlockSize), 0);
      const auto *tail = static_cast<const uint8_t *>(output) + kBlockSize;
      EXPECT_EQ(std::memcmp(tail, source.data() + kBlockSize, kTailSize), 0);
      EXPECT_TRUE(std::all_of(tail + kTailSize, tail + kBlockSize,
                              [](uint8_t value) { return value == 0; }));
    }
  }
  std::free(backing);
}

TEST_F(IoUringLoaderTest,
       StagingAcceptsUnalignedDestinationsWithoutNativeRing) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  constexpr size_t tail = 123;
  std::vector<uint8_t> source(kBlockSize + tail, 0x6d);
  ASSERT_EQ(static_cast<ssize_t>(source.size()),
            ::pwrite(file.fd(), source.data(), source.size(), 0));
  IoUringStagingTestPeer peer;
  IoUringRing ring;
  peer.attach(ring, file.fd());
  void *backing = allocate_aligned(3 * kBlockSize);
  ASSERT_NE(nullptr, backing);
  std::unique_ptr<void, decltype(&std::free)> cleanup(backing, std::free);
  for (bool asynchronous : {true, false}) {
    for (size_t bias : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(::testing::Message() << asynchronous << ":" << bias);
      auto *output = static_cast<uint8_t *>(backing) + bias;
      std::memset(output, 0xac, 2 * kBlockSize);
      const IoUringRead reads[] = {
          {0, kBlockSize, output},
          {kBlockSize, kBlockSize, output + kBlockSize, tail},
      };
      if (asynchronous) {
        ASSERT_EQ(0, ring.submit_reads(file.fd(), reads, 2));
        uint32_t completed[2]{};
        ASSERT_EQ(2, ring.get_completed_reads(2, completed, 2));
      } else {
        ASSERT_EQ(0, ring.execute(file.fd(), reads, 2));
      }
      EXPECT_EQ(0, std::memcmp(output, source.data(), source.size()));
      EXPECT_TRUE(std::all_of(output + source.size(), output + 2 * kBlockSize,
                              [](uint8_t value) { return value == 0; }));
      EXPECT_EQ(0u, pool_.used());
    }
  }

  auto &budget = MemoryLimitPool::get_instance();
  std::array<char *, 2> pins{};
  for (auto &pin : pins)
    ASSERT_TRUE(budget.try_acquire_buffer(kVectorPageSize, pin));
  auto release = ScopeGuard::Make([&] {
    for (char *pin : pins) budget.release_buffer(pin, kVectorPageSize);
  });
  for (size_t remaining : {size_t{0}, size_t{1024}}) {
    SCOPED_TRACE(remaining);
    ASSERT_GE(budget.available(), remaining);
    std::array<BudgetedAlignedBuffer, 32> pressure;
    for (auto &buffer : pressure) {
      const size_t bytes =
          std::min(budget.available() - remaining, size_t{128} << 10);
      buffer.resize(bytes, AILEGO_CACHE_LINE_SIZE);
      if (bytes) std::memset(buffer.data(), 'p', bytes);
    }
    ASSERT_EQ(remaining, budget.available());
    const auto before = budget.stats();
    auto *output = static_cast<uint8_t *>(backing) + 1;
    std::memset(output, 0xac, 2 * kBlockSize);
    const IoUringRead read{0, kBlockSize, output};
    EXPECT_EQ(-1, ring.execute(file.fd(), &read, 1));
    EXPECT_EQ(-1, ring.submit_reads(file.fd(), &read, 1));
    EXPECT_FALSE(ring.has_pending_reads());
    EXPECT_TRUE(ring.is_valid());
    EXPECT_EQ(before.capacity_waits, budget.stats().capacity_waits);
    EXPECT_TRUE(std::all_of(output, output + 2 * kBlockSize,
                            [](uint8_t value) { return value == 0xac; }));
    EXPECT_EQ(before.used, budget.used());
  }
  const IoUringRead retry{0, kBlockSize, backing};
  ASSERT_EQ(0, ring.execute(file.fd(), &retry, 1));
  EXPECT_EQ(0, std::memcmp(backing, source.data(), kBlockSize));
}

TEST_F(IoUringLoaderTest, StalledSubmissionDrainsAcceptedReadsBeforeRetry) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::array<char, 2 * kBlockSize> source;
  source.fill('x');
  ASSERT_EQ(static_cast<ssize_t>(source.size()),
            ::pwrite(file.fd(), source.data(), source.size(), 0));
  IoUringStagingTestPeer peer;
  IoUringRing ring;
  peer.attach(ring, file.fd());
  peer.stall_after_first_submission();

  std::array<char, 2 * kBlockSize> output{};
  const IoUringRead reads[] = {
      {0, kBlockSize, output.data()},
      {kBlockSize, kBlockSize, output.data() + kBlockSize},
  };
  ASSERT_EQ(-1, ring.execute(file.fd(), reads, 2));
  EXPECT_EQ('x', output.front());
  EXPECT_EQ(0, output.back());
  EXPECT_TRUE(ring.is_valid());
  EXPECT_EQ(0u, pool_.used());

  peer.resume_submissions();
  ASSERT_EQ(0, ring.execute(file.fd(), reads, 2));
  EXPECT_EQ(source, output);
}

TEST_F(IoUringLoaderTest, WritesScatteredBatch) {
  IoUringRing ring;
  if (!ring.setup(/*entries=*/8)) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(kBlockCount * kBlockSize);
  for (size_t block = 0; block < kBlockCount; ++block) {
    std::memset(source.data() + block * kBlockSize, static_cast<int>(block + 1),
                kBlockSize);
  }

  const std::array<size_t, kBlockCount> order = {3, 0, 2, 1};
  std::vector<IoUringWrite> requests;
  for (size_t block : order) {
    requests.emplace_back(block * kBlockSize, kBlockSize,
                          source.data() + block * kBlockSize);
  }
  ASSERT_EQ(ring.execute_writes(file.fd(), requests), 0);
  EXPECT_EQ(
      BudgetedMappedBuffer::mapping_size(kBlockCount * kIoUringStagingAlign),
      pool_.external_used());

  std::vector<uint8_t> output(source.size());
  ASSERT_EQ(::pread(file.fd(), output.data(), output.size(), 0),
            static_cast<ssize_t>(output.size()));
  EXPECT_EQ(output, source);
}

}  // namespace
}  // namespace ailego
}  // namespace zvec

#endif  // (defined(__linux) || defined(__linux__)) && !defined(__ANDROID__)

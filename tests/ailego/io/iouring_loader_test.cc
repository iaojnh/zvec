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
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include "../buffer/pool_payload_budget.h"

namespace zvec {
namespace ailego {
class IoUringStagingTestPeer {
 public:
  static void bind_async(IoUringRing &ring, int fd, io_uring_sqe *submissions,
                         io_uring_cqe *completions, unsigned *indices,
                         unsigned *sq_tail, unsigned *cq_head,
                         unsigned *cq_tail, unsigned *mask,
                         int (*enter)(int, unsigned, unsigned, unsigned)) {
    ring.ring_fd_ = ::dup(fd);
    ring.sq_entries_ = 4;
    ring.sq_tail_ = sq_tail;
    ring.sq_ring_mask_ = mask;
    ring.sq_array_ = indices;
    ring.sqes_ = submissions;
    ring.cq_head_ = cq_head;
    ring.cq_tail_ = cq_tail;
    ring.cq_ring_mask_ = mask;
    ring.cqes_ = completions;
    ring.enter_override_ = enter;
  }
  static void *data(const IoUringRing &ring) {
    return ring.staging_;
  }
  static size_t size(const IoUringRing &ring) {
    return ring.staging_size_;
  }
  static int execute_with_duplicate_completion(IoUringRing &ring, int fd,
                                               const IoUringWrite *writes) {
    std::array<io_uring_sqe, 2> submissions{};
    std::array<io_uring_cqe, 2> completions{};
    for (auto &event : completions) event.res = writes[0].len;
    unsigned indices[] = {0, 1};
    unsigned sq_tail = 0, cq_head = 0, cq_tail = 2, mask = 1;
    ring.ring_fd_ = ::dup(fd);
    ring.sq_entries_ = 2;
    ring.sq_tail_ = &sq_tail;
    ring.sq_ring_mask_ = &mask;
    ring.sq_array_ = indices;
    ring.sqes_ = submissions.data();
    ring.cq_head_ = &cq_head;
    ring.cq_tail_ = &cq_tail;
    ring.cq_ring_mask_ = &mask;
    ring.cqes_ = completions.data();
    return ring.execute_writes(fd, writes, 2);
  }
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

struct FakeAsyncRing {
  std::array<io_uring_sqe, 4> submissions{};
  std::array<io_uring_cqe, 4> completions{};
  unsigned indices[4]{0, 1, 2, 3};
  unsigned tail{0}, head{0}, complete_tail{0}, mask{3}, accepted{0};
  bool partial_failure{false}, short_read{false};
  static FakeAsyncRing *active;
  static int enter(int, unsigned count, unsigned minimum, unsigned) {
    auto &queue = *active;
    if (count) {
      if (queue.partial_failure && queue.accepted) {
        errno = EIO;
        return -1;
      }
      const unsigned submitted = queue.partial_failure ? 1 : count;
      queue.accepted += submitted;
      return submitted;
    }
    if (minimum && queue.complete_tail == 0) {
      // Deliberately complete in reverse order. submit() never waits here.
      for (unsigned n = 0; n < queue.accepted; ++n) {
        const unsigned i = queue.accepted - n - 1;
        const auto &request = queue.submissions[i];
        std::memset(reinterpret_cast<void *>(request.addr),
                    0x31 + request.off / 512, request.len);
        queue.completions[n].user_data = request.user_data;
        queue.completions[n].res =
            queue.short_read && n == 0 ? request.len - 1 : request.len;
      }
      queue.complete_tail = queue.accepted;
    }
    return 0;
  }
  void bind(IoUringRing &ring, int fd) {
    active = this;
    IoUringStagingTestPeer::bind_async(ring, fd, submissions.data(),
                                       completions.data(), indices, &tail,
                                       &head, &complete_tail, &mask, enter);
  }
};
FakeAsyncRing *FakeAsyncRing::active = nullptr;

TEST_F(IoUringLoaderTest, AsyncCancellationAndShortReadsReleaseAllBacking) {
  const int fd = ::open("/dev/null", O_RDONLY);
  ASSERT_GE(fd, 0);
  const auto close_fd = ScopeGuard::Make([&] { ::close(fd); });
  for (unsigned mode : {1u, 2u}) {
    SCOPED_TRACE(mode);
    FakeAsyncRing queue;
    queue.partial_failure = mode == 1;
    queue.short_read = mode == 2;
    IoUringRing ring;
    queue.bind(ring, fd);
    alignas(4096) std::array<unsigned char, 1024> output{};
    const std::array<IoUringRead, 2> reads{
        {{0, 512, output.data()}, {512, 512, output.data() + 512}}};
    const int submitted = ring.submit_reads(fd, reads.data(), reads.size());
    if (mode == 1)
      EXPECT_NE(0, submitted);
    else {
      ASSERT_EQ(0, submitted);
      std::array<uint32_t, 2> completed{};
      EXPECT_LT(ring.get_completed_reads(1, completed.data(), 2), 0);
    }
    EXPECT_FALSE(ring.has_pending_reads());
    EXPECT_EQ(0, output[0]);
    EXPECT_EQ(0, output[512]);
    EXPECT_EQ(0u, pool_.used());
  }
}

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

TEST_F(IoUringLoaderTest, DuplicateWriteCompletionQuarantinesStaging) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  IoUringRing ring;
  ASSERT_TRUE(ring.ensure_staging(2 * kIoUringStagingAlign));
  void *staging = IoUringStagingTestPeer::data(ring);
  const size_t charge = IoUringStagingTestPeer::size(ring);
  const auto cleanup = ScopeGuard::Make([&] {
    // The fake completion queue never submitted a kernel operation.
    if (!IoUringStagingTestPeer::data(ring)) {
      File::MemoryUnmap(staging, charge);
      pool_.release_external(charge);
    }
  });
  std::array<char, kBlockSize> data{};
  IoUringWrite writes[] = {{0, data.size(), data.data()},
                          {data.size(), data.size(), data.data()}};
  EXPECT_EQ(IoUringRing::kQuarantined,
            IoUringStagingTestPeer::execute_with_duplicate_completion(
                ring, file.fd(), writes));
  EXPECT_FALSE(ring.is_valid());
  EXPECT_EQ(charge, pool_.external_used());
}

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

  void *output = allocate_aligned(2 * kBlockSize);
  ASSERT_NE(output, nullptr);
  std::vector<IoUringRead> requests;
  requests.emplace_back(/*offset=*/0, kBlockSize, output);
  requests.emplace_back(kBlockSize, kBlockSize,
                        static_cast<uint8_t *>(output) + kBlockSize, kTailSize);

  for (bool asynchronous : {false, true}) {
    SCOPED_TRACE(asynchronous);
    std::memset(output, 0xac, 2 * kBlockSize);
    if (!asynchronous) {
      ASSERT_EQ(0, ring.execute(file.fd(), requests));
    } else {
      ASSERT_EQ(0, ring.submit_reads(file.fd(), requests.data(), requests.size()));
      std::array<bool, 2> seen{};
      size_t total = 0;
      while (total < requests.size()) {
        std::array<uint32_t, 2> completed{};
        const int count = ring.get_completed_reads(1, completed.data(), completed.size());
        ASSERT_GT(count, 0);
        for (int i = 0; i < count; ++i) {
          const size_t id = completed[i];
          ASSERT_LT(id, requests.size());
          ASSERT_FALSE(seen[id]);
          seen[id] = true;
          const size_t bytes = id == 0 ? kBlockSize : kTailSize;
          EXPECT_EQ(0, std::memcmp(static_cast<uint8_t *>(output) + id * kBlockSize,
                                   source.data() + id * kBlockSize, bytes));
        }
        total += count;
      }
    }
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0u, IoUringStagingTestPeer::size(ring));
    EXPECT_EQ(std::memcmp(output, source.data(), kBlockSize), 0);
    const auto *tail = static_cast<const uint8_t *>(output) + kBlockSize;
    EXPECT_EQ(std::memcmp(tail, source.data() + kBlockSize, kTailSize), 0);
    EXPECT_TRUE(std::all_of(tail + kTailSize, tail + kBlockSize,
                            [](uint8_t value) { return value == 0; }));
  }
  std::free(output);
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

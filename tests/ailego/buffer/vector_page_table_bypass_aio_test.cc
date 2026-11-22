// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <ailego/io/libaio_loader.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "pool_payload_budget.h"

namespace zvec::ailego {

struct VecBufferPoolTestPeer {
  static void use_libaio(VecBufferPool &pool) {
    pool.aio_enabled_ = true;
    pool.io_backend_type_ = IOBackendType::kLibAio;
  }
  static bool load_pages(VecBufferPool &pool) {
    const block_id_t pages[] = {0, 1, 2, 3};
    return pool.load_pages_aio(pages, 4, VecBufferPool::kLowPriority);
  }
  static void evict_pages(VecBufferPool &pool) {
    pool.page_table_.force_evict_all_loaded();
  }
};

namespace {

struct FakeAio {
  enum class Mode {
    SUCCESS,
    PARTIAL,
    SHORT_READ,
    WAIT_ERROR,
    DESTROY_ERROR,
    DUPLICATE,
    REJECTED_RING
  };
  static FakeAio *active;
  Mode mode{Mode::SUCCESS};
  std::array<uint32_t, 8> header{0, 16, 0, 0, 0xa10a10a1, 0, 0, 32};
  std::array<iocb, 16> requests{};
  std::array<iocb *, 16> request_addresses{};
  size_t submitted{0};
  size_t pending{0};
  size_t setup_calls{0};
  size_t submit_calls{0};
  size_t destroy_calls{0};

  static int setup(int, io_context_t *context) {
    ++active->setup_calls;
    if (active->mode == Mode::REJECTED_RING) active->header[4] = 0;
    *context = reinterpret_cast<io_context_t>(active->header.data());
    return 0;
  }
  static int destroy(io_context_t) {
    ++active->destroy_calls;
    if ((active->mode == Mode::DESTROY_ERROR ||
         active->mode == Mode::DUPLICATE ||
         active->mode == Mode::REJECTED_RING) &&
        active->destroy_calls == 1)
      return -EIO;
    active->pending = 0;
    return 0;
  }
  static int submit(io_context_t, long count, iocb **requests) {
    ++active->submit_calls;
    EXPECT_EQ(0u, active->pending);
    const size_t accepted =
        active->mode == Mode::PARTIAL
            ? std::min<size_t>(2, static_cast<size_t>(count))
            : static_cast<size_t>(count);
    for (size_t i = 0; i < accepted; ++i) {
      active->requests[i] = *requests[i];
      active->request_addresses[i] = requests[i];
    }
    active->submitted = active->pending = accepted;
    return static_cast<int>(accepted);
  }
  static int getevents(io_context_t, long, long, io_event *events, timespec *) {
    if (active->mode == Mode::WAIT_ERROR || active->mode == Mode::DESTROY_ERROR)
      return -EIO;
    // Deliberately return one completion at a time, in reverse request order.
    size_t index = --active->pending;
    if (active->mode == Mode::DUPLICATE && index == 0)
      index = active->submitted - 1;
    const auto &request = active->requests[index];
    const ssize_t bytes = ::pread(request.aio_fildes, request.u.c.buf,
                                  request.u.c.nbytes, request.u.c.offset);
    events[0] = {};
    events[0].data = request.data;
    events[0].obj = active->request_addresses[index];
    events[0].res = active->mode == Mode::SHORT_READ ? bytes - 512 : bytes;
    return 1;
  }
};

FakeAio *FakeAio::active = nullptr;

class BypassAioTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &loader = LibAioLoader::Instance();
    if (!loader.load()) GTEST_SKIP() << "libaio not installed";
    setup_ = loader.io_setup;
    destroy_ = loader.io_destroy;
    submit_ = loader.io_submit;
    getevents_ = loader.io_getevents;
    loader.io_setup = FakeAio::setup;
    loader.io_destroy = FakeAio::destroy;
    loader.io_submit = FakeAio::submit;
    loader.io_getevents = FakeAio::getevents;
    FakeAio::active = &fake_;
    ASSERT_EQ(0, PoolPayloadBudget::get_instance().init(8UL * 1024 * 1024));
    char path[] = "bypass_aio_test_XXXXXX";
    const int fd = ::mkstemp(path);
    ASSERT_GE(fd, 0);
    path_ = path;
    std::vector<char> page(kVectorPageSize);
    for (size_t i = 0; i < 4; ++i) {
      std::fill(page.begin(), page.end(), static_cast<char>(i + 1));
      EXPECT_EQ(static_cast<ssize_t>(page.size()),
                ::write(fd, page.data(), page.size()));
    }
    EXPECT_EQ(0, ::close(fd));
  }
  void TearDown() override {
    // Every test joins its worker before restoring callbacks: the worker's
    // TLS context must use the same destroy function that created it.
    if (setup_) {
      auto &loader = LibAioLoader::Instance();
      loader.io_setup = setup_;
      loader.io_destroy = destroy_;
      loader.io_submit = submit_;
      loader.io_getevents = getevents_;
    }
    FakeAio::active = nullptr;
    if (!path_.empty()) std::remove(path_.c_str());
  }
  void check_read(VecBufferPool &pool, size_t cached_pages) {
    std::vector<char> output(4 * kVectorPageSize + 1, 'X');
    char *base = output.data();
    if (reinterpret_cast<uintptr_t>(base) % kVectorPageSize == 0) ++base;
    std::array<VecBufferPool::ReadRange, 4> ranges{};
    for (size_t i = 0; i < ranges.size(); ++i)
      ranges[i] = {i * kVectorPageSize, kVectorPageSize,
                   base + i * kVectorPageSize};
    ASSERT_TRUE(pool.read_ranges_bypass(ranges.data(), ranges.size()));
    for (size_t i = 0; i < ranges.size(); ++i) {
      EXPECT_TRUE(std::all_of(
          base + i * kVectorPageSize, base + (i + 1) * kVectorPageSize,
          [i](char byte) { return byte == static_cast<char>(i + 1); }));
      EXPECT_EQ(i < cached_pages, pool.is_page_resident(i));
    }
  }
  void run(FakeAio::Mode mode) {
    fake_.mode = mode;
    VecBufferPool pool(path_, false);
    ASSERT_EQ(0, pool.init());
    VecBufferPoolTestPeer::use_libaio(pool);
    // Create backing once before the lifetime assertion; slab headers remain
    // charged after returning their page slots to the free list.
    std::array<char *, 4> warm{};
    auto &memory = MemoryLimitPool::get_instance();
    ASSERT_EQ(warm.size(), memory.batch_acquire_buffers(
                               kVectorPageSize, warm.data(), warm.size()));
    for (char *page : warm) memory.release_buffer(page, kVectorPageSize);
    auto &payload = PoolPayloadBudget::get_instance();
    const size_t before = payload.used();
    const size_t cached_pages = mode == FakeAio::Mode::PARTIAL ? 2 : 0;
    const size_t quarantined_pages =
        mode == FakeAio::Mode::DESTROY_ERROR ||
                mode == FakeAio::Mode::DUPLICATE
            ? 4
            : 0;
    const bool retried_destroy =
        quarantined_pages || mode == FakeAio::Mode::REJECTED_RING;
    std::thread worker([&] {
      EXPECT_FALSE(VecBufferPoolTestPeer::load_pages(pool));
      EXPECT_EQ(before + (cached_pages + quarantined_pages) * kVectorPageSize,
                payload.used());
      check_read(pool, cached_pages);
      if (retried_destroy) {
        const auto calls = fake_.submit_calls;
        EXPECT_FALSE(VecBufferPoolTestPeer::load_pages(pool));
        EXPECT_EQ(calls,
                  fake_.submit_calls);  // Quarantined context not reused.
      }
    });
    worker.join();
    EXPECT_EQ(mode == FakeAio::Mode::REJECTED_RING ? 0u : 1u,
              fake_.submit_calls);
    EXPECT_EQ(0u, fake_.pending);
    VecBufferPoolTestPeer::evict_pages(pool);
    EXPECT_EQ(before, payload.used());
    EXPECT_EQ(retried_destroy ? 2u : 1u, fake_.destroy_calls);
  }

  FakeAio fake_;
  std::string path_;
  aio_setup_fn setup_{nullptr};
  aio_destroy_fn destroy_{nullptr};
  aio_submit_fn submit_{nullptr};
  aio_getevents_fn getevents_{nullptr};
};

TEST_F(BypassAioTest, CacheAioDrainsPartialSubmissionBeforeFallback) {
  run(FakeAio::Mode::PARTIAL);
}
TEST_F(BypassAioTest, CacheAioRejectsShortReadWithoutPublishingPages) {
  run(FakeAio::Mode::SHORT_READ);
}
TEST_F(BypassAioTest, CacheAioRejectsRingWithoutSubmittingRequests) {
  run(FakeAio::Mode::REJECTED_RING);
}
TEST_F(BypassAioTest, CacheAioRetiresContextOnCompletionError) {
  run(FakeAio::Mode::WAIT_ERROR);
}
TEST_F(BypassAioTest, CacheAioQuarantinesUntilDestroyWithoutReusingContext) {
  run(FakeAio::Mode::DESTROY_ERROR);
}
TEST_F(BypassAioTest, CacheAioDuplicateCompletionKeepsKernelDestinations) {
  run(FakeAio::Mode::DUPLICATE);
}
TEST_F(BypassAioTest, BypassWithNativeBackendReadsCrossPageAtFullBudget) {
  VecBufferPool pool(path_, false);
  ASSERT_EQ(0, pool.init());
  VecBufferPoolTestPeer::use_libaio(pool);
  auto &memory = MemoryLimitPool::get_instance();
  std::array<BudgetedAlignedBuffer, 64> pressure;
  for (auto &buffer : pressure) {
    const size_t bytes = std::min(memory.available(), size_t{128} << 10);
    buffer.resize(bytes, AILEGO_CACHE_LINE_SIZE);
    if (bytes) std::memset(buffer.data(), 'p', bytes);
  }
  ASSERT_EQ(0u, memory.available());

  // The aligned file ranges were eligible for native AIO even though their
  // destinations require a bounce page with O_DIRECT.
  check_read(pool, 0);

  std::array<char, 53> across{}, other{};
  const VecBufferPool::ReadRange ranges[] = {
      {kVectorPageSize - 19, across.size(), across.data()},
      {3 * kVectorPageSize + 7, other.size(), other.data()}};
  ASSERT_TRUE(pool.read_ranges_bypass(ranges, 2));
  EXPECT_EQ(std::string(19, '\1') + std::string(across.size() - 19, '\2'),
            std::string(across.data(), across.size()));
  EXPECT_EQ(std::string(other.size(), '\4'),
            std::string(other.data(), other.size()));
  EXPECT_EQ(0u, fake_.setup_calls);
  EXPECT_EQ(0u, fake_.submit_calls);
  EXPECT_EQ(0u, memory.available());
}
}  // namespace
}  // namespace zvec::ailego

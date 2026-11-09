// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <ailego/pattern/defer.h>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/io/file.h>
#if defined(__APPLE__)
#include <sys/mman.h>
#include <TargetConditionals.h>
#endif
#if defined(__linux__) && !defined(__ANDROID__)
#include <sys/mman.h>
#include <ailego/io/iouring_loader.h>
#endif

namespace zvec::ailego {

// Pause between the real admission steps without sleeps or production hooks.
struct MemoryLimitPoolTestPeer {
  static size_t reserve_margin(MemoryLimitPool &pool, size_t capacity) {
    return pool.reserve_margin(capacity);
  }
  static bool reserve_page(MemoryLimitPool &pool, size_t bytes) {
    return pool.try_reserve_page_used(bytes);
  }
  static bool reserve_backing(MemoryLimitPool &pool, size_t bytes) {
    return pool.try_reserve_committed(bytes);
  }
  static void release_backing(MemoryLimitPool &pool, size_t bytes) {
    pool.release_committed(bytes);
  }
  static void stop_background(MemoryLimitPool &pool) {
    pool.stop_background_evictor();
  }
  static bool configure_rss(MemoryLimitPool &pool, size_t limit, size_t rss) {
    rss_sample = rss;
    return pool.process_rss_.configure(limit, +[] { return rss_sample; });
  }
  static void sample_rss(MemoryLimitPool &pool, size_t rss) {
    rss_sample = rss;
    pool.process_rss_.refresh(true);
  }
  static size_t rss_available(MemoryLimitPool &pool) {
    return pool.process_rss_.available();
  }
  static ProcessRssBudget::ResidentSample slab_resident_sample() {
    return MemoryLimitPool::read_slab_resident_sample();
  }
  static ProcessRssBudget::ResidentSample sample_with_slab_lock(
      MemoryLimitPool &pool) {
    std::lock_guard<std::mutex> lock(pool.slab_mutex_);
    return pool.read_slab_resident_sample();
  }
  static void refresh_rss(MemoryLimitPool &pool) {
    pool.process_rss_.refresh(true);
  }
  static size_t trim_retained(MemoryLimitPool &pool, size_t bytes) {
    return pool.trim_retained_blocks(bytes);
  }
  static size_t trim_free(MemoryLimitPool &pool, size_t bytes) {
    return pool.trim_free_buffers(bytes);
  }
  inline static size_t rss_sample = 0;
};

namespace {

class PendingReservation {
 public:
  PendingReservation(MemoryLimitPool &pool, size_t bytes, bool backing = false)
      : pool_(pool), backing_(backing) {
    if (backing ? MemoryLimitPoolTestPeer::reserve_backing(pool, bytes)
                : MemoryLimitPoolTestPeer::reserve_page(pool, bytes)) {
      bytes_ = bytes;
    }
  }
  ~PendingReservation() {
    if (!bytes_) return;
    if (backing_) {
      MemoryLimitPoolTestPeer::release_backing(pool_, bytes_);
    } else {
      pool_.release_buffer(nullptr, bytes_);
    }
  }
  bool acquired() const {
    return bytes_ != 0;
  }

 private:
  MemoryLimitPool &pool_;
  bool backing_;
  size_t bytes_{0};
};

class PoolAdmissionTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(0));
    baseline_ = pool_.queue_allocated();
    ASSERT_EQ(0, pool_.init(baseline_ + 4 * page_));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
    EXPECT_EQ(pool_.queue_metadata_used() + pool_.stats().slab_header_bytes,
              pool_.used());
    expect_bounded();
    EXPECT_EQ(0, pool_.init(0));
  }
  void expect_bounded() const {
    EXPECT_LE(pool_.used(), pool_.capacity());
    EXPECT_LE(pool_.committed(), pool_.capacity());
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  const size_t page_ = MemoryLimitPool::page_buffer_size();
  size_t baseline_{0};
};

#if defined(__APPLE__) && TARGET_OS_OSX
TEST_F(PoolAdmissionTest, NativeResidencyWitnessTracksDiscardReuseAndBusySample) {
  ASSERT_EQ(0, pool_.init(64UL << 20));
  MemoryLimitPoolTestPeer::stop_background(pool_);
  const auto release = [this](char *data) {
    pool_.release_buffer(data, page_);
  };
  const auto acquire = [&] {
    char *data = nullptr;
    pool_.try_acquire_buffer(page_, data);
    return std::unique_ptr<char, decltype(release)>(data, release);
  };
  auto page = acquire();
  auto neighbor = acquire();
  auto untouched = acquire();
  ASSERT_NE(nullptr, page);
  ASSERT_NE(nullptr, neighbor);
  ASSERT_NE(nullptr, untouched);
  std::memset(page.get(), 7, page_);
  std::memset(neighbor.get(), 11, page_);
  const auto full = MemoryLimitPoolTestPeer::slab_resident_sample();
  ASSERT_EQ(3 * page_, full.owned_resident);  // Header and both touched pages.
  const auto busy = MemoryLimitPoolTestPeer::sample_with_slab_lock(pool_);
  EXPECT_GT(busy.rss, 0u);
  EXPECT_EQ(0u, busy.owned_resident);
  char *address = page.get();
  page.reset();
  const size_t charge = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(charge));
  char residency = 0;
  EXPECT_EQ(0, ::mincore(address, page_, &residency));
  EXPECT_EQ(0, residency & 1);
  const auto discarded = MemoryLimitPoolTestPeer::slab_resident_sample();
  EXPECT_EQ(page_, full.owned_resident - discarded.owned_resident);
  EXPECT_EQ(11, neighbor.get()[page_ - 1]);
  pool_.release_external(charge);
  page.reset(acquire().release());
  ASSERT_NE(nullptr, page);
  std::memset(page.get(), 13, page_);
  const auto reused = MemoryLimitPoolTestPeer::slab_resident_sample();
  EXPECT_EQ(full.owned_resident, reused.owned_resident);
  EXPECT_LE(reused.owned_resident, reused.rss);
}
#endif

TEST_F(PoolAdmissionTest, FixedConsumersRejectPendingPageReservations) {
  // Plain function wrappers also work under ARM64 UBSan's indirect-call
  // instrumentation; the test still exercises each real reservation API.
  using Charge = bool (*)(MemoryLimitPool &, size_t);
  using Release = void (*)(MemoryLimitPool &, size_t);
  struct Consumer {
    Charge charge;
    Release release;
  };
  const auto release_external = +[](MemoryLimitPool &pool, size_t bytes) {
    pool.release_external(bytes);
  };
  const auto release_metadata = +[](MemoryLimitPool &pool, size_t bytes) {
    pool.release_metadata(bytes);
  };
  const std::array<Consumer, 4> consumers{{
      {+[](MemoryLimitPool &pool, size_t bytes) {
         return pool.try_charge_external(bytes);
       },
       release_external},
      {+[](MemoryLimitPool &pool, size_t bytes) {
         return pool.try_charge_workspace(bytes);
       },
       release_external},
      {+[](MemoryLimitPool &pool, size_t bytes) {
         return pool.try_charge_metadata(bytes);
       },
       release_metadata},
      {+[](MemoryLimitPool &pool, size_t bytes) {
         return pool.try_charge_metadata_without_reclaim(bytes);
       },
       release_metadata},
  }};
  for (const auto &consumer : consumers) {
    {
      PendingReservation pending(pool_, pool_.available());
      ASSERT_TRUE(pending.acquired());
      const bool charged = consumer.charge(pool_, page_);
      EXPECT_FALSE(charged) << "Pending logical page budget was reused";
      expect_bounded();
      if (charged) consumer.release(pool_, page_);
    }
    ASSERT_TRUE(consumer.charge(pool_, page_));
    consumer.release(pool_, page_);
    EXPECT_EQ(baseline_, pool_.used());
    EXPECT_EQ(baseline_, pool_.committed());
  }
}

TEST_F(PoolAdmissionTest, MandatoryConsumersWaitWhileCapacityMakesProgress) {
  for (const bool metadata : {false, true}) {
    auto first = std::make_unique<PendingReservation>(pool_, page_);
    ASSERT_TRUE(first->acquired());
    auto remaining =
        std::make_unique<PendingReservation>(pool_, pool_.available());
    ASSERT_TRUE(remaining->acquired());
    const uint64_t initial_waits = pool_.stats().capacity_waits;
    std::atomic<bool> observed_second_wait{false};
    std::thread reclaimer([&] {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (pool_.stats().capacity_waits == initial_waits &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      // Partial progress cannot satisfy the request in the first wait. The
      // rest is returned only when admission waits again, without I/O sleeps.
      first.reset();
      while (pool_.stats().capacity_waits < initial_waits + 2 &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      observed_second_wait = pool_.stats().capacity_waits >= initial_waits + 2;
      remaining.reset();
    });
    const size_t bytes = 3 * page_;
    const bool charged = metadata ? pool_.try_charge_metadata(bytes)
                                  : pool_.try_charge_workspace(bytes);
    reclaimer.join();
    EXPECT_TRUE(observed_second_wait.load());
    EXPECT_TRUE(charged);
    expect_bounded();
    if (charged) {
      if (metadata)
        pool_.release_metadata(bytes);
      else
        pool_.release_external(bytes);
    }
  }
}


TEST_F(PoolAdmissionTest, MandatoryConsumersReleasePriorityAfterFailure) {
  for (const bool metadata : {false, true}) {
    {
      PendingReservation pinned(pool_, pool_.available());
      ASSERT_TRUE(pinned.acquired());
      const bool charged = metadata ? pool_.try_charge_metadata(page_)
                                    : pool_.try_charge_workspace(page_);
      EXPECT_FALSE(charged);
      if (charged) {
        if (metadata)
          pool_.release_metadata(page_);
        else
          pool_.release_external(page_);
      }
    }
    PendingReservation next_pages(pool_, pool_.available());
    EXPECT_TRUE(next_pages.acquired());
    expect_bounded();
  }
}

TEST_F(PoolAdmissionTest, BackingRefusalRollsBackUsedReservation) {
  {
    PendingReservation pending(pool_, pool_.available(), true);
    ASSERT_TRUE(pending.acquired());
    EXPECT_FALSE(pool_.try_charge_external(page_));
    EXPECT_EQ(baseline_, pool_.used());
    EXPECT_EQ(pool_.capacity(), pool_.committed());
    EXPECT_FALSE(pool_.try_charge_metadata_without_reclaim(page_));
    EXPECT_EQ(baseline_, pool_.used());
    expect_bounded();
  }
  ASSERT_TRUE(pool_.try_charge_external(page_));
  pool_.release_external(page_);
}

TEST_F(PoolAdmissionTest, FailedWorkspaceAllocationRestoresRssHeadroom) {
  ASSERT_EQ(0, pool_.init(baseline_ + 64 * page_));
  MemoryLimitPoolTestPeer::stop_background(pool_);
  const size_t initial_rss = baseline_ + 128 * page_;
  const size_t resident_bytes = 32 * page_;
  const size_t current_rss = initial_rss + resident_bytes;
  const size_t headroom = 4 * page_;
  ASSERT_TRUE(MemoryLimitPoolTestPeer::configure_rss(
      pool_, current_rss + headroom, initial_rss));
  MemoryReservation resident;
  resident.add(resident_bytes);
  MemoryLimitPoolTestPeer::sample_rss(pool_, current_rss);
  const size_t used = pool_.used();
  const size_t committed = pool_.committed();
  for (int i = 0; i < 10; ++i) {
    bool attempted = false;
    MemoryReservation workspace;
    EXPECT_THROW(workspace.add_for_allocation(
                     page_,
                     [&] {
                       attempted = true;
                       MemoryLimitPoolTestPeer::sample_rss(pool_, current_rss);
                       EXPECT_EQ(headroom - page_,
                                 MemoryLimitPoolTestPeer::rss_available(pool_));
                       throw std::bad_alloc();
                     }),
                 std::bad_alloc);
    ASSERT_TRUE(attempted);
    EXPECT_EQ(0u, workspace.bytes());
    EXPECT_EQ(used, pool_.used());
    EXPECT_EQ(committed, pool_.committed());
    EXPECT_EQ(headroom, MemoryLimitPoolTestPeer::rss_available(pool_));
  }
}

#if defined(__linux__) && !defined(__ANDROID__)
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#endif

TEST_F(PoolAdmissionTest, RequiredWritePageUsesRemainingRssRoom) {
  ASSERT_EQ(0, pool_.init(baseline_ + 128 * page_));
  MemoryLimitPoolTestPeer::stop_background(pool_);
  const size_t limit = baseline_ + 256 * page_;
  ASSERT_TRUE(
      MemoryLimitPoolTestPeer::configure_rss(pool_, limit, limit - 4 * page_));
  ASSERT_TRUE(pool_.under_process_rss_pressure());
  const auto timeout = std::chrono::milliseconds(1);
  char *buffer = nullptr;
  EXPECT_FALSE(pool_.try_acquire_buffer(page_, buffer));
  EXPECT_FALSE(pool_.wait_for_available(page_, timeout));
  EXPECT_TRUE(pool_.wait_for_available(page_, timeout, /*required=*/true));
  ASSERT_TRUE(pool_.try_acquire_buffer(page_, buffer, /*required=*/true));
  pool_.release_buffer(buffer, page_);

  using Pressure = ProcessRssBudget::PressureLevel;
  pool_.set_memory_pressure(Pressure::kCritical);
  EXPECT_FALSE(pool_.try_acquire_buffer(page_, buffer, /*required=*/true));
  pool_.set_memory_pressure(Pressure::kNormal);
}

TEST_F(PoolAdmissionTest, ConcurrentMixedAdmissionStaysWithinBothLimits) {
  ASSERT_EQ(0, pool_.init(baseline_ + 32 * page_));
  std::atomic<bool> stop{false};
  std::atomic<size_t> violations{0};
  const auto observe = [&] {
    if (pool_.used() > pool_.capacity() || pool_.committed() > pool_.capacity())
      ++violations;
  };
  std::thread observer([&] {
    while (!stop.load()) {
      observe();
      std::this_thread::yield();
    }
  });
  std::vector<std::thread> workers;
  for (size_t thread = 0; thread < 8; ++thread) {
    workers.emplace_back([&, thread] {
      for (size_t round = 0; round < 300; ++round) {
        std::array<char *, 4> pages{};
        const size_t count =
            pool_.batch_acquire_buffers(page_, pages.data(), pages.size());
        observe();
        for (size_t i = 0; i < count; ++i) {
          pages[i][0] = static_cast<char>(thread);
          pool_.release_buffer(pages[i], page_);
        }
        if (pool_.try_charge_external(page_)) {
          observe();
          pool_.release_external(page_);
        }
        if (pool_.try_charge_metadata_without_reclaim(page_)) {
          observe();
          pool_.release_metadata(page_);
        }
        void *retained = pool_.acquire_retained_block(1);
        observe();
        pool_.release_retained_block(retained);
        void *queued = EvictionQueueTraits::malloc(64);
        observe();
        EvictionQueueTraits::free(queued);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  stop.store(true);
  observer.join();
  EXPECT_EQ(0u, violations.load());
  expect_bounded();
}

}  // namespace
}  // namespace zvec::ailego

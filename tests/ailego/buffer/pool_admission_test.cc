// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_reservation.h>

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
  void expect_workspace_priority(bool bulk);
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  const size_t page_ = MemoryLimitPool::page_buffer_size();
  size_t baseline_{0};
};

TEST_F(PoolAdmissionTest, SmallPoolWatermarksKeepFittingWorkingSets) {
  const std::array<size_t, 9> capacities{0,          page_,
                                         1UL << 20,  (16UL << 20) - page_,
                                         16UL << 20, (16UL << 20) + page_,
                                         32UL << 20, 64UL << 20,
                                         1UL << 30};
  for (size_t capacity : capacities) {
    SCOPED_TRACE(capacity);
    const size_t margin =
        MemoryLimitPoolTestPeer::reserve_margin(pool_, capacity);
    EXPECT_LE(margin, capacity / 8);
    EXPECT_GE(capacity - 2 * margin, capacity - capacity / 4);
  }
}

TEST_F(PoolAdmissionTest, WaitIncludesHeaderWhenEverySlabSlotIsOccupied) {
  ASSERT_EQ(0, pool_.init(baseline_ + MemoryLimitPool::slab_size() + page_));
  MemoryLimitPoolTestPeer::stop_background(pool_);
  auto release = [this](char *buffer) { pool_.release_buffer(buffer, page_); };
  std::vector<std::unique_ptr<char, decltype(release)>> held;
  const size_t pages = MemoryLimitPool::slab_size() / page_ - 1;
  held.reserve(pages);
  for (size_t i = 0; i < pages; ++i) {
    char *buffer = nullptr;
    ASSERT_TRUE(pool_.try_acquire_buffer(page_, buffer));
    held.emplace_back(buffer, release);
  }
  EXPECT_EQ(page_, pool_.available());
  char *extra = nullptr;
  EXPECT_FALSE(pool_.try_acquire_buffer(page_, extra));
  EXPECT_FALSE(pool_.wait_for_available(page_, std::chrono::milliseconds(1)));
  held.pop_back();
  EXPECT_TRUE(pool_.wait_for_available(page_, std::chrono::milliseconds(1)));
  expect_bounded();
}

TEST_F(PoolAdmissionTest, FixedConsumersRejectPendingPageReservations) {
  using Charge = bool (MemoryLimitPool::*)(size_t);
  using Release = void (MemoryLimitPool::*)(size_t);
  struct Consumer {
    Charge charge;
    Release release;
  };
  const std::array<Consumer, 4> consumers{
      {{&MemoryLimitPool::try_charge_external,
        &MemoryLimitPool::release_external},
       {&MemoryLimitPool::try_charge_workspace,
        &MemoryLimitPool::release_external},
       {&MemoryLimitPool::try_charge_metadata,
        &MemoryLimitPool::release_metadata},
       {&MemoryLimitPool::try_charge_metadata_without_reclaim,
        &MemoryLimitPool::release_metadata}}};
  for (const auto &consumer : consumers) {
    {
      PendingReservation pending(pool_, pool_.available());
      ASSERT_TRUE(pending.acquired());
      const bool charged = (pool_.*consumer.charge)(page_);
      EXPECT_FALSE(charged) << "Pending logical page budget was reused";
      expect_bounded();
      if (charged) (pool_.*consumer.release)(page_);
    }
    ASSERT_TRUE((pool_.*consumer.charge)(page_));
    (pool_.*consumer.release)(page_);
    EXPECT_EQ(baseline_, pool_.used());
    EXPECT_EQ(baseline_, pool_.committed());
  }
}

TEST_F(PoolAdmissionTest, QueueMetadataRejectsPendingPageReservations) {
  {
    PendingReservation pending(pool_, pool_.available());
    ASSERT_TRUE(pending.acquired());
    void *allocation = EvictionQueueTraits::malloc(1);
    EXPECT_EQ(nullptr, allocation);
    expect_bounded();
    EvictionQueueTraits::free(allocation);
  }
  void *allocation = EvictionQueueTraits::malloc(1);
  ASSERT_NE(nullptr, allocation);
  EvictionQueueTraits::free(allocation);
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

TEST_F(PoolAdmissionTest, RetainedReuseRejectsPendingPageReservations) {
  void *block = pool_.acquire_retained_block(1);
  ASSERT_NE(nullptr, block);
  pool_.release_retained_block(block);
  {
    PendingReservation pending(pool_, pool_.available());
    ASSERT_TRUE(pending.acquired());
    block = pool_.acquire_retained_block(1);
    EXPECT_EQ(nullptr, block);
    expect_bounded();
    pool_.release_retained_block(block);
  }
  block = pool_.acquire_retained_block(1);
  ASSERT_NE(nullptr, block);
  pool_.release_retained_block(block);
}

void PoolAdmissionTest::expect_workspace_priority(bool bulk) {
  if (bulk) {
    // Reuse existing backing so a missing priority check cannot be masked by
    // a separate slab-header allocation failing in the competing reader.
    char *buffer = nullptr;
    ASSERT_TRUE(pool_.try_acquire_buffer(page_, buffer));
    pool_.release_buffer(buffer, page_);
  }
  for (const bool metadata : {false, true}) {
    auto first = std::make_unique<PendingReservation>(pool_, page_);
    ASSERT_TRUE(first->acquired());
    auto remaining =
        std::make_unique<PendingReservation>(pool_, pool_.available());
    ASSERT_TRUE(remaining->acquired());
    const auto initial_waits = pool_.stats().capacity_waits;
    std::atomic<bool> page_readmitted{false};
    std::atomic<bool> saw_wait{false};
    std::thread reclaimer([&] {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (pool_.stats().capacity_waits == initial_waits &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      saw_wait = pool_.stats().capacity_waits != initial_waits;
      first.reset();
      // The mandatory consumer needs three pages and cannot win yet. Model
      // a page reader racing to reuse the first reclaimed page instead.
      if (bulk) {
        char *buffer = nullptr;
        page_readmitted = pool_.batch_acquire_buffers(page_, &buffer, 1) != 0;
        if (buffer) pool_.release_buffer(buffer, page_);
      } else {
        PendingReservation competing_page(pool_, page_);
        page_readmitted = competing_page.acquired();
      }
      while (pool_.stats().capacity_waits < initial_waits + 2 &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      remaining.reset();
    });
    const size_t bytes = 3 * page_;
    const bool charged = metadata ? pool_.try_charge_metadata(bytes)
                                  : pool_.try_charge_workspace(bytes);
    reclaimer.join();
    EXPECT_TRUE(saw_wait.load());
    EXPECT_FALSE(page_readmitted.load());
    EXPECT_TRUE(charged);
    expect_bounded();
    if (charged) {
      if (metadata)
        pool_.release_metadata(bytes);
      else
        pool_.release_external(bytes);
    }
    // Priority must disappear on success so later page admission still works.
    PendingReservation next_page(pool_, page_);
    EXPECT_TRUE(next_page.acquired());
  }
}

TEST_F(PoolAdmissionTest, MandatoryConsumersKeepReclaimedHeadroom) {
  expect_workspace_priority(false);
}

TEST_F(PoolAdmissionTest,
       MandatoryConsumersKeepReclaimedHeadroomFromBulkReads) {
  expect_workspace_priority(true);
}

TEST_F(PoolAdmissionTest, SinglePageWorkspaceWaitIgnoresPageAdmissionLimit) {
  auto first = std::make_unique<PendingReservation>(pool_, page_);
  ASSERT_TRUE(first->acquired());
  PendingReservation remaining(pool_, pool_.available());
  ASSERT_TRUE(remaining.acquired());
  const auto before = pool_.stats();
  std::thread reclaimer([&] {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (pool_.stats().capacity_waits == before.capacity_waits &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    first.reset();
  });
  const bool charged = pool_.try_charge_workspace(page_);
  reclaimer.join();
  EXPECT_TRUE(charged);
  EXPECT_EQ(before.capacity_wait_timeouts,
            pool_.stats().capacity_wait_timeouts);
  if (charged) pool_.release_external(page_);
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

TEST_F(PoolAdmissionTest, SlabHeaderRejectsPendingPageReservations) {
  {
    // The data page fits logically, but its new slab header does not.
    PendingReservation pending(pool_, pool_.available() - page_);
    ASSERT_TRUE(pending.acquired());
    char *page = nullptr;
    const bool acquired = pool_.try_acquire_buffer(page_, page);
    EXPECT_FALSE(acquired);
    expect_bounded();
    if (page) pool_.release_buffer(page, page_);
  }
  char *page = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(page_, page));
  ASSERT_NE(nullptr, page);
  pool_.release_buffer(page, page_);
}

TEST_F(PoolAdmissionTest, MandatoryConsumersWaitThroughClockOnlyPasses) {
  ASSERT_EQ(0, pool_.init(baseline_ + 64 * page_));
  // Advance CLOCK on this thread only; the reclaimer controls the release.
  // This also avoids unrelated producer metadata allocation under pressure.
  MemoryLimitPoolTestPeer::stop_background(pool_);
  class DeferredPages : public EvictableBlockOwner {
   public:
    DeferredPages() {
      auto &queue = BlockEvictionQueue::get_instance();
      queue.set_valid(this);
      for (size_t key = 0; key < 8; ++key) enqueue(key);
    }
    ~DeferredPages() override {
      BlockEvictionQueue::get_instance().set_invalid(this);
    }
    bool is_dead_block(eviction_key_t, version_t) override {
      return false;
    }
    bool evict_block(eviction_key_t key) override {
      // A CLOCK pass can change an entry without releasing bytes yet.
      enqueue(key);
      return false;
    }
    void enqueue(eviction_key_t key) {
      BlockEvictionQueue::BlockType item;
      item.owner = this;
      item.owner_key = key;
      EXPECT_TRUE(BlockEvictionQueue::get_instance().add_single_block(item, 0));
    }
  } deferred;
  auto pending = std::make_unique<PendingReservation>(pool_, pool_.available());
  ASSERT_TRUE(pending->acquired());
  const auto initial_waits = pool_.stats().capacity_waits;
  std::atomic<bool> observed_second_wait{false};
  std::thread reclaimer([&] {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (pool_.stats().capacity_waits < initial_waits + 2 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    observed_second_wait = pool_.stats().capacity_waits >= initial_waits + 2;
    pending.reset();
  });
  const bool charged = pool_.try_charge_workspace(page_);
  reclaimer.join();
  EXPECT_TRUE(observed_second_wait.load());
  EXPECT_TRUE(charged);
  expect_bounded();
  if (charged) pool_.release_external(page_);
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
    EXPECT_THROW(workspace.add_for_allocation(page_, [&] {
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

TEST_F(PoolAdmissionTest, RetainedBackingReuseObeysMeasuredHeadroom) {
  ASSERT_EQ(0, pool_.init(baseline_ + 128 * page_));
  MemoryLimitPoolTestPeer::stop_background(pool_);
  const size_t request = page_ - MemoryLimitPool::kRetainedBlockHeaderBytes;
  void *block = pool_.acquire_retained_block(request);
  ASSERT_NE(nullptr, block);
  pool_.release_retained_block(block);
  ASSERT_EQ(page_, pool_.stats().retained_idle_bytes);

  const size_t limit = baseline_ + 256 * page_;
  ASSERT_TRUE(MemoryLimitPoolTestPeer::configure_rss(pool_, limit, limit - 1));
  block = pool_.acquire_retained_block(request);
  EXPECT_EQ(nullptr, block);
  if (block) pool_.release_retained_block(block);

  MemoryLimitPoolTestPeer::sample_rss(pool_, limit - 16 * page_);
  block = pool_.acquire_retained_block(request);
  ASSERT_NE(nullptr, block);
  pool_.release_retained_block(block);
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

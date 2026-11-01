// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <atomic>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {

// Pause between the real admission steps without sleeps or production hooks.
struct MemoryLimitPoolTestPeer {
  static bool reserve_page(MemoryLimitPool &pool, size_t bytes) {
    return pool.try_reserve_page_used(bytes);
  }
  static bool reserve_backing(MemoryLimitPool &pool, size_t bytes) {
    return pool.try_reserve_committed(bytes);
  }
  static void release_backing(MemoryLimitPool &pool, size_t bytes) {
    pool.committed_size_.fetch_sub(bytes, std::memory_order_relaxed);
  }
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

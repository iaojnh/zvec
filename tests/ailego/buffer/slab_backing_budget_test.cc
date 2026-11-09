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
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static void stop_background(MemoryLimitPool &pool) {
    pool.stop_background_evictor();
  }
  static ProcessRssBudget::ResidentSample resident_sample() {
    return MemoryLimitPool::read_slab_resident_sample();
  }
  static ProcessRssBudget::ResidentSample sample_while_busy(MemoryLimitPool &pool) {
    std::lock_guard<std::mutex> lock(pool.slab_mutex_);
    return resident_sample();
  }
};
namespace {
class SlabBackingBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(64UL << 20));
    baseline_ = pool_.committed();
  }
  void TearDown() override {
    EXPECT_EQ(pool_.queue_metadata_used() + pool_.stats().slab_header_bytes,
              pool_.used());
    EXPECT_LE(pool_.committed(), pool_.capacity());
    EXPECT_EQ(0, pool_.init(0));
    EXPECT_EQ(0u, pool_.committed());
    EXPECT_EQ(0u, pool_.stats().slab_header_bytes);
  }
  struct PageDeleter {
    void operator()(char *data) const {
      MemoryLimitPool::get_instance().release_buffer(
          data, MemoryLimitPool::page_buffer_size());
    }
  };
  using Page = std::unique_ptr<char, PageDeleter>;
  Page acquire() {
    char *data = nullptr;
    pool_.try_acquire_buffer(page_size_, data);
    return Page(data);
  }
  void expect_discarded(char *data) {
#if defined(_WIN32)
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(0u, ::VirtualQuery(data, &info, sizeof(info)));
    EXPECT_EQ(MEM_RESERVE, info.State);
#else
#if defined(__linux__)
    unsigned char residency = 0;
#else
    char residency = 0;
#endif
    ASSERT_EQ(0, ::mincore(data, page_size_, &residency));
    EXPECT_EQ(0, residency & 1);
#endif
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  const size_t page_size_ = MemoryLimitPool::page_buffer_size();
  size_t baseline_{0};
};

TEST_F(SlabBackingBudgetTest, HeaderAdmissionFailureRollsBackDataPage) {
  ASSERT_EQ(0, pool_.init(0));
  ASSERT_EQ(0, pool_.init(baseline_ + page_size_));
  EXPECT_EQ(nullptr, acquire());
  EXPECT_EQ(baseline_, pool_.committed());
  EXPECT_EQ(baseline_, pool_.used());
  EXPECT_EQ(0u, pool_.stats().slab_count);
}

TEST_F(SlabBackingBudgetTest, IdlePageKeepsBackingChargeAndReusesIt) {
  auto page = acquire();
  ASSERT_NE(nullptr, page);
  char *address = page.get();
  std::memset(address, 7, page_size_);
  EXPECT_EQ(baseline_ + 2 * page_size_, pool_.committed());
  page.reset();
  EXPECT_EQ(baseline_ + page_size_, pool_.used());
  EXPECT_EQ(baseline_ + 2 * page_size_, pool_.committed());
  const auto reused = pool_.stats().alloc_from_freelist;
  page = acquire();
  ASSERT_EQ(address, page.get());
  EXPECT_EQ(7, page.get()[page_size_ - 1]);
  EXPECT_EQ(baseline_ + 2 * page_size_, pool_.committed());
  EXPECT_EQ(reused + 1, pool_.stats().alloc_from_freelist);
}

#if defined(__APPLE__) && TARGET_OS_OSX
TEST_F(SlabBackingBudgetTest, NativeResidencyWitnessTracksDiscardAndReuse) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  auto page = acquire();
  auto neighbor = acquire();
  auto untouched = acquire();
  ASSERT_NE(nullptr, page);
  ASSERT_NE(nullptr, neighbor);
  ASSERT_NE(nullptr, untouched);
  std::memset(page.get(), 7, page_size_);
  std::memset(neighbor.get(), 11, page_size_);
  const auto full = MemoryLimitPoolTestPeer::resident_sample();
  ASSERT_EQ(3 * page_size_, full.owned_resident);  // Header and both pages.
  char *address = page.get();
  page.reset();
  const size_t charge = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(charge));
  expect_discarded(address);
  const auto discarded = MemoryLimitPoolTestPeer::resident_sample();
  EXPECT_EQ(page_size_, full.owned_resident - discarded.owned_resident);
  EXPECT_EQ(11, neighbor.get()[page_size_ - 1]);
  pool_.release_external(charge);
  page = acquire();
  ASSERT_NE(nullptr, page);
  std::memset(page.get(), 13, page_size_);
  const auto reused = MemoryLimitPoolTestPeer::resident_sample();
  EXPECT_EQ(full.owned_resident, reused.owned_resident);
  EXPECT_LE(reused.owned_resident, reused.rss);
}

TEST_F(SlabBackingBudgetTest, BusyNativeResidencySampleKeepsAdmissionConservative) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  auto page = acquire();
  ASSERT_NE(nullptr, page);
  std::memset(page.get(), 17, page_size_);
  const auto sample = MemoryLimitPoolTestPeer::sample_while_busy(pool_);
  EXPECT_GT(sample.rss, 0u);
  EXPECT_EQ(0u, sample.owned_resident);
}
#endif

TEST_F(SlabBackingBudgetTest,
       PressureDiscardsBackingWithoutChangingLiveNeighbor) {
  auto live = acquire();
  auto idle = acquire();
  ASSERT_NE(nullptr, live);
  ASSERT_NE(nullptr, idle);
  std::memset(live.get(), 17, page_size_);
  std::memset(idle.get(), 23, page_size_);
  char *discarded = idle.get();
  idle.reset();
  const size_t charge = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(charge));
  expect_discarded(discarded);
  EXPECT_EQ(17, live.get()[0]);
  EXPECT_EQ(17, live.get()[page_size_ - 1]);
  pool_.release_external(charge);
  idle = acquire();
  ASSERT_EQ(discarded, idle.get());
  // Newly committed backing must not contain data from the refunded page.
  EXPECT_EQ(0, idle.get()[0]);
  EXPECT_EQ(0, idle.get()[page_size_ - 1]);
  std::memset(idle.get(), 29, page_size_);
  EXPECT_EQ(17, live.get()[page_size_ - 1]);
}

TEST_F(SlabBackingBudgetTest,
       MixedPageAndRetainedRetirementPreservesOwnersAndSharedReserve) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  constexpr size_t kReserve = 8UL << 20;
  constexpr size_t kWorkers = 4;
  constexpr size_t kPages = 64;
  const auto release = [this](void *data) {
    pool_.release_retained_block(data);
  };
  const size_t live_bytes =
      (1UL << 20) - MemoryLimitPool::kRetainedBlockHeaderBytes;
  std::unique_ptr<void, decltype(release)> live(
      pool_.acquire_retained_block(live_bytes, true), release);
  ASSERT_NE(nullptr, live);
  std::memset(live.get(), 117, live_bytes);
  std::unique_ptr<void, decltype(release)> idle(
      pool_.acquire_retained_block(kReserve -
                                   MemoryLimitPool::kRetainedBlockHeaderBytes),
      release);
  ASSERT_NE(nullptr, idle);
  std::array<std::vector<Page>, kWorkers> groups;
  for (auto &group : groups) {
    for (size_t i = 0; i < kPages; ++i) {
      group.push_back(acquire());
      ASSERT_NE(nullptr, group.back());
      std::memset(group.back().get(), 121, page_size_);
    }
  }
  idle.reset();
  std::array<std::thread, kWorkers> workers;
  for (size_t i = 0; i < kWorkers; ++i) {
    workers[i] = std::thread([&, i] {
      for (size_t round = 0; round < 32; ++round) {
        for (auto &page : groups[i]) page.reset();
        for (auto &page : groups[i]) {
          page = acquire();
          ASSERT_NE(nullptr, page);
          std::memset(page.get(), 121, page_size_);
        }
        EXPECT_LE(pool_.committed(), pool_.capacity());
        EXPECT_EQ(117, static_cast<char *>(live.get())[live_bytes - 1]);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  for (auto &group : groups) group.clear();
  EXPECT_LE(pool_.committed() - pool_.used(), kReserve + kWorkers * page_size_);
  EXPECT_EQ(117, static_cast<char *>(live.get())[0]);
  EXPECT_EQ(117, static_cast<char *>(live.get())[live_bytes - 1]);
}
class QueueMappingBudgetTest : public testing::Test {
 protected:
  struct Deleter {
    void operator()(void *ptr) const {
      EvictionQueueTraits::free(ptr);
    }
  };
  using Allocation = std::unique_ptr<void, Deleter>;
  struct MetadataCharge {
    MetadataCharge(MemoryLimitPool &owner, size_t size)
        : pool(owner),
          bytes(size),
          acquired(owner.try_charge_metadata_without_reclaim(size)) {}
    ~MetadataCharge() {
      if (acquired) pool.release_metadata(bytes);
    }
    MemoryLimitPool &pool;
    size_t bytes;
    bool acquired;
  };
  void SetUp() override {
    (void)BlockEvictionQueue::get_instance();
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(64UL << 20));
    MemoryLimitPoolTestPeer::stop_background(pool_);
    baseline_ = pool_.stats();
  }
  void TearDown() override {
    EXPECT_EQ(baseline_.queue_allocated, pool_.queue_allocated());
    EXPECT_EQ(baseline_.used, pool_.used());
    EXPECT_EQ(baseline_.committed, pool_.committed());
    EXPECT_EQ(0, pool_.init(0));
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  MemoryLimitPool::PoolStats baseline_;
};

TEST_F(QueueMappingBudgetTest, FreedSlotReusesBackingAndLastOwnerUnmaps) {
  Allocation first(EvictionQueueTraits::malloc(997));
  Allocation second(EvictionQueueTraits::malloc(997));
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  void *address = first.get();
  const auto charged = pool_.stats();
  first.reset();
  Allocation reused(EvictionQueueTraits::malloc(997));
  ASSERT_EQ(address, reused.get());
  EXPECT_EQ(charged.queue_allocated, pool_.queue_allocated());
  EXPECT_EQ(charged.committed, pool_.committed());
  std::memset(reused.get(), 73, 997);
  second.reset();
  EXPECT_EQ(charged.queue_allocated, pool_.queue_allocated());
  EXPECT_EQ(73, static_cast<char *>(reused.get())[996]);
  reused.reset();
  EXPECT_EQ(baseline_.queue_allocated, pool_.queue_allocated());
#if defined(__linux__)
  const size_t page = MemoryLimitPool::page_buffer_size();
  auto *mapping = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(address) & ~(page - 1));
  unsigned char resident = 0;
  errno = 0;
  EXPECT_EQ(-1, ::mincore(mapping, page, &resident));
  EXPECT_EQ(ENOMEM, errno);
#endif
}

TEST_F(QueueMappingBudgetTest, PaddingAdmissionRejectsBeforeMapping) {
  const size_t page = MemoryLimitPool::page_buffer_size();
  ASSERT_GE(pool_.available(), page);
  const size_t charge = pool_.available() - page + 1;
  MetadataCharge reservation(pool_, charge);
  ASSERT_TRUE(reservation.acquired);
  const auto before = pool_.stats();
  Allocation denied(EvictionQueueTraits::malloc(997));
  EXPECT_EQ(nullptr, denied);
  EXPECT_EQ(before.used, pool_.used());
  EXPECT_EQ(before.committed, pool_.committed());
  EXPECT_EQ(before.queue_allocated, pool_.queue_allocated());
  denied.reset();
}

}  // namespace
}  // namespace zvec::ailego

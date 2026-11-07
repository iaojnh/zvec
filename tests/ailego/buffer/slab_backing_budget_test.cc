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

TEST_F(SlabBackingBudgetTest, HeaderStaysChargedAfterDataDiscard) {
  auto page = acquire();
  ASSERT_NE(nullptr, page);
  std::memset(page.get(), 3, page_size_);
  page.reset();
  // Data backing is reclaimable, but the live slab header is not.
  EXPECT_FALSE(pool_.try_charge_external(pool_.capacity() - baseline_));
  EXPECT_EQ(baseline_ + page_size_, pool_.committed());
  const size_t charge = pool_.capacity() - baseline_ - page_size_;
  ASSERT_TRUE(pool_.try_charge_external(charge));
  EXPECT_EQ(pool_.capacity(), pool_.committed());
  pool_.release_external(charge);
  EXPECT_EQ(baseline_ + page_size_, pool_.committed());
}

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

TEST_F(SlabBackingBudgetTest, RepeatedPressureDoesNotLeaveResidentOldPages) {
  constexpr size_t kPages = 16;
  for (size_t round = 0; round < 32; ++round) {
    std::vector<Page> pages;
    std::vector<char *> addresses;
    for (size_t i = 0; i < kPages; ++i) {
      auto page = acquire();
      ASSERT_NE(nullptr, page);
      std::memset(page.get(), 31, page_size_);
      addresses.push_back(page.get());
      pages.push_back(std::move(page));
    }
    pages.clear();
    const size_t charge = pool_.capacity() - pool_.used();
    ASSERT_TRUE(pool_.try_charge_external(charge));
    EXPECT_EQ(0u, pool_.stats().free_buffers);
    for (auto *address : addresses) expect_discarded(address);
    pool_.release_external(charge);
    EXPECT_EQ(baseline_ + page_size_, pool_.committed());
    EXPECT_EQ(1u, pool_.stats().slab_count);
  }
}

TEST_F(SlabBackingBudgetTest, EachSlabHeaderIsChargedBeforeMapping) {
  const size_t count = MemoryLimitPool::slab_size() / page_size_;
  std::vector<Page> pages;
  for (size_t i = 0; i < count; ++i) {
    pages.push_back(acquire());
    ASSERT_NE(nullptr, pages.back());
  }
  const auto stats = pool_.stats();
  EXPECT_EQ(2u, stats.slab_count);
  EXPECT_EQ(2 * page_size_, stats.slab_header_bytes);
  EXPECT_EQ(baseline_ + (count + 2) * page_size_, stats.committed);
  pages.clear();
  EXPECT_EQ(baseline_ + (count + 2) * page_size_, pool_.committed());
}

TEST_F(SlabBackingBudgetTest, ConcurrentPressureKeepsLivePagesIntact) {
  std::vector<std::thread> workers;
  for (size_t thread = 0; thread < 4; ++thread) {
    workers.emplace_back([this, thread] {
      for (size_t round = 0; round < 100; ++round) {
        auto live = acquire();
        if (!live) continue;
        const char value = static_cast<char>(thread + 1);
        std::memset(live.get(), value, page_size_);
        auto idle = acquire();
        idle.reset();
        const size_t charge = pool_.capacity() - baseline_ - 16 * page_size_;
        if (pool_.try_charge_external(charge)) {
          EXPECT_LE(pool_.committed(), pool_.capacity());
          pool_.release_external(charge);
        }
        EXPECT_EQ(value, live.get()[0]);
        EXPECT_EQ(value, live.get()[page_size_ - 1]);
      }
    });
  }
  for (auto &worker : workers) worker.join();
}

TEST_F(SlabBackingBudgetTest, BackgroundReturnsClosedPageBackingToReserve) {
  // More than one maintenance batch, without reaching the admission limit.
  // Repeating the lifecycle must not need a new allocation to drive cleanup.
  for (size_t round = 0; round < 3; ++round) {
    std::vector<Page> pages;
    const size_t bytes = 24UL << 20;
    for (size_t i = 0; i < bytes / page_size_; ++i) {
      auto page = acquire();
      ASSERT_NE(nullptr, page);
      std::memset(page.get(), 7, page_size_);
      pages.push_back(std::move(page));
    }
    pages.clear();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const size_t reserve = 8UL << 20;
    while (pool_.committed() > pool_.used() + reserve &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_LE(pool_.committed(), pool_.used() + reserve);
    EXPECT_LE(pool_.stats().free_buffers * page_size_, reserve);
  }
}

TEST_F(SlabBackingBudgetTest, ActiveCacheRetirementBoundsIdleBackingInline) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  auto live = acquire();
  ASSERT_NE(nullptr, live);
  std::memset(live.get(), 79, page_size_);
  std::vector<Page> pages;
  constexpr size_t kBytes = 24UL << 20;
  constexpr size_t kReserve = 8UL << 20;
  for (size_t i = 0; i < kBytes / page_size_; ++i) {
    pages.push_back(acquire());
    ASSERT_NE(nullptr, pages.back());
    std::memset(pages.back().get(), 83, page_size_);
  }
  char *last = pages.back().get();
  const auto reclaimed = pool_.stats().slab_reclaimed_pages;
  for (auto &page : pages) page.reset();
  // No sleep, new allocation, budget pressure or background worker is needed.
  EXPECT_LE(pool_.committed() - pool_.used(), kReserve);
  EXPECT_GT(pool_.stats().slab_reclaimed_pages, reclaimed);
  expect_discarded(last);
  EXPECT_EQ(79, live.get()[0]);
  EXPECT_EQ(79, live.get()[page_size_ - 1]);
  const auto reused = pool_.stats().alloc_from_freelist;
  auto spare = acquire();
  ASSERT_NE(nullptr, spare);
  EXPECT_EQ(reused + 1, pool_.stats().alloc_from_freelist);
}

TEST_F(SlabBackingBudgetTest, ConcurrentRetirementKeepsSmallReuseReserve) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  constexpr size_t kWorkers = 4;
  constexpr size_t kReserve = 8UL << 20;
  std::array<std::vector<Page>, kWorkers> pages;
  for (auto &group : pages) {
    for (size_t i = 0; i < (6UL << 20) / page_size_; ++i) {
      group.push_back(acquire());
      ASSERT_NE(nullptr, group.back());
      std::memset(group.back().get(), 89, page_size_);
    }
  }
  std::array<std::thread, kWorkers> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      for (auto &page : pages[i]) page.reset();
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_LE(pool_.committed() - pool_.used(), kReserve + kWorkers * page_size_);
  EXPECT_LE(pool_.stats().free_buffers * page_size_,
            kReserve + kWorkers * page_size_);
  EXPECT_GT(pool_.stats().free_buffers, 0u);
  EXPECT_GT(pool_.stats().slab_reclaimed_pages, 0u);
}

TEST_F(SlabBackingBudgetTest, RetainedRetirementBoundsSharedIdleBackingInline) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  constexpr size_t kReserve = 8UL << 20;
  constexpr size_t kPayload = (1UL << 20) -
                            MemoryLimitPool::kRetainedBlockHeaderBytes;
  void *live = pool_.acquire_retained_block(kPayload, true);
  ASSERT_NE(nullptr, live);
  std::memset(live, 97, kPayload);
  std::vector<void *> blocks;
  for (size_t i = 0; i < 24; ++i) {
    blocks.push_back(pool_.acquire_retained_block(kPayload));
    ASSERT_NE(nullptr, blocks.back());
    std::memset(blocks.back(), 101, kPayload);
  }
  for (auto *block : blocks) pool_.release_retained_block(block);
  // Large retired mappings share the page reserve; no paced worker, sleep,
  // subsequent allocation or budget pressure is required to enforce it.
  EXPECT_LE(pool_.committed() - pool_.used(), kReserve);
  EXPECT_LE(pool_.stats().retained_idle_bytes, kReserve);
  EXPECT_EQ(97, static_cast<char *>(live)[kPayload - 1]);
  const auto reuses = pool_.stats().retained_block_reuses;
  void *reused = pool_.acquire_retained_block(kPayload);
  ASSERT_NE(nullptr, reused);
  EXPECT_EQ(reuses + 1, pool_.stats().retained_block_reuses);
  pool_.release_retained_block(reused);
  pool_.release_retained_block(live);
}

TEST_F(SlabBackingBudgetTest,
       RetainedRetirementDoesNotDuplicateIdlePageReserve) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  constexpr size_t kReserve = 8UL << 20;
  constexpr size_t kPayload =
      (1UL << 20) - MemoryLimitPool::kRetainedBlockHeaderBytes;
  void *block = pool_.acquire_retained_block(kPayload);
  ASSERT_NE(nullptr, block);
  std::vector<Page> pages;
  for (size_t i = 0; i < kReserve / page_size_; ++i) {
    pages.push_back(acquire());
    ASSERT_NE(nullptr, pages.back());
    std::memset(pages.back().get(), 103, page_size_);
  }
  pages.clear();
  const auto before = pool_.committed();
  pool_.release_retained_block(block);
  EXPECT_EQ(before - (1UL << 20), pool_.committed());
  EXPECT_EQ(0u, pool_.stats().retained_idle_bytes);
  EXPECT_LE(pool_.committed() - pool_.used(), kReserve);
}

TEST_F(SlabBackingBudgetTest, ConcurrentRetainedRetirementKeepsSharedReserve) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  constexpr size_t kPayload =
      (1UL << 20) - MemoryLimitPool::kRetainedBlockHeaderBytes;
  std::array<std::vector<void *>, 4> groups;
  for (auto &group : groups) {
    for (size_t i = 0; i < 6; ++i) {
      group.push_back(pool_.acquire_retained_block(kPayload));
      ASSERT_NE(nullptr, group.back());
      std::memset(group.back(), 107, kPayload);
    }
  }
  std::array<std::thread, 4> workers;
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::thread([&, i] {
      for (auto *block : groups[i]) pool_.release_retained_block(block);
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_LE(pool_.stats().retained_idle_bytes, 8UL << 20);
  EXPECT_LE(pool_.committed() - pool_.used(), 8UL << 20);
}

TEST_F(SlabBackingBudgetTest, IdleMappingsYieldSharedReserveToReusedPages) {
  MemoryLimitPoolTestPeer::stop_background(pool_);
  constexpr size_t kReserve = 8UL << 20;
  constexpr size_t kPages = 64;
  constexpr size_t kRounds = 32;
  const auto release = [this](void *data) {
    pool_.release_retained_block(data);
  };
  const size_t live_bytes =
      (64UL << 10) - MemoryLimitPool::kRetainedBlockHeaderBytes;
  std::unique_ptr<void, decltype(release)> live(
      pool_.acquire_retained_block(live_bytes, true), release);
  ASSERT_NE(nullptr, live);
  std::memset(live.get(), 109, live_bytes);
  std::unique_ptr<void, decltype(release)> idle(
      pool_.acquire_retained_block(kReserve -
                                   MemoryLimitPool::kRetainedBlockHeaderBytes),
      release);
  ASSERT_NE(nullptr, idle);
  std::vector<Page> pages;
  for (size_t i = 0; i < kPages; ++i) {
    pages.push_back(acquire());
    ASSERT_NE(nullptr, pages.back());
    std::memset(pages.back().get(), 113, page_size_);
  }
  idle.reset();
  ASSERT_EQ(kReserve, pool_.stats().retained_idle_bytes);
  const auto before = pool_.stats();
  for (size_t round = 0; round < kRounds; ++round) {
    for (auto &page : pages) page.reset();
    EXPECT_LE(pool_.committed() - pool_.used(), kReserve);
    for (auto &page : pages) {
      page = acquire();
      ASSERT_NE(nullptr, page);
      std::memset(page.get(), 113, page_size_);
    }
    EXPECT_EQ(109, static_cast<char *>(live.get())[live_bytes - 1]);
  }
  const auto after = pool_.stats();
  EXPECT_EQ(0u, after.slab_reclaimed_pages - before.slab_reclaimed_pages);
  EXPECT_EQ(kRounds * kPages,
            after.alloc_from_freelist - before.alloc_from_freelist);
  EXPECT_LT(after.retained_idle_bytes, kReserve);
  EXPECT_LE(after.committed, pool_.capacity());
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

TEST_F(QueueMappingBudgetTest, SmallObjectFreeKeepsBackingCharged) {
  Allocation first(EvictionQueueTraits::malloc(997));
  Allocation second(EvictionQueueTraits::malloc(997));
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  std::memset(second.get(), 71, 997);
  const size_t backing = pool_.queue_allocated();
  first.reset();
  // A neighbor still owns the same allocation region. Object destruction
  // must not let another consumer spend that region's physical budget.
  EXPECT_EQ(backing, pool_.queue_allocated());
  EXPECT_EQ(71, static_cast<char *>(second.get())[996]);
}

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

TEST_F(QueueMappingBudgetTest,
       SmallAndLargeAllocationsPreserveAlignmentAndBytes) {
  const std::array<size_t, 10> sizes{
      {1, 17, 31, 64, 997, 1112, 4080, 4081, 65567, 1048607}};
  std::vector<Allocation> memory;
  for (size_t size : sizes) {
    memory.emplace_back(EvictionQueueTraits::malloc(size));
    ASSERT_NE(nullptr, memory.back());
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(memory.back().get()) %
                      alignof(std::max_align_t));
    std::memset(memory.back().get(), 79, size);
    EXPECT_LE(pool_.committed(), pool_.capacity());
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    EXPECT_EQ(79, static_cast<char *>(memory[i].get())[sizes[i] - 1]);
  }
}

TEST_F(QueueMappingBudgetTest, ConcurrentSlotReusePreservesLiveNeighbors) {
  std::atomic<size_t> failures{0};
  std::array<std::thread, 8> workers;
  for (size_t worker = 0; worker < workers.size(); ++worker) {
    workers[worker] = std::thread([&, worker] {
      for (size_t round = 0; round < 100; ++round) {
        std::array<Allocation, 8> memory;
        for (auto &item : memory) {
          item.reset(EvictionQueueTraits::malloc(997));
          if (!item) {
            ++failures;
            return;
          }
          std::memset(item.get(), static_cast<int>(worker + 1), 997);
        }
        for (size_t i = 0; i < memory.size(); i += 2) memory[i].reset();
        for (size_t i = 1; i < memory.size(); i += 2) {
          if (static_cast<char *>(memory[i].get())[996] !=
              static_cast<char>(worker + 1))
            ++failures;
        }
        if (pool_.committed() > pool_.capacity()) ++failures;
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0u, failures.load());
}
TEST_F(QueueMappingBudgetTest, PageFallbackPreservesBackingAndBudget) {
  const size_t page = MemoryLimitPool::page_buffer_size();
  const size_t charge = pool_.available() - 2 * page;
  MetadataCharge reservation(pool_, charge);
  ASSERT_TRUE(reservation.acquired);
  Allocation first(EvictionQueueTraits::malloc(997));
  Allocation second(EvictionQueueTraits::malloc(997));
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  EXPECT_EQ(baseline_.queue_allocated + page, pool_.queue_allocated());
  first.reset();
  EXPECT_EQ(baseline_.queue_allocated + page, pool_.queue_allocated());
  second.reset();
  EXPECT_EQ(baseline_.queue_allocated, pool_.queue_allocated());
}

TEST_F(QueueMappingBudgetTest, FullSingleSlotRetirementPreservesNonfullSlab) {
  const size_t page = MemoryLimitPool::page_buffer_size();
  if (page != 4096)
    GTEST_SKIP() << "Single-slot fallback requires 4KiB system pages";
  std::vector<Allocation> normal;
  normal.emplace_back(EvictionQueueTraits::malloc(4080));
  ASSERT_NE(nullptr, normal.back());
  const size_t backing = pool_.queue_allocated();
  const size_t charge = pool_.available() - 2 * page;
  MetadataCharge reservation(pool_, charge);
  ASSERT_TRUE(reservation.acquired);
  Allocation fallback;
  for (size_t count = 0; count < 64; ++count) {
    Allocation next(EvictionQueueTraits::malloc(4080));
    ASSERT_NE(nullptr, next);
    if (pool_.queue_allocated() > backing) {
      fallback = std::move(next);
      break;
    }
    normal.push_back(std::move(next));
  }
  ASSERT_NE(nullptr, fallback);
  EXPECT_EQ(backing + 2 * page, pool_.queue_allocated());
  void *address = normal.front().get();
  normal.front().reset();
  fallback.reset();
  EXPECT_EQ(backing, pool_.queue_allocated());
  Allocation reused(EvictionQueueTraits::malloc(4080));
  EXPECT_EQ(address, reused.get());
  EXPECT_EQ(backing, pool_.queue_allocated());
}

}  // namespace
}  // namespace zvec::ailego

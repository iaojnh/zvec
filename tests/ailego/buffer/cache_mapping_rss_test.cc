// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
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

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static void stop_background(MemoryLimitPool &pool) {
    pool.stop_background_evictor();
  }
  static size_t mappings(MemoryLimitPool &pool) {
    std::lock_guard<std::mutex> lock(pool.cache_mapping_mutex_);
    size_t count = 0;
    MemoryLimitPool::CacheBuffer *previous = nullptr;
    for (auto *mapping = pool.cache_mappings_; mapping;
         mapping = mapping->mapped_next) {
      EXPECT_EQ(previous, mapping->mapped_prev);
      if (previous)
        EXPECT_LT(reinterpret_cast<uintptr_t>(previous->data),
                  reinterpret_cast<uintptr_t>(mapping->data));
      previous = mapping;
      ++count;
    }
    return count;
  }
  static size_t trim(MemoryLimitPool &pool, size_t bytes) {
    return pool.trim_cache_buffers(bytes);
  }
  static size_t trim_retained(MemoryLimitPool &pool, size_t bytes) {
    return pool.trim_retained_blocks(bytes);
  }
  static ProcessRssBudget::ResidentSample sample() {
    return MemoryLimitPool::read_slab_resident_sample();
  }
  static ProcessRssBudget::ResidentSample background_sample() {
    return MemoryLimitPool::read_background_resident_sample();
  }
  static ProcessRssBudget::ResidentSample sample_while_busy(
      MemoryLimitPool &pool) {
    std::lock_guard<std::mutex> lock(pool.cache_mapping_mutex_);
    return sample();
  }
  static ProcessRssBudget::ResidentSample background_while_registry_busy(
      MemoryLimitPool &pool) {
    return background_while_mutex_busy(pool.cache_mapping_mutex_);
  }
  static ProcessRssBudget::ResidentSample background_while_slabs_busy(
      MemoryLimitPool &pool) {
    return background_while_mutex_busy(pool.slab_mutex_);
  }
  static ProcessRssBudget::ResidentSample background_while_retained_busy(
      MemoryLimitPool &pool) {
    return background_while_mutex_busy(pool.retained_mutex_);
  }
  static ProcessRssBudget::ResidentSample background_while_mutex_busy(
      std::mutex &mutex) {
    std::atomic<bool> locked{false}, release{false};
    std::thread holder([&] {
      std::lock_guard<std::mutex> lock(mutex);
      locked.store(true, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
    });
    while (!locked.load(std::memory_order_acquire)) std::this_thread::yield();
    const auto result = background_sample();
    release.store(true, std::memory_order_release);
    holder.join();
    return result;
  }
};
namespace {
constexpr size_t kMiB = 1UL << 20;
size_t sampled_rss = 0, sampled_resident = 0;
ProcessRssBudget::ResidentSample ReadResident() {
  return {sampled_rss, sampled_resident};
}

TEST(CacheMappingBudgetTest, ConfirmationKeepsUntouchedAndPendingBytesCharged) {
  ProcessRssBudget budget;
  sampled_rss = 64 * kMiB;
  sampled_resident = 0;
  ASSERT_TRUE(budget.configure(512 * kMiB, nullptr, nullptr, 0, ReadResident));
  ASSERT_TRUE(budget.try_acquire(256 * kMiB));
  sampled_rss = 320 * kMiB;
  sampled_resident = 256 * kMiB;
  budget.refresh(true);
  ASSERT_TRUE(budget.try_acquire(32 * kMiB));  // Untouched backing.
  budget.release(128 * kMiB);
  ASSERT_TRUE(budget.try_acquire(128 * kMiB));
  budget.release(8 * kMiB);
  ASSERT_TRUE(budget.try_acquire(8 * kMiB));
  ASSERT_TRUE(budget.pressure());
  EXPECT_TRUE(budget.needs_resident_confirmation());
  EXPECT_GT(budget.resident_release_generation(), 0u);
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));  // Allocation still in flight.
  budget.refresh(true);
  EXPECT_FALSE(budget.pressure());
  EXPECT_FALSE(budget.needs_resident_confirmation());
  EXPECT_EQ(152 * kMiB, budget.available());
  budget.finish_reservation(8 * kMiB, false);
  EXPECT_EQ(160 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(161 * kMiB));
  sampled_rss = 500 * kMiB;  // Actual RSS pressure must remain effective.
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  EXPECT_FALSE(budget.needs_resident_confirmation());
  budget.release(288 * kMiB);
}

class CacheMappingRssTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(128 * kMiB));
    MemoryLimitPoolTestPeer::stop_background(pool_);
    ASSERT_TRUE(pool_.configure_process_rss_limit(0));
    ASSERT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.evictable_cache_used());
    EXPECT_EQ(0, pool_.init(0));
    EXPECT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
  }
  struct Deleter {
    void operator()(MemoryLimitPool::CacheBuffer *buffer) const {
      MemoryLimitPool::get_instance().release_cache_buffer(buffer);
    }
  };
  using Buffer = std::unique_ptr<MemoryLimitPool::CacheBuffer, Deleter>;
  Buffer acquire(size_t bytes = kMiB) {
    return Buffer(pool_.try_acquire_cache_buffer(bytes));
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
};

TEST_F(CacheMappingRssTest, IdleReuseRetainsRegistrationUntilSuccessfulUnmap) {
  auto first = acquire();
  auto second = acquire(2 * kMiB);
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  EXPECT_EQ(2u, MemoryLimitPoolTestPeer::mappings(pool_));
  auto *address = first.get();
  first.reset();
  EXPECT_EQ(2u, MemoryLimitPoolTestPeer::mappings(pool_));
  first = acquire();
  EXPECT_EQ(address, first.get());
  EXPECT_EQ(2u, MemoryLimitPoolTestPeer::mappings(pool_));
  first.reset();
  EXPECT_EQ(kMiB, MemoryLimitPoolTestPeer::trim(pool_, kMiB));
  EXPECT_EQ(1u, MemoryLimitPoolTestPeer::mappings(pool_));
}

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST_F(CacheMappingRssTest, NativeWitnessExcludesUntouchedAndIncludesIdleBacking) {
  auto touched = acquire(8 * kMiB);
  auto untouched = acquire(8 * kMiB);
  ASSERT_NE(nullptr, touched);
  ASSERT_NE(nullptr, untouched);
  const auto empty = MemoryLimitPoolTestPeer::sample();
  std::memset(touched->data, 17, touched->bytes);
  const auto full = MemoryLimitPoolTestPeer::sample();
  EXPECT_EQ(8 * kMiB, full.owned_resident - empty.owned_resident);
  EXPECT_EQ(full.owned_resident,
            MemoryLimitPoolTestPeer::background_sample().owned_resident);
  EXPECT_LE(full.owned_resident, full.rss);
  touched.reset();
  EXPECT_EQ(full.owned_resident,
            MemoryLimitPoolTestPeer::sample().owned_resident);
  EXPECT_EQ(8 * kMiB, MemoryLimitPoolTestPeer::trim(pool_, 8 * kMiB));
  EXPECT_EQ(empty.owned_resident,
            MemoryLimitPoolTestPeer::sample().owned_resident);
}

TEST_F(CacheMappingRssTest, RetainedMetadataWitnessTracksTouchReuseAndUnmap) {
  constexpr size_t bytes = 4 * kMiB;
  const size_t charge = MemoryLimitPool::retained_block_size(bytes);
  const size_t initial_metadata = pool_.metadata_used();
  const size_t initial_external = pool_.external_used();
  void *touched = pool_.acquire_retained_block(bytes, true);
  void *untouched = pool_.acquire_retained_block(bytes, true);
  ASSERT_NE(nullptr, touched);
  ASSERT_NE(nullptr, untouched);
  const auto empty = MemoryLimitPoolTestPeer::background_sample();
  std::memset(touched, 43, bytes);
  const auto full = MemoryLimitPoolTestPeer::background_sample();
  EXPECT_EQ(bytes, full.owned_resident - empty.owned_resident);
  EXPECT_EQ(full.owned_resident,
            MemoryLimitPoolTestPeer::sample().owned_resident);
  pool_.release_retained_block(touched);
  EXPECT_EQ(full.owned_resident,
            MemoryLimitPoolTestPeer::background_sample().owned_resident);
  void *workspace = pool_.acquire_retained_block(bytes, false);
  EXPECT_EQ(touched, workspace);
  EXPECT_EQ(initial_external + charge, pool_.external_used());
  EXPECT_EQ(initial_metadata + charge, pool_.metadata_used());
  pool_.release_retained_block(workspace);
  EXPECT_EQ(charge, MemoryLimitPoolTestPeer::trim_retained(pool_, charge));
  EXPECT_LT(MemoryLimitPoolTestPeer::background_sample().owned_resident,
            full.owned_resident);
  pool_.release_retained_block(untouched);
}

TEST_F(CacheMappingRssTest, BusyRetainedRegistryKeepsIndependentCacheProof) {
  auto touched = acquire(8 * kMiB);
  ASSERT_NE(nullptr, touched);
  std::memset(touched->data, 47, touched->bytes);
  const auto sample =
      MemoryLimitPoolTestPeer::background_while_retained_busy(pool_);
  EXPECT_GE(sample.owned_resident, touched->bytes);
  EXPECT_LE(sample.owned_resident, sample.rss);
}

TEST_F(CacheMappingRssTest, ConcurrentRetainedSamplesKeepMappingLifetimesSafe) {
  std::atomic<bool> done{false};
  std::atomic<size_t> samples{0};
  std::thread sampler([&] {
    while (!done.load()) {
      const auto sample = MemoryLimitPoolTestPeer::background_sample();
      EXPECT_LE(sample.owned_resident, sample.rss);
      ++samples;
      std::this_thread::yield();
    }
  });
  while (!samples.load()) std::this_thread::yield();
  for (size_t i = 0; i < 256; ++i) {
    void *data = pool_.acquire_retained_block(16UL << 10, i % 2);
    EXPECT_NE(nullptr, data);
    if (!data) break;
    std::memset(data, static_cast<int>(i), 16UL << 10);
    pool_.release_retained_block(data);
    (void)MemoryLimitPoolTestPeer::trim_retained(pool_, 128UL << 10);
  }
  done = true;
  sampler.join();
  EXPECT_GT(samples.load(), 0u);
}

TEST_F(CacheMappingRssTest, BusyRegistryPreservesIndependentSlabWitness) {
  std::vector<char *> pages;
  const size_t page_size = MemoryLimitPool::page_buffer_size();
  for (size_t i = 0; i < 16; ++i) {
    char *page = nullptr;
    ASSERT_TRUE(pool_.try_acquire_buffer(page_size, page));
    ASSERT_NE(nullptr, page);
    std::memset(page, 31, page_size);
    pages.push_back(page);
  }
  const auto sample =
      MemoryLimitPoolTestPeer::background_while_registry_busy(pool_);
  EXPECT_GE(sample.owned_resident, pages.size() * page_size);
  EXPECT_LE(sample.owned_resident, sample.rss);
  for (auto *page : pages) pool_.release_buffer(page, page_size);
}

TEST_F(CacheMappingRssTest, BusySlabListPreservesIndependentMappingWitness) {
  auto touched = acquire(8 * kMiB);
  ASSERT_NE(nullptr, touched);
  std::memset(touched->data, 23, touched->bytes);
  const auto sample = MemoryLimitPoolTestPeer::background_while_slabs_busy(pool_);
  EXPECT_GE(sample.owned_resident, touched->bytes);
  EXPECT_LE(sample.owned_resident, sample.rss);
}

TEST_F(CacheMappingRssTest, BusyRegistrySuppliesNoStaleWitness) {
  auto buffer = acquire();
  ASSERT_NE(nullptr, buffer);
  std::memset(buffer->data, 23, buffer->bytes);
  const auto sample = MemoryLimitPoolTestPeer::sample_while_busy(pool_);
  EXPECT_GT(sample.rss, 0u);
  EXPECT_EQ(0u, sample.owned_resident);
}

TEST_F(CacheMappingRssTest, BackgroundSnapshotHonorsSlabChargedPageMask) {
  const size_t bytes = MemoryLimitPool::page_buffer_size();
  char *touched = nullptr, *untouched = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(bytes, touched));
  ASSERT_TRUE(pool_.try_acquire_buffer(bytes, untouched));
  std::memset(touched, 29, bytes);
  const auto raw = MemoryLimitPoolTestPeer::sample();
  const auto background = MemoryLimitPoolTestPeer::background_sample();
  EXPECT_EQ(raw.owned_resident, background.owned_resident);
  EXPECT_GT(background.owned_resident, 0u);
  pool_.release_buffer(touched, bytes);
  pool_.release_buffer(untouched, bytes);
}

TEST_F(CacheMappingRssTest, ConcurrentNativeSamplesKeepMappingLifetimesSafe) {
  std::atomic<bool> done{false};
  std::atomic<size_t> samples{0};
  std::thread sampler([&] {
    while (!done.load()) {
      const auto sample = samples.load() % 2
                              ? MemoryLimitPoolTestPeer::sample()
                              : MemoryLimitPoolTestPeer::background_sample();
      EXPECT_LE(sample.owned_resident, sample.rss);
      ++samples;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  for (size_t i = 0; i < 256; ++i) {
    auto buffer = acquire();
    EXPECT_NE(nullptr, buffer);
    if (!buffer) break;
    std::memset(buffer->data, static_cast<int>(i), buffer->bytes);
    buffer.reset();
    EXPECT_EQ(kMiB, MemoryLimitPoolTestPeer::trim(pool_, kMiB));
  }
  done = true;
  sampler.join();
  EXPECT_GT(samples.load(), 0u);
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
}
#endif
}  // namespace
}  // namespace zvec::ailego

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

#include "db/common/global_resource.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/io/file.h>
#include <zvec/db/config.h>
#include "db/common/native_image_memory.h"
#include "db/common/native_worker_memory.h"
#if defined(__linux__)
#include <dlfcn.h>
#endif

using namespace zvec;

#if GTEST_HAS_DEATH_TEST
namespace {

struct ResourceSnapshot {
  ailego::ThreadPool *query_pool{nullptr};
  ailego::ThreadPool *optimize_pool{nullptr};
  arrow::internal::ThreadPool *arrow_pool{nullptr};
  std::shared_ptr<rocksdb::Cache> cache;
  std::shared_ptr<rocksdb::WriteBufferManager> write_buffer_manager;
  uint64_t rocksdb_capacity{0};
  bool consistent{false};
};

ResourceSnapshot ReadPublishedResources(GlobalResource &resource) {
  ResourceSnapshot snapshot;
  snapshot.query_pool = resource.query_thread_pool();
  snapshot.optimize_pool = resource.optimize_thread_pool();
  snapshot.arrow_pool = resource.arrow_query_thread_pool();
  snapshot.cache = resource.rocksdb_block_cache();
  snapshot.write_buffer_manager = resource.rocksdb_write_buffer_manager();
  snapshot.rocksdb_capacity = resource.rocksdb_memory_capacity();
  if (!snapshot.query_pool || !snapshot.optimize_pool || !snapshot.cache ||
      !snapshot.write_buffer_manager || !snapshot.arrow_pool) {
    return snapshot;
  }

  for (int iteration = 0; iteration < 100; ++iteration) {
    if (resource.initialize() != 0 ||
        resource.query_thread_pool() != snapshot.query_pool ||
        resource.optimize_thread_pool() != snapshot.optimize_pool ||
        resource.arrow_query_thread_pool() != snapshot.arrow_pool ||
        resource.rocksdb_block_cache() != snapshot.cache ||
        resource.rocksdb_write_buffer_manager() !=
            snapshot.write_buffer_manager ||
        resource.rocksdb_memory_capacity() != snapshot.rocksdb_capacity) {
      return snapshot;
    }
  }
  snapshot.consistent = true;
  return snapshot;
}

void CheckConcurrentLazyResourceAccess(bool existing_pool) {
  constexpr uint64_t kExistingPoolBytes = 100ULL * 1024 * 1024;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_FALSE(pool.initialized());
  if (existing_pool) {
    ASSERT_EQ(0, pool.init(kExistingPoolBytes));
  }
  const auto &config = GlobalConfig::Instance();
  const auto total_capacity =
      existing_pool ? kExistingPoolBytes : config.memory_limit_bytes();
  const auto expected_pool_capacity =
      existing_pool
          ? kExistingPoolBytes
          : GlobalResource::calculate_buffer_pool_memory_budget(total_capacity);
  auto &resource = GlobalResource::Instance();

  std::promise<void> start;
  const auto started = start.get_future().share();
  std::vector<std::future<ResourceSnapshot>> readers;
  try {
    for (int worker = 0; worker < 8; ++worker) {
      readers.emplace_back(
          std::async(std::launch::async, [&resource, started]() {
            started.wait();
            return ReadPublishedResources(resource);
          }));
    }
  } catch (...) {
    start.set_value();
    throw;
  }
  start.set_value();

  std::vector<ResourceSnapshot> snapshots;
  for (auto &reader : readers) {
    snapshots.push_back(reader.get());
  }

  const auto max_workers = std::max(std::thread::hardware_concurrency(), 1u);
  const auto expected_rocksdb_capacity =
      GlobalResource::calculate_rocksdb_memory_budget(total_capacity);
  for (const auto &snapshot : snapshots) {
    ASSERT_TRUE(snapshot.consistent);
    EXPECT_EQ(snapshots.front().query_pool, snapshot.query_pool);
    EXPECT_EQ(snapshots.front().optimize_pool, snapshot.optimize_pool);
    EXPECT_EQ(snapshots.front().arrow_pool, snapshot.arrow_pool);
    EXPECT_EQ(snapshot.query_pool->count(), snapshot.arrow_pool->GetCapacity());
    EXPECT_EQ(snapshots.front().cache, snapshot.cache);
    EXPECT_EQ(snapshots.front().write_buffer_manager,
              snapshot.write_buffer_manager);
    EXPECT_EQ(std::min(config.query_thread_count(), max_workers),
              snapshot.query_pool->count());
    EXPECT_EQ(std::min(config.optimize_thread_count(), max_workers),
              snapshot.optimize_pool->count());
    EXPECT_EQ(expected_rocksdb_capacity, snapshot.rocksdb_capacity);
    EXPECT_EQ(expected_rocksdb_capacity, snapshot.cache->GetCapacity());
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    ASSERT_NE(nullptr, snapshot.cache->memory_allocator());
    EXPECT_STREQ("ZvecMappedCacheAllocator",
                 snapshot.cache->memory_allocator()->Name());
#else
    EXPECT_EQ(nullptr, snapshot.cache->memory_allocator());
#endif
    EXPECT_EQ(GlobalResource::calculate_rocksdb_write_buffer_budget(
                  expected_rocksdb_capacity),
              snapshot.write_buffer_manager->buffer_size());
    EXPECT_TRUE(snapshot.write_buffer_manager->cost_to_cache());
  }
  EXPECT_EQ(expected_pool_capacity, pool.capacity());
}

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
void CheckStartupHostMemoryRecovery() {
  constexpr size_t kMiB = 1UL << 20;
  void *host = ailego::File::MemoryMap(128 * kMiB, 0);
  ASSERT_NE(nullptr, host);
  auto cleanup = ailego::ScopeGuard::Make([&] {
    if (host) ailego::File::MemoryUnmap(host, 128 * kMiB);
  });
  std::memset(host, 0x6a, 128 * kMiB);
  const size_t rss = ailego::MemoryHelper::SelfRSS();
  ASSERT_GT(rss, 0u);
  GlobalConfig::ConfigData config;
  config.memory_limit_bytes = rss + 4 * kMiB;
  config.query_thread_count = config.optimize_thread_count = 1;
  ASSERT_TRUE(GlobalConfig::Instance().initialize(config).ok());
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_GT(pool.queue_metadata_used(), 0u);
  ASSERT_GT(detail::NativeWorkerStackReservation().bytes(), 0u);
  ailego::File::MemoryUnmap(host, 128 * kMiB);
  host = nullptr;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (pool.under_process_rss_pressure() &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_FALSE(pool.under_process_rss_pressure());
  ASSERT_LT(pool.process_rss() + 16 * kMiB, config.memory_limit_bytes);
  ASSERT_TRUE(pool.try_charge_workspace(16 * kMiB));
  pool.release_external(16 * kMiB);
  // Persistent metadata and worker allowances remain live throughout recovery.
  EXPECT_GT(pool.committed(), 0u);
}
#endif

}  // namespace
#endif  // GTEST_HAS_DEATH_TEST

class GlobalResourceDeathTest : public ::testing::TestWithParam<bool> {};

TEST_P(GlobalResourceDeathTest,
       ConcurrentLazyInitializationPublishesResources) {
#if GTEST_HAS_DEATH_TEST
  // Re-exec to keep the first lazy initialization independent of the explicit
  // configuration and process-wide pool used by the remaining tests.
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        CheckConcurrentLazyResourceAccess(GetParam());
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
#else
  GTEST_SKIP()
      << "Process-isolated exit tests are not supported on this platform";
#endif
}

// Exercise both a fresh default budget and an existing core-owned pool.
INSTANTIATE_TEST_SUITE_P(PoolInitialization, GlobalResourceDeathTest,
                         ::testing::Values(false, true));

#if (defined(__linux__) || defined(__APPLE__) || defined(_WIN32)) && \
    GTEST_HAS_DEATH_TEST
TEST(GlobalResourceStartupDeathTest, HostReleaseRestoresWorkspaceAdmission) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        CheckStartupHostMemoryRecovery();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}

TEST(GlobalResourceStartupDeathTest,
     PartialBudgetFailureCannotBecomeStandalone) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        constexpr size_t limit = 128UL << 20;
        if (pool.init(
                GlobalResource::calculate_buffer_pool_memory_budget(limit)))
          std::_Exit(1);
        // Simulate existing mandatory backing exhausting the pool exactly at
        // native image/worker reservation, after pool setup has succeeded.
        const size_t occupied = pool.available();
        if (!pool.try_charge_external(occupied)) std::_Exit(2);
        GlobalConfig::ConfigData config;
        config.memory_limit_bytes = limit;
        config.query_thread_count = config.optimize_thread_count = 1;
        if (GlobalConfig::Instance().initialize(config).ok()) std::_Exit(3);
        pool.release_external(occupied);
        auto &resources = GlobalResource::Instance();
        if (resources.initialize() == 0 || resources.initialized() ||
            resources.query_thread_pool() || resources.rocksdb_block_cache())
          std::_Exit(4);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}
#endif

#if defined(__linux__)
TEST(NativeImageMemory, CountsResidentCapacityNotFileSizeOrSegmentGaps) {
  ElfW(Phdr) headers[4]{};
  headers[0].p_type = PT_LOAD;
  headers[0].p_vaddr = 0x1003;
  headers[0].p_filesz = 16;
  headers[0].p_memsz = 0x1800;
  headers[1].p_type = PT_LOAD;
  headers[1].p_vaddr = 0x2f00;
  headers[1].p_memsz = 0x300;
  headers[2].p_type = PT_TLS;
  headers[2].p_memsz = 0x100000;
  headers[3].p_type = PT_LOAD;
  headers[3].p_vaddr = 0x8001;
  headers[3].p_memsz = 1;
  size_t bytes = 0;
  ASSERT_TRUE(detail::ElfLoadImageBytes(headers, 4, 4096, &bytes));
  EXPECT_EQ(4u * 4096, bytes);
  ASSERT_TRUE(detail::ElfLoadImageBytes(headers, 4, 16384, &bytes));
  EXPECT_EQ(2u * 16384, bytes);
}

TEST(NativeImageMemory, RejectsInvalidRangesWithoutPublishingPartialResult) {
  ElfW(Phdr) headers[2]{};
  size_t bytes = 73;
  EXPECT_FALSE(detail::ElfLoadImageBytes(headers, 2, 4096, &bytes));
  EXPECT_FALSE(detail::ElfLoadImageBytes(nullptr, 2, 4096, &bytes));
  EXPECT_FALSE(detail::ElfLoadImageBytes(headers, 2, 0, &bytes));
  EXPECT_FALSE(detail::ElfLoadImageBytes(headers, 2, 4096, nullptr));
  headers[0].p_type = PT_LOAD;
  headers[0].p_vaddr = std::numeric_limits<uintptr_t>::max() - 1;
  headers[0].p_memsz = 4;
  EXPECT_FALSE(detail::ElfLoadImageBytes(headers, 2, 4096, &bytes));
  headers[0].p_memsz = 1;
  EXPECT_FALSE(detail::ElfLoadImageBytes(headers, 2, 4096, &bytes));
  headers[0].p_vaddr = 0x2000;
  headers[1].p_type = PT_LOAD;
  headers[1].p_vaddr = 0x1000;
  headers[1].p_memsz = 1;
  EXPECT_FALSE(detail::ElfLoadImageBytes(headers, 2, 4096, &bytes));
  EXPECT_EQ(73u, bytes);
}

TEST(NativeImageMemory, ResolvesSharedOwnerAndExcludesCallerExecutable) {
  static const char caller_anchor = 0;
  EXPECT_EQ(0u, detail::NativeImageBytesForAddress(&caller_anchor));
  // A non-PIE main executable may expose a function's canonical PLT address
  // through RTLD_DEFAULT. RTLD_NEXT excludes that executable so this tests the
  // actual shared image.
  auto *symbol = dlsym(RTLD_NEXT, "malloc");
  ASSERT_NE(nullptr, symbol);
  const auto bytes = detail::NativeImageBytesForAddress(symbol);
  EXPECT_GT(bytes, 0u);
  EXPECT_EQ(0u, bytes % static_cast<size_t>(sysconf(_SC_PAGESIZE)));
  EXPECT_THROW(detail::NativeImageBytesForAddress(nullptr), std::bad_alloc);
  const char stack_anchor = 0;
  EXPECT_THROW(detail::NativeImageBytesForAddress(&stack_anchor), std::bad_alloc);
}

TEST(NativeImageMemoryDeathTest, ReservesBeforeAdmissionAndDoesNotRefundLiveImage) {
#if GTEST_HAS_DEATH_TEST
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        if (pool.init(128u * 1024 * 1024) != 0) std::_Exit(1);
        auto *symbol = dlsym(RTLD_NEXT, "malloc");
        const auto bytes = detail::NativeImageBytesForAddress(symbol);
        if (!bytes) std::_Exit(5);
        const auto before = pool.stats();
        detail::ReserveNativeImage(symbol);
        const auto after = pool.stats();
        if (after.external_used != before.external_used + bytes ||
            after.committed != before.committed + bytes ||
            detail::NativeImageReservation().bytes() != bytes)
          std::_Exit(2);
        detail::ReserveNativeImage(symbol);
        if (pool.stats().committed != after.committed) std::_Exit(3);
        // Reject a reservation beyond available capacity without changing
        // ownership of the existing image reservation.
        bool rejected = false;
        try {
          detail::NativeImageReservation().add(pool.capacity());
        } catch (const std::bad_alloc &) {
          rejected = true;
        }
        if (!rejected || detail::NativeImageReservation().bytes() != bytes ||
            pool.stats().committed != after.committed)
          std::_Exit(4);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
#else
  GTEST_SKIP() << "Process-isolated tests are not supported on this platform";
#endif
}
#endif

TEST(GlobalResource, ReservesFifteenPercentForRocksDb) {
  constexpr uint64_t kOneHundredMiB = 100ULL * 1024 * 1024;
  constexpr uint64_t kTwoGiB = 2ULL * 1024 * 1024 * 1024;

  EXPECT_EQ(15ULL * 1024 * 1024,
            GlobalResource::calculate_rocksdb_memory_budget(kOneHundredMiB));
  EXPECT_EQ(
      85ULL * 1024 * 1024,
      GlobalResource::calculate_buffer_pool_memory_budget(kOneHundredMiB));
  EXPECT_EQ(kTwoGiB,
            GlobalResource::calculate_rocksdb_memory_budget(kTwoGiB) +
                GlobalResource::calculate_buffer_pool_memory_budget(kTwoGiB));
}

TEST(GlobalResource, LeavesCacheHeadroomForCompaction) {
  constexpr uint64_t kMiB = 1024 * 1024;
  const auto rocksdb_bytes =
      GlobalResource::calculate_rocksdb_memory_budget(512 * kMiB);
  EXPECT_EQ(
      rocksdb_bytes / 2,
      GlobalResource::calculate_rocksdb_write_buffer_budget(rocksdb_bytes));
  EXPECT_EQ(0u, GlobalResource::calculate_rocksdb_write_buffer_budget(0));
  EXPECT_EQ(1u, GlobalResource::calculate_rocksdb_write_buffer_budget(1));
  EXPECT_EQ(1u, GlobalResource::calculate_rocksdb_write_buffer_budget(3));
  EXPECT_EQ(std::numeric_limits<uint64_t>::max() / 2,
            GlobalResource::calculate_rocksdb_write_buffer_budget(
                std::numeric_limits<uint64_t>::max()));
}

TEST(GlobalResource, SizesShardsForPinnedWriteBufferCharges) {
  constexpr uint64_t kMiB = 1024 * 1024;
  EXPECT_EQ(0, GlobalResource::calculate_rocksdb_cache_shard_bits(0));
  EXPECT_EQ(0,
            GlobalResource::calculate_rocksdb_cache_shard_bits(32 * kMiB - 1));
  EXPECT_EQ(1, GlobalResource::calculate_rocksdb_cache_shard_bits(32 * kMiB));
  EXPECT_EQ(2,
            GlobalResource::calculate_rocksdb_cache_shard_bits(
                GlobalResource::calculate_rocksdb_memory_budget(512 * kMiB)));
  EXPECT_EQ(6, GlobalResource::calculate_rocksdb_cache_shard_bits(1024 * kMiB));
  EXPECT_EQ(6, GlobalResource::calculate_rocksdb_cache_shard_bits(
                   std::numeric_limits<uint64_t>::max()));
}

TEST(GlobalResource, CacheAdmitsIndexBlockWhileWriteBufferIsPinned) {
  constexpr size_t kMiB = 1024 * 1024;
  const auto capacity =
      GlobalResource::calculate_rocksdb_memory_budget(512 * kMiB);
  auto cache = rocksdb::NewLRUCache(
      capacity, GlobalResource::calculate_rocksdb_cache_shard_bits(capacity),
      /*strict_capacity_limit=*/true);
  ASSERT_NE(nullptr, cache);
  EXPECT_TRUE(cache->HasStrictCapacityLimit());
  // Model the pinned cache charges without allocating a full memtable.
  // Releasing the manager also releases these dummy handles on early failure.
  auto manager = std::make_unique<rocksdb::WriteBufferManager>(
      GlobalResource::calculate_rocksdb_write_buffer_budget(capacity), cache,
      /*allow_stall=*/true);
  manager->ReserveMem(manager->buffer_size());
  EXPECT_TRUE(manager->ShouldFlush());
  EXPECT_TRUE(manager->ShouldStall());
  EXPECT_GE(manager->dummy_entries_in_cache_usage(), manager->memory_usage());
  rocksdb::Cache::Handle *handle = nullptr;
  const auto status =
      cache->Insert("compaction-index", nullptr, &rocksdb::kNoopCacheItemHelper,
                    2 * kMiB, &handle);
  EXPECT_TRUE(status.ok()) << status.ToString();
  EXPECT_NE(nullptr, handle);
  EXPECT_LE(cache->GetUsage(), capacity);
  if (handle) {
    cache->Release(handle);
  }
  // Retain strict rejection for genuinely oversized entries.
  handle = nullptr;
  const auto oversized =
      cache->Insert("oversized", nullptr, &rocksdb::kNoopCacheItemHelper,
                    capacity + 1, &handle);
  EXPECT_TRUE(oversized.IsMemoryLimit()) << oversized.ToString();
  if (handle) {
    cache->Release(handle);
  }
  manager->ScheduleFreeMem(manager->buffer_size());
  manager->FreeMem(manager->buffer_size());
  EXPECT_FALSE(manager->ShouldStall());
}

TEST(GlobalResource, UsesEffectiveCountsAndDisablesBindingByDefault) {
  GlobalConfig::ConfigData config;
  config.query_thread_count = 2;
  config.optimize_thread_count = 1;
  EXPECT_FALSE(config.query_thread_binding);
  EXPECT_FALSE(config.optimize_thread_binding);

  const auto status = GlobalConfig::Instance().initialize(config);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FALSE(GlobalConfig::Instance().query_thread_binding());
  EXPECT_FALSE(GlobalConfig::Instance().optimize_thread_binding());

  const auto max_workers = std::max(std::thread::hardware_concurrency(), 1u);
  const auto expected_query_workers =
      std::min(GlobalConfig::Instance().query_thread_count(), max_workers);
  const auto expected_optimize_workers =
      std::min(GlobalConfig::Instance().optimize_thread_count(), max_workers);

  EXPECT_EQ(expected_query_workers,
            GlobalResource::Instance().query_thread_pool()->count());
  EXPECT_EQ(expected_optimize_workers,
            GlobalResource::Instance().optimize_thread_pool()->count());
  auto *arrow_pool = GlobalResource::Instance().arrow_query_thread_pool();
  ASSERT_NE(nullptr, arrow_pool);
  EXPECT_EQ(expected_query_workers, arrow_pool->GetCapacity());
  EXPECT_NE(arrow::internal::GetCpuThreadPool(), arrow_pool);
  auto *io_pool = GlobalResource::Instance().arrow_io_thread_pool();
  ASSERT_NE(nullptr, io_pool);
  EXPECT_EQ(expected_query_workers, io_pool->GetCapacity());
  EXPECT_NE(arrow_pool, io_pool);
  auto task =
      arrow_pool->Submit([arrow_pool] { return arrow_pool->OwnsThisThread(); });
  ASSERT_TRUE(task.ok());
  ASSERT_TRUE(task->result().ok());
  EXPECT_TRUE(*task->result());

  const uint64_t total_capacity = config.memory_limit_bytes;
  const uint64_t expected_rocksdb_capacity =
      GlobalResource::calculate_rocksdb_memory_budget(total_capacity);
  const uint64_t expected_buffer_capacity =
      GlobalResource::calculate_buffer_pool_memory_budget(total_capacity);
  EXPECT_EQ(expected_buffer_capacity,
            ailego::MemoryLimitPool::get_instance().capacity());

#if defined(__APPLE__) || defined(__linux__) || defined(_WIN32)
  const auto stack_bytes = detail::DefaultNativeWorkerStackBytes();
  const auto native_workers =
      3 * expected_query_workers + expected_optimize_workers +
      detail::RocksDbBackgroundThreads(config.optimize_thread_count) + 2;
  EXPECT_GT(stack_bytes, 0u);
  EXPECT_EQ(native_workers * stack_bytes,
            detail::NativeWorkerStackReservation().bytes());
  EXPECT_GE(ailego::MemoryLimitPool::get_instance().external_used(),
            native_workers * stack_bytes);
  // Repeated access/initialization cannot charge the same process workers
  // again, and collection close does not refund live worker-stack capacity.
  const auto reserved = detail::NativeWorkerStackReservation().bytes();
  EXPECT_EQ(0, GlobalResource::Instance().initialize());
  EXPECT_EQ(reserved, detail::NativeWorkerStackReservation().bytes());
#endif

  const auto image_reserved = detail::NativeImageReservation().bytes();
  EXPECT_GE(ailego::MemoryLimitPool::get_instance().external_used(),
            image_reserved);
  EXPECT_EQ(0, GlobalResource::Instance().initialize());
  EXPECT_EQ(image_reserved, detail::NativeImageReservation().bytes());
  EXPECT_EQ(total_capacity,
            ailego::MemoryLimitPool::get_instance().process_rss_limit());
  EXPECT_GT(ailego::MemoryLimitPool::get_instance().process_rss(), 0u);

  const auto rocksdb_cache = GlobalResource::Instance().rocksdb_block_cache();
  const auto write_buffer_manager =
      GlobalResource::Instance().rocksdb_write_buffer_manager();
  ASSERT_NE(nullptr, rocksdb_cache);
  ASSERT_NE(nullptr, write_buffer_manager);
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  ASSERT_NE(nullptr, rocksdb_cache->memory_allocator());
  EXPECT_STREQ("ZvecMappedCacheAllocator",
               rocksdb_cache->memory_allocator()->Name());
#else
  EXPECT_EQ(nullptr, rocksdb_cache->memory_allocator());
#endif
  EXPECT_EQ(expected_rocksdb_capacity, rocksdb_cache->GetCapacity());
  EXPECT_EQ(GlobalResource::calculate_rocksdb_write_buffer_budget(
                expected_rocksdb_capacity),
            write_buffer_manager->buffer_size());
  EXPECT_TRUE(write_buffer_manager->cost_to_cache());
}

TEST(GlobalResource, RssPressureReclaimsOnlyUnpinnedRocksDbEntries) {
  auto &resource = GlobalResource::Instance();
  ASSERT_EQ(0, resource.initialize());
  auto cache = resource.rocksdb_block_cache();
  ASSERT_NE(nullptr, cache);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t original_limit = pool.process_rss_limit();
  ASSERT_GT(original_limit, 0u);
  auto restore = ailego::ScopeGuard::Make(
      [&] { pool.configure_process_rss_limit(original_limit); });
  rocksdb::Cache::Handle *cold = nullptr;
  rocksdb::Cache::Handle *pinned = nullptr;
  auto release = ailego::ScopeGuard::Make([&] {
    if (cold) cache->Release(cold);
    if (pinned) cache->Release(pinned);
  });
  ASSERT_TRUE(cache
                  ->Insert("rss-cold", nullptr, &rocksdb::kNoopCacheItemHelper,
                           4096, &cold)
                  .ok());
  ASSERT_TRUE(cache
                  ->Insert("rss-pinned", nullptr,
                           &rocksdb::kNoopCacheItemHelper, 4096, &pinned)
                  .ok());
  cache->Release(cold);
  cold = nullptr;
  const auto pinned_usage = cache->GetPinnedUsage();
  ASSERT_GT(cache->GetUsage(), pinned_usage);
  ASSERT_TRUE(pool.configure_process_rss_limit(1));
  const auto wait_until = [](auto ready) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!ready() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return ready();
  };
  ASSERT_TRUE(wait_until([&] { return cache->GetUsage() == pinned_usage; }));
  EXPECT_EQ(pinned_usage, cache->GetPinnedUsage());
  cache->Release(pinned);
  pinned = nullptr;
  EXPECT_TRUE(wait_until([&] { return cache->GetUsage() == 0; }));
}

TEST(GlobalResource, HostPressureReclaimsCacheWithoutChangingConfiguredLimit) {
  auto &resource = GlobalResource::Instance();
  ASSERT_EQ(0, resource.initialize());
  auto cache = resource.rocksdb_block_cache();
  ASSERT_NE(nullptr, cache);
  auto &config = GlobalConfig::Instance();
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t limit = pool.process_rss_limit();
  using Pressure = GlobalConfig::MemoryPressure;
  auto restore = ailego::ScopeGuard::Make(
      [&] { config.set_memory_pressure(Pressure::kNormal); });
  EXPECT_FALSE(config.set_memory_pressure(static_cast<Pressure>(255)).ok());
  rocksdb::Cache::Handle *pinned = nullptr;
  auto release = ailego::ScopeGuard::Make([&] {
    if (pinned) cache->Release(pinned);
  });
  ASSERT_TRUE(cache->Insert("host-cold", nullptr, &rocksdb::kNoopCacheItemHelper,
                            4096, nullptr).ok());
  ASSERT_TRUE(cache->Insert("host-pinned", nullptr,
                            &rocksdb::kNoopCacheItemHelper, 4096, &pinned).ok());
  ASSERT_TRUE(config.set_memory_pressure(Pressure::kBackground).ok());
  EXPECT_TRUE(pool.under_process_rss_pressure());
  EXPECT_EQ(limit, pool.process_rss_limit());
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (cache->GetUsage() > cache->GetPinnedUsage() &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(cache->GetPinnedUsage(), cache->GetUsage());
  EXPECT_GT(cache->GetPinnedUsage(), 0u);
  ASSERT_TRUE(config.set_memory_pressure(Pressure::kCritical).ok());
  EXPECT_FALSE(pool.try_charge_external(4096));
  ASSERT_TRUE(config.set_memory_pressure(Pressure::kNormal).ok());
  ASSERT_TRUE(pool.try_charge_external(4096));
  pool.release_external(4096);
}

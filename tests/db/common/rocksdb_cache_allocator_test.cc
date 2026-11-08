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
#include "db/common/rocksdb_cache_allocator.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <new>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/table.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/common/rocksdb_block_cache.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec {
namespace {

TEST(RocksdbCacheAllocatorTest, CoversCapacityAndAlignment) {
  RocksdbCacheAllocator allocator;
  for (size_t size :
       {0u, 1u, 4095u, 4096u, 16383u, 16384u, 16385u, 65536u, 1048576u}) {
    void *data = allocator.Allocate(size);
    ASSERT_NE(nullptr, data);
    EXPECT_EQ(0u,
              reinterpret_cast<uintptr_t>(data) % alignof(std::max_align_t));
    EXPECT_GE(allocator.UsableSize(data, size), size + sizeof(size_t));
    std::memset(data, 0xa5, size);
    if (size != 0) {
      EXPECT_EQ(0xa5, static_cast<unsigned char *>(data)[size - 1]);
    }
    allocator.Deallocate(data);
  }
  allocator.Deallocate(nullptr);
}

TEST(RocksdbCacheAllocatorTest, RejectsOverflowBeforeMapping) {
  RocksdbCacheAllocator allocator;
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  EXPECT_THROW(allocator.Allocate(kMax), std::bad_alloc);
  EXPECT_THROW(allocator.Allocate(kMax - sizeof(std::max_align_t)),
               std::bad_alloc);
  // A failed allocation must not poison subsequent allocations.
  void *data = allocator.Allocate(1024);
  allocator.Deallocate(data);
}

TEST(RocksdbCacheAllocatorTest,
     ProcessPressureAllowsNecessaryReadsWithoutLosingOwners) {
  RocksdbCacheAllocator allocator;
  void *live = allocator.Allocate(4096);
  auto release = ailego::ScopeGuard::Make([&] { allocator.Deallocate(live); });
  std::memset(live, 0x31, 4096);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.process_rss_limit();
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(pool.configure_process_rss_limit(previous)); });
  ASSERT_TRUE(pool.configure_process_rss_limit(1));
  void *temporary = nullptr;
  ASSERT_NO_THROW(temporary = allocator.Allocate(4096));
  ASSERT_NE(nullptr, temporary);
  allocator.Deallocate(temporary);
  EXPECT_EQ(0x31, static_cast<unsigned char *>(live)[4095]);
  ASSERT_TRUE(pool.configure_process_rss_limit(0));
  void *retry = allocator.Allocate(4096);
  ASSERT_NE(nullptr, retry);
  allocator.Deallocate(retry);
}

TEST(RocksdbCacheAllocatorTest, SupportsCrossThreadRelease) {
  auto allocator = std::make_shared<RocksdbCacheAllocator>();
  std::array<std::future<void>, 4> workers;
  for (auto &worker : workers) {
    worker = std::async(std::launch::async, [allocator]() {
      for (size_t round = 0; round < 100; ++round) {
        void *data = allocator->Allocate(4096 + round);
        std::memset(data, 0x31, 4096 + round);
        auto release = std::async(std::launch::async, [allocator, data]() {
          EXPECT_EQ(0x31, static_cast<unsigned char *>(data)[4095]);
          allocator->Deallocate(data);
        });
        release.get();
      }
    });
  }
  for (auto &worker : workers) {
    worker.get();
  }
}

void DeleteCachedBlock(void *value, rocksdb::MemoryAllocator *allocator) {
  allocator->Deallocate(value);
}

const rocksdb::Cache::CacheItemHelper kMappedCacheBlockHelper{
    rocksdb::CacheEntryRole::kMisc, DeleteCachedBlock};

void DeleteChargeTestValue(void *, rocksdb::MemoryAllocator *) {}

const rocksdb::Cache::CacheItemHelper kFilterChargeHelper{
    rocksdb::CacheEntryRole::kFilterBlock, DeleteChargeTestValue};
const rocksdb::Cache::CacheItemHelper kDataChargeHelper{
    rocksdb::CacheEntryRole::kDataBlock, DeleteChargeTestValue};

std::shared_ptr<RocksdbBlockCache> CreateChargeTestCache(size_t capacity) {
  return std::make_shared<RocksdbBlockCache>(
      rocksdb::NewLRUCache(capacity, 0, true, 0.5, nullptr, false,
                           rocksdb::kDontChargeCacheMetadata));
}

void CountDeletedBlock(void *value, rocksdb::MemoryAllocator *) {
  ++*static_cast<int *>(value);
}
const rocksdb::Cache::CacheItemHelper kCountedDataHelper{
    rocksdb::CacheEntryRole::kDataBlock, CountDeletedBlock};
const rocksdb::Cache::CacheItemHelper kWriteBufferChargeHelper{
    rocksdb::CacheEntryRole::kWriteBuffer, DeleteChargeTestValue};

TEST(RocksdbBlockCacheTest, LastContextReleasesUnpinnedBlocks) {
  auto cache = CreateChargeTestCache(4096);
  auto first = RocksdbBlockCache::AcquireContext(cache);
  auto second = RocksdbBlockCache::AcquireContext(cache);
  int deleted = 0;
  ASSERT_TRUE(
      cache->Insert("closed-db", &deleted, &kCountedDataHelper, 64).ok());
  first.reset();
  EXPECT_EQ(0, deleted);
  EXPECT_EQ(64u, cache->GetUsage());
  second.reset();
  EXPECT_EQ(1, deleted);
  EXPECT_EQ(0u, cache->GetUsage());
}

TEST(RocksdbBlockCacheTest, PinnedBlockSurvivesCloseAndFreesOnFinalRelease) {
  auto cache = CreateChargeTestCache(4096);
  auto context = RocksdbBlockCache::AcquireContext(cache);
  int deleted = 0;
  rocksdb::Cache::Handle *handle = nullptr;
  ASSERT_TRUE(
      cache->Insert("held", &deleted, &kCountedDataHelper, 64, &handle).ok());
  ASSERT_TRUE(cache->Ref(handle));
  context.reset();
  EXPECT_EQ(0, deleted);
  EXPECT_EQ(&deleted, cache->Value(handle));
  EXPECT_EQ(64u, cache->GetUsage());
  EXPECT_FALSE(cache->Release(handle));
  EXPECT_EQ(0, deleted);
  EXPECT_TRUE(cache->Release(handle));
  EXPECT_EQ(1, deleted);
  EXPECT_EQ(0u, cache->GetUsage());
}

TEST(RocksdbBlockCacheTest, RepeatedContextsDoNotAccumulateClosedDbBlocks) {
  auto cache = CreateChargeTestCache(4096);
  int deleted = 0;
  for (int cycle = 0; cycle < 100; ++cycle) {
    auto context = RocksdbBlockCache::AcquireContext(cache);
    ASSERT_TRUE(
        cache->Insert(std::to_string(cycle), &deleted, &kCountedDataHelper, 64)
            .ok());
    context.reset();
    EXPECT_EQ(cycle + 1, deleted);
    EXPECT_EQ(0u, cache->GetUsage());
  }
}

TEST(RocksdbBlockCacheTest, PressureUsesChargedTransientHandles) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.process_rss_limit();
  auto restore = ailego::ScopeGuard::Make(
      [&] { pool.configure_process_rss_limit(previous); });
  auto cache = CreateChargeTestCache(256);
  ASSERT_TRUE(pool.configure_process_rss_limit(1));
  int deleted = 0;
  rocksdb::Cache::Handle *handle = nullptr;
  ASSERT_TRUE(
      cache->Insert("read", &deleted, &kCountedDataHelper, 64, &handle).ok());
  EXPECT_EQ(nullptr, cache->Lookup("read", &kCountedDataHelper, nullptr));
  EXPECT_EQ(64u, cache->GetUsage());
  cache->EraseUnRefEntries();
  EXPECT_EQ(0, deleted);  // Active data must not be reclaimed.
  rocksdb::Cache::Handle *rejected = nullptr;
  EXPECT_TRUE(
      cache->Insert("full", &deleted, &kCountedDataHelper, 256, &rejected)
          .IsMemoryLimit());
  EXPECT_EQ(nullptr, rejected);
  EXPECT_EQ(0, deleted);  // Failed admission leaves ownership with the caller.
  EXPECT_TRUE(cache->Release(handle));
  EXPECT_EQ(1, deleted);
  EXPECT_EQ(0u, cache->GetUsage());
  ASSERT_TRUE(
      cache->Insert("prefetch", &deleted, &kCountedDataHelper, 64).ok());
  EXPECT_EQ(2, deleted);
  EXPECT_EQ(0u, cache->GetUsage());

  int value = 0;
  ASSERT_TRUE(
      cache->Insert("wbm", &value, &kWriteBufferChargeHelper, 64, &handle)
          .ok());
  auto *found = cache->Lookup("wbm", &kWriteBufferChargeHelper, nullptr);
  ASSERT_NE(nullptr, found);  // Write-buffer accounting is not a read cache.
  cache->Release(found);
  cache->Release(handle);
  cache->EraseUnRefEntries();

  ASSERT_TRUE(pool.configure_process_rss_limit(0));
  ASSERT_TRUE(
      cache->Insert("recovered", &deleted, &kCountedDataHelper, 64, &handle)
          .ok());
  cache->Release(handle);
  found = cache->Lookup("recovered", &kCountedDataHelper, nullptr);
  ASSERT_NE(nullptr, found);
  cache->Release(found);
  cache->EraseUnRefEntries();
  EXPECT_EQ(3, deleted);
}

#if (defined(__linux__) || defined(__APPLE__) || defined(_WIN32)) && \
    GTEST_HAS_DEATH_TEST
void CheckReadsAndBackgroundCompactionUnderPressure() {
  rocksdb::Options options;
  options.create_if_missing = true;
  options.disable_auto_compactions = true;
  options.compression = rocksdb::kNoCompression;
  auto allocator = std::make_shared<RocksdbCacheAllocator>();
  auto cache = std::make_shared<RocksdbBlockCache>(
      rocksdb::NewLRUCache(8UL << 20, 0, true, 0.5, allocator));
  rocksdb::BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.cache_index_and_filter_blocks = true;
  table_options.cache_index_and_filter_blocks_with_high_priority = true;
  table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));
  options.write_buffer_manager =
      std::make_shared<rocksdb::WriteBufferManager>(4UL << 20, cache, true);
  options.table_factory.reset(
      rocksdb::NewBlockBasedTableFactory(table_options));
#if defined(_WIN32)
  char temporary[MAX_PATH + 1]{};
  const DWORD path_length = ::GetTempPathA(sizeof(temporary), temporary);
  ASSERT_GT(path_length, 0u);
  ASSERT_LT(path_length, sizeof(temporary));
  char directory[MAX_PATH + 1]{};
  ASSERT_NE(0u, ::GetTempFileNameA(temporary, "zvr", 0, directory));
  ASSERT_TRUE(::DeleteFileA(directory));
  ASSERT_TRUE(::CreateDirectoryA(directory, nullptr));
#else
  // CTest/device runners provide a writable working directory; Android does
  // not provide the desktop /tmp directory.
  char directory[] = "./zvec-rss-rocksdb-XXXXXX";
  ASSERT_NE(nullptr, ::mkdtemp(directory));
#endif
  auto cleanup = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(rocksdb::DestroyDB(directory, options).ok()); });
  rocksdb::DB *raw = nullptr;
  ASSERT_TRUE(rocksdb::DB::Open(options, directory, &raw).ok());
  std::unique_ptr<rocksdb::DB> db(raw);
  rocksdb::FlushOptions flush;
  flush.wait = true;
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < 200; ++i) {
      ASSERT_TRUE(db->Put(rocksdb::WriteOptions(), std::to_string(i),
                          std::string(4096, 'a' + round))
                      .ok());
    }
    ASSERT_TRUE(db->Flush(flush).ok());
  }
  cache->EraseUnRefEntries();
  auto &pool = ailego::MemoryLimitPool::get_instance();
  auto restore =
      ailego::ScopeGuard::Make([&] { pool.configure_process_rss_limit(0); });
  ASSERT_TRUE(pool.configure_process_rss_limit(1));
  for (bool fill_cache : {false, true}) {
    rocksdb::ReadOptions read;
    read.fill_cache = fill_cache;
    std::string value;
    ASSERT_TRUE(db->Get(read, "10", &value).ok());
    EXPECT_EQ(std::string(4096, 'b'), value);
    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(read));
    size_t rows = 0;
    for (it->SeekToFirst(); it->Valid(); it->Next()) ++rows;
    EXPECT_TRUE(it->status().ok());
    EXPECT_EQ(200u, rows);
  }
  ASSERT_TRUE(db->Put(rocksdb::WriteOptions(), "new", "value").ok());
  ASSERT_TRUE(db->Flush(flush).ok());
  ASSERT_TRUE(
      db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr).ok());
  ASSERT_TRUE(pool.configure_process_rss_limit(0));
  std::string value;
  ASSERT_TRUE(db->Get(rocksdb::ReadOptions(), "new", &value).ok());
  EXPECT_EQ("value", value);
}

TEST(RocksdbBlockCacheDeathTest, PressureDoesNotEscapeBackgroundThreads) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        CheckReadsAndBackgroundCompactionUnderPressure();
        std::exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}
#endif

TEST(RocksdbBlockCacheTest, ChargesOnlyFilterMetadata) {
  auto cache = CreateChargeTestCache(4096);
  int value = 0;
  rocksdb::Cache::Handle *filter = nullptr;
  rocksdb::Cache::Handle *data = nullptr;
  ASSERT_TRUE(
      cache->Insert("filter", &value, &kFilterChargeHelper, 64, &filter).ok());
  ASSERT_TRUE(
      cache->Insert("data", &value, &kDataChargeHelper, 64, &data).ok());
  EXPECT_EQ(64u + RocksdbBlockCache::kFilterMetadataCharge,
            cache->GetCharge(filter));
  EXPECT_EQ(64u, cache->GetCharge(data));
  EXPECT_EQ(128u + RocksdbBlockCache::kFilterMetadataCharge, cache->GetUsage());
  cache->Release(filter);
  cache->Release(data);
}

TEST(RocksdbBlockCacheTest, ErasedPinnedFilterKeepsChargeUntilFinalRelease) {
  auto cache = CreateChargeTestCache(4096);
  int value = 0;
  rocksdb::Cache::Handle *handle = nullptr;
  ASSERT_TRUE(
      cache->Insert("filter", &value, &kFilterChargeHelper, 64, &handle).ok());
  const size_t charge = cache->GetUsage();
  ASSERT_TRUE(cache->Ref(handle));
  cache->Erase("filter");
  EXPECT_EQ(charge, cache->GetUsage());
  EXPECT_FALSE(cache->Release(handle));
  EXPECT_EQ(charge, cache->GetUsage());
  EXPECT_TRUE(cache->Release(handle));
  EXPECT_EQ(0u, cache->GetUsage());
}

TEST(RocksdbBlockCacheTest, RejectsFilterWhenOnlyMetadataExceedsCapacity) {
  auto cache = CreateChargeTestCache(RocksdbBlockCache::kFilterMetadataCharge);
  int value = 0;
  rocksdb::Cache::Handle *handle = nullptr;
  const auto status =
      cache->Insert("filter", &value, &kFilterChargeHelper, 1, &handle);
  EXPECT_TRUE(status.IsMemoryLimit());
  EXPECT_EQ(nullptr, handle);
  EXPECT_EQ(0u, cache->GetUsage());
  ASSERT_TRUE(
      cache->Insert("data", &value, &kDataChargeHelper, 1, &handle).ok());
  cache->Release(handle);
}

TEST(RocksdbBlockCacheTest, StandaloneFilterCannotEscapeCharge) {
  auto cache = CreateChargeTestCache(4096);
  int value = 0;
  auto *handle =
      cache->CreateStandalone("filter", &value, &kFilterChargeHelper, 64, true);
  ASSERT_NE(nullptr, handle);
  EXPECT_EQ(64u + RocksdbBlockCache::kFilterMetadataCharge, cache->GetUsage());
  EXPECT_TRUE(cache->Release(handle));
  EXPECT_EQ(0u, cache->GetUsage());
  cache->SetCapacity(1);
  EXPECT_EQ(nullptr, cache->CreateStandalone("rejected", &value,
                                             &kFilterChargeHelper, 1, true));
  EXPECT_EQ(0u, cache->GetUsage());
}

TEST(RocksdbBlockCacheTest, RejectsOverflowAndPreservesMappedAllocatorHook) {
  auto allocator = std::make_shared<RocksdbCacheAllocator>();
  auto cache = std::make_shared<RocksdbBlockCache>(
      rocksdb::NewLRUCache(4096, 0, true, 0.5, allocator));
  EXPECT_EQ(allocator.get(), cache->memory_allocator());
  EXPECT_TRUE(cache->HasStrictCapacityLimit());
  int value = 0;
  rocksdb::Cache::Handle *handle = nullptr;
  EXPECT_TRUE(cache
                  ->Insert("overflow", &value, &kFilterChargeHelper,
                           std::numeric_limits<size_t>::max(), &handle)
                  .IsMemoryLimit());
  EXPECT_EQ(nullptr, handle);
  EXPECT_EQ(nullptr,
            cache->CreateStandalone("overflow", &value, &kFilterChargeHelper,
                                    std::numeric_limits<size_t>::max(), true));
  EXPECT_EQ(0u, cache->GetUsage());
}

TEST(RocksdbCacheAllocatorTest,
     StrictCacheChargesMappedSizeAndReleasesOnErase) {
  auto allocator = std::make_shared<RocksdbCacheAllocator>();
  auto release = [allocator](void *data) { allocator->Deallocate(data); };
  void *first = allocator->Allocate(4096);
  const size_t charge = allocator->UsableSize(first, 4096);
  std::unique_ptr<void, decltype(release)> entry(first, release);
  rocksdb::Cache::Handle *handle = nullptr;
  // Disable metadata charging here so the exact payload capacity is the
  // independent variable; production keeps RocksDB's metadata charging on.
  auto cache = rocksdb::NewLRUCache(charge, 0, true, 0.5, allocator, false,
                                    rocksdb::kDontChargeCacheMetadata);
  ASSERT_TRUE(cache
                  ->Insert("first", entry.get(), &kMappedCacheBlockHelper,
                           charge, &handle)
                  .ok());
  entry.release();
  EXPECT_EQ(charge, cache->GetUsage());
  void *second = allocator->Allocate(4096);
  std::unique_ptr<void, decltype(release)> rejected(second, release);
  rocksdb::Cache::Handle *second_handle = nullptr;
  const auto status =
      cache->Insert("second", rejected.get(), &kMappedCacheBlockHelper, charge,
                    &second_handle);
  if (status.ok()) {
    rejected.release();
    cache->Release(second_handle);
  }
  EXPECT_FALSE(status.ok());
  // On failed insertion ownership remains with the caller.
  rejected.reset();
  cache->Erase("first");
  EXPECT_EQ(charge, cache->GetUsage());  // still pinned
  cache->Release(handle);
  EXPECT_EQ(0u, cache->GetUsage());
}

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
TEST(RocksdbCacheAllocatorTest, ChargesNativePageRounding) {
  RocksdbCacheAllocator allocator;
#if defined(_WIN32)
  SYSTEM_INFO system{};
  ::GetSystemInfo(&system);
  const size_t page_size = system.dwPageSize;
#else
  const auto page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
#endif
  ASSERT_GT(page_size, 0u);
  for (size_t size : {page_size - 1, page_size, page_size + 1}) {
    void *data = allocator.Allocate(size);
    const size_t charge = allocator.UsableSize(data, size);
    EXPECT_EQ(0u, charge % page_size);
    EXPECT_GT(charge, size);
    EXPECT_LE(charge, size + page_size + alignof(std::max_align_t));
    allocator.Deallocate(data);
  }
}

TEST(RocksdbCacheAllocatorTest, DeallocationUnmapsBackingPages) {
  RocksdbCacheAllocator allocator;
  void *data = allocator.Allocate(65536);
#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION region{};
  ASSERT_NE(0u, ::VirtualQuery(data, &region, sizeof(region)));
  ASSERT_EQ(DWORD{MEM_COMMIT}, region.State);
  ASSERT_EQ(DWORD{MEM_PRIVATE}, region.Type);
  const void *base = region.AllocationBase;
  allocator.Deallocate(data);
  ASSERT_NE(0u, ::VirtualQuery(base, &region, sizeof(region)));
  EXPECT_EQ(DWORD{MEM_FREE}, region.State);
#else
  const auto page_size = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
  const uintptr_t base =
      reinterpret_cast<uintptr_t>(data) / page_size * page_size;
#if defined(__APPLE__)
  // Unlike Linux, macOS mincore can succeed for an unmapped address. Query
  // the mapping itself, not its residency bits.
  auto is_mapped = [base]() {
    vm_address_t address = base;
    vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::vm_region_64(
        ::mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) {
      ::mach_port_deallocate(::mach_task_self(), object);
    }
    EXPECT_TRUE(result == KERN_SUCCESS || result == KERN_INVALID_ADDRESS);
    return result == KERN_SUCCESS && address <= base && base - address < size;
  };
  ASSERT_TRUE(is_mapped());
  allocator.Deallocate(data);
  EXPECT_FALSE(is_mapped());
#else
  unsigned char resident = 0;
  ASSERT_EQ(0, ::mincore(reinterpret_cast<void *>(base), page_size, &resident));
  allocator.Deallocate(data);
  errno = 0;
  EXPECT_EQ(-1,
            ::mincore(reinterpret_cast<void *>(base), page_size, &resident));
  EXPECT_EQ(ENOMEM, errno);
#endif
#endif
}
#endif

}  // namespace
}  // namespace zvec

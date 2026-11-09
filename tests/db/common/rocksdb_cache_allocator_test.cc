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
#endif

}  // namespace
}  // namespace zvec

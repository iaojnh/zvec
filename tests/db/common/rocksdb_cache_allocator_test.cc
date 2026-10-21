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
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/cache.h>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace zvec {
namespace {

TEST(RocksdbCacheAllocatorTest, CoversCapacityAndAlignment) {
  RocksdbCacheAllocator allocator;
  for (size_t size : {0u, 1u, 4095u, 4096u, 65536u, 1048576u}) {
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

#if defined(__linux__)
TEST(RocksdbCacheAllocatorTest, DeallocationUnmapsBackingPages) {
  RocksdbCacheAllocator allocator;
  void *data = allocator.Allocate(65536);
  const auto page_size = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
  const uintptr_t base =
      reinterpret_cast<uintptr_t>(data) / page_size * page_size;
  unsigned char resident = 0;
  ASSERT_EQ(0, ::mincore(reinterpret_cast<void *>(base), page_size, &resident));
  allocator.Deallocate(data);
  errno = 0;
  EXPECT_EQ(-1,
            ::mincore(reinterpret_cast<void *>(base), page_size, &resident));
  EXPECT_EQ(ENOMEM, errno);
}
#endif

}  // namespace
}  // namespace zvec

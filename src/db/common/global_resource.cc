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
#include <exception>
#include <mutex>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/cache.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>
#include "db/common/rocksdb_cache_allocator.h"

namespace zvec {

uint64_t GlobalResource::calculate_rocksdb_memory_budget(
    uint64_t total_bytes) noexcept {
  return (total_bytes / 100) * detail::kRocksDbMemoryPercent +
         ((total_bytes % 100) * detail::kRocksDbMemoryPercent) / 100;
}

uint64_t GlobalResource::calculate_buffer_pool_memory_budget(
    uint64_t total_bytes) noexcept {
  return total_bytes - calculate_rocksdb_memory_budget(total_bytes);
}

uint64_t GlobalResource::calculate_rocksdb_write_buffer_budget(
    uint64_t rocksdb_bytes) noexcept {
  // Memtables charge pinned entries to the same cache used by SST readers.
  // Start flushing before they consume the space needed for compaction index
  // blocks. This is a limit inside the cache budget, not an extra reservation;
  // read-only workloads can still use the entire cache.
  return rocksdb_bytes > 1 ? rocksdb_bytes / 2 : rocksdb_bytes;
}

int GlobalResource::calculate_rocksdb_cache_shard_bits(
    uint64_t rocksdb_bytes) noexcept {
  // Strict capacity is enforced per shard, without borrowing from other
  // shards. RocksDB's default 512 KiB minimum is too small when pinned WBM
  // charges (256 KiB each) share the cache with compaction index blocks.
  constexpr uint64_t kMinimumShardBytes = 16ULL * 1024 * 1024;
  constexpr int kMaximumShardBits = 6;
  int bits = 0;
  while (bits < kMaximumShardBits && rocksdb_bytes / 2 >= kMinimumShardBytes) {
    rocksdb_bytes /= 2;
    ++bits;
  }
  return bits;
}

int GlobalResource::initialize() {
  // Resource handles are immutable after publication. Queries need an acquire
  // load once initialization has completed, not the shared initialization lock.
  if (initialized_.load(std::memory_order_acquire)) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(initialization_mutex_);
  if (query_thread_pool_ && optimize_thread_pool_) {
    return 0;
  }
  // Keep the initial budget snapshot and resource creation under one lock.
  // Another lazy caller must not mistake our newly created pool for a
  // preconfigured standalone pool.
  const auto &config = GlobalConfig::Instance();
  auto &memory_pool = zvec::ailego::MemoryLimitPool::get_instance();
  // Standalone/core users may configure the process-wide pool before the DB
  // layer lazily creates its thread pools. In that case the first published
  // pool budget remains authoritative; a lazy accessor must not try to resize
  // a live cache merely because GlobalConfig still contains its default.
  const bool preserve_existing_pool = memory_pool.initialized();
  const uint64_t effective_memory_limit = preserve_existing_pool
                                              ? memory_pool.capacity()
                                              : config.memory_limit_bytes();
  return initialize_locked(
      effective_memory_limit, config.query_thread_count(),
      config.query_thread_binding(), config.optimize_thread_count(),
      config.optimize_thread_binding(), preserve_existing_pool);
}

int GlobalResource::initialize(uint64_t memory_limit_bytes,
                               uint32_t query_thread_count,
                               bool query_thread_binding,
                               uint32_t optimize_thread_count,
                               bool optimize_thread_binding,
                               bool preserve_existing_pool) {
  std::lock_guard<std::mutex> lock(initialization_mutex_);
  return initialize_locked(memory_limit_bytes, query_thread_count,
                           query_thread_binding, optimize_thread_count,
                           optimize_thread_binding, preserve_existing_pool);
}

int GlobalResource::initialize_locked(uint64_t memory_limit_bytes,
                                      uint32_t query_thread_count,
                                      bool query_thread_binding,
                                      uint32_t optimize_thread_count,
                                      bool optimize_thread_binding,
                                      bool preserve_existing_pool) {
  try {
    auto &memory_pool = zvec::ailego::MemoryLimitPool::get_instance();
    const uint64_t rocksdb_memory_capacity =
        calculate_rocksdb_memory_budget(memory_limit_bytes);
    const uint64_t buffer_pool_capacity =
        preserve_existing_pool
            ? memory_pool.capacity()
            : calculate_buffer_pool_memory_budget(memory_limit_bytes);
    if (query_thread_pool_ && optimize_thread_pool_) {
      if (memory_limit_bytes_ == memory_limit_bytes &&
          query_thread_count_ == query_thread_count &&
          optimize_thread_count_ == optimize_thread_count &&
          query_thread_binding_ == query_thread_binding &&
          optimize_thread_binding_ == optimize_thread_binding) {
        return 0;
      }
      LOG_ERROR(
          "GlobalResource::initialize rejected configuration change after "
          "thread pools were created");
      return -1;
    }

    // An explicit GlobalConfig initialization must never publish a memory
    // limit different from an already configured lower-level pool. Reject it
    // before thread-pool publication; lazy initialize() above deliberately
    // passes the existing capacity and therefore remains
    // compatible with standalone/core callers.
    if (!preserve_existing_pool && memory_pool.initialized() &&
        memory_pool.capacity() != buffer_pool_capacity) {
      LOG_ERROR(
          "GlobalResource::initialize rejected memory limit change after "
          "MemoryLimitPool was configured: requested_buffer_capacity=%llu "
          "current_capacity=%llu",
          static_cast<unsigned long long>(buffer_pool_capacity),
          static_cast<unsigned long long>(memory_pool.capacity()));
      return -1;
    }

    auto query_thread_pool = std::make_unique<ailego::ThreadPool>(
        query_thread_count, query_thread_binding);
    auto optimize_thread_pool = std::make_unique<ailego::ThreadPool>(
        optimize_thread_count, optimize_thread_binding);
    std::shared_ptr<rocksdb::MemoryAllocator> rocksdb_cache_allocator;
#if defined(__linux__)
    // Linux malloc can retain flush/compaction buffers around live cache
    // payloads even after all memtables have been freed. Use RocksDB's public
    // allocator hook; keep other platforms' previously validated path.
    rocksdb_cache_allocator = std::make_shared<RocksdbCacheAllocator>();
#endif
    auto rocksdb_block_cache = rocksdb::NewLRUCache(
        static_cast<size_t>(rocksdb_memory_capacity),
        calculate_rocksdb_cache_shard_bits(rocksdb_memory_capacity),
        /*strict_capacity_limit=*/true, /*high_pri_pool_ratio=*/0.5,
        std::move(rocksdb_cache_allocator));
    if (!rocksdb_block_cache) {
      LOG_ERROR("Failed to create the process-wide RocksDB block cache");
      return -1;
    }
    auto rocksdb_write_buffer_manager =
        std::make_shared<rocksdb::WriteBufferManager>(
            static_cast<size_t>(
                calculate_rocksdb_write_buffer_budget(rocksdb_memory_capacity)),
            rocksdb_block_cache,
            /*allow_stall=*/true);
    if (memory_pool.init(static_cast<size_t>(buffer_pool_capacity)) != 0) {
      return -1;
    }

    memory_limit_bytes_ = memory_limit_bytes;
    rocksdb_memory_capacity_ = rocksdb_memory_capacity;
    query_thread_count_ = query_thread_count;
    optimize_thread_count_ = optimize_thread_count;
    query_thread_binding_ = query_thread_binding;
    optimize_thread_binding_ = optimize_thread_binding;
    this->query_thread_pool_ = std::move(query_thread_pool);
    this->optimize_thread_pool_ = std::move(optimize_thread_pool);
    rocksdb_block_cache_ = std::move(rocksdb_block_cache);
    rocksdb_write_buffer_manager_ = std::move(rocksdb_write_buffer_manager);
    initialized_.store(true, std::memory_order_release);
    LOG_INFO(
        "Managed memory initialized: total=%llu buffer_pool=%llu "
        "rocksdb=%llu rocksdb_percent=%u rocksdb_write_buffer=%llu "
        "rocksdb_cache_shard_bits=%d",
        static_cast<unsigned long long>(memory_limit_bytes_),
        static_cast<unsigned long long>(buffer_pool_capacity),
        static_cast<unsigned long long>(rocksdb_memory_capacity_),
        detail::kRocksDbMemoryPercent,
        static_cast<unsigned long long>(
            rocksdb_write_buffer_manager_->buffer_size()),
        calculate_rocksdb_cache_shard_bits(rocksdb_memory_capacity_));
    return 0;
  } catch (const std::exception &e) {
    LOG_ERROR("GlobalResource::initialize failed: %s", e.what());
  } catch (...) {
    LOG_ERROR("GlobalResource::initialize failed with an unknown exception");
  }
  return -1;
}

}  // namespace zvec

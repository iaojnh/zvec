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
#include <limits>
#include <mutex>
#include <arrow/util/thread_pool.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/cache.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>
#include "db/common/native_image_memory.h"
#include "db/common/native_worker_memory.h"
#include "db/common/rocksdb_block_cache.h"
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
  if (initialization_failed_) return -1;
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
  if (initialization_failed_) return -1;
  if (memory_limit_bytes > (std::numeric_limits<size_t>::max)()) {
    LOG_ERROR("Requested process memory budget exceeds the address space");
    return -1;
  }
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

    const uint32_t query_workers =
        ailego::ThreadPool::count_for_request(query_thread_count);
    const uint32_t optimize_workers =
        ailego::ThreadPool::count_for_request(optimize_thread_count);
    std::shared_ptr<rocksdb::MemoryAllocator> rocksdb_cache_allocator;
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    // malloc can retain freed SST blocks around live cache payloads after
    // flush/compaction. On macOS these pages can remain resident even after
    // being marked reusable. Use RocksDB's allocator hook to unmap released
    // payloads and include native page rounding in their cache charge.
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
    rocksdb_block_cache =
        std::make_shared<RocksdbBlockCache>(std::move(rocksdb_block_cache));
    auto rocksdb_write_buffer_manager =
        std::make_shared<rocksdb::WriteBufferManager>(
            static_cast<size_t>(
                calculate_rocksdb_write_buffer_budget(rocksdb_memory_capacity)),
            rocksdb_block_cache,
            /*allow_stall=*/true);
    // From here failures can leave a configured pool, native reservations or
    // an RSS reclaimer behind. Keep all later access failed rather than
    // mistaking that partial DB state for an independently configured pool.
    // Incompatible standalone configurations were rejected above, unchanged.
    initialization_failed_ = true;
    if (memory_pool.init(static_cast<size_t>(buffer_pool_capacity)) != 0) {
      return -1;
    }

    if (!preserve_existing_pool) {
      // Locate the image with an anchor in a product translation unit so the
      // caller's main executable is not charged. Shared-library code, read-only
      // data, and BSS outlive collection.close; reserve them before publishing.
      static const char image_anchor = 0;
      detail::ReserveNativeImage(&image_anchor);
      // Admit startup stacks before creating the two worker pools. The
      // dedicated Arrow pools use the same query worker count.
      detail::ReserveNativeWorkerStacks(
          query_workers, optimize_workers, optimize_thread_count,
          2 * static_cast<size_t>(query_workers));
    }
    auto query_thread_pool = std::make_unique<ailego::ThreadPool>(
        query_workers, query_thread_binding);
    auto optimize_thread_pool = std::make_unique<ailego::ThreadPool>(
        optimize_workers, optimize_thread_binding);
    if (!preserve_existing_pool) {
      // RSS already includes resident images, stacks, and caller-owned memory.
      // Use it as a separate admission/eviction signal, never another fixed
      // reservation that would count those bytes twice.
      memory_pool.set_process_rss_reclaimer(
          [cache = std::weak_ptr<rocksdb::Cache>(rocksdb_block_cache)] {
            if (auto owner = cache.lock()) owner->EraseUnRefEntries();
          });
      const size_t permanent_backing =
          detail::NativeImageReservation().bytes() +
          detail::NativeWorkerStackReservation().bytes();
      if (!memory_pool.configure_process_rss_limit(memory_limit_bytes,
                                                   permanent_backing)) {
        LOG_ERROR("Failed to sample process RSS for memory admission");
        return -1;
      }
    }

    // MakeEternal supports singleton destruction late at process exit. Its
    // lazy workers are admitted above before any plan can start using them.
    auto arrow_pool = arrow::internal::ThreadPool::MakeEternal(
        static_cast<int>(query_thread_pool->count()));
    if (!arrow_pool.ok()) {
      LOG_ERROR("Failed to create managed Arrow executor: %s",
                arrow_pool.status().ToString().c_str());
      return -1;
    }

    auto arrow_io_pool = arrow::internal::ThreadPool::MakeEternal(
        static_cast<int>(query_thread_pool->count()));
    if (!arrow_io_pool.ok()) {
      LOG_ERROR("Failed to create managed Arrow I/O executor: %s",
                arrow_io_pool.status().ToString().c_str());
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
    arrow_query_thread_pool_ = std::move(*arrow_pool);
    arrow_io_thread_pool_ = std::move(*arrow_io_pool);
    rocksdb_block_cache_ = std::move(rocksdb_block_cache);
    rocksdb_write_buffer_manager_ = std::move(rocksdb_write_buffer_manager);
    initialization_failed_ = false;
    initialized_.store(true, std::memory_order_release);
    LOG_INFO(
        "Managed memory initialized: total=%llu buffer_pool=%llu "
        "rocksdb=%llu rocksdb_percent=%u rocksdb_write_buffer=%llu "
        "rocksdb_cache_shard_bits=%d rss_limit=%llu rss_headroom=%llu",
        static_cast<unsigned long long>(memory_limit_bytes_),
        static_cast<unsigned long long>(buffer_pool_capacity),
        static_cast<unsigned long long>(rocksdb_memory_capacity_),
        detail::kRocksDbMemoryPercent,
        static_cast<unsigned long long>(
            rocksdb_write_buffer_manager_->buffer_size()),
        calculate_rocksdb_cache_shard_bits(rocksdb_memory_capacity_),
        static_cast<unsigned long long>(memory_pool.process_rss_limit()),
        static_cast<unsigned long long>(ailego::ProcessRssBudget::headroom(
            memory_pool.process_rss_limit())));
    return 0;
  } catch (const std::exception &e) {
    LOG_ERROR("GlobalResource::initialize failed: %s", e.what());
  } catch (...) {
    LOG_ERROR("GlobalResource::initialize failed with an unknown exception");
  }
  return -1;
}

}  // namespace zvec

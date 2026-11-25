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
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>
#include "db/common/native_image_memory.h"
#include "db/common/native_worker_memory.h"

namespace zvec {

int GlobalResource::initialize() {
  // Resource handles are immutable after publication. Queries need an acquire
  // load once initialization has completed, not the shared initialization lock.
  if (initialized_.load(std::memory_order_acquire)) {
    return 0;
  }

  // An explicit configuration owns startup once it starts. Keep the
  // configuration lock through lazy startup so that a racing explicit call
  // cannot publish settings for resources being created from the defaults.
  auto &config = GlobalConfig::Instance();
  std::unique_lock config_lock(config.initialization_mutex_);
  if (config.initializing_thread_ == std::this_thread::get_id()) return -1;
  config.initialization_cv_.wait(config_lock, [&config] {
    return config.initialization_state_ !=
           GlobalConfig::InitializationState::kInitializing;
  });
  if (initialized_.load(std::memory_order_acquire)) return 0;
  if (config.initialization_state_ ==
      GlobalConfig::InitializationState::kFailed)
    return -1;

  std::unique_lock lock(initialization_mutex_);
  if (initialization_failed_) return -1;
  // Keep the initial budget snapshot and resource creation under one lock.
  // Another lazy caller must not mistake our newly created pool for a
  // preconfigured standalone pool.
  const auto snapshot = config.config_snapshot();
  auto &memory_pool = zvec::ailego::MemoryLimitPool::get_instance();
  // Standalone/core users may configure the process-wide pool before the DB
  // layer lazily creates its thread pools. In that case the first published
  // pool budget remains authoritative; a lazy accessor must not try to resize
  // a live cache merely because GlobalConfig still contains its default.
  const bool preserve_existing_pool = memory_pool.initialized();
  const uint64_t effective_memory_limit = preserve_existing_pool
                                              ? memory_pool.capacity()
                                              : snapshot->memory_limit_bytes;
  const int result = initialize_locked(
      effective_memory_limit, snapshot->query_thread_count,
      snapshot->query_thread_binding, snapshot->optimize_thread_count,
      snapshot->optimize_thread_binding, preserve_existing_pool);
  if (result == 0) initialized_.store(true, std::memory_order_release);
  lock.unlock();
  config_lock.unlock();
  if (result == 0) log_initialized();
  return result;
}

int GlobalResource::prepare(uint64_t memory_limit_bytes,
                            uint32_t query_thread_count,
                            bool query_thread_binding,
                            uint32_t optimize_thread_count,
                            bool optimize_thread_binding) {
  std::lock_guard<std::mutex> lock(initialization_mutex_);
  return initialize_locked(memory_limit_bytes, query_thread_count,
                           query_thread_binding, optimize_thread_count,
                           optimize_thread_binding,
                           /*preserve_existing_pool=*/false);
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
    if (!preserve_existing_pool && !memory_policy_ready_ &&
        memory_pool.process_rss_limit() != 0) {
      // A lower-level RSS controller already owns live projections. Do not
      // reset it or double-charge workers admitted through that controller.
      LOG_ERROR("Cannot replace an independently configured RSS policy");
      return -1;
    }
    if (query_thread_pool_ && optimize_thread_pool_) {
      if (memory_limit_bytes_ == memory_limit_bytes &&
          query_thread_count_ == query_thread_count &&
          optimize_thread_count_ == optimize_thread_count &&
          query_thread_binding_ == query_thread_binding &&
          optimize_thread_binding_ == optimize_thread_binding) {
        if (!preserve_existing_pool && !memory_policy_ready_) {
          // These pools are already published. If policy admission fails,
          // explicit startup must fail while existing callers retain their
          // valid standalone resource handles.
          reserve_native_memory(query_thread_pool_->count(),
                                optimize_thread_pool_->count());
          if (configure_memory_policy(memory_limit_bytes) != 0) return -1;
        }
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
        memory_pool.capacity() != memory_limit_bytes) {
      LOG_ERROR(
          "GlobalResource::initialize rejected memory limit change after "
          "MemoryLimitPool was configured: requested_capacity=%llu "
          "current_capacity=%llu",
          static_cast<unsigned long long>(memory_limit_bytes),
          static_cast<unsigned long long>(memory_pool.capacity()));
      return -1;
    }

    const uint32_t query_workers =
        ailego::ThreadPool::count_for_request(query_thread_count);
    const uint32_t optimize_workers =
        ailego::ThreadPool::count_for_request(optimize_thread_count);
    // From here failures can leave a configured pool or native reservations
    // behind. Keep all later access failed rather than mistaking that partial
    // DB state for an independently configured pool. Incompatible standalone
    // configurations were rejected above, unchanged.
    initialization_failed_ = true;
    if (memory_pool.init(static_cast<size_t>(memory_limit_bytes)) != 0) {
      return -1;
    }

    if (!preserve_existing_pool) {
      reserve_native_memory(query_workers, optimize_workers);
    }
    auto query_thread_pool = std::make_unique<ailego::ThreadPool>(
        query_workers, query_thread_binding);
    auto optimize_thread_pool = std::make_unique<ailego::ThreadPool>(
        optimize_workers, optimize_thread_binding);
    if (!preserve_existing_pool) {
      if (configure_memory_policy(memory_limit_bytes) != 0) return -1;
    }

    memory_limit_bytes_ = memory_limit_bytes;
    query_thread_count_ = query_thread_count;
    optimize_thread_count_ = optimize_thread_count;
    query_thread_binding_ = query_thread_binding;
    optimize_thread_binding_ = optimize_thread_binding;
    this->query_thread_pool_ = std::move(query_thread_pool);
    this->optimize_thread_pool_ = std::move(optimize_thread_pool);
    initialization_failed_ = false;
    return 0;
  } catch (const std::exception &e) {
    LOG_ERROR("GlobalResource::initialize failed: %s", e.what());
  } catch (...) {
    LOG_ERROR("GlobalResource::initialize failed with an unknown exception");
  }
  return -1;
}

void GlobalResource::reserve_native_memory(uint32_t query_workers,
                                           uint32_t optimize_workers) {
  // Anchor in the product image rather than an inline caller's executable.
  static const char image_anchor = 0;
  detail::ReserveNativeImage(&image_anchor);
  detail::ReserveNativeWorkerStacks(query_workers, optimize_workers);
}

int GlobalResource::configure_memory_policy(uint64_t memory_limit_bytes) {
  // RSS includes native backing and caller memory. Exclude the permanent
  // lifetime allowance from projections so native bytes are not counted twice.
  const size_t permanent_backing =
      detail::NativeImageReservation().bytes() +
      detail::NativeWorkerStackReservation().bytes();
  if (!ailego::MemoryLimitPool::get_instance().configure_process_rss_limit(
          memory_limit_bytes, permanent_backing)) {
    LOG_ERROR("Failed to sample process RSS for memory admission");
    return -1;
  }
  memory_policy_ready_ = true;
  return 0;
}

void GlobalResource::log_initialized() const noexcept try {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto rss_limit = pool.process_rss_limit();
  LOG_INFO(
      "Managed memory initialized: total=%llu buffer_pool=%llu "
      "rss_limit=%llu rss_headroom=%llu",
      static_cast<unsigned long long>(memory_limit_bytes_),
      static_cast<unsigned long long>(pool.capacity()),
      static_cast<unsigned long long>(rss_limit),
      static_cast<unsigned long long>(
          ailego::ProcessRssBudget::headroom(rss_limit)));
} catch (...) {
  // Logging after publication cannot change the startup result.
}

}  // namespace zvec

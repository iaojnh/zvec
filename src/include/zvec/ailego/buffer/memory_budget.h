// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <zvec/export.h>

namespace zvec {
namespace ailego {

//! Process-wide memory partitioning and accounting.
//!
//! Shared-cache and RocksDB capacities are enforced by their native cache
//! implementations. The shared-cache partition covers VecBufferPool pages
//! plus external consumers such as the DiskANN node cache. Query and
//! resident-metadata allocations use try_charge() and release() directly.
//! SafetyReserve is deliberately unavailable to allocators: its capacity is
//! subtracted from the other partitions.
class ZVEC_AILEGO_API MemoryBudgetManager {
 public:
  enum class Category : uint8_t {
    BufferCache = 0,
    RocksDbBlockCache,
    QueryWorking,
    ResidentMetadata,
    SafetyReserve,
    Count
  };

  struct Config {
    uint64_t total_bytes{0};
    // False keeps standalone/core-only users backward compatible. DB
    // initialization sets this when at least one explicit partition is used.
    bool enforce_accounting{false};
    uint64_t buffer_cache_bytes{0};
    uint64_t rocksdb_block_cache_bytes{0};
    uint64_t query_working_bytes{0};
    uint64_t resident_metadata_bytes{0};
    uint64_t safety_reserve_bytes{0};
  };

  struct Snapshot {
    Config config{};
    uint64_t query_working_used{0};
    uint64_t resident_metadata_used{0};
    uint64_t query_working_rejections{0};
    uint64_t resident_metadata_rejections{0};
  };

  //! Retrieve the process-wide budget manager.
  //!
  //! zvec component libraries are also shipped as static archives and may be
  //! embedded in more than one DSO in the same process (for example, a host
  //! library plus the standalone DiskANN library).  A conventional function
  //! local object would then be constructed once per DSO.  Keep only a
  //! constant-initialized atomic pointer in COMDAT storage instead: ELF and
  //! Mach-O loaders coalesce that weak storage across images, so every copy of
  //! the accessor implementation publishes and observes the same heap object.
  //! The object is intentionally process-lifetime, avoiding both
  //! guard-variable splitting and cross-DSO destruction-order hazards.
  static MemoryBudgetManager &get_instance();

  MemoryBudgetManager(const MemoryBudgetManager &) = delete;
  MemoryBudgetManager &operator=(const MemoryBudgetManager &) = delete;

  //! Configure partitions during process initialization. Returns false if
  //! their sum does not exactly match total_bytes.
  bool configure(const Config &config) {
    const uint64_t parts[] = {
        config.buffer_cache_bytes, config.rocksdb_block_cache_bytes,
        config.query_working_bytes, config.resident_metadata_bytes,
        config.safety_reserve_bytes};
    uint64_t sum = 0;
    for (uint64_t part : parts) {
      if (part > config.total_bytes - sum) {
        return false;
      }
      sum += part;
    }
    if (sum != config.total_bytes) {
      return false;
    }

    config_ = config;
    for (auto &used : used_bytes_) {
      used.store(0, std::memory_order_relaxed);
    }
    for (auto &rejections : rejected_charges_) {
      rejections.store(0, std::memory_order_relaxed);
    }
    configured_.store(true, std::memory_order_release);
    return true;
  }

  //! Reserve bytes from an explicitly-accounted category. Accounting is
  //! bypassed only for backward-compatible standalone/core-only users. Once
  //! DB initialization enables explicit accounting, a zero category capacity
  //! rejects every non-empty allocation from that category.
  bool try_charge(Category category, uint64_t bytes) {
    if (bytes == 0 || !configured_.load(std::memory_order_acquire) ||
        !config_.enforce_accounting) {
      return true;
    }
    if (category == Category::Count) {
      return false;
    }
    const uint64_t limit = capacity(category);
    if (limit == 0 || category == Category::SafetyReserve ||
        category == Category::BufferCache ||
        category == Category::RocksDbBlockCache) {
      rejected_charges_[index(category)].fetch_add(1,
                                                   std::memory_order_relaxed);
      return false;
    }

    auto &used = used_bytes_[index(category)];
    uint64_t current = used.load(std::memory_order_relaxed);
    while (current <= limit && bytes <= limit - current) {
      if (used.compare_exchange_weak(current, current + bytes,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
        return true;
      }
    }
    rejected_charges_[index(category)].fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  void release(Category category, uint64_t bytes) {
    if (bytes == 0 || !configured_.load(std::memory_order_acquire) ||
        !config_.enforce_accounting) {
      return;
    }
    auto &used = used_bytes_[index(category)];
    const uint64_t previous = used.fetch_sub(bytes, std::memory_order_relaxed);
    (void)previous;
    assert(previous >= bytes);
  }

  uint64_t capacity(Category category) const {
    switch (category) {
      case Category::BufferCache:
        return config_.buffer_cache_bytes;
      case Category::RocksDbBlockCache:
        return config_.rocksdb_block_cache_bytes;
      case Category::QueryWorking:
        return config_.query_working_bytes;
      case Category::ResidentMetadata:
        return config_.resident_metadata_bytes;
      case Category::SafetyReserve:
        return config_.safety_reserve_bytes;
      case Category::Count:
        return 0;
    }
    return 0;
  }

  uint64_t used(Category category) const {
    return used_bytes_[index(category)].load(std::memory_order_relaxed);
  }

  Snapshot snapshot() const {
    Snapshot result;
    result.config = config_;
    result.query_working_used = used(Category::QueryWorking);
    result.resident_metadata_used = used(Category::ResidentMetadata);
    result.query_working_rejections =
        rejected_charges_[index(Category::QueryWorking)].load(
            std::memory_order_relaxed);
    result.resident_metadata_rejections =
        rejected_charges_[index(Category::ResidentMetadata)].load(
            std::memory_order_relaxed);
    return result;
  }

  bool configured() const {
    return configured_.load(std::memory_order_acquire);
  }

 private:
  MemoryBudgetManager() = default;

  //! Weak COMDAT storage shared by every out-of-line get_instance() copy.
  //! Keeping this inline preserves the existing exported accessor ABI while
  //! ensuring copies pulled from a static archive still coordinate through
  //! one process-wide pointer.
  static std::atomic<MemoryBudgetManager *> &instance_slot() {
    static std::atomic<MemoryBudgetManager *> instance{nullptr};
    return instance;
  }

  static constexpr size_t index(Category category) {
    return static_cast<size_t>(category);
  }

  Config config_{};
  std::array<std::atomic<uint64_t>, static_cast<size_t>(Category::Count)>
      used_bytes_{};
  std::array<std::atomic<uint64_t>, static_cast<size_t>(Category::Count)>
      rejected_charges_{};
  std::atomic<bool> configured_{false};
};

}  // namespace ailego
}  // namespace zvec

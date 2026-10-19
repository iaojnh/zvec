// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cassert>
#include <limits>
#include <mutex>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::detail {

// Shared planning allowance for third-party metadata using ordinary heap
// containers. Retain the concurrent high water across close/reopen because
// object destruction does not guarantee that malloc returns backing pages.
// This is not exact allocator accounting or a process RSS limit.
class RetainedMetadataBudget {
 public:
  class Lease {
   public:
    explicit Lease(RetainedMetadataBudget &owner) : owner_(owner) {}
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    ~Lease() {
      retain_at_most(0);
    }

    void acquire(size_t bytes) {
      std::lock_guard<std::mutex> lock(owner_.mutex_);
      auto &pool = ailego::MemoryLimitPool::get_instance();
      if (!pool.capacity()) return;
      if (bytes > std::numeric_limits<size_t>::max() - owner_.active_)
        throw std::bad_alloc();
      const size_t target = owner_.active_ + bytes;
      // One cap for RocksDB instances/CFs and Parquet footers together.
      // A rejected open must leave existing readers room to decode/query.
      if (target > pool.capacity() / 2) throw std::bad_alloc();
      if (target > owner_.reserved_) {
        if (!pool.try_charge_metadata(target - owner_.reserved_))
          throw std::bad_alloc();
        owner_.reserved_ = target;
      }
      owner_.active_ = target;
      bytes_ += bytes;
    }

    size_t bytes() const {
      return bytes_;
    }
    void retain_at_most(size_t bytes) {
      std::lock_guard<std::mutex> lock(owner_.mutex_);
      if (bytes >= bytes_) return;
      assert(owner_.active_ >= bytes_ - bytes);
      owner_.active_ -= bytes_ - bytes;
      bytes_ = bytes;
    }

   private:
    RetainedMetadataBudget &owner_;
    size_t bytes_{0};
  };

  ~RetainedMetadataBudget() {
    assert(active_ == 0);
    if (reserved_)
      ailego::MemoryLimitPool::get_instance().release_metadata(reserved_);
  }
  size_t reserved() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reserved_;
  }
  static RetainedMetadataBudget &instance() {
    static auto *budget = new RetainedMetadataBudget;
    return *budget;
  }

 private:
  friend class MemoryBudgetTestPeer;
  mutable std::mutex mutex_;
  size_t active_{0};
  size_t reserved_{0};
};

}  // namespace zvec::detail

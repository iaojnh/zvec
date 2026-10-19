// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cassert>
#include <limits>
#include <mutex>
#include <zvec/ailego/buffer/memory_reservation.h>

namespace zvec::detail {

// Planning allowance for results and write staging that use ordinary heap
// objects (including Doc copies, WAL serialization and FTS workspaces).
// This is not an allocator or an RSS measurement. Retain the concurrent high
// water mark: destroying an object need not return its backing pages to the OS.
// The production owner lives for the process lifetime; short-lived owners are
// useful only in isolated tests. Caller-retained results are outside this
// scope.
class ResultMemoryAllowance {
 public:
  class Lease {
   public:
    explicit Lease(ResultMemoryAllowance *owner,
                   bool serialize_materialization = false)
        : owner_(owner),
          serialize_materialization_(serialize_materialization) {}
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    ~Lease() {
      if (bytes_ != 0) owner_->release_active(bytes_);
    }

    bool enabled() const {
      return owner_ != nullptr;
    }
    void add(size_t bytes) {
      if (!owner_ || bytes == 0) return;
      // Take the materialization slot before charging the first chunk, not
      // after a partially built result has exhausted the shared allowance.
      // Keep it through every chunk of this API call. Otherwise concurrent
      // calls can all retain partial results and wait for one another to end.
      // Callers must hold neither the GIL nor database locks while waiting.
      if (serialize_materialization_ && !materialization_.owns_lock()) {
        materialization_ =
            std::unique_lock<std::mutex>(owner_->materialization_mutex_);
      }
      bytes_ += owner_->acquire(bytes);
    }

   private:
    ResultMemoryAllowance *owner_;
    size_t bytes_{0};
    bool serialize_materialization_;
    // Destruction releases active bytes above before allowing the next call.
    std::unique_lock<std::mutex> materialization_;
  };

  size_t active_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
  }
  size_t reserved_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reservation_.bytes();
  }

 private:
  friend class MemoryBudgetTestPeer;
  size_t acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t capacity = ailego::MemoryLimitPool::get_instance().capacity();
    if (!capacity) return 0;
    // One result-equivalent for active objects, another for allocation growth
    // and previously freed backing. This is a conservative planning policy,
    // not a claim that an arbitrary system allocator has bounded fragmentation.
    constexpr size_t kBackingMultiplier = 2;
    const auto max_active =
        std::numeric_limits<size_t>::max() / kBackingMultiplier;
    if (bytes > max_active || active_ > max_active - bytes) {
      throw std::bad_alloc();
    }
    const size_t target = (active_ + bytes) * kBackingMultiplier;
    // A failed oversized request must not leave a permanent allowance that
    // consumes the whole cache. Keep at least three quarters for native
    // storage, metadata and decoding; callers can split large result batches.
    if (target > capacity / 4) throw std::bad_alloc();
    if (target > reservation_.bytes()) {
      // Admission may evict native cache pages. Bindings must release the GIL
      // before calling add(); no Python code executes while this lock is held.
      reservation_.add(target - reservation_.bytes());
    }
    active_ += bytes;
    return bytes;
  }

  void release_active(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(active_ >= bytes);
    active_ -= bytes;
    // Do not refund reservation_: later calls reuse this high-water allowance.
  }

  mutable std::mutex mutex_;
  std::mutex materialization_mutex_;
  ailego::MemoryReservation reservation_;
  size_t active_{0};
};

// All forward-result paths share one high water and one active limit. Separate
// per-API allowances could each consume a quarter of the pool and leave no
// room for decoding after alternating Query and Fetch calls.
inline ResultMemoryAllowance &SharedResultMemoryAllowance() {
  (void)ailego::MemoryLimitPool::get_instance();
  static ResultMemoryAllowance allowance;
  return allowance;
}

}  // namespace zvec::detail

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
// useful only in isolated tests. Returned payloads can retain a shared Lease;
// release the materialization slot before transferring it to another thread.
class ResultMemoryAllowance {
 public:
  class Lease {
   public:
    explicit Lease(ResultMemoryAllowance *owner,
                   bool serialize_materialization = false,
                   size_t backing_quantum = 1)
        : owner_(owner),
          serialize_materialization_(serialize_materialization),
          backing_quantum_(backing_quantum) {}
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    ~Lease() {
      if (bytes_ != 0) owner_->release_active(bytes_);
    }

    bool enabled() const {
      return owner_ != nullptr;
    }
    void end_materialization() {
      // A result may be destroyed on a different caller thread. Never let its
      // lifetime extend ownership of the producing thread's mutex.
      if (materialization_.owns_lock()) materialization_.unlock();
      serialize_materialization_ = false;
    }
    void begin_materialization() {
      if (!owner_) return;
      // Enter before fetching the first native chunk, not just before Python
      // conversion. Otherwise waiting calls retain already materialized Docs
      // and grow ordinary-heap backing despite serializing their conversion.
      // Keep the slot through every chunk of this API call.
      // Callers must hold neither the GIL nor database locks while waiting.
      if (serialize_materialization_ && !materialization_.owns_lock()) {
        materialization_ =
            std::unique_lock<std::mutex>(owner_->materialization_mutex_);
      }
    }
    void add(size_t bytes) {
      add(bytes, bytes);
    }

    // A retained allowance can cover some of the new allocation's lifetime,
    // while the entire allocation still consumes RSS room when materialized.
    void add(size_t bytes, size_t materialization) {
      if (!owner_ || (!bytes && !materialization)) return;
      begin_materialization();
      bytes_ += owner_->acquire(bytes, backing_quantum_, materialization);
    }

    // Transfer admission to an owner allocated only AFTER that admission.
    // Both leases refer to the same allowance and hold no materialization lock.
    void absorb(Lease &other) noexcept {
      assert(owner_ == other.owner_);
      assert(!materialization_.owns_lock() && !other.materialization_.owns_lock());
      bytes_ += std::exchange(other.bytes_, 0);
    }

    size_t bytes() const noexcept {
      return bytes_;
    }

    void retain_at_most(size_t bytes) {
      if (bytes >= bytes_) return;
      owner_->release_active(bytes_ - bytes);
      bytes_ = bytes;
    }

   private:
    ResultMemoryAllowance *owner_;
    size_t bytes_{0};
    bool serialize_materialization_;
    size_t backing_quantum_;
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
  size_t acquire(size_t bytes, size_t backing_quantum,
                 size_t materialization) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    const size_t capacity = pool.capacity();
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
    if (backing_quantum == 0) throw std::bad_alloc();
    const size_t high_water = active_ + bytes;
    // Result allocators may retain backing in arenas rather than individual
    // object-sized pieces. Round each backing equivalent before admission;
    // otherwise native cache pages can consume an uncharged partial arena.
    const size_t remainder = high_water % backing_quantum;
    const size_t padding = remainder ? backing_quantum - remainder : 0;
    if (padding > max_active - high_water) throw std::bad_alloc();
    const size_t target = (high_water + padding) * kBackingMultiplier;
    // A failed oversized request must not leave a permanent allowance that
    // consumes the whole cache. Keep at least three quarters for native
    // storage, metadata and decoding; callers can split large result batches.
    if (target > capacity / 4) throw std::bad_alloc();
    if (target > reservation_.bytes()) {
      // Admission may evict native cache pages. Bindings must release the GIL
      // before calling add(); no Python code executes while this lock is held.
      reservation_.add(target - reservation_.bytes(), materialization);
    } else if (!pool.try_materialize(materialization)) {
      throw std::bad_alloc();
    }
    active_ += bytes;
    return bytes;
  }

  void release_active(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(active_ >= bytes);
    active_ -= bytes;
    // Normally keep the high water for reuse. Under process pressure an
    // entirely idle result allowance must yield to mandatory I/O and decode.
    // Retained heap pages still count in the process's measured RSS.
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (!active_ && pool.process_rss_limit() &&
        pool.under_process_rss_pressure())
      reservation_.retain_at_most(0);
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

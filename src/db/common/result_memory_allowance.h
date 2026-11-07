// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cassert>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
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
  // Scoped copies provide their physical size independently of planning quota.
  // Retain that payload as RSS backing across samples and page-out/refault.
  // Work without a scoped copy keeps the full conservative projection.
  class BackingReservation {
   public:
    ~BackingReservation() {
      retain_at_most(0);
    }
    size_t bytes() const {
      return bytes_;
    }
    void acquire(size_t target, size_t rss_target, size_t materialization,
                 bool pending_copy) {
      auto &pool = ailego::MemoryLimitPool::get_instance();
      const size_t growth = target > bytes_ ? target - bytes_ : 0;
      const size_t rss_growth =
          rss_target > rss_bytes_ ? rss_target - rss_bytes_ : 0;
      if (growth && !pool.try_charge_workspace_plan(growth))
        throw std::bad_alloc();
      if (!pool.try_charge_process_backing(rss_growth, materialization,
                                           pending_copy)) {
        if (growth) pool.release_workspace_plan(growth);
        throw std::bad_alloc();
      }
      bytes_ += growth;
      rss_bytes_ += rss_growth;
    }
    void retain_at_most(size_t bytes, size_t rss = 0) {
      bytes = std::min(bytes, bytes_);
      rss = std::min(rss, rss_bytes_);
      auto &pool = ailego::MemoryLimitPool::get_instance();
      if (rss_bytes_ > rss) pool.release_process_backing(rss_bytes_ - rss);
      if (bytes_ > bytes) pool.release_workspace_plan(bytes_ - bytes);
      bytes_ = bytes;
      rss_bytes_ = rss;
    }

   private:
    size_t bytes_{0};
    size_t rss_bytes_{0};
  };

 public:
  // Keep measured headroom while the producer is between admission and copy.
  // Cancel unused admission. After starting, a failed partial copy can still
  // leave resident allocator backing, so keep its projection until a sample.
  class PendingCopy {
   public:
    PendingCopy() = default;
    PendingCopy(const PendingCopy &) = delete;
    PendingCopy &operator=(const PendingCopy &) = delete;
    ~PendingCopy() {
      finish();
    }

    void start() noexcept {
      started_ = true;
    }
    void finish() {
      if (const size_t bytes = std::exchange(bytes_, 0))
        ailego::MemoryLimitPool::get_instance().finish_materialization(
            bytes, started_);
      started_ = false;
    }

   private:
    friend class ResultMemoryAllowance;

    size_t bytes_{0};
    bool started_{false};
  };

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
      if (bytes_ || copy_bytes_)
        owner_->release_active(bytes_, native_bytes_, copy_bytes_);
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
      if (owner_->acquire(bytes, backing_quantum_, materialization, nullptr)) {
        bytes_ += bytes;
        native_bytes_ += bytes;
      }
    }

    void add_pending(size_t bytes, PendingCopy &pending) {
      add_pending(bytes, bytes, pending);
    }

    void add_pending(size_t bytes, size_t materialization,
                     PendingCopy &pending) {
      if (!owner_ || (!bytes && !materialization)) return;
      begin_materialization();
      if (owner_->acquire(bytes, backing_quantum_, materialization, &pending)) {
        bytes_ += bytes;
        copy_bytes_ += materialization;
      }
    }

    // Transfer admission to an owner allocated only AFTER that admission.
    // Both leases refer to the same allowance and hold no materialization lock.
    void absorb(Lease &other) noexcept {
      assert(owner_ == other.owner_);
      assert(!materialization_.owns_lock() &&
             !other.materialization_.owns_lock());
      bytes_ += std::exchange(other.bytes_, 0);
      native_bytes_ += std::exchange(other.native_bytes_, 0);
      copy_bytes_ += std::exchange(other.copy_bytes_, 0);
    }

    size_t bytes() const noexcept {
      return bytes_;
    }

    void retain_at_most(size_t bytes) {
      if (bytes >= bytes_ && (bytes || !copy_bytes_)) return;
      const size_t native = std::min(native_bytes_, bytes);
      // A partial logical trim does not prove which physical copies died.
      const size_t copy = bytes ? copy_bytes_ : 0;
      owner_->release_active(bytes_ - bytes, native_bytes_ - native,
                             copy_bytes_ - copy);
      bytes_ = bytes;
      native_bytes_ = native;
      copy_bytes_ = copy;
    }

   private:
    ResultMemoryAllowance *owner_;
    size_t bytes_{0};
    size_t native_bytes_{0};
    size_t copy_bytes_{0};
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

  void reclaim_idle() {
    // Background eviction must not wait for an admission that itself needs
    // cache reclamation to make progress.
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock) return;
    // Reclaim unused high water even while a small result is alive. Otherwise
    // that result can pin a previous large request's entire planning allowance
    // and prevent mandatory index metadata from being admitted.
    const size_t target = rounded(active_, backing_quantum_) * 2;
    const size_t native =
        std::min(target, rounded(native_active_, backing_quantum_) * 2);
    const size_t rss =
        native +
        std::min(rounded(copy_active_, backing_quantum_), target - native);
    reservation_.retain_at_most(target, rss);
  }

 private:
  friend class MemoryBudgetTestPeer;
  static size_t rounded(size_t bytes, size_t quantum) {
    const size_t remainder = bytes % quantum;
    return bytes + (remainder ? quantum - remainder : 0);
  }

  bool acquire(size_t bytes, size_t backing_quantum, size_t materialization,
               PendingCopy *pending) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    const size_t capacity = pool.capacity();
    if (!capacity) return false;
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
    const size_t native_high_water = native_active_ + (pending ? 0 : bytes);
    const size_t native_remainder = native_high_water % backing_quantum;
    const size_t native_padding =
        native_remainder ? backing_quantum - native_remainder : 0;
    if (native_padding > max_active - native_high_water) throw std::bad_alloc();
    const size_t copy_growth = pending ? materialization : 0;
    if (copy_growth > max_active || copy_active_ > max_active - copy_growth)
      throw std::bad_alloc();
    const size_t copy_high_water = copy_active_ + copy_growth;
    const size_t copy_remainder = copy_high_water % backing_quantum;
    const size_t copy_padding =
        copy_remainder ? backing_quantum - copy_remainder : 0;
    if (copy_padding > max_active - copy_high_water) throw std::bad_alloc();
    const size_t native_target = (native_high_water + native_padding) * 2;
    const size_t rss_target =
        native_target +
        std::min(copy_high_water + copy_padding, target - native_target);
    if (pending &&
        materialization > std::numeric_limits<size_t>::max() - pending->bytes_)
      throw std::bad_alloc();
    reservation_.acquire(target, rss_target, materialization,
                         pending != nullptr);
    if (pending) pending->bytes_ += materialization;
    active_ += bytes;
    if (!pending) native_active_ += bytes;
    copy_active_ += copy_growth;
    backing_quantum_ = std::max(backing_quantum_, backing_quantum);
    return true;
  }

  void release_active(size_t bytes, size_t native, size_t copy) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(active_ >= bytes);
    active_ -= bytes;
    assert(native_active_ >= native);
    native_active_ -= native;
    assert(copy_active_ >= copy);
    copy_active_ -= copy;
    // Normally keep the high water for reuse. Under process pressure an
    // entirely idle result allowance must yield to mandatory I/O and decode.
    // Retained heap pages still count in the process's measured RSS.
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (!active_ && !copy_active_ && pool.process_rss_limit() &&
        pool.under_process_rss_pressure())
      reservation_.retain_at_most(0);
  }

  mutable std::mutex mutex_;
  std::mutex materialization_mutex_;
  BackingReservation reservation_;
  size_t active_{0};
  size_t native_active_{0};
  size_t copy_active_{0};
  size_t backing_quantum_{1};
};

// All forward-result paths share one high water and one active limit. Separate
// per-API allowances could each consume a quarter of the pool and leave no
// room for decoding after alternating Query and Fetch calls.
inline const std::shared_ptr<ResultMemoryAllowance> &
SharedResultMemoryAllowanceOwner() {
  (void)ailego::MemoryLimitPool::get_instance();
  static const auto owner = std::make_shared<ResultMemoryAllowance>();
  return owner;
}

inline ResultMemoryAllowance &SharedResultMemoryAllowance() {
  return *SharedResultMemoryAllowanceOwner();
}

}  // namespace zvec::detail

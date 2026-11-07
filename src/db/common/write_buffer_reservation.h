// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/db/status.h>

namespace zvec {

namespace detail {

// Ordinary heap backing can outlive a writer's live objects. Retain one
// process-wide concurrent high-water allowance after actual writes, not one
// permanent reservation per Collection/flush. Later writers reuse that
// allowance before requesting more. This remains a planning estimate, not
// allocator-level ownership or a hard RSS guarantee.
class WriteBufferMemoryAllowance {
  // The flush plan includes Arrow buffers with their own allocator charges.
  // Only this allowance's ordinary-heap backing belongs in the RSS ledger.
  class BackingReservation {
   public:
    ~BackingReservation() {
      retain_at_most(0);
    }
    size_t bytes() const {
      return bytes_;
    }
    void acquire(size_t target, size_t rss_target, size_t materialization) {
      auto &pool = ailego::MemoryLimitPool::get_instance();
      const size_t growth = target > bytes_ ? target - bytes_ : 0;
      const size_t rss_growth =
          rss_target > rss_bytes_ ? rss_target - rss_bytes_ : 0;
      if (growth && !pool.try_charge_workspace_plan(growth))
        throw std::bad_alloc();
      if (!pool.try_materialize(0) || !pool.try_charge_process_backing(
                                          rss_growth, materialization, false)) {
        if (growth) pool.release_workspace_plan(growth);
        throw std::bad_alloc();
      }
      bytes_ += growth;
      rss_bytes_ += rss_growth;
    }
    void retain_at_most(size_t bytes, size_t rss = 0) noexcept {
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
  size_t acquire(size_t bytes, size_t backing) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(backing <= bytes);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (!pool.capacity()) return 0;
    if (bytes > std::numeric_limits<size_t>::max() - active_)
      throw std::bad_alloc();
    const size_t target = active_ + bytes;
    const size_t retained = std::max(retained_, active_backing_ + backing);
    reservation_.acquire(target, retained, backing);
    active_ = target;
    active_backing_ += backing;
    retained_ = retained;
    return bytes;
  }

  void retain_backing(size_t previous, size_t backing) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(backing >= previous);
    const size_t target = active_backing_ + backing - previous;
    assert(target <= active_);
    const size_t retained = std::max(retained_, target);
    reservation_.acquire(reservation_.bytes(), retained, backing - previous);
    active_backing_ = target;
    retained_ = retained;
  }

  void release(size_t bytes, size_t backing) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(active_ >= bytes);
    assert(active_backing_ >= backing);
    active_ -= bytes;
    active_backing_ -= backing;
    reservation_.retain_at_most(std::max(active_, retained_), retained_);
  }

  size_t active_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
  }

  size_t reserved_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reservation_.bytes();
  }

  void reclaim_idle() {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock || active_) return;
    retained_ = 0;
    reservation_.retain_at_most(0);
  }

 private:
  friend class MemoryBudgetTestPeer;
  mutable std::mutex mutex_;
  BackingReservation reservation_;
  size_t active_{0};
  size_t active_backing_{0};
  size_t retained_{0};
};

inline const std::shared_ptr<WriteBufferMemoryAllowance> &
SharedWriteBufferMemoryAllowanceOwner() {
  (void)ailego::MemoryLimitPool::get_instance();
  static const auto owner = std::make_shared<WriteBufferMemoryAllowance>();
  return owner;
}

inline WriteBufferMemoryAllowance &SharedWriteBufferMemoryAllowance() {
  return *SharedWriteBufferMemoryAllowanceOwner();
}

}  // namespace detail

// A planning allowance, not an allocator or a hard RSS limit. A Collection
// serializes admission with writes and flush/segment rotation. Keep the
// active lease while a write buffer is live, including failed flushes. After
// flush/close, actual writes retain the shared backing allowance;
// admission-only create/clean-open checks return their temporary excess. WAL
// replay uses the managed-write policy because its record and decoded-Doc
// workspaces have separate ordinary-heap allowances.
class WriteBufferReservation {
 public:
  WriteBufferReservation() = default;
  WriteBufferReservation(const WriteBufferReservation &) = delete;
  WriteBufferReservation &operator=(const WriteBufferReservation &) = delete;
  ~WriteBufferReservation() {
    release();
  }

  Status acquire(size_t max_buffer_size, bool retain_backing = false) {
    return acquire_impl(
        max_buffer_size,
        retain_backing ? std::numeric_limits<size_t>::max() : 0);
  }

  // Arrow payload/encoding buffers have their own budgeted allocator. The
  // Doc staging cache and WAL serialization also retain their own ordinary-
  // heap allowance. Keep peak admission while writing, but only retain the
  // fixed small-workspace headroom after flush, not four Arrow equivalents.
  Status acquire_managed(size_t max_buffer_size) {
    return acquire_impl(max_buffer_size, kEncodingAllowance);
  }

  void release() noexcept {
    if (bytes_ == 0) return;
    detail::SharedWriteBufferMemoryAllowance().release(bytes_, backing_);
    bytes_ = 0;
    backing_ = 0;
  }

  size_t bytes() const noexcept {
    return bytes_;
  }

 private:
  static constexpr size_t kEncodingAllowance = 8 * 1024 * 1024;

  Status acquire_impl(size_t max_buffer_size, size_t backing) {
    auto &allowance = detail::SharedWriteBufferMemoryAllowance();
    if (bytes_ != 0) {
      // An admission-only lease has not projected any encoder backing yet.
      // Debit each promotion once, and still check current pressure when a
      // writer returns with the same workspace.
      const size_t next_backing = std::max(backing_, std::min(backing, bytes_));
      try {
        allowance.retain_backing(backing_, next_backing);
      } catch (const std::bad_alloc &) {
        return Status(
            StatusCode::RESOURCE_EXHAUSTED,
            "Write buffer materialization blocked by memory pressure");
      }
      backing_ = next_backing;
      return Status::OK();
    }

    // Docs/Arrow batches overlap merge and Parquet encoding buffers. Reserve
    // four buffer-equivalents (including allocation/reuse headroom) plus a
    // fixed allowance for small writers. Two equivalents did not cover the
    // measured flush peak. This remains a planning estimate: oversized docs,
    // wide schemas, retained results and allocator fragmentation can exceed it.
    constexpr size_t kBufferMultiplier = 4;
    if (max_buffer_size >
        (std::numeric_limits<size_t>::max() - kEncodingAllowance) /
            kBufferMultiplier) {
      return Status::InvalidArgument("write buffer reservation size overflow");
    }
    const size_t bytes =
        max_buffer_size * kBufferMultiplier + kEncodingAllowance;
    try {
      backing = std::min(backing, bytes);
      bytes_ = allowance.acquire(bytes, backing);
      backing_ = bytes_ ? backing : 0;
    } catch (const std::bad_alloc &) {
      return Status(
          StatusCode::RESOURCE_EXHAUSTED,
          "Insufficient memory budget for write buffer reservation (" +
              std::to_string(bytes) +
              " bytes). Reduce max_buffer_size or the number of "
              "writable collections, or increase memory_limit.");
    }
    return Status::OK();
  }

  size_t bytes_{0};
  size_t backing_{0};
};

}  // namespace zvec

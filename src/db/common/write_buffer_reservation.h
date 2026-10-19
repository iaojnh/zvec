// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
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
 public:
  size_t acquire(size_t bytes, size_t backing) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(backing <= bytes);
    if (!ailego::MemoryLimitPool::get_instance().capacity()) return 0;
    if (bytes > std::numeric_limits<size_t>::max() - active_)
      throw std::bad_alloc();
    const size_t target = active_ + bytes;
    if (target > reservation_.bytes()) {
      reservation_.add(target - reservation_.bytes());
    }
    active_ = target;
    active_backing_ += backing;
    retained_ = std::max(retained_, active_backing_);
    return bytes;
  }

  void retain_backing(size_t previous, size_t backing) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(backing >= previous);
    active_backing_ += backing - previous;
    assert(active_backing_ <= active_);
    retained_ = std::max(retained_, active_backing_);
  }

  void release(size_t bytes, size_t backing) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(active_ >= bytes);
    assert(active_backing_ >= backing);
    active_ -= bytes;
    active_backing_ -= backing;
    reservation_.retain_at_most(std::max(active_, retained_));
  }

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
  mutable std::mutex mutex_;
  ailego::MemoryReservation reservation_;
  size_t active_{0};
  size_t active_backing_{0};
  size_t retained_{0};
};

inline WriteBufferMemoryAllowance &SharedWriteBufferMemoryAllowance() {
  (void)ailego::MemoryLimitPool::get_instance();
  static WriteBufferMemoryAllowance allowance;
  return allowance;
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
      backing = std::max(backing_, std::min(backing, bytes_));
      allowance.retain_backing(backing_, backing);
      backing_ = backing;
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

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <new>
#include <utility>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {

// Use only while calling shared, allocation-only helpers with a fixed
// BudgetedAllocator type. Never span storage calls or user callbacks.
class ProcessRssAllocationScope {
 public:
  explicit ProcessRssAllocationScope(bool enabled = true)
      : previous_(std::exchange(state(), enabled)) {}
  ProcessRssAllocationScope(const ProcessRssAllocationScope &) = delete;
  ProcessRssAllocationScope &operator=(const ProcessRssAllocationScope &) =
      delete;
  ~ProcessRssAllocationScope() {
    state() = previous_;
  }

  static bool active() {
    return state();
  }

 private:
  static ZVEC_AILEGO_API bool &state();
  bool previous_;
};

// Owns process RSS backing for interfaces that require a standard allocator.
// Declare the reservation before the destination so it is destroyed last.
class ProcessRssReservation {
 public:
  ProcessRssReservation() = default;
  ProcessRssReservation(const ProcessRssReservation &) = delete;
  ProcessRssReservation &operator=(const ProcessRssReservation &) = delete;
  ProcessRssReservation(ProcessRssReservation &&other) noexcept {
    swap(other);
  }
  ProcessRssReservation &operator=(ProcessRssReservation &&other) noexcept {
    ProcessRssReservation replacement(std::move(other));
    swap(replacement);
    return *this;
  }
  ~ProcessRssReservation() {
    if (bytes_) {
      MemoryLimitPool::get_instance().release_heap_process_backing(bytes_,
                                                                   projection_);
    }
  }

  template <typename Allocate>
  void allocate(size_t limit, Allocate &&allocate) {
    using Pending = MemoryLimitPool::PendingAllocation;
    Pending pending(MemoryLimitPool::get_instance(),
                    Pending::Kind::kProcessRss);
    if (limit > std::numeric_limits<size_t>::max() - bytes_ ||
        !pending.reserve(limit))
      throw std::bad_alloc();
    const size_t bytes = allocate();
    if (bytes > limit) throw std::bad_alloc();
    projection_.append(
        ProcessRssBudget::HeapProjection::Full(bytes, pending.commit(bytes)));
    bytes_ += bytes;
  }

  void swap(ProcessRssReservation &other) noexcept {
    std::swap(bytes_, other.bytes_);
    std::swap(projection_, other.projection_);
  }

 private:
  size_t bytes_{0};
  ProcessRssBudget::HeapProjection projection_{};
};

}  // namespace zvec::ailego

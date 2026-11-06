// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <limits>
#include <mutex>

namespace zvec::ailego {

// A second admission signal alongside allocation accounting. RSS includes
// caller-owned memory and allocator retention; it is never added to used().
// Sampling and uninstrumented allocations make this a soft process target.
class ProcessRssBudget {
 public:
  using Reader = size_t (*)();

  static size_t headroom(size_t limit) {
    if (!limit) return 0;
    return std::min(std::max(size_t{1}, limit / 8),
                    std::clamp(limit / 20, size_t{8} << 20, size_t{64} << 20));
  }

  bool configure(size_t limit, Reader reader) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t rss = limit && reader ? reader() : 0;
    if (limit && !rss) return false;
    // Changing an active limit must not forget live, possibly untouched
    // backing. The initial sample already includes pre-existing allocations.
    if (limit && !this->limit()) {
      baseline_rss_ = rss;
      baseline_backing_ = backing_;
    }
    reader_ = reader;
    limit_.store(limit, std::memory_order_relaxed);
    pressure_.store(false, std::memory_order_relaxed);
    publish(rss);
    return true;
  }

  // Keep new backing charged for its entire lifetime, including before it is
  // faulted in. Sampling must not repeatedly admit the same unused headroom.
  // Existing backing reused by the pool is already included in this ledger.
  bool try_acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_locked(false);
    const size_t available = available_.load(std::memory_order_relaxed);
    if (bytes > available ||
        bytes > std::numeric_limits<size_t>::max() - backing_) {
      if (limit()) pressure_.store(true, std::memory_order_relaxed);
      return false;
    }
    backing_ += bytes;
    update_available();
    return true;
  }

  // Some consumers (RocksDB's block reader) cannot safely reject allocation
  // here. Their cache owns admission; still include temporary and live backing
  // in the process estimate so other consumers stop growing under pressure.
  void account(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(bytes <= std::numeric_limits<size_t>::max() - backing_);
    backing_ += bytes;
    update_available();
  }

  void release(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(bytes <= backing_);
    backing_ -= bytes;
    update_available();
  }

  void refresh(bool force = false) {
    if (!limit()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_locked(force);
  }

  size_t limit() const {
    return limit_.load(std::memory_order_relaxed);
  }
  size_t available() const {
    return available_.load(std::memory_order_relaxed);
  }
  size_t rss() const {
    return rss_.load(std::memory_order_relaxed);
  }
  bool pressure() const {
    return pressure_.load(std::memory_order_relaxed);
  }

 private:
  void refresh_locked(bool force) {
    if (!limit()) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - sampled_at_ < std::chrono::milliseconds(20)) return;
    const size_t rss = reader_ ? reader_() : 0;
    // A failed sample must not replenish admission headroom. Retry on the
    // next maintenance pass, keeping the last successful pressure signal.
    if (rss) publish(rss);
  }

  void publish(size_t rss) {
    // Only rebase down with no outstanding backing. A decrease in total
    // backing alone does not prove which owners retired: resident startup
    // allocations may have been replaced by untouched mappings of equal size.
    if (backing_ == 0 && rss < baseline_rss_) {
      baseline_rss_ = rss;
      baseline_backing_ = backing_;
    }
    sampled_backing_ = backing_;
    rss_.store(rss, std::memory_order_relaxed);
    update_available();
    sampled_at_ = std::chrono::steady_clock::now();
  }

  static size_t projected(size_t rss, size_t baseline, size_t backing) {
    const size_t growth = backing > baseline ? backing - baseline : 0;
    return rss + std::min(growth, std::numeric_limits<size_t>::max() - rss);
  }

  void update_available() {
    const size_t cap = limit();
    // Take the larger estimate, never add managed bytes to measured RSS.
    // The lifetime estimate preserves lazy allocations across samples; the
    // latest sample catches host growth and includes subsequent admissions.
    const size_t usage =
        std::max(projected(baseline_rss_, baseline_backing_, backing_),
                 projected(rss_.load(std::memory_order_relaxed),
                           sampled_backing_, backing_));
    const size_t available = cap ? (usage < cap ? cap - usage : 0)
                                 : std::numeric_limits<size_t>::max();
    const size_t margin = headroom(cap);
    const bool pressure = cap && (pressure_.load(std::memory_order_relaxed)
                                      ? available < 2 * margin
                                      : available <= margin);
    available_.store(available, std::memory_order_relaxed);
    pressure_.store(pressure, std::memory_order_relaxed);
  }

  std::mutex mutex_;
  Reader reader_{nullptr};
  size_t backing_{0};
  size_t baseline_rss_{0};
  size_t baseline_backing_{0};
  size_t sampled_backing_{0};
  std::chrono::steady_clock::time_point sampled_at_{};
  std::atomic<size_t> limit_{0};
  std::atomic<size_t> available_{std::numeric_limits<size_t>::max()};
  std::atomic<size_t> rss_{0};
  std::atomic<bool> pressure_{false};
};
}  // namespace zvec::ailego

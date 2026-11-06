// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <limits>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/parallel/worker_memory.h>

namespace zvec::detail {

using ailego::DefaultNativeWorkerStackBytes;
using ailego::NativeWorkerStackAllowance;
#if defined(_WIN32)
using ailego::WindowsWorkerStackBytes;
#endif

inline uint32_t RocksDbBackgroundThreads(uint32_t optimize_threads) noexcept {
  return std::clamp<uint32_t>(optimize_threads, 2, 16);
}

inline ailego::MemoryReservation &NativeWorkerStackReservation() {
  // GlobalResource's process-wide worker pools are destroyed during static
  // teardown, potentially after MemoryLimitPool. Keep their small accounting
  // owner process-lifetime as well; never refund while worker backing remains
  // live or call a destroyed pool from another singleton's destructor.
  static auto *reservation = new ailego::MemoryReservation;
  return *reservation;
}

inline void ReserveNativeWorkerStacks(size_t query_workers,
                                      size_t optimize_workers,
                                      uint32_t optimize_threads,
                                      size_t arrow_workers = 0) {
  const size_t stack_bytes = DefaultNativeWorkerStackBytes();
  if (stack_bytes == 0) return;
  // Include the dedicated Arrow CPU/I/O executors, the two zvec pools,
  // RocksDB's low-priority pool, its high-priority flush worker and the shared
  // BufferPool eviction worker. Caller-
  // owned Python/application threads are not charged here.
  const size_t other_workers = RocksDbBackgroundThreads(optimize_threads) + 2;
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  if (query_workers > kMax - optimize_workers ||
      query_workers + optimize_workers > kMax - other_workers ||
      arrow_workers > kMax - query_workers - optimize_workers - other_workers)
    throw std::bad_alloc();
  const size_t workers =
      query_workers + optimize_workers + other_workers + arrow_workers;
  if (workers > kMax / stack_bytes) throw std::bad_alloc();
  const size_t bytes = workers * stack_bytes;
  auto &reservation = NativeWorkerStackReservation();
  if (bytes > reservation.bytes()) reservation.add(bytes - reservation.bytes());
}

}  // namespace zvec::detail

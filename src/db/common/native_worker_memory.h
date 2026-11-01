// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <zvec/ailego/buffer/memory_reservation.h>
#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace zvec::detail {

inline uint32_t RocksDbBackgroundThreads(uint32_t optimize_threads) noexcept {
  return std::clamp<uint32_t>(optimize_threads, 2, 16);
}

// macOS secondary std::threads use the pthread default stack capacity. Reserve
// that capacity, not an RSS sample or an arbitrary per-worker estimate. Guard
// pages cannot become resident and are deliberately excluded. Other platforms
// retain their existing behavior until their native stack policy is covered.
inline size_t DefaultNativeWorkerStackBytes() {
#if defined(__APPLE__)
  pthread_attr_t attributes;
  if (pthread_attr_init(&attributes) != 0) throw std::bad_alloc();
  size_t bytes = 0;
  const int result = pthread_attr_getstacksize(&attributes, &bytes);
  const int destroyed = pthread_attr_destroy(&attributes);
  if (result != 0 || destroyed != 0 || bytes == 0) throw std::bad_alloc();
  return bytes;
#else
  return 0;
#endif
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
                                      uint32_t optimize_threads) {
  const size_t stack_bytes = DefaultNativeWorkerStackBytes();
  if (stack_bytes == 0) return;
  // Besides the two zvec pools: RocksDB's low-priority pool, its one high-
  // priority flush worker and the shared BufferPool eviction worker. Caller-
  // owned Python/application threads are not charged here.
  const size_t other_workers = RocksDbBackgroundThreads(optimize_threads) + 2;
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  if (query_workers > kMax - optimize_workers ||
      query_workers + optimize_workers > kMax - other_workers)
    throw std::bad_alloc();
  const size_t workers = query_workers + optimize_workers + other_workers;
  if (workers > kMax / stack_bytes) throw std::bad_alloc();
  const size_t bytes = workers * stack_bytes;
  auto &reservation = NativeWorkerStackReservation();
  if (bytes > reservation.bytes()) reservation.add(bytes - reservation.bytes());
}

}  // namespace zvec::detail

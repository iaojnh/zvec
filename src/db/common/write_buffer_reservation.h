// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <string>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/db/status.h>

namespace zvec {

// A planning allowance, not an allocator or a hard RSS limit. A Collection
// serializes its writes and keeps this allowance across flush/segment rotation.
// Reserve before opening its writer (including WAL recovery), not per document.
class WriteBufferReservation {
 public:
  WriteBufferReservation() = default;
  WriteBufferReservation(const WriteBufferReservation &) = delete;
  WriteBufferReservation &operator=(const WriteBufferReservation &) = delete;

  Status acquire(size_t max_buffer_size) {
    if (reservation_.bytes() != 0) return Status::OK();

    // Docs/Arrow batches overlap merge and Parquet encoding buffers. Reserve
    // four buffer-equivalents (including allocation/reuse headroom) plus a
    // fixed allowance for small writers. Two equivalents did not cover the
    // measured flush peak. This remains a planning estimate: oversized docs,
    // wide schemas, retained results and allocator fragmentation can exceed it.
    constexpr size_t kEncodingAllowance = 8 * 1024 * 1024;
    constexpr size_t kBufferMultiplier = 4;
    if (max_buffer_size >
        (std::numeric_limits<size_t>::max() - kEncodingAllowance) /
            kBufferMultiplier) {
      return Status::InvalidArgument("write buffer reservation size overflow");
    }
    const size_t bytes =
        max_buffer_size * kBufferMultiplier + kEncodingAllowance;
    try {
      reservation_.add(bytes);
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

  void release() noexcept {
    reservation_.retain_at_most(0);
  }

  size_t bytes() const noexcept {
    return reservation_.bytes();
  }

 private:
  ailego::MemoryReservation reservation_;
};

}  // namespace zvec

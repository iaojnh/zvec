// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

// One parser callback's copied input. Reuse exclusive funded backing when it
// fits; otherwise admit fresh backing only after the previous input retires.
// Large inputs prefer unmapping, with an exact heap alternative when page or
// descriptor overhead prevents input/private admission. No optional cache is
// populated and a private retry can change the owned representation once.
class IndexParsingInput {
 public:
  explicit IndexParsingInput(void *funded = nullptr, size_t bytes = 0)
      : funded_(funded), funded_bytes_(bytes) {}

  bool prepare(size_t len, bool retry = false) {
    if (retry) {
      if (exact_ || !mapping_.size()) return false;
      exact_ = true;
    } else {
      exact_ = false;
    }
    clear();
    if (!exact_ && funded_ && len <= funded_bytes_) {
      data_ = funded_;
      return true;
    }
    const bool heap =
        exact_ || len < ailego::MemoryLimitPool::page_buffer_size();
    if (!heap && mapping_.resize(len) == 0) {
      data_ = mapping_.data();
      return true;
    }
    try {
      heap_ = IndexStorage::MemoryBlock::AllocateOwned(len);
      data_ = const_cast<void *>(heap_.data());
      return true;
    } catch (const std::bad_alloc &) {
      // A small heap refusal may still permit a mapping. A mapping already
      // refused above must not be attempted repeatedly for the same input.
      if (heap && !exact_ && mapping_.resize(len) == 0) {
        data_ = mapping_.data();
        return true;
      }
      return false;
    }
  }

  void clear() noexcept {
    heap_.reset();
    mapping_.clear();
    data_ = nullptr;
  }
  void *data() const noexcept {
    return data_;
  }

 private:
  IndexStorage::MemoryBlock heap_;
  ailego::BudgetedMappedBuffer mapping_;
  void *funded_{nullptr}, *data_{nullptr};
  size_t funded_bytes_{0};
  bool exact_{false};
};
}  // namespace zvec::core

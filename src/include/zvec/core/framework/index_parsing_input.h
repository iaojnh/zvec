// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <utility>
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

  IndexParsingInput(IndexParsingInput &&other) noexcept {
    swap(other);
  }
  IndexParsingInput &operator=(IndexParsingInput &&other) noexcept {
    if (this != &other) {
      IndexParsingInput replacement(std::move(other));
      swap(replacement);
    }
    return *this;
  }

  bool prepare(size_t len, bool retry = false) {
    // Fixed-size checksum/scanning reads can reuse exactly the same funded
    // representation. A length change still retires the previous input so an
    // oversized buffer cannot compete with the next private allocation.
    if (!retry && data_ && size_ == len) {
      exact_ = false;
      return true;
    }
    if (retry) {
      if (exact_ || !mapping_.size()) return false;
      exact_ = true;
    } else {
      exact_ = false;
    }
    clear();
    size_ = len;
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
    size_ = 0;
  }
  void *data() const noexcept {
    return data_;
  }

 private:
  void swap(IndexParsingInput &other) noexcept {
    std::swap(heap_, other.heap_);
    mapping_.swap(other.mapping_);
    std::swap(funded_, other.funded_);
    std::swap(data_, other.data_);
    std::swap(funded_bytes_, other.funded_bytes_);
    std::swap(size_, other.size_);
    std::swap(exact_, other.exact_);
  }

  IndexStorage::MemoryBlock heap_;
  ailego::BudgetedMappedBuffer mapping_;
  void *funded_{nullptr}, *data_{nullptr};
  size_t funded_bytes_{0};
  size_t size_{0};
  bool exact_{false};
};
}  // namespace zvec::core

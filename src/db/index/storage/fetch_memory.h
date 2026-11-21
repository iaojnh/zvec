// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <arrow/api.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "parquet_memory_pool.h"

namespace zvec::detail {

// A per-batch limit on top of the shared allocator, not a second reservation.
// All arrays allocated here must be destroyed before this stack-owned pool.
class FetchMemoryPool final : public arrow::MemoryPool {
 public:
  FetchMemoryPool(arrow::MemoryPool *parent, int64_t limit)
      : parent_(parent), limit_(limit) {}

  arrow::Status Allocate(int64_t size, int64_t alignment,
                         uint8_t **out) override {
    if (size < 0 || size > limit_ - stats_.bytes_allocated()) {
      exceeded_ = true;
      return arrow::Status::OutOfMemory("Fetch batch byte limit exceeded");
    }
    ARROW_RETURN_NOT_OK(parent_->Allocate(size, alignment, out));
    stats_.DidAllocateBytes(size);
    return arrow::Status::OK();
  }
  arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                           int64_t alignment, uint8_t **ptr) override {
    // A managed mapping can shrink in place even when the batch is full.
    // All other reallocations must leave room for a second buffer.
    if (auto *parquet = dynamic_cast<ParquetMemoryPool *>(parent_)) {
      if (auto status = parquet->try_reallocate_in_place(old_size, new_size,
                                                         alignment, ptr)) {
        if (status->ok() && old_size != new_size) {
          stats_.DidReallocateBytes(old_size, new_size);
        }
        return *status;
      }
    }
    if (new_size < 0 || new_size > limit_ - stats_.bytes_allocated()) {
      exceeded_ = true;
      return arrow::Status::OutOfMemory("Fetch batch byte limit exceeded");
    }
    ARROW_RETURN_NOT_OK(
        parent_->Reallocate(old_size, new_size, alignment, ptr));
    stats_.DidAllocateBytes(new_size);
    stats_.DidFreeBytes(old_size);
    return arrow::Status::OK();
  }
  void Free(uint8_t *buffer, int64_t size, int64_t alignment) override {
    parent_->Free(buffer, size, alignment);
    stats_.DidFreeBytes(size);
  }
  int64_t bytes_allocated() const override {
    return stats_.bytes_allocated();
  }
  int64_t max_memory() const override {
    return stats_.max_memory();
  }
  int64_t total_bytes_allocated() const override {
    return stats_.total_bytes_allocated();
  }
  int64_t num_allocations() const override {
    return stats_.num_allocations();
  }
  std::string backend_name() const override {
    return "fetch_bounded";
  }
  bool exceeded() const {
    return exceeded_;
  }

 private:
  arrow::MemoryPool *parent_;
  int64_t limit_;
  arrow::internal::MemoryPoolStats stats_;
  bool exceeded_{false};
};

// Synchronous batch scratch: admit the entire backing through the ordinary
// bounded pool once, and keep it charged until every temporary has been freed.
// Each batch has its own cursor, so suballocations need no shared allocator or
// RSS-ledger lock. Larger buffers retain the parent's overlap admission rules.
class FetchScratchMemoryPool final : public arrow::MemoryPool {
 public:
  explicit FetchScratchMemoryPool(FetchMemoryPool *parent, bool enable)
      : parent_(parent) {
    if (!enable ||
        ailego::MemoryLimitPool::get_instance().prefetch_available() <
            16 * kScratchBytes)
      return;
    // Optional preparation must not prevent the original small-buffer path.
    try {
      if (!parent_->Allocate(kScratchBytes, 64, &scratch_).ok())
        scratch_ = nullptr;
    } catch (const std::bad_alloc &) {
      scratch_ = nullptr;
    }
  }
  ~FetchScratchMemoryPool() override {
    assert(stats_.bytes_allocated() == 0);
    if (scratch_) parent_->Free(scratch_, kScratchBytes, 64);
  }
  FetchScratchMemoryPool(const FetchScratchMemoryPool &) = delete;
  FetchScratchMemoryPool &operator=(const FetchScratchMemoryPool &) = delete;
  bool enabled() const {
    return scratch_ != nullptr;
  }
  arrow::Status Allocate(int64_t size, int64_t alignment,
                         uint8_t **out) override {
    if (!out) return arrow::Status::Invalid("Null fetch scratch output");
    if (size > 0 && size <= 16 * 1024 && alignment > 0 && alignment <= 64 &&
        (alignment & (alignment - 1)) == 0 && scratch_ &&
        size <= kScratchBytes - next_) {
      *out = scratch_ + next_;
      next_ += (size + 63) & ~int64_t{63};
    } else {
      ARROW_RETURN_NOT_OK(parent_->Allocate(size, alignment, out));
    }
    stats_.DidAllocateBytes(size);
    return arrow::Status::OK();
  }
  arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                           int64_t alignment, uint8_t **ptr) override {
    if (new_size < 0)
      return arrow::Status::Invalid("Negative fetch allocation");
    if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
      return arrow::Status::Invalid("Invalid fetch alignment");
    if (owns(*ptr)) {
      if (new_size <= old_size &&
          reinterpret_cast<uintptr_t>(*ptr) % alignment == 0) {
        stats_.DidReallocateBytes(old_size, new_size);
        return arrow::Status::OK();
      }
      uint8_t *replacement = nullptr;
      ARROW_RETURN_NOT_OK(Allocate(new_size, alignment, &replacement));
      std::memcpy(replacement, *ptr, std::min(old_size, new_size));
      Free(*ptr, old_size, alignment);
      *ptr = replacement;
      return arrow::Status::OK();
    }
    ARROW_RETURN_NOT_OK(
        parent_->Reallocate(old_size, new_size, alignment, ptr));
    stats_.DidReallocateBytes(old_size, new_size);
    return arrow::Status::OK();
  }
  void Free(uint8_t *buffer, int64_t size, int64_t alignment) override {
    if (!owns(buffer)) parent_->Free(buffer, size, alignment);
    stats_.DidFreeBytes(size);
  }
  int64_t bytes_allocated() const override {
    return stats_.bytes_allocated();
  }
  int64_t max_memory() const override {
    return stats_.max_memory();
  }
  int64_t total_bytes_allocated() const override {
    return stats_.total_bytes_allocated();
  }
  int64_t num_allocations() const override {
    return stats_.num_allocations();
  }
  std::string backend_name() const override {
    return "fetch_scratch";
  }

 private:
  bool owns(const uint8_t *buffer) const {
    const auto address = reinterpret_cast<uintptr_t>(buffer);
    const auto begin = reinterpret_cast<uintptr_t>(scratch_);
    return scratch_ && address >= begin && address - begin < kScratchBytes;
  }
  static constexpr int64_t kScratchBytes = 64 * 1024;
  FetchMemoryPool *parent_;
  uint8_t *scratch_{nullptr};
  int64_t next_{0};
  arrow::internal::MemoryPoolStats stats_;
};

// Arrow GetScalar copies binary/string values through std::string, bypassing
// the array's allocator. Keep a view instead, retaining its original owner.
inline arrow::Result<std::shared_ptr<arrow::Scalar>> FetchScalarView(
    const arrow::ChunkedArray &column, int64_t row) {
  if (row < 0 || row >= column.length()) {
    return arrow::Status::IndexError("Fetch row out of bounds");
  }
  for (const auto &chunk : column.chunks()) {
    if (row >= chunk->length()) {
      row -= chunk->length();
      continue;
    }
    if (chunk->IsNull(row)) return arrow::MakeNullScalar(chunk->type());
    if (chunk->type_id() == arrow::Type::STRING ||
        chunk->type_id() == arrow::Type::BINARY) {
      const auto &binary = static_cast<const arrow::BinaryArray &>(*chunk);
      auto bytes =
          arrow::SliceBuffer(binary.value_data(), binary.value_offset(row),
                             binary.value_length(row));
      if (chunk->type_id() == arrow::Type::STRING) {
        return std::make_shared<arrow::StringScalar>(std::move(bytes));
      }
      return std::make_shared<arrow::BinaryScalar>(std::move(bytes));
    }
    return chunk->GetScalar(row);
  }
  return arrow::Status::IndexError("Fetch chunk out of bounds");
}

inline arrow::Status AppendFetchRow(arrow::ArrayBuilder *builder,
                                    const arrow::ChunkedArray &column,
                                    int64_t row) {
  if (row < 0 || row >= column.length()) {
    return arrow::Status::IndexError("Fetch row out of bounds");
  }
  for (const auto &chunk : column.chunks()) {
    if (row >= chunk->length()) {
      row -= chunk->length();
      continue;
    }
    return builder->AppendArraySlice(*chunk->data(), row, 1);
  }
  return arrow::Status::IndexError("Fetch chunk out of bounds");
}

}  // namespace zvec::detail

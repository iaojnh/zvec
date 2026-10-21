// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <arrow/api.h>

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
    // The parent may allocate-and-copy; count both buffers at the peak.
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

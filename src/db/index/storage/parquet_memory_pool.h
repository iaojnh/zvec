// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <arrow/memory_pool.h>

namespace parquet {
class FileMetaData;
}

namespace zvec {
namespace detail {

// Isolate Parquet buffers from the general-purpose heap when budgeted. Small
// buffers share reusable backing blocks; large buffers use mappings. Backing
// capacity is admitted before decoding, including temporary buffers.
class ParquetMemoryPool final : public arrow::MemoryPool {
 public:
  // An internal initial cutoff, not a claim of an optimal tuning value.
  static constexpr int64_t kMappedAllocationThreshold = 256 * 1024;

  explicit ParquetMemoryPool(
      arrow::MemoryPool *small_pool = arrow::default_memory_pool(),
      bool enforce_budget = false)
      : small_pool_(small_pool), enforce_budget_(enforce_budget) {}

  using arrow::MemoryPool::Allocate;
  using arrow::MemoryPool::Free;
  using arrow::MemoryPool::Reallocate;

  arrow::Status Allocate(int64_t size, int64_t alignment,
                         uint8_t **out) override;
  arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                           int64_t alignment, uint8_t **ptr) override;
  void Free(uint8_t *buffer, int64_t size, int64_t alignment) override;

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
    return "parquet_mapped";
  }

  // Includes OS page rounding, alignment padding, and the mapping header.
  // This is mapped capacity, not RSS and not an additional budget charge.
  int64_t mapped_bytes() const {
    return mapped_bytes_.load(std::memory_order_relaxed);
  }
  int64_t num_mapped_allocations() const {
    return num_mapped_allocations_.load(std::memory_order_relaxed);
  }
  int64_t total_mapped_bytes_allocated() const {
    return total_mapped_bytes_allocated_.load(std::memory_order_relaxed);
  }
  size_t charged_bytes() const {
    return charged_bytes_.load(std::memory_order_relaxed);
  }

 private:
  struct SmallChunk;
  static constexpr size_t kSmallClasses = 7;  // 128 through 8192 bytes
  arrow::Status allocate_small(int64_t size, int64_t alignment, uint8_t **out);
  void free_small(uint8_t *buffer);

  arrow::MemoryPool *small_pool_;
  const bool enforce_budget_;
  arrow::internal::MemoryPoolStats stats_;
  std::atomic<int64_t> mapped_bytes_{0};
  std::atomic<int64_t> num_mapped_allocations_{0};
  std::atomic<int64_t> total_mapped_bytes_allocated_{0};
  std::atomic<size_t> charged_bytes_{0};
  std::mutex small_mutex_;
  std::array<SmallChunk *, kSmallClasses> small_chunks_{};
};

// Cached Arrow buffers retain this owner because Arrow stores a raw pool
// pointer. Do not replace the shared owner with a loader-local pool.
std::shared_ptr<ParquetMemoryPool> GetParquetMemoryPool();

// ColumnChunkReader accepts top-level fields, but the streaming reader takes
// physical leaf indices. Expand nested fields without dropping their children.
std::vector<int> ParquetLeafColumns(const parquet::FileMetaData &metadata,
                                    const std::vector<int> &fields);

// A batch-sizing hint only: compressed/repeated values may decode to more
// bytes. The allocator, not this estimate, enforces admission.
int64_t ParquetDecodeBatchRows(const parquet::FileMetaData &metadata,
                               int row_group, const std::vector<int> &columns);

}  // namespace detail
}  // namespace zvec

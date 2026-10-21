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

#include "parquet_memory_pool.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <parquet/metadata.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

namespace zvec {
namespace detail {
namespace {

constexpr int64_t kMinimumAlignment = 64;
constexpr size_t kSmallChunkBytes = 64 * 1024;

enum class BackingKind { Heap, Mapping, Retained, SmallChunk };

struct MappingHeader {
  void *base;
  size_t size;
  size_t charged;
  BackingKind kind{BackingKind::Mapping};
};

arrow::Status ValidateRequest(int64_t size, int64_t alignment) {
  if (size < 0) {
    return arrow::Status::Invalid("Negative Parquet allocation size");
  }
  if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
    return arrow::Status::Invalid("Parquet alignment must be a power of two");
  }
  if (static_cast<uint64_t>(size) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(alignment) > std::numeric_limits<size_t>::max()) {
    return arrow::Status::OutOfMemory(
        "Parquet allocation exceeds address space");
  }
  return arrow::Status::OK();
}

size_t SystemPageSize() {
#if defined(_WIN32)
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  return info.dwPageSize;
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 0;
#endif
}

arrow::Status MappingSize(int64_t size, int64_t alignment, size_t *out) {
  const size_t page_size = SystemPageSize();
  if (page_size == 0) {
    return arrow::Status::IOError("Cannot determine OS page size");
  }
  const uint64_t limit = std::min<uint64_t>(
      std::numeric_limits<size_t>::max(), std::numeric_limits<int64_t>::max());
  const uint64_t padding =
      static_cast<uint64_t>(alignment - 1) + sizeof(MappingHeader);
  if (padding > limit || static_cast<uint64_t>(size) > limit - padding ||
      static_cast<uint64_t>(size) + padding > limit - (page_size - 1)) {
    return arrow::Status::OutOfMemory("Parquet mapping size overflow");
  }
  const size_t needed = static_cast<size_t>(size) + padding;
  *out = ((needed + page_size - 1) / page_size) * page_size;
  return arrow::Status::OK();
}

void *MapMemory(size_t size) {
#if defined(_WIN32)
  return ::VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE);
#else
  void *base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return base == MAP_FAILED ? nullptr : base;
#endif
}

bool UnmapMemory(void *base, size_t size) {
#if defined(_WIN32)
  (void)size;
  return ::VirtualFree(base, 0, MEM_RELEASE) != 0;
#else
  return ::munmap(base, size) == 0;
#endif
}

}  // namespace

struct alignas(64) ParquetMemoryPool::SmallChunk {
  SmallChunk *next{nullptr};
  void *free_head{nullptr};
  size_t stride{0};
  size_t next_slot{0};
  size_t live_slots{0};
};

arrow::Status ParquetMemoryPool::allocate_small(int64_t size, int64_t alignment,
                                                uint8_t **out) {
  auto &budget = ailego::MemoryLimitPool::get_instance();
  const size_t needed = static_cast<size_t>(size) + alignment;
  // Unusual alignment and medium buffers get their own reusable mapping.
  // Record the original pointer: a budget may be enabled after allocation.
  if (budget.capacity() == 0) {
    uint8_t *base = nullptr;
    ARROW_RETURN_NOT_OK(small_pool_->Allocate(needed, alignment, &base));
    *out = base + alignment;
    new (*out - sizeof(MappingHeader))
        MappingHeader{base, needed, 0, BackingKind::Heap};
    return arrow::Status::OK();
  }
  constexpr size_t max_stride = 128UL << (kSmallClasses - 1);
  if (alignment > kMinimumAlignment || needed > max_stride) {
    void *base = budget.acquire_retained_block(needed);
    if (!base) {
      return arrow::Status::OutOfMemory("Cannot reserve Parquet backing");
    }
    const auto start =
        reinterpret_cast<uintptr_t>(base) + sizeof(MappingHeader);
    const auto aligned = (start + alignment - 1) & ~(alignment - 1);
    *out = reinterpret_cast<uint8_t *>(aligned);
    const auto charge = budget.retained_block_size(needed);
    new (*out - sizeof(MappingHeader))
        MappingHeader{base, needed, charge, BackingKind::Retained};
    charged_bytes_.fetch_add(charge, std::memory_order_relaxed);
    return arrow::Status::OK();
  }
  size_t index = 0, stride = 128;
  while (stride < needed) {
    stride *= 2;
    ++index;
  }
  constexpr size_t payload =
      kSmallChunkBytes - ailego::MemoryLimitPool::kRetainedBlockHeaderBytes;
  const size_t slots = (payload - sizeof(SmallChunk)) / stride;
  std::unique_lock<std::mutex> lock(small_mutex_);
  auto *chunk = small_chunks_[index];
  while (chunk && !chunk->free_head && chunk->next_slot == slots) {
    chunk = chunk->next;
  }
  if (!chunk) {
    // Admission may evict a cached Arrow array and reenter free_small().
    lock.unlock();
    void *base = budget.acquire_retained_block(payload);
    if (!base) {
      return arrow::Status::OutOfMemory("Cannot reserve Parquet small chunk");
    }
    chunk = new (base) SmallChunk{};
    chunk->stride = stride;
    charged_bytes_.fetch_add(kSmallChunkBytes, std::memory_order_relaxed);
    lock.lock();
    chunk->next = small_chunks_[index];
    small_chunks_[index] = chunk;
  }
  void *slot = chunk->free_head;
  if (slot) {
    chunk->free_head = *static_cast<void **>(slot);
  } else {
    slot = reinterpret_cast<char *>(chunk + 1) + chunk->next_slot++ * stride;
  }
  ++chunk->live_slots;
  *out = static_cast<uint8_t *>(slot) + kMinimumAlignment;
  new (*out - sizeof(MappingHeader))
      MappingHeader{chunk, 0, 0, BackingKind::SmallChunk};
  return arrow::Status::OK();
}

void ParquetMemoryPool::free_small(uint8_t *buffer) {
  const auto header =
      *reinterpret_cast<const MappingHeader *>(buffer - sizeof(MappingHeader));
  auto &budget = ailego::MemoryLimitPool::get_instance();
  if (header.kind == BackingKind::Retained) {
    budget.release_retained_block(header.base);
    charged_bytes_.fetch_sub(header.charged, std::memory_order_relaxed);
    return;
  }
  assert(header.kind == BackingKind::SmallChunk);
  auto *chunk = static_cast<SmallChunk *>(header.base);
  std::unique_lock<std::mutex> lock(small_mutex_);
  void *slot = buffer - kMinimumAlignment;
  new (slot) void *(chunk->free_head);
  chunk->free_head = slot;
  if (--chunk->live_slots != 0) return;
  size_t index = 0;
  for (size_t stride = 128; stride < chunk->stride; stride *= 2) ++index;
  auto **link = &small_chunks_[index];
  while (*link != chunk) link = &(*link)->next;
  *link = chunk->next;
  lock.unlock();
  budget.release_retained_block(chunk);
  charged_bytes_.fetch_sub(kSmallChunkBytes, std::memory_order_relaxed);
  // The global pool keeps idle backing committed until reuse or unmapping.
}

arrow::Status ParquetMemoryPool::Allocate(int64_t size, int64_t alignment,
                                          uint8_t **out) {
  if (out == nullptr) {
    return arrow::Status::Invalid("Null Parquet allocation output");
  }
  ARROW_RETURN_NOT_OK(ValidateRequest(size, alignment));
  alignment = std::max(alignment, kMinimumAlignment);
  auto &budget = ailego::MemoryLimitPool::get_instance();
  if (size < kMappedAllocationThreshold) {
    // Arrow requires a non-null data address even for empty binary/string
    // buffers. Let its allocator provide and own the zero-size sentinel.
    if (!enforce_budget_ || size == 0) {
      ARROW_RETURN_NOT_OK(small_pool_->Allocate(size, alignment, out));
    } else {
      // The prefix records the actual charge, even for allocations made
      // before a process budget is configured. Never infer it at Free time.
      if (size > std::numeric_limits<int64_t>::max() - alignment) {
        return arrow::Status::OutOfMemory("Parquet allocation size overflow");
      }
      const auto backing_size = size + alignment;
      if (static_cast<uint64_t>(backing_size) >
          std::numeric_limits<size_t>::max()) {
        return arrow::Status::OutOfMemory(
            "Parquet allocation exceeds address space");
      }
      ARROW_RETURN_NOT_OK(allocate_small(size, alignment, out));
    }
  } else {
    size_t mapped_size = 0;
    ARROW_RETURN_NOT_OK(MappingSize(size, alignment, &mapped_size));
    const size_t charge =
        enforce_budget_ && budget.capacity() ? mapped_size : 0;
    if (charge && !budget.try_charge_workspace(charge)) {
      return arrow::Status::OutOfMemory(
          "Parquet decode exceeds shared budget: requested=", charge,
          " used=", budget.used(), " capacity=", budget.capacity());
    }
    void *base = MapMemory(mapped_size);
    if (base == nullptr) {
      budget.release_external(charge);
      return arrow::Status::OutOfMemory("Cannot map Parquet buffer of ", size,
                                        " bytes");
    }
    const uintptr_t start =
        reinterpret_cast<uintptr_t>(base) + sizeof(MappingHeader);
    const uintptr_t aligned = (start + static_cast<uintptr_t>(alignment - 1)) &
                              ~static_cast<uintptr_t>(alignment - 1);
    auto *buffer = reinterpret_cast<uint8_t *>(aligned);
    new (buffer - sizeof(MappingHeader))
        MappingHeader{base, mapped_size, charge};
    *out = buffer;
    charged_bytes_.fetch_add(charge, std::memory_order_relaxed);
    mapped_bytes_.fetch_add(static_cast<int64_t>(mapped_size),
                            std::memory_order_relaxed);
    num_mapped_allocations_.fetch_add(1, std::memory_order_relaxed);
    total_mapped_bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
  }
  stats_.DidAllocateBytes(size);
  return arrow::Status::OK();
}

arrow::Status ParquetMemoryPool::Reallocate(int64_t old_size, int64_t new_size,
                                            int64_t alignment, uint8_t **ptr) {
  if (ptr == nullptr || (old_size != 0 && *ptr == nullptr)) {
    return arrow::Status::Invalid("Null Parquet reallocation input");
  }
  ARROW_RETURN_NOT_OK(ValidateRequest(old_size, alignment));
  ARROW_RETURN_NOT_OK(ValidateRequest(new_size, alignment));
  alignment = std::max(alignment, kMinimumAlignment);
  if (old_size == new_size) {
    return arrow::Status::OK();
  }
  if (!enforce_budget_ && old_size != 0 &&
      old_size < kMappedAllocationThreshold &&
      new_size < kMappedAllocationThreshold) {
    ARROW_RETURN_NOT_OK(
        small_pool_->Reallocate(old_size, new_size, alignment, ptr));
    stats_.DidReallocateBytes(old_size, new_size);
    return arrow::Status::OK();
  }

  // Keep the old buffer intact if allocation fails. Account for both buffers
  // while copying rather than hiding the transient peak behind a size delta.
  uint8_t *replacement = nullptr;
  ARROW_RETURN_NOT_OK(Allocate(new_size, alignment, &replacement));
  if (old_size != 0) {
    std::memcpy(replacement, *ptr,
                static_cast<size_t>(std::min(old_size, new_size)));
  }
  Free(*ptr, old_size, alignment);
  *ptr = replacement;
  return arrow::Status::OK();
}

void ParquetMemoryPool::Free(uint8_t *buffer, int64_t size, int64_t alignment) {
  if (buffer == nullptr) {
    assert(size == 0);
    return;
  }
  assert(size >= 0 && alignment > 0 && (alignment & (alignment - 1)) == 0);
  alignment = std::max(alignment, kMinimumAlignment);
  if (size < kMappedAllocationThreshold) {
    if (!enforce_budget_ || size == 0) {
      small_pool_->Free(buffer, size, alignment);
    } else {
      const auto header = *reinterpret_cast<const MappingHeader *>(
          buffer - sizeof(MappingHeader));
      if (header.kind == BackingKind::Heap) {
        small_pool_->Free(static_cast<uint8_t *>(header.base), header.size,
                          alignment);
      } else {
        free_small(buffer);
      }
    }
  } else {
    const auto header = *reinterpret_cast<const MappingHeader *>(
        buffer - sizeof(MappingHeader));
    const bool released = UnmapMemory(header.base, header.size);
    assert(released);
    if (!released) {
      return;
    }
    mapped_bytes_.fetch_sub(static_cast<int64_t>(header.size),
                            std::memory_order_relaxed);
    ailego::MemoryLimitPool::get_instance().release_external(header.charged);
    charged_bytes_.fetch_sub(header.charged, std::memory_order_relaxed);
  }
  stats_.DidFreeBytes(size);
}

std::shared_ptr<ParquetMemoryPool> GetParquetMemoryPool() {
  static auto pool = std::make_shared<ParquetMemoryPool>(
      arrow::default_memory_pool(), /*enforce_budget=*/true);
  return pool;
}

int64_t ParquetDecodeBatchRows(const parquet::FileMetaData &metadata,
                               int row_group, const std::vector<int> &columns) {
  constexpr int64_t kTargetBytes = 512 * 1024;
  auto group = metadata.RowGroup(row_group);
  if (group->num_rows() <= 0) return 8192;
  int64_t row_bytes = 0;
  for (int column : columns) {
    if (column < 0 || column >= group->num_columns()) continue;
    const auto bytes = group->ColumnChunk(column)->total_uncompressed_size();
    if (bytes <= 0) continue;
    const int64_t width = std::min(bytes / group->num_rows(), kTargetBytes) + 1;
    row_bytes += std::min(width, kTargetBytes);
    if (row_bytes >= kTargetBytes) return 1;
  }
  return std::min<int64_t>(8192,
                           kTargetBytes / std::max<int64_t>(1, row_bytes));
}

std::vector<int> ParquetLeafColumns(const parquet::FileMetaData &metadata,
                                    const std::vector<int> &fields) {
  const auto *schema = metadata.schema();
  std::vector<int> leaves;
  for (int field : fields) {
    if (field < 0 || field >= schema->group_node()->field_count()) continue;
    const auto *root = schema->group_node()->field(field).get();
    for (int leaf = 0; leaf < schema->num_columns(); ++leaf) {
      if (schema->GetColumnRoot(leaf) == root &&
          std::find(leaves.begin(), leaves.end(), leaf) == leaves.end()) {
        leaves.push_back(leaf);
      }
    }
  }
  return leaves;
}

}  // namespace detail
}  // namespace zvec

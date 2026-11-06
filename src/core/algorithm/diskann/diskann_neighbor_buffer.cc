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

#include "diskann_neighbor_buffer.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_error.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace zvec {
namespace core {
namespace {
bool RoundToPages(size_t bytes, size_t *rounded) {
  const size_t page = ailego::MemoryHelper::PageSize();
  if (bytes > std::numeric_limits<size_t>::max() - (page - 1)) return false;
  *rounded = (bytes + page - 1) / page * page;
  return true;
}
}  // namespace

DiskAnnNeighborBuffer::~DiskAnnNeighborBuffer() {
  clear();
}

int DiskAnnNeighborBuffer::allocate(size_t bytes) {
  size_t rounded;
  if (!RoundToPages(bytes, &rounded)) return IndexError_Overflow;
  if (rounded == 0) return release_mapping() ? 0 : IndexError_Runtime;

  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t charge = pool.capacity() ? rounded : 0;
  using Pending = ailego::MemoryLimitPool::PendingAllocation;
  // The owning graph may be holding its chunk lock. Reclaim idle backing
  // only, so admission cannot reenter an eviction owner from under the lock.
  Pending pending(pool, Pending::Kind::kLockedExternal);
  if (!pending.reserve(charge)) return IndexError_NoMemory;
  DiskAnnNeighborBuffer replacement;
  replacement.data_ = static_cast<char *>(ailego::File::MemoryMap(rounded, 0));
  if (!replacement.data_) return IndexError_NoMemory;
  pending.commit();
  replacement.size_ = rounded;
  replacement.charged_bytes_ = charge;

  // Keep the old owner intact if its mapping cannot be released.
  if (!release_mapping()) return IndexError_Runtime;
  std::swap(data_, replacement.data_);
  std::swap(size_, replacement.size_);
  std::swap(charged_bytes_, replacement.charged_bytes_);
  return 0;
}

int DiskAnnNeighborBuffer::shrink(size_t bytes) {
  size_t rounded;
  if (!RoundToPages(bytes, &rounded)) return IndexError_Overflow;
  if (rounded > size_) return IndexError_InvalidLength;
  if (rounded == size_) return 0;
  if (rounded == 0) return release_mapping() ? 0 : IndexError_Runtime;
#if defined(_WIN32)
  // Keep the reservation until clear(); only committed pages consume memory.
  if (!::VirtualFree(data_ + rounded, size_ - rounded, MEM_DECOMMIT)) {
    return IndexError_Runtime;
  }
#else
  if (::munmap(data_ + rounded, size_ - rounded) != 0) {
    return IndexError_Runtime;
  }
#endif
  const size_t refund = charged_bytes_ ? size_ - rounded : 0;
  charged_bytes_ -= refund;
  size_ = rounded;
  if (refund) ailego::MemoryLimitPool::get_instance().release_external(refund);
  return 0;
}

bool DiskAnnNeighborBuffer::release_mapping() {
  if (!data_) return true;
  if (!ailego::File::TryMemoryUnmap(data_, size_)) {
    LOG_ERROR("Failed to unmap DiskANN neighbor backing; size=%zu", size_);
    return false;
  }
  if (charged_bytes_)
    ailego::MemoryLimitPool::get_instance().release_external(charged_bytes_);
  data_ = nullptr;
  size_ = charged_bytes_ = 0;
  return true;
}

void DiskAnnNeighborBuffer::clear() {
  (void)release_mapping();
}

struct DiskAnnNeighborPool::Row {
  Chunk *chunk;
  uint32_t count;
  uint32_t id;
};

struct DiskAnnNeighborPool::Chunk {
  static void destroy(Chunk *chunk) {
    const size_t charge = chunk->charged_bytes;
    delete chunk;
    if (charge) ailego::MemoryLimitPool::get_instance().release_external(charge);
  }

  DiskAnnNeighborBuffer buffer;
  size_t charged_bytes{0};
  Chunk *prev{nullptr};
  Chunk *next{nullptr};
  Chunk *available_prev{nullptr};
  Chunk *available_next{nullptr};
  size_t bucket{0};
  size_t stride{0};
  size_t packed_bytes{0};
  uint32_t capacity{0};
  uint32_t slots{0};
  uint32_t used{0};
  uint32_t live{0};
  uint32_t free_head{UINT32_MAX};
};

DiskAnnNeighborPool::~DiskAnnNeighborPool() {
  clear();
}

int DiskAnnNeighborPool::reserve(size_t nodes, uint32_t max_degree) {
  if (max_degree == 0) return IndexError_InvalidArgument;
  if (static_cast<size_t>(max_degree) >
      (std::numeric_limits<size_t>::max() - sizeof(Row) - alignof(Row)) /
          sizeof(uint32_t))
    return IndexError_Overflow;
  if (nodes > rows_.max_size()) return IndexError_Overflow;
  ailego::BudgetedVector<uint8_t *> rows;
  ailego::BudgetedVector<Chunk *> available;
  const size_t classes =
      (static_cast<size_t>(max_degree) - 1) / kCapacityStep + 1;
  try {
    rows.resize(nodes, nullptr);
    available.resize(classes, nullptr);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
  clear();
  rows_.swap(rows);
  available_.swap(available);
  max_degree_ = max_degree;
  // Avoid one large chunk per class for small indexes. For larger indexes,
  // cap chunks at 1 MiB so entirely empty chunks can be returned during build.
  const size_t page = ailego::MemoryHelper::PageSize();
  const size_t stride =
      sizeof(Row) + static_cast<size_t>(max_degree) * sizeof(uint32_t);
  constexpr size_t max_chunk = 1024u * 1024u;
  const size_t scale = classes * 4;
  const size_t target = nodes / scale > max_chunk / stride
                            ? max_chunk
                            : std::max(page, (nodes / scale) * stride);
  chunk_bytes_ = std::min(max_chunk, (target + page - 1) / page * page);
  return 0;
}

void DiskAnnNeighborPool::add_available(Chunk *chunk) {
  Chunk *&head = available_[chunk->bucket];
  chunk->available_prev = nullptr;
  chunk->available_next = head;
  if (head) head->available_prev = chunk;
  head = chunk;
}

void DiskAnnNeighborPool::remove_available(Chunk *chunk) {
  if (chunk->available_prev) {
    chunk->available_prev->available_next = chunk->available_next;
  } else {
    available_[chunk->bucket] = chunk->available_next;
  }
  if (chunk->available_next) {
    chunk->available_next->available_prev = chunk->available_prev;
  }
  chunk->available_prev = chunk->available_next = nullptr;
}

DiskAnnNeighborPool::Row *DiskAnnNeighborPool::acquire(uint32_t id,
                                                       uint32_t count) {
  const size_t bucket = (static_cast<size_t>(count) - 1) / kCapacityStep;
  std::lock_guard<std::mutex> lock(mutex_);
  Chunk *chunk = available_[bucket];
  if (!chunk) {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    const size_t charge = pool.capacity() ? sizeof(Chunk) : 0;
    using Pending = ailego::MemoryLimitPool::PendingAllocation;
    Pending pending(pool, Pending::Kind::kLockedExternal);
    if (!pending.reserve(charge)) return nullptr;
    std::unique_ptr<Chunk, decltype(&Chunk::destroy)> next(
        new (std::nothrow) Chunk, &Chunk::destroy);
    if (!next) return nullptr;
    pending.commit();
    next->charged_bytes = charge;
    next->bucket = bucket;
    next->capacity = static_cast<uint32_t>(std::min<uint64_t>(
        (uint64_t{bucket} + 1) * kCapacityStep, max_degree_));
    const size_t payload =
        static_cast<size_t>(next->capacity) * sizeof(uint32_t);
    if (payload >
        std::numeric_limits<size_t>::max() - sizeof(Row) - alignof(Row)) {
      return nullptr;
    }
    next->stride = (sizeof(Row) + payload + alignof(Row) - 1) / alignof(Row) *
                   alignof(Row);
    if (next->buffer.allocate(std::max(chunk_bytes_, next->stride)) != 0) {
      return nullptr;
    }
    next->slots = static_cast<uint32_t>(next->buffer.size() / next->stride);
    next->next = chunks_;
    if (chunks_) chunks_->prev = next.get();
    chunks_ = next.release();
    chunk = chunks_;
    mapped_bytes_ += chunk->buffer.size();
    ++chunk_count_;
    add_available(chunk);
  }

  uint32_t slot;
  if (chunk->free_head != UINT32_MAX) {
    slot = chunk->free_head;
    auto *row =
        reinterpret_cast<Row *>(chunk->buffer.data() + slot * chunk->stride);
    chunk->free_head = row->count;
  } else {
    slot = chunk->used++;
  }
  auto *row =
      new (chunk->buffer.data() + slot * chunk->stride) Row{chunk, count, id};
  if (++chunk->live == chunk->slots) remove_available(chunk);
  return row;
}

void DiskAnnNeighborPool::release(Row *row) {
  std::lock_guard<std::mutex> lock(mutex_);
  Chunk *chunk = row->chunk;
  const bool was_full = chunk->live == chunk->slots;
  const auto slot = static_cast<uint32_t>(
      (reinterpret_cast<char *>(row) - chunk->buffer.data()) / chunk->stride);
  row->id = UINT32_MAX;
  row->count = chunk->free_head;
  chunk->free_head = slot;
  --chunk->live;
  if (was_full) add_available(chunk);
  if (chunk->live == 0) {
    remove_available(chunk);
    if (chunk->prev)
      chunk->prev->next = chunk->next;
    else
      chunks_ = chunk->next;
    if (chunk->next) chunk->next->prev = chunk->prev;
    mapped_bytes_ -= chunk->buffer.size();
    --chunk_count_;
    Chunk::destroy(chunk);
  }
}

std::pair<uint32_t, const uint32_t *> DiskAnnNeighborPool::get(
    uint32_t id) const {
  static const uint32_t empty = 0;
  const uint8_t *data = rows_[id];
  if (!data) return {0, &empty};
  if (compacted_) {
    return {*reinterpret_cast<const uint32_t *>(data),
            reinterpret_cast<const uint32_t *>(data + sizeof(uint32_t))};
  }
  const auto *row = reinterpret_cast<const Row *>(data);
  return {row->count, reinterpret_cast<const uint32_t *>(row + 1)};
}

int DiskAnnNeighborPool::set(uint32_t id, const uint32_t *neighbors,
                             size_t count) {
  if (compacted_) return IndexError_NoReady;
  if (id >= rows_.size() || count > max_degree_)
    return IndexError_InvalidLength;
  auto *old = reinterpret_cast<Row *>(rows_[id]);
  if (count == 0) {
    rows_[id] = nullptr;
    if (old) release(old);
    return 0;
  }
  const size_t bucket = (count - 1) / kCapacityStep;
  Row *row = old;
  if (!old || old->chunk->bucket != bucket) {
    row = acquire(id, static_cast<uint32_t>(count));
    if (!row) return IndexError_NoMemory;
  }
  std::memmove(row + 1, neighbors, count * sizeof(uint32_t));
  row->count = static_cast<uint32_t>(count);
  rows_[id] = reinterpret_cast<uint8_t *>(row);
  if (old && row != old) release(old);
  return 0;
}

int DiskAnnNeighborPool::append(uint32_t id, uint32_t neighbor) {
  if (compacted_) return IndexError_NoReady;
  if (id >= rows_.size()) return IndexError_InvalidLength;
  auto *old = reinterpret_cast<Row *>(rows_[id]);
  const uint32_t count = old ? old->count : 0;
  if (count >= max_degree_) return IndexError_InvalidLength;
  Row *row = old;
  if (!old || count == old->chunk->capacity) {
    row = acquire(id, count + 1);
    if (!row) return IndexError_NoMemory;
    if (old) std::memcpy(row + 1, old + 1, count * sizeof(uint32_t));
  }
  reinterpret_cast<uint32_t *>(row + 1)[count] = neighbor;
  row->count = count + 1;
  rows_[id] = reinterpret_cast<uint8_t *>(row);
  if (old && row != old) release(old);
  return 0;
}

int DiskAnnNeighborPool::compact(uint32_t final_max_degree) {
  if (!compacted_) {
    // Check every node before changing the mutable layout.
    for (uint8_t *data : rows_) {
      if (data &&
          reinterpret_cast<const Row *>(data)->count > final_max_degree) {
        return IndexError_InvalidLength;
      }
    }
    for (Chunk *chunk = chunks_; chunk; chunk = chunk->next) {
      size_t offset = 0;
      for (uint32_t slot = 0; slot < chunk->used; ++slot) {
        auto *row = reinterpret_cast<const Row *>(chunk->buffer.data() +
                                                  slot * chunk->stride);
        if (row->id == UINT32_MAX) continue;
        const uint32_t id = row->id;
        const uint32_t count = row->count;
        char *packed = chunk->buffer.data() + offset;
        // Visit source slots in ascending address order. The packed row is
        // always shorter, so moves cannot overwrite a later live slot.
        std::memmove(packed + sizeof(count), row + 1, count * sizeof(uint32_t));
        std::memcpy(packed, &count, sizeof(count));
        rows_[id] = reinterpret_cast<uint8_t *>(packed);
        offset += sizeof(count) + count * sizeof(uint32_t);
      }
      chunk->packed_bytes = offset;
    }
    compacted_ = true;
  }
  // All rows already have the packed layout if an OS shrink needs retrying.
  for (Chunk *chunk = chunks_; chunk; chunk = chunk->next) {
    const size_t before = chunk->buffer.size();
    int ret = chunk->buffer.shrink(chunk->packed_bytes);
    if (ret != 0) return ret;
    mapped_bytes_ -= before - chunk->buffer.size();
  }
  return 0;
}

size_t DiskAnnNeighborPool::memory_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return mapped_bytes_ + rows_.capacity() * sizeof(uint8_t *) +
         available_.capacity() * sizeof(Chunk *) + chunk_count_ * sizeof(Chunk);
}

void DiskAnnNeighborPool::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  while (chunks_) {
    Chunk *next = chunks_->next;
    Chunk::destroy(chunks_);
    chunks_ = next;
  }
  decltype(rows_)().swap(rows_);
  decltype(available_)().swap(available_);
  max_degree_ = 0;
  chunk_bytes_ = mapped_bytes_ = chunk_count_ = 0;
  compacted_ = false;
}

}  // namespace core
}  // namespace zvec

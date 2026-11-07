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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <ailego/utility/mobile_memory.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>

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
namespace ailego {

namespace {

constexpr size_t kMinimumPageBytes = 4096UL;
constexpr size_t kSlabBytes = MemoryLimitPool::slab_size();
constexpr size_t kSlabAlignment = MemoryLimitPool::slab_alignment();
constexpr size_t kMaxSlabDataPages = kSlabBytes / kMinimumPageBytes - 1;

constexpr size_t kQueueSmallBytes = 4096;
constexpr size_t kQueueSlabBytes = 64UL << 10;
constexpr size_t kQueueAlignment = alignof(std::max_align_t);

static_assert(kQueueSmallBytes % kQueueAlignment == 0,
              "queue size classes must preserve maximum alignment");

void *map_queue_memory(size_t bytes) {
#if defined(_WIN32)
  return ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE);
#else
  void *data = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return data == MAP_FAILED ? nullptr : data;
#endif
}

bool unmap_queue_memory(void *data, size_t bytes) {
#if defined(_WIN32)
  (void)bytes;
  return ::VirtualFree(data, 0, MEM_RELEASE) != 0;
#else
  return ::munmap(data, bytes) == 0;
#endif
}

static_assert((kSlabAlignment & (kSlabAlignment - 1)) == 0,
              "slab alignment must be a power of two");
static_assert(kSlabBytes == kSlabAlignment,
              "slab owner lookup relies on size matching alignment");
static_assert(kMaxSlabDataPages <= UINT16_MAX,
              "slab page indexes must fit in uint16_t");

struct AlignedSlabMapping {
  char *base{nullptr};
  void *reservation_base{nullptr};
  size_t reservation_size{0};
};

AlignedSlabMapping reserve_aligned_slab(size_t page_size) {
  constexpr size_t kReservationBytes = kSlabBytes + kSlabAlignment;
#if defined(_WIN32)
  void *reservation =
      ::VirtualAlloc(nullptr, kReservationBytes, MEM_RESERVE, PAGE_READWRITE);
  if (reservation == nullptr) {
    return {};
  }
  const uintptr_t raw = reinterpret_cast<uintptr_t>(reservation);
  const uintptr_t aligned = (raw + kSlabAlignment - 1) & ~(kSlabAlignment - 1);
  void *header = ::VirtualAlloc(reinterpret_cast<void *>(aligned), page_size,
                                MEM_COMMIT, PAGE_READWRITE);
  if (header == nullptr) {
    ::VirtualFree(reservation, 0, MEM_RELEASE);
    return {};
  }
  return {reinterpret_cast<char *>(aligned), reservation, kReservationBytes};
#else
  (void)page_size;
  void *reservation = ::mmap(nullptr, kReservationBytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reservation == MAP_FAILED) {
    return {};
  }

  const uintptr_t raw = reinterpret_cast<uintptr_t>(reservation);
  const uintptr_t aligned = (raw + kSlabAlignment - 1) & ~(kSlabAlignment - 1);
  const size_t prefix = aligned - raw;
  const size_t suffix = kReservationBytes - prefix - kSlabBytes;
  if (prefix != 0 && ::munmap(reservation, prefix) != 0) {
    ::munmap(reservation, kReservationBytes);
    return {};
  }
  if (suffix != 0 &&
      ::munmap(reinterpret_cast<void *>(aligned + kSlabBytes), suffix) != 0) {
    ::munmap(reinterpret_cast<void *>(aligned), kSlabBytes + suffix);
    return {};
  }
  return {reinterpret_cast<char *>(aligned), reinterpret_cast<void *>(aligned),
          kSlabBytes};
#endif
}

bool commit_slab_page(char *page, size_t page_size) {
#if defined(_WIN32)
  return ::VirtualAlloc(page, page_size, MEM_COMMIT, PAGE_READWRITE) == page;
#else
  (void)page;
  (void)page_size;
  return true;
#endif
}

bool discard_slab_page(char *page, size_t page_size) {
#if defined(_WIN32)
  return ::VirtualFree(page, page_size, MEM_DECOMMIT) != 0;
#elif defined(__APPLE__)
  // MADV_FREE_REUSABLE is only lazy reclamation: it can leave resident pages
  // behind after their budget has been spent by another consumer. Replace
  // this exclusively owned, free page with fresh demand-zero backing before
  // refunding it. MAP_FIXED preserves the slab's address reservation without
  // an unmap/remap gap; never apply it to a live or pinned cache page.
  return ::mmap(page, page_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page;
#else
  return ::madvise(page, page_size, MADV_DONTNEED) == 0;
#endif
}

}  // namespace

struct alignas(std::max_align_t) MemoryLimitPool::QueueAllocationHeader {
  size_t bytes{0};
  union {
    QueueMemorySlab *owner;
    QueueAllocationHeader *next_free;
  };
};

struct MemoryLimitPool::QueueMappingHeader {
  ProcessRssBudget::MaterializationStamp stamp{};
  QueueAllocationHeader allocation{};
};

struct MemoryLimitPool::QueueMemorySlab {
  QueueMemorySlab(size_t allocation_size, size_t region_bytes)
      : slot_size(allocation_size),
        mapping_bytes(region_bytes),
        next_unused((sizeof(QueueMemorySlab) + kQueueAlignment - 1) &
                    ~(kQueueAlignment - 1)) {}

  bool has_space() const {
    return free_head != nullptr || slot_size <= mapping_bytes - next_unused;
  }
  void link(QueueMemorySlab *&head) {
    previous = nullptr;
    next = head;
    if (head) head->previous = this;
    head = this;
  }
  void unlink(QueueMemorySlab *&head) {
    if (previous)
      previous->next = next;
    else
      head = next;
    if (next) next->previous = previous;
    previous = next = nullptr;
  }

  QueueMemorySlab *previous{nullptr};
  QueueMemorySlab *next{nullptr};
  size_t slot_size;
  size_t mapping_bytes;
  size_t next_unused;
  size_t live_slots{0};
  QueueAllocationHeader *free_head{nullptr};
  ProcessRssBudget::MappingProjection projection{};
};

void *EvictionQueueTraits::malloc(size_t bytes) {
  return MemoryLimitPool::get_instance().allocate_queue_memory(bytes);
}

void EvictionQueueTraits::free(void *ptr) {
  MemoryLimitPool::get_instance().free_queue_memory(ptr);
}

void *MemoryLimitPool::allocate_queue_memory(size_t bytes) {
  if (bytes > std::numeric_limits<size_t>::max() -
                  sizeof(QueueAllocationHeader) - (kQueueAlignment - 1)) {
    return nullptr;
  }
  const size_t total =
      (sizeof(QueueAllocationHeader) + bytes + kQueueAlignment - 1) &
      ~(kQueueAlignment - 1);
  std::lock_guard<std::mutex> lock(queue_memory_mutex_);
  const bool charged = pool_size_.load(std::memory_order_relaxed) != 0;
  const auto reserve = [this, charged](size_t size) {
    if (size > std::numeric_limits<size_t>::max() -
                   queue_allocated_size_.load(std::memory_order_relaxed)) {
      return false;
    }
    if (charged && !try_reserve_allocation(size, Backing::kPending))
      return false;
    if (charged) {
      metadata_used_size_.fetch_add(size, std::memory_order_relaxed);
      queue_charged_size_.fetch_add(size, std::memory_order_relaxed);
    }
    return true;
  };
  const auto rollback = [this, charged](size_t size) {
    if (charged) {
      queue_charged_size_.fetch_sub(size, std::memory_order_relaxed);
      release_fixed(size, &metadata_used_size_, Backing::kPending);
    }
  };
  const auto denied = [this]() -> void * {
    // Do not call try_charge_metadata here: that can recurse into this same
    // queue with owner/lifecycle locks held. Existing enqueue-failure recovery
    // retries registration after capacity becomes available.
    high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
    bg_cv_.notify_one();
    return nullptr;
  };
  const size_t page = page_buffer_size();
  if (total <= kQueueSmallBytes) {
    const size_t index = total / kQueueAlignment - 1;
    auto *slab = queue_small_slabs_[index];
    if (charged && slab && !slab->free_head) {
      // Advancing into a new page of an old slab can raise RSS even though
      // the whole mapping is already charged. Previously used slots remain
      // available so queue registration can help reclaim pages under pressure.
      const size_t previous_page = (slab->next_unused - 1) / page;
      const size_t last_page = (slab->next_unused + total - 1) / page;
      if (last_page != previous_page &&
          !process_rss_.try_extend_mapping((last_page - previous_page) * page,
                                           slab->mapping_bytes,
                                           slab->projection))
        return denied();
    }
    if (!slab) {
      const size_t slab_header =
          (sizeof(QueueMemorySlab) + kQueueAlignment - 1) &
          ~(kQueueAlignment - 1);
      const size_t minimum = (slab_header + total + page - 1) & ~(page - 1);
      size_t region_bytes = std::max(kQueueSlabBytes, page);
      if (!reserve(region_bytes)) {
        region_bytes = minimum;
        if (region_bytes >= std::max(kQueueSlabBytes, page) ||
            !reserve(region_bytes)) {
          return denied();
        }
      }
      void *raw = map_queue_memory(region_bytes);
      if (!raw) {
        rollback(region_bytes);
        return nullptr;
      }
      slab = new (raw) QueueMemorySlab(total, region_bytes);
      if (charged) {
        slab->projection = ProcessRssBudget::MappingProjection::Full(
            region_bytes, process_rss_.finish_reservation(region_bytes, true));
      }
      slab->link(queue_small_slabs_[index]);
      queue_allocated_size_.fetch_add(region_bytes, std::memory_order_relaxed);
    }
    QueueAllocationHeader *header = slab->free_head;
    if (header) {
      slab->free_head = header->next_free;
    } else {
      header = new (reinterpret_cast<char *>(slab) + slab->next_unused)
          QueueAllocationHeader{};
      slab->next_unused += total;
    }
    header->bytes = total;
    header->owner = slab;
    ++slab->live_slots;
    if (!slab->has_space()) slab->unlink(queue_small_slabs_[index]);
    return header + 1;
  }
  static_assert(offsetof(QueueMappingHeader, allocation) +
                    sizeof(QueueAllocationHeader) ==
                sizeof(QueueMappingHeader));
  constexpr size_t prefix = offsetof(QueueMappingHeader, allocation);
  if (total > std::numeric_limits<size_t>::max() - prefix - (page - 1)) {
    return nullptr;
  }
  const size_t region_bytes = (prefix + total + page - 1) & ~(page - 1);
  if (!reserve(region_bytes)) return denied();
  void *raw = map_queue_memory(region_bytes);
  if (!raw) {
    rollback(region_bytes);
    return nullptr;
  }
  auto *mapping = new (raw) QueueMappingHeader{};
  if (charged)
    mapping->stamp = process_rss_.finish_reservation(region_bytes, true);
  auto *header = &mapping->allocation;
  header->bytes = region_bytes;
  header->owner = nullptr;
  queue_allocated_size_.fetch_add(region_bytes, std::memory_order_relaxed);
  return header + 1;
}

void MemoryLimitPool::free_queue_memory(void *ptr) {
  if (!ptr) {
    return;
  }
  auto *header = static_cast<QueueAllocationHeader *>(ptr) - 1;
  std::lock_guard<std::mutex> lock(queue_memory_mutex_);
  size_t bytes = header->bytes;
  ProcessRssBudget::MappingProjection projection;
  if (auto *slab = header->owner) {
    const size_t index = slab->slot_size / kQueueAlignment - 1;
    const bool was_full = !slab->has_space();
    if (--slab->live_slots != 0) {
      header->next_free = slab->free_head;
      slab->free_head = header;
      if (was_full) slab->link(queue_small_slabs_[index]);
      return;
    }
    bytes = slab->mapping_bytes;
    projection = slab->projection;
    auto *previous = slab->previous;
    auto *next = slab->next;
    // Save links before unmapping; do not access any slab/header byte after
    // releasing the region. Other live neighbors always retain the charge.
    if (!unmap_queue_memory(slab, bytes)) {
      header->next_free = slab->free_head;
      slab->free_head = header;
      if (was_full) slab->link(queue_small_slabs_[index]);
      LOG_ERROR("Failed to unmap queue slab; backing remains charged");
      return;
    }
    if (!was_full) {
      if (previous)
        previous->next = next;
      else
        queue_small_slabs_[index] = next;
      if (next) next->previous = previous;
    }
  } else {
    auto *mapping = reinterpret_cast<QueueMappingHeader *>(
        reinterpret_cast<char *>(header) -
        offsetof(QueueMappingHeader, allocation));
    projection =
        ProcessRssBudget::MappingProjection::Full(bytes, mapping->stamp);
    if (!unmap_queue_memory(mapping, bytes)) {
      LOG_ERROR("Failed to unmap queue allocation; backing remains charged");
      return;
    }
  }
  queue_allocated_size_.fetch_sub(bytes, std::memory_order_relaxed);
  if (pool_size_.load(std::memory_order_relaxed) != 0) {
    queue_charged_size_.fetch_sub(bytes, std::memory_order_relaxed);
    release_fixed(bytes, &metadata_used_size_, Backing::kLive, projection);
  }
}

struct MemoryLimitPool::ReclaimableSlab {
  static constexpr uint64_t kMagic = 0x5A564543534C4142ULL;

  ReclaimableSlab(MemoryLimitPool *pool, void *reservation,
                  size_t reservation_bytes, size_t system_page_size)
      : owner(pool),
        reservation_base(reservation),
        reservation_size(reservation_bytes),
        page_size(system_page_size),
        data_page_count(
            static_cast<uint16_t>(kSlabBytes / system_page_size - 1)) {}

  bool unmap_empty() {
    assert(committed_pages == 0);
    auto *pool = owner;
    auto *following = next;
    void *base = reservation_base;
    const size_t bytes = reservation_size;
    const size_t page = page_size;
    const auto projection = header_projection;
    this->~ReclaimableSlab();
    if (unmap_queue_memory(base, bytes)) return true;

    // The pages were discarded before teardown. Restore an empty slab so
    // the old pool can keep using its charged header and address range.
    auto *restored = new (this) ReclaimableSlab(pool, base, bytes, page);
    restored->next = following;
    restored->header_projection = projection;
    LOG_ERROR("MemoryLimitPool: failed to unmap an empty slab");
    return false;
  }

  uint64_t magic{kMagic};
  MemoryLimitPool *owner{nullptr};
  ReclaimableSlab *next{nullptr};
  void *reservation_base{nullptr};
  size_t reservation_size{0};
  size_t page_size{0};
  std::mutex mutex;
  uint16_t next_unused{1};
  uint16_t reclaimed_count{0};
  uint16_t data_page_count{0};
  size_t committed_pages{0};
  uint16_t reclaimed_pages[kMaxSlabDataPages]{};
  ProcessRssBudget::ReusablePages<kMaxSlabDataPages> projections;
  ProcessRssBudget::MaterializationStamp header_projection{};
};

struct MemoryLimitPool::RetainedBlock {
  RetainedBlock *next{nullptr};
  size_t bytes{0};
  bool metadata{false};
  ProcessRssBudget::MaterializationStamp projection{};
  ProcessRssBudget::MaterializationStamp creation{};
};

size_t MemoryLimitPool::retained_block_size(size_t bytes) {
  const size_t page = page_buffer_size();
  if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() -
                                kRetainedBlockHeaderBytes - (page - 1)) {
    return 0;
  }
  return ((bytes + kRetainedBlockHeaderBytes + page - 1) / page) * page;
}

void *MemoryLimitPool::acquire_retained_block(size_t bytes, bool metadata) {
  static_assert(sizeof(RetainedBlock) <= kRetainedBlockHeaderBytes,
                "retained block header must fit its reserved prefix");
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  const size_t charge = retained_block_size(bytes);
  if (charge == 0 || capacity() == 0) return nullptr;
  auto *counter = metadata ? &metadata_used_size_ : &external_used_size_;
  {
    std::lock_guard<std::mutex> lock(retained_mutex_);
    for (auto **link = &retained_free_; *link; link = &(*link)->next) {
      auto *block = *link;
      if (block->bytes != charge) continue;
      // An idle mapping can contain pages that were never faulted in or were
      // reclaimed by the OS. Assigning a new owner can make them resident.
      // Backing is already charged, but a pending page may own the logical
      // headroom. Leave the idle block linked until admission succeeds.
      if (!try_reserve_used(charge)) break;
      if (!process_rss_.try_materialize_reusable(charge, block->projection)) {
        release_buffer(nullptr, charge);
        break;
      }
      *link = block->next;
      block->next = nullptr;
      block->metadata = metadata;
      retained_idle_bytes_.fetch_sub(charge, std::memory_order_relaxed);
      counter->fetch_add(charge, std::memory_order_relaxed);
      retained_block_reuses_.fetch_add(1, std::memory_order_relaxed);
      return reinterpret_cast<char *>(block) + kRetainedBlockHeaderBytes;
    }
  }
  // No allocator lock may be held across eviction or writeback callbacks.
  if (!try_charge_fixed_with_wait(charge, counter, Backing::kPending))
    return nullptr;
#if defined(_WIN32)
  void *raw =
      ::VirtualAlloc(nullptr, charge, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
  void *raw = ::mmap(nullptr, charge, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) raw = nullptr;
#endif
  if (!raw) {
    release_fixed(charge, counter, Backing::kPending);
    return nullptr;
  }
  const auto projection = process_rss_.finish_reservation(charge, true);
  auto *block = new (raw)
      RetainedBlock{nullptr, charge, metadata, projection, projection};
  retained_block_bytes_.fetch_add(charge, std::memory_order_relaxed);
  return reinterpret_cast<char *>(block) + kRetainedBlockHeaderBytes;
}

bool MemoryLimitPool::try_materialize_retained_block(void *data) {
  auto *block = reinterpret_cast<RetainedBlock *>(static_cast<char *>(data) -
                                                  kRetainedBlockHeaderBytes);
  return process_rss_.try_materialize_reusable(block->bytes, block->projection);
}

void MemoryLimitPool::release_retained_block(void *data) {
  if (!data) return;
  auto *block = reinterpret_cast<RetainedBlock *>(static_cast<char *>(data) -
                                                  kRetainedBlockHeaderBytes);
  {
    std::lock_guard<std::mutex> lock(retained_mutex_);
    const size_t bytes = block->bytes;
    const auto projection = ProcessRssBudget::MappingProjection::ForReuse(
        bytes, block->projection, block->creation);
    auto *counter =
        block->metadata ? &metadata_used_size_ : &external_used_size_;
    // Retained mappings share the idle reserve with free data pages. Returning
    // a burst of Arrow chunks must not leave excess backing resident until
    // paced background maintenance catches up. The owner has relinquished
    // this entire block; live/pinned allocations are never reclaimed here.
    const size_t committed = committed_size_.load(std::memory_order_relaxed);
    const size_t used = used_size_.load(std::memory_order_relaxed);
    const size_t idle = committed > used ? committed - used : 0;
    const size_t reserve =
        process_rss_.pressure() ? 0 : reserve_margin(capacity());
    bool unmapped = false;
    if (idle > reserve || bytes > reserve - idle) {
#if defined(_WIN32)
      unmapped = ::VirtualFree(block, 0, MEM_RELEASE) != 0;
#else
      unmapped = ::munmap(block, bytes) == 0;
#endif
    }
    if (unmapped) {
      retained_block_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
      release_committed(bytes, Backing::kLive, projection);
    } else {
      // A failed physical release keeps the backing and its committed charge.
      block->next = retained_free_;
      retained_free_ = block;
      retained_idle_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    counter->fetch_sub(bytes, std::memory_order_relaxed);
    // Publish idle backing or finish unmapping before init() sees no owners.
    used_size_.fetch_sub(bytes, std::memory_order_relaxed);
  }
  { std::lock_guard<std::mutex> lock(capacity_mutex_); }
  capacity_cv_.notify_all();
}

size_t MemoryLimitPool::trim_retained_blocks(size_t bytes_needed) {
  size_t released = 0;
  std::lock_guard<std::mutex> lock(retained_mutex_);
  while (retained_free_ && released < bytes_needed) {
    auto *block = retained_free_;
    auto *next = block->next;
    const size_t bytes = block->bytes;
    const auto projection = ProcessRssBudget::MappingProjection::ForReuse(
        bytes, block->projection, block->creation);
#if defined(_WIN32)
    const bool unmapped = ::VirtualFree(block, 0, MEM_RELEASE) != 0;
#else
    const bool unmapped = ::munmap(block, bytes) == 0;
#endif
    if (!unmapped) {
      // A failed OS release must not create reusable budget.
      LOG_ERROR("MemoryLimitPool: failed to unmap retained backing block");
      break;
    }
    retained_free_ = next;
    retained_block_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    retained_idle_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    release_committed(bytes, Backing::kLive, projection);
    released += bytes;
  }
  return released;
}

size_t MemoryLimitPool::page_buffer_size() {
  static const size_t page_size = []() -> size_t {
#if defined(_WIN32)
    SYSTEM_INFO info;
    ::GetSystemInfo(&info);
    return static_cast<size_t>(info.dwPageSize);
#else
    return static_cast<size_t>(::getpagesize());
#endif
  }();
  assert(page_size >= kMinimumPageBytes && page_size < kSlabBytes &&
         (page_size & (page_size - 1)) == 0);
  return page_size;
}

bool BlockEvictionQueue::evict_single_block(BlockType &item) {
  return evict_single_block(item, /*age_protected=*/false);
}

bool BlockEvictionQueue::evict_single_block(
    BlockType &item, bool age_protected,
    std::array<size_t, kQueueCount> *scan_remaining) {
  if (age_protected) {
    const size_t probation = approximate_queue_sizes_[kProbationPriority].load(
        std::memory_order_relaxed);
    const size_t protected_pages =
        approximate_queue_sizes_[kProtectedPriority].load(
            std::memory_order_relaxed);
    const size_t dominance_threshold =
        probation > (std::numeric_limits<size_t>::max() - 1) /
                        kProtectedDominanceRatio
            ? std::numeric_limits<size_t>::max()
            : (probation + 1) * kProtectedDominanceRatio;
    if (protected_pages > dominance_threshold &&
        (!scan_remaining || (*scan_remaining)[kProtectedPriority] != 0) &&
        evict_queues_[kProtectedPriority].try_dequeue(item)) {
      if (scan_remaining) --(*scan_remaining)[kProtectedPriority];
      approximate_queue_sizes_[kProtectedPriority].fetch_sub(
          1, std::memory_order_relaxed);
      protected_aging_dequeues_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  for (size_t i = 0; i < kQueueCount; i++) {
    if (scan_remaining && (*scan_remaining)[i] == 0) continue;
    if (evict_queues_[i].try_dequeue(item)) {
      if (scan_remaining) --(*scan_remaining)[i];
      approximate_queue_sizes_[i].fetch_sub(1, std::memory_order_relaxed);
      return true;
    }
  }
  return false;
}

bool BlockEvictionQueue::evict_block(BlockType &item) {
  size_t attempts = 0;
  bool age_protected = true;
  return evict_block(item, attempts, std::numeric_limits<size_t>::max(),
                     age_protected);
}

bool BlockEvictionQueue::evict_block(
    BlockType &item, size_t &attempts, size_t max_attempts, bool &age_protected,
    std::array<size_t, kQueueCount> *scan_remaining) {
  while (attempts < max_attempts) {
    if (!evict_single_block(item, age_protected, scan_remaining)) {
      return false;
    }
    // Protected aging is deliberately a once-per-reclaim-batch decision.
    age_protected = false;
    ++attempts;
    std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
    if (item.owner == nullptr ||
        valid_owners_.find(item.owner) == valid_owners_.end() ||
        item.owner->is_dead_block(item.owner_key, item.version)) {
      continue;
    }
    const uint8_t current_priority =
        item.owner->eviction_priority(item.owner_key);
    if (item.priority != current_priority) {
      item.priority = current_priority;
      if (!add_single_block(item, static_cast<int>(current_priority))) {
        item.owner->eviction_requeue_failed(item.owner_key, item.version);
      }
      continue;
    }
    return true;
  }
  return false;
}

void BlockEvictionQueue::recycle() {
  BlockType item;
  // A foreground page fault must not scan the whole global queue while its
  // page remains in kLoadingRefCount. Try a small CLOCK sample and let the
  // caller fall back or retry; background reclaim handles deep queue walks.
  static constexpr size_t kForegroundReclaimAttempts = 20;
  const size_t max_attempts = kForegroundReclaimAttempts;
  size_t attempts = 0;
  bool recovered = false;
  bool age_protected = true;
  // Page allocations can hit their admission limit before the process-wide
  // pool is full because part of the budget is reserved for external cache
  // consumers. Stop immediately after enough room for one page is reclaimed.
  while (MemoryLimitPool::get_instance().is_page_full() &&
         attempts < max_attempts) {
    if (!evict_block(item, attempts, max_attempts, age_protected)) {
      if (attempts >= max_attempts) {
        break;
      }
      if (recovered || recover_owner_queues() == 0) {
        break;
      }
      recovered = true;
      continue;
    }
    {
      std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
      if (item.owner != nullptr &&
          valid_owners_.find(item.owner) != valid_owners_.end() &&
          !item.owner->is_dead_block(item.owner_key, item.version)) {
        item.owner->evict_block(item.owner_key);
      }
    }
  }
}

size_t BlockEvictionQueue::recycle_inactive(size_t max_attempts) {
  // Sampling must progress even when ordinary pages are continuously
  // replenished. Collect the sample before requeueing so a hot/pinned entry
  // is not inspected twice in the same pass. Never scan a whole cache here.
  std::array<BlockType, 8> items;
  const size_t attempts = std::min(max_attempts, items.size());
  size_t found = 0;
  for (; found < attempts; ++found) {
    if (!evict_queues_[kExplicitHotPriority].try_dequeue(items[found])) break;
    approximate_queue_sizes_[kExplicitHotPriority].fetch_sub(
        1, std::memory_order_relaxed);
  }
  size_t evicted = 0;
  for (size_t i = 0; i < found; ++i) {
    const BlockType &item = items[i];
    std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
    if (item.owner == nullptr || valid_owners_.count(item.owner) == 0 ||
        item.owner->is_dead_block(item.owner_key, item.version)) {
      continue;
    }
    const uint8_t priority = item.owner->eviction_priority(item.owner_key);
    if (priority == kExplicitHotPriority) {
      inactive_probes_.fetch_add(1, std::memory_order_relaxed);
      if (item.owner->evict_inactive_block(item.owner_key)) {
        ++evicted;
        inactive_evictions_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
    }
    if (!add_single_block(item, static_cast<int>(priority))) {
      item.owner->eviction_requeue_failed(item.owner_key, item.version);
    }
  }
  return evicted;
}

size_t BlockEvictionQueue::batch_recycle(size_t count) {
  // Small foreground reservations preserve the existing cost preference.
  // Larger pressure batches give inactive decoded payloads bounded CLOCK
  // attention, rather than letting them monopolize the pool indefinitely.
  size_t evicted = count >= kProtectedAgingMinBatch ? recycle_inactive(8) : 0;
  // Bound work when no page is currently evictable.
  const size_t max_attempts =
      count > (std::numeric_limits<size_t>::max() - 16) / 4
          ? std::numeric_limits<size_t>::max()
          : count * 4 + 16;
  size_t attempts = 0;
  bool recovered = false;
  bool age_protected = count >= kProtectedAgingMinBatch;
  // Pinned entries are requeued. Do not spend the whole bounded scan on
  // those same low-priority entries while reclaimable decoded columns wait
  // in a higher-priority queue. Snapshot each tier and reserve scan work for
  // every tier; successful low-priority eviction still satisfies count first.
  std::array<size_t, kQueueCount> scan_remaining{};
  const auto reset_scan = [&] {
    const size_t per_queue = std::max<size_t>(1, max_attempts / kQueueCount);
    for (size_t i = 0; i < kQueueCount; ++i) {
      const size_t depth = std::max<size_t>(
          1, approximate_queue_sizes_[i].load(std::memory_order_relaxed));
      // Allow CLOCK's second chance (and cross-tier promotion) within a
      // batch, without letting one pinned entry monopolize all scan work.
      scan_remaining[i] = depth >= per_queue / 2 ? per_queue : depth * 2;
    }
  };
  reset_scan();
  while (evicted < count && attempts < max_attempts) {
    BlockType item;
    if (!evict_block(item, attempts, max_attempts, age_protected,
                     &scan_remaining)) {
      if (attempts >= max_attempts) {
        break;
      }
      if (recovered || recover_owner_queues() == 0) {
        break;
      }
      recovered = true;
      reset_scan();
      continue;
    }
    std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
    if (item.owner != nullptr &&
        valid_owners_.find(item.owner) != valid_owners_.end() &&
        !item.owner->is_dead_block(item.owner_key, item.version) &&
        item.owner->evict_block(item.owner_key)) {
      ++evicted;
    }
  }
  return evicted;
}

size_t BlockEvictionQueue::recover_owner_queues() {
  std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
  size_t recovered = 0;
  for (EvictableBlockOwner *owner : valid_owners_) {
    recovered += owner->recover_eviction_queue();
  }
  return recovered;
}

size_t BlockEvictionQueue::cleanup_dead_entries(size_t max_attempts) {
  if (max_attempts == 0) return 0;
  std::unique_lock<std::mutex> cleanup_lock(cleanup_mutex_, std::try_to_lock);
  if (!cleanup_lock.owns_lock()) return 0;
  if (std::all_of(cleanup_scan_remaining_.begin(),
                  cleanup_scan_remaining_.end(),
                  [](size_t remaining) { return remaining == 0; })) {
    const auto generation = cleanup_generation_.load(std::memory_order_relaxed);
    if (generation == cleanup_generation_seen_) return 0;
    cleanup_generation_seen_ = generation;
    for (size_t i = 0; i < kQueueCount; ++i) {
      cleanup_scan_remaining_[i] =
          approximate_queue_sizes_[i].load(std::memory_order_relaxed);
    }
  }

  size_t attempts = 0;
  size_t cleaned = 0;
  std::array<size_t, kQueueCount> budgets{1, 1, 1};
  size_t first_queue = cleanup_next_queue_;
  cleanup_next_queue_ = (cleanup_next_queue_ + 1) % kQueueCount;
  if (max_attempts >= kQueueCount) {
    // Most retired pages accumulate in probation. Reserve smaller shares for
    // protected/hot entries while giving probation the first and largest scan.
    budgets[kProtectedPriority] = std::max<size_t>(1, max_attempts / 8);
    budgets[kExplicitHotPriority] = std::max<size_t>(1, max_attempts / 16);
    budgets[kProbationPriority] = max_attempts - budgets[kProtectedPriority] -
                                  budgets[kExplicitHotPriority];
    first_queue = kProbationPriority;
  }
  // Spend unused shares in priority order. The snapshot remaining counts
  // prevent live requeued entries from being scanned twice in this pass.
  for (size_t i = 0; i < 2 * kQueueCount && attempts < max_attempts; ++i) {
    const size_t priority = (first_queue + i) % kQueueCount;
    const size_t budget =
        i < kQueueCount ? budgets[priority] : max_attempts - attempts;
    size_t remaining = std::min(budget, cleanup_scan_remaining_[priority]);
    while (remaining != 0 && attempts < max_attempts) {
      // Tokenless dequeue favors a few producer streams. Requeuing live
      // entries onto the cleaner's stream can then hide old producer streams
      // indefinitely. Rotate producers and dequeue a batch before requeuing.
      std::array<BlockType, 64> items;
      const size_t requested =
          std::min({items.size(), remaining, max_attempts - attempts});
      const size_t found = evict_queues_[priority].try_dequeue_bulk(
          *cleanup_consumers_[priority], items.begin(), requested);
      if (found == 0) {
        cleanup_scan_remaining_[priority] = 0;
        break;
      }
      remaining -= found;
      cleanup_scan_remaining_[priority] -= found;
      approximate_queue_sizes_[priority].fetch_sub(found,
                                                   std::memory_order_relaxed);
      attempts += found;
      std::shared_lock<std::shared_mutex> owner_lock(valid_owners_mutex_);
      for (size_t entry = 0; entry < found; ++entry) {
        const auto &item = items[entry];
        if (item.owner == nullptr || valid_owners_.count(item.owner) == 0 ||
            item.owner->is_dead_block(item.owner_key, item.version)) {
          ++cleaned;
          continue;
        }
        // Keep the original priority/membership and CLOCK state. Only ordinary
        // pressure-driven eviction decides whether a live page should go.
        if (!add_single_block(item, static_cast<int>(priority))) {
          item.owner->eviction_requeue_failed(item.owner_key, item.version);
        }
      }
    }
  }
  cleanup_attempts_.fetch_add(attempts, std::memory_order_relaxed);
  dead_entries_cleaned_.fetch_add(cleaned, std::memory_order_relaxed);
  return cleaned;
}

bool BlockEvictionQueue::add_single_block(const BlockType &block,
                                          int queue_index) {
  if (queue_index < 0 || queue_index >= static_cast<int>(kQueueCount)) {
    LOG_ERROR("invalid eviction priority: %d", queue_index);
    return false;
  }
  BlockType queued = block;
  queued.priority = static_cast<uint8_t>(queue_index);
  // Publish the depth first so a concurrent consumer cannot dequeue the item
  // before its accounting is visible. Roll it back if enqueue fails.
  approximate_queue_sizes_[queue_index].fetch_add(1, std::memory_order_relaxed);
  bool ok = evict_queues_[queue_index].enqueue(queued);
  if (!ok) {
    approximate_queue_sizes_[queue_index].fetch_sub(1,
                                                    std::memory_order_relaxed);
    LOG_DEBUG("Eviction queue enqueue rejected under allocation pressure.");
    return false;
  }
  return true;
}

MemoryLimitPool::~MemoryLimitPool() {
  stop_background_evictor();
  drain_free_list();
}

bool MemoryLimitPool::drain_free_list() {
  trim_free_buffers(std::numeric_limits<size_t>::max());
  return release_all_slabs_locked();
}

size_t MemoryLimitPool::pick_shard() {
  // Keep each thread on one shard for locality.
  thread_local size_t idx = shard_seq_.fetch_add(1, std::memory_order_relaxed);
  return idx % kNumFreeShards;
}

bool MemoryLimitPool::try_reserve_used(size_t bytes) {
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  size_t used = used_size_.load(std::memory_order_relaxed);
  while (used <= capacity && bytes <= capacity - used) {
    if (used_size_.compare_exchange_weak(used, used + bytes,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

bool MemoryLimitPool::try_reserve_page_used(size_t bytes, bool required) {
  process_rss_.refresh();
  if (!required && process_rss_.pressure()) return false;
  const size_t page_limit = page_admission_limit();
  size_t used = used_size_.load(std::memory_order_relaxed);
  while (used <= page_limit && bytes <= page_limit - used) {
    if (used_size_.compare_exchange_weak(used, used + bytes,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

bool MemoryLimitPool::try_reserve_committed(size_t bytes, Backing backing,
                                            size_t materialization) {
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  size_t committed = committed_size_.load(std::memory_order_relaxed);
  while (committed <= capacity && bytes <= capacity - committed) {
    if (committed_size_.compare_exchange_weak(committed, committed + bytes,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
      bool admitted;
      if (backing == Backing::kPending) {
        admitted = process_rss_.try_reserve(bytes);
      } else if (backing == Backing::kPendingCopy) {
        admitted =
            process_rss_.try_acquire_with_pending_copy(bytes, materialization);
      } else {
        admitted = process_rss_.try_acquire(bytes, materialization);
      }
      if (!admitted) {
        committed_size_.fetch_sub(bytes, std::memory_order_relaxed);
        bg_cv_.notify_one();
        return false;
      }
      return true;
    }
  }
  return false;
}

size_t MemoryLimitPool::release_committed(
    size_t bytes, Backing backing,
    ProcessRssBudget::MappingProjection projection) {
  const size_t previous =
      committed_size_.fetch_sub(bytes, std::memory_order_relaxed);
  assert(previous >= bytes);
  if (backing == Backing::kPending) {
    process_rss_.finish_reservation(bytes, false);
  } else {
    process_rss_.release_projected_mapping(bytes, projection);
  }
  return previous;
}

bool MemoryLimitPool::try_reserve_allocation(size_t bytes, Backing backing,
                                             size_t materialization) {
  // Pages reserve logical slots before acquiring backing. Checking only
  // committed capacity here would spend those pending slots a second time.
  // Keep both counters individually bounded without holding a lock across
  // allocation, eviction, or writeback callbacks.
  if (!try_reserve_used(bytes)) return false;
  if (!try_reserve_committed(bytes, backing, materialization)) {
    release_buffer(nullptr, bytes);
    return false;
  }
  return true;
}

bool MemoryLimitPool::is_cacheable_buffer_size(size_t buffer_size) {
  return buffer_size == page_buffer_size();
}

char *MemoryLimitPool::pop_free_buffer(size_t start_shard) {
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    size_t shard = (start_shard + i) % kNumFreeShards;
    std::lock_guard<std::mutex> lock(free_shards_[shard].mutex);
    char *buffer = free_shards_[shard].head;
    if (buffer) {
      free_shards_[shard].head = *reinterpret_cast<char **>(buffer);
      free_shards_[shard].count.fetch_sub(1, std::memory_order_relaxed);
      return buffer;
    }
  }
  return nullptr;
}

void MemoryLimitPool::push_free_buffer(char *buffer, size_t shard) {
  shard %= kNumFreeShards;
  std::lock_guard<std::mutex> lock(free_shards_[shard].mutex);
  *reinterpret_cast<char **>(buffer) = free_shards_[shard].head;
  free_shards_[shard].head = buffer;
  free_shards_[shard].count.fetch_add(1, std::memory_order_relaxed);
}

char *MemoryLimitPool::acquire_slab_buffer() {
  static_assert(sizeof(ReclaimableSlab) <= kMinimumPageBytes,
                "slab metadata must fit in its header page");

  std::lock_guard<std::mutex> slabs_lock(slab_mutex_);
  auto acquire_from = [this](ReclaimableSlab *slab) -> char * {
    std::lock_guard<std::mutex> slab_lock(slab->mutex);
    uint16_t page_index = 0;
    bool reclaimed = false;
    if (slab->reclaimed_count != 0) {
      reclaimed = true;
      page_index = slab->reclaimed_pages[--slab->reclaimed_count];
    } else if (slab->next_unused <= slab->data_page_count) {
      page_index = slab->next_unused++;
    } else {
      return nullptr;
    }

    char *page = reinterpret_cast<char *>(slab) +
                 static_cast<size_t>(page_index) * slab->page_size;
    if (!commit_slab_page(page, slab->page_size)) {
      if (reclaimed) {
        slab->reclaimed_pages[slab->reclaimed_count++] = page_index;
      } else {
        --slab->next_unused;
      }
      return nullptr;
    }
    ++slab->committed_pages;
    process_rss_.commit_reusable_page(slab->page_size, slab->projections,
                                      page_index - 1);
    return page;
  };

  if (allocation_slab_ != nullptr) {
    if (char *page = acquire_from(allocation_slab_)) {
      return page;
    }
  }
  for (ReclaimableSlab *slab = slabs_; slab != nullptr; slab = slab->next) {
    if (slab == allocation_slab_) {
      continue;
    }
    if (char *page = acquire_from(slab)) {
      allocation_slab_ = slab;
      return page;
    }
  }

  const size_t page_size = page_buffer_size();
  // The header is resident-capable backing too. Keep it charged even after
  // all data pages have been returned or discarded, until slab teardown.
  // Do not evict here while holding allocator locks.
  if (!try_reserve_allocation(page_size, Backing::kPending)) {
    return nullptr;
  }
  metadata_used_size_.fetch_add(page_size, std::memory_order_relaxed);
  AlignedSlabMapping mapping = reserve_aligned_slab(page_size);
  if (mapping.base == nullptr) {
    release_fixed(page_size, &metadata_used_size_, Backing::kPending);
    LOG_ERROR("MemoryLimitPool: failed to reserve an aligned slab");
    return nullptr;
  }
  ReclaimableSlab *slab = new (mapping.base) ReclaimableSlab(
      this, mapping.reservation_base, mapping.reservation_size, page_size);
  slab->header_projection = process_rss_.finish_reservation(page_size, true);
  slab->next = slabs_;
  slabs_ = slab;
  allocation_slab_ = slab;
  slab_count_.fetch_add(1, std::memory_order_relaxed);
  slab_mapped_bytes_.fetch_add(mapping.reservation_size,
                               std::memory_order_relaxed);
  char *page = acquire_from(slab);
  if (page != nullptr) {
    return page;
  }

  ReclaimableSlab *next = slab->next;
  const auto projection = slab->header_projection;
  if (slab->unmap_empty()) {
    slabs_ = next;
    allocation_slab_ = nullptr;
    slab_count_.fetch_sub(1, std::memory_order_relaxed);
    slab_mapped_bytes_.fetch_sub(mapping.reservation_size,
                                 std::memory_order_relaxed);
    release_unmapped_metadata(page_size, projection);
  }
  return nullptr;
}

bool MemoryLimitPool::reclaim_slab_buffer(char *buffer) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(buffer);
  const uintptr_t slab_address = address & ~(kSlabAlignment - 1);
  auto *slab = reinterpret_cast<ReclaimableSlab *>(slab_address);
  const size_t offset = address - slab_address;
  if (slab->magic != ReclaimableSlab::kMagic || slab->owner != this ||
      offset < slab->page_size || offset >= kSlabBytes ||
      offset % slab->page_size != 0) {
    LOG_ERROR("MemoryLimitPool: invalid page returned to slab allocator");
    return false;
  }

  const auto page_index = static_cast<uint16_t>(offset / slab->page_size);
  std::lock_guard<std::mutex> slab_lock(slab->mutex);
  if (!discard_slab_page(buffer, slab->page_size)) {
    LOG_ERROR("MemoryLimitPool: failed to discard a free slab page");
    return false;
  }

  assert(slab->committed_pages != 0);
  --slab->committed_pages;
  const size_t previous =
      committed_size_.fetch_sub(slab->page_size, std::memory_order_relaxed);
  (void)previous;
  assert(previous >= slab->page_size);
  process_rss_.release_reusable_page(slab->page_size, slab->projections,
                                     page_index - 1);
  assert(slab->reclaimed_count < slab->data_page_count);
  slab->reclaimed_pages[slab->reclaimed_count++] = page_index;
  slab_reclaimed_pages_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool MemoryLimitPool::release_all_slabs_locked() {
  std::lock_guard<std::mutex> slabs_lock(slab_mutex_);
  allocation_slab_ = nullptr;
  while (slabs_) {
    ReclaimableSlab *slab = slabs_;
    if (slab->committed_pages != 0) return false;
    ReclaimableSlab *next = slab->next;
    const size_t page = slab->page_size;
    const size_t bytes = slab->reservation_size;
    const auto projection = slab->header_projection;
    if (!slab->unmap_empty()) return false;
    slabs_ = next;
    slab_count_.fetch_sub(1, std::memory_order_relaxed);
    slab_mapped_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    release_unmapped_metadata(page, projection);
  }
  return true;
}

size_t MemoryLimitPool::trim_free_buffers(size_t bytes_needed) {
  if (bytes_needed == 0) {
    return 0;
  }

  size_t released_bytes = trim_retained_blocks(bytes_needed);
  size_t shard = pick_shard();
  while (released_bytes < bytes_needed) {
    char *buffer = pop_free_buffer(shard);
    if (!buffer) {
      break;
    }
    // Only publish capacity after the physical page has been discarded.
    if (!reclaim_slab_buffer(buffer)) {
      push_free_buffer(buffer, shard);
      break;
    }
    released_bytes += page_buffer_size();
  }
  return released_bytes;
}

bool MemoryLimitPool::trim_idle_buffers() {
  // These buffers have no owners and no cached contents. Keep a small reuse
  // reserve, not the historical high-water allocation of a finished build.
  // The counters are only a hint: pop/unmap still obtains exclusive ownership
  // before discarding backing and refunding committed capacity.
  size_t idle = retained_idle_bytes_.load(std::memory_order_relaxed);
  for (const auto &shard : free_shards_) {
    idle += shard.count.load(std::memory_order_relaxed) * page_buffer_size();
  }
  const size_t reserve =
      process_rss_.pressure() ? 0 : reserve_margin(capacity());
  if (idle > reserve) {
    // Bound each maintenance pass. A retained mapping is released whole;
    // live pages and live retained blocks are never touched by this path.
    const size_t released =
        trim_free_buffers(std::min(idle - reserve, 64 * page_buffer_size()));
    if (released != 0) {
      {
        std::lock_guard<std::mutex> lock(capacity_mutex_);
      }
      capacity_cv_.notify_all();
      return true;
    }
  }
  return false;
}

bool MemoryLimitPool::configure_process_rss_limit(size_t bytes) {
  return configure_process_rss_limit(bytes, 0);
}

bool MemoryLimitPool::configure_process_rss_limit(size_t bytes,
                                                  size_t permanent_backing) {
  const bool configured = process_rss_.configure(
      bytes, &MemoryHelper::SelfRSS, NativeApplicationMemoryReader(),
      permanent_backing);
  bg_cv_.notify_one();
  return configured;
}

void MemoryLimitPool::set_memory_pressure(
    ProcessRssBudget::PressureLevel level) {
  process_rss_.set_external_pressure(level);
  bg_cv_.notify_one();
  capacity_cv_.notify_all();
}

void MemoryLimitPool::set_process_rss_reclaimer(
    std::function<void()> reclaimer) {
  auto callback =
      reclaimer
          ? std::make_shared<const std::function<void()>>(std::move(reclaimer))
          : nullptr;
  std::lock_guard<std::mutex> lock(process_reclaimer_mutex_);
  process_reclaimer_ = std::move(callback);
}

void MemoryLimitPool::reclaim_process_cache() {
  std::shared_ptr<const std::function<void()>> callback;
  {
    std::lock_guard<std::mutex> lock(process_reclaimer_mutex_);
    callback = process_reclaimer_;
  }
  // Copying this owner allocates nothing on the pressure path. It shares the
  // same pool across DSOs, and no pool lock is held while calling another
  // cache.
  if (callback) (*callback)();
}

int MemoryLimitPool::init(size_t pool_size) {
  // Queue storage can exist before a pool is configured. Construct it first
  // and adopt all retained allocations when publishing a non-zero budget.
  // Keep the existing preallocation policy and reject an insufficient budget.
  (void)BlockEvictionQueue::get_instance();
  std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  // Re-publishing the active process-wide budget is a safe no-op.
  if (initialized_.load(std::memory_order_acquire) &&
      pool_size_.load(std::memory_order_relaxed) == pool_size) {
    return 0;
  }
  const size_t used = used_size_.load(std::memory_order_relaxed);
  const size_t external = external_used_size_.load(std::memory_order_relaxed);
  const size_t metadata = metadata_used_size_.load(std::memory_order_relaxed);
  const size_t queue_charge =
      queue_charged_size_.load(std::memory_order_relaxed);
  const size_t allocator_charge =
      queue_charge +
      slab_count_.load(std::memory_order_relaxed) * page_buffer_size();
  if (used != allocator_charge || external != 0 ||
      metadata != allocator_charge) {
    LOG_ERROR(
        "MemoryLimitPool reinitialization rejected while cache memory is "
        "active: requested_capacity=%zu current_capacity=%zu used=%zu "
        "external_used=%zu metadata_used=%zu",
        pool_size, pool_size_.load(std::memory_order_relaxed), used, external,
        metadata);
    return -1;
  }

  // Tear down the background evictor first: it reads pool_size_ and touches
  // the free-list, both of which we are about to reset.
  const bool was_running = bg_running_.load();
  stop_background_evictor();
  std::unique_lock<std::mutex> queue_lock(queue_memory_mutex_);
  const size_t queue_bytes =
      queue_allocated_size_.load(std::memory_order_relaxed);
  auto restart_previous_pool = [&] {
    queue_lock.unlock();
    if (was_running) {
      try {
        start_background_evictor();
      } catch (const std::exception &e) {
        LOG_ERROR("Failed to restart the shared-cache evictor: %s", e.what());
      } catch (...) {
        LOG_ERROR("Failed to restart the shared-cache evictor");
      }
    }
    return -1;
  };
  if (pool_size != 0 && pool_size < queue_bytes) {
    LOG_ERROR(
        "Shared cache budget cannot fit eviction queues: capacity=%zu "
        "queue_bytes=%zu",
        pool_size, queue_bytes);
    return restart_previous_pool();
  }
  // No owner allocations remain. Keep the existing budget active until the
  // old allocator releases its backing, so failure leaves it usable.
  if (!drain_free_list() ||
      retained_block_bytes_.load(std::memory_order_relaxed) != 0) {
    return restart_previous_pool();
  }
  const size_t old_charge = queue_charged_size_.exchange(0);
  if (old_charge != 0) {
    release_metadata(old_charge);
  }
  pool_size_.store(0, std::memory_order_relaxed);
  process_rss_.configure(0, nullptr);
  set_process_rss_reclaimer({});
  pool_size_.store(pool_size, std::memory_order_relaxed);
  if (pool_size != 0) {
    queue_charged_size_.store(queue_bytes, std::memory_order_relaxed);
    metadata_used_size_.fetch_add(queue_bytes, std::memory_order_relaxed);
    used_size_.fetch_add(queue_bytes, std::memory_order_relaxed);
    committed_size_.fetch_add(queue_bytes, std::memory_order_relaxed);
    process_rss_.account(queue_bytes);
  }
  initialized_.store(true, std::memory_order_release);
  queue_lock.unlock();
  LOG_INFO("Shared cache initialized with capacity: %zu", pool_size);
  if (pool_size > 0) {
    try {
      start_background_evictor();
    } catch (const std::exception &e) {
      std::lock_guard<std::mutex> lock(queue_memory_mutex_);
      release_metadata(queue_charged_size_.exchange(0));
      pool_size_.store(0, std::memory_order_relaxed);
      initialized_.store(false, std::memory_order_release);
      LOG_ERROR("Failed to start the shared-cache evictor: %s", e.what());
      return -1;
    } catch (...) {
      std::lock_guard<std::mutex> lock(queue_memory_mutex_);
      release_metadata(queue_charged_size_.exchange(0));
      pool_size_.store(0, std::memory_order_relaxed);
      initialized_.store(false, std::memory_order_release);
      LOG_ERROR(
          "Failed to start the shared-cache evictor with an unknown error");
      return -1;
    }
  }
  return 0;
}

void MemoryLimitPool::start_background_evictor() {
  bool expected = false;
  if (!bg_running_.compare_exchange_strong(expected, true)) {
    return;  // already running
  }
  try {
    bg_thread_ = std::thread([this] { background_evict_loop(); });
  } catch (...) {
    bg_running_.store(false, std::memory_order_release);
    throw;
  }
}

void MemoryLimitPool::stop_background_evictor() {
  if (!bg_running_.exchange(false)) {
    return;  // not running
  }
  { std::lock_guard<std::mutex> lk(bg_mutex_); }
  bg_cv_.notify_all();
  if (bg_thread_.joinable()) {
    bg_thread_.join();
  }
}

void MemoryLimitPool::background_evict_loop() {
  using std::chrono::milliseconds;
  bool idle_progress = false;
  auto last_process_reclaim = std::chrono::steady_clock::time_point{};
  const auto page_admission_pressure = [this] {
    // Reserved headroom can block page admission below both general
    // watermarks. Keep the background queue progressing at that boundary,
    // but do not reclaim when only non-evictable fixed reservations remain.
    return used_size_.load(std::memory_order_relaxed) > fixed_used() &&
           (is_page_full() || process_rss_.pressure());
  };
  while (bg_running_.load()) {
    {
      std::unique_lock<std::mutex> lk(bg_mutex_);
      bg_cv_.wait_for(lk, milliseconds(5),
                      [this, &page_admission_pressure, &idle_progress] {
                        return !bg_running_.load() || idle_progress ||
                               should_background_reclaim() ||
                               page_admission_pressure();
                      });
    }
    if (!bg_running_.load()) break;
    if (pool_size_.load(std::memory_order_relaxed) == 0) continue;
    process_rss_.refresh();
    if (process_rss_.limit()) capacity_cv_.notify_all();
    const auto now = std::chrono::steady_clock::now();
    if (process_rss_.pressure() &&
        now - last_process_reclaim >= milliseconds(20)) {
      // RocksDB has a separate cache budget. Retire its unpinned entries too,
      // including when vector pages cannot make further reclamation progress.
      reclaim_process_cache();
      process_rss_.refresh(true);
      last_process_reclaim = now;
    }
    // Flush/compaction can retire many owners without reaching a watermark.
    // Reuse queue blocks before stale memberships cause unbounded growth.
    const size_t retired =
        BlockEvictionQueue::get_instance().cleanup_dead_entries(256);
    // Page/entry retirement can leave committed backing resident even when
    // used is far below the eviction watermark. Return only idle backing.
    // After the final page owner closes, finish returning idle backing to the
    // existing reuse reserve without sleeping between every small batch. An
    // active cache retains its paced page maintenance. Keep removing retired
    // memberships while cleanup makes progress, even below the watermarks.
    // No progress (including an OS discard failure) restores the wait.
    const bool trimmed = trim_idle_buffers();
    idle_progress =
        retired != 0 ||
        (trimmed && used_size_.load(std::memory_order_relaxed) == fixed_used());
    const size_t low = low_watermark();
    if (used_size_.load() > low || page_admission_pressure()) {
      bg_evict_rounds_.fetch_add(1, std::memory_order_relaxed);
    }
    // Reclaim to the low watermark and make room for page admission so the
    // foreground path finds ready buffers without a deep inline queue walk.
    while (bg_running_.load() &&
           (used_size_.load() > low || page_admission_pressure())) {
      size_t n = BlockEvictionQueue::get_instance().batch_recycle(64);
      if (n == 0) {
        // Back off when pressure remains but eviction makes no progress.
        bg_no_progress_sleeps_.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<std::mutex> lk(bg_mutex_);
        bg_cv_.wait_for(lk, milliseconds(5),
                        [this] { return !bg_running_.load(); });
        break;
      }
      bg_evicted_buffers_.fetch_add(n, std::memory_order_relaxed);
      if (process_rss_.pressure()) {
        trim_idle_buffers();
        process_rss_.refresh(true);
      }
    }
  }
}

char *MemoryLimitPool::pop_reusable_buffer(size_t shard, size_t bytes) {
  char *buffer = pop_free_buffer(shard);
  if (!buffer) return nullptr;
  const uintptr_t address = reinterpret_cast<uintptr_t>(buffer);
  const uintptr_t slab_address = address & ~(kSlabAlignment - 1);
  auto *slab = reinterpret_cast<ReclaimableSlab *>(slab_address);
  const size_t page_index = (address - slab_address) / bytes;
  if (process_rss_.try_materialize_reusable_page(bytes, slab->projections,
                                                 page_index - 1))
    return buffer;
  push_free_buffer(buffer, shard);
  return nullptr;
}

bool MemoryLimitPool::try_acquire_buffer(const size_t buffer_size,
                                         char *&buffer) {
  return try_acquire_buffer(buffer_size, buffer, false);
}

bool MemoryLimitPool::try_acquire_buffer(size_t buffer_size, char *&buffer,
                                         bool required) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  buffer = nullptr;
  const bool cacheable = is_cacheable_buffer_size(buffer_size);
  if (buffer_size == 0 ||
      !(cacheable ? try_reserve_page_used(buffer_size, required)
                  : try_reserve_used(buffer_size))) {
    // Out of budget: wake the background evictor so the next attempt is
    // more likely to find a free buffer without inline eviction.
    high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
    bg_cv_.notify_one();
    return false;
  }

  if (cacheable) {
    buffer = pop_reusable_buffer(pick_shard(), buffer_size);
    if (buffer) {
      alloc_from_freelist_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  if (!try_reserve_committed(buffer_size, Backing::kPending)) {
    // This is primarily useful for a non-cacheable size. For the normal page
    // size, all currently visible free buffers were already checked above.
    trim_free_buffers(buffer_size);
    if (!try_reserve_committed(buffer_size, Backing::kPending)) {
      release_buffer(nullptr, buffer_size);
      high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
  }
  buffer = cacheable ? acquire_slab_buffer()
                     : static_cast<char *>(ailego_aligned_malloc(
                           buffer_size, kMinimumPageBytes));
  if (!buffer) {
    release_committed(buffer_size, Backing::kPending);
    release_buffer(nullptr, buffer_size);
    return false;
  }
  if (!cacheable) process_rss_.finish_reservation(buffer_size, true);
  alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool MemoryLimitPool::wait_for_available(const size_t buffer_size,
                                         std::chrono::milliseconds timeout) {
  return wait_for_available(buffer_size, timeout, false);
}

bool MemoryLimitPool::wait_for_available(size_t buffer_size,
                                         std::chrono::milliseconds timeout,
                                         bool required) {
  return wait_for_available_impl(buffer_size, timeout, true, required);
}

bool MemoryLimitPool::wait_for_available_impl(size_t buffer_size,
                                              std::chrono::milliseconds timeout,
                                              bool page_admission,
                                              bool required) {
  if (buffer_size == 0) {
    return true;
  }
  capacity_waits_.fetch_add(1, std::memory_order_relaxed);
  std::unique_lock<std::mutex> lock(capacity_mutex_);
  const bool available = capacity_cv_.wait_for(
      lock, timeout, [this, buffer_size, page_admission, required] {
        const size_t capacity = pool_size_.load(std::memory_order_relaxed);
        const size_t used = used_size_.load(std::memory_order_relaxed);
        if (capacity < used || buffer_size > capacity - used) {
          return false;
        }
        if (buffer_size > process_rss_.available()) return false;
        if (page_admission && !required && process_rss_.pressure())
          return false;
        // A page can be denied below the total capacity: unconsumed external
        // cache headroom is not available to page admission. Use the same
        // page-limit predicate as eviction instead of turning backpressure
        // into immediate successful waits while pages still cannot fit.
        if (!page_admission || !is_cacheable_buffer_size(buffer_size))
          return true;
        const size_t limit = page_admission_limit();
        if (used >= limit || buffer_size > limit - used) return false;
        for (const auto &shard : free_shards_)
          if (shard.count.load(std::memory_order_relaxed) != 0) return true;
        const size_t committed =
            committed_size_.load(std::memory_order_relaxed);
        const size_t fixed = fixed_used();
        const size_t backed = committed > fixed ? committed - fixed : 0;
        const size_t slots_per_slab = slab_size() / buffer_size - 1;
        const size_t slabs = slab_count_.load(std::memory_order_relaxed);
        // A full slab needs another resident header as well as the payload.
        // Otherwise an apparent one-page allowance causes immediate waits
        // to spin out before the dirty-writeback worker can run. Outstanding
        // non-page backing only makes this hint conservative; allocation is
        // retried after a timeout, without taking allocator locks under CV.
        const bool has_slot =
            slabs != 0 && backed / buffer_size / slabs < slots_per_slab;
        const size_t extra = has_slot ? 0 : buffer_size;
        return committed <= capacity && buffer_size <= capacity - committed &&
               extra <= capacity - committed - buffer_size &&
               extra <= capacity - used - buffer_size;
      });
  if (!available) {
    capacity_wait_timeouts_.fetch_add(1, std::memory_order_relaxed);
  }
  return available;
}

bool MemoryLimitPool::try_charge_external(const size_t buffer_size) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed(buffer_size, &external_used_size_);
}

bool MemoryLimitPool::try_charge_workspace(const size_t buffer_size) {
  return try_charge_workspace(buffer_size, 0);
}

bool MemoryLimitPool::try_charge_workspace(size_t buffer_size,
                                           size_t materialization) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed_with_wait(buffer_size, &external_used_size_,
                                    Backing::kLive, materialization);
}

bool MemoryLimitPool::try_charge_workspace_with_pending_copy(
    size_t bytes, size_t materialization) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed_with_wait(bytes, &external_used_size_,
                                    Backing::kPendingCopy, materialization);
}

bool MemoryLimitPool::try_charge_metadata(const size_t buffer_size) {
  return try_charge_metadata(buffer_size, 0);
}

bool MemoryLimitPool::try_charge_metadata(size_t buffer_size,
                                          size_t materialization) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_fixed_with_wait(buffer_size, &metadata_used_size_,
                                    Backing::kLive, materialization);
}

bool MemoryLimitPool::try_charge_metadata_without_reclaim(size_t bytes) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  return try_charge_locked(bytes, &metadata_used_size_, Backing::kLive);
}

bool MemoryLimitPool::try_charge_locked(size_t bytes,
                                        std::atomic<size_t> *counter,
                                        Backing backing) {
  if (!bytes) return true;
  if (!try_reserve_allocation(bytes, backing)) {
    // Discard only idle backing; never call an eviction owner under its lock.
    trim_free_buffers(bytes);
    if (!try_reserve_allocation(bytes, backing)) return false;
  }
  counter->fetch_add(bytes, std::memory_order_relaxed);
  bg_cv_.notify_one();
  return true;
}

MemoryLimitPool::PendingAllocation::PendingAllocation(MemoryLimitPool &pool,
                                                      Kind kind)
    : pool_(pool),
      kind_(kind),
      counter_(kind == Kind::kMetadata || kind == Kind::kLockedMetadata
                   ? &pool.metadata_used_size_
                   : &pool.external_used_size_) {}

MemoryLimitPool::PendingAllocation::~PendingAllocation() {
  if (bytes_) pool_.release_fixed(bytes_, counter_, Backing::kPending);
}

bool MemoryLimitPool::PendingAllocation::reserve(size_t bytes) {
  assert(bytes_ == 0);
  if (!bytes) return true;
  std::shared_lock<std::shared_mutex> lifecycle_lock(pool_.lifecycle_mutex_);
  bool reserved = false;
  switch (kind_) {
    case Kind::kExternal:
      reserved = pool_.try_charge_fixed(bytes, counter_, Backing::kPending);
      break;
    case Kind::kWorkspace:
    case Kind::kMetadata:
      reserved =
          pool_.try_charge_fixed_with_wait(bytes, counter_, Backing::kPending);
      break;
    case Kind::kLockedMetadata:
    case Kind::kLockedExternal:
      reserved = pool_.try_charge_locked(bytes, counter_, Backing::kPending);
      break;
  }
  if (reserved) bytes_ = bytes;
  return reserved;
}

void MemoryLimitPool::PendingAllocation::commit(size_t bytes) {
  (void)commit_mapping(bytes);
}

ProcessRssBudget::MaterializationStamp
MemoryLimitPool::PendingAllocation::commit_mapping() {
  return commit_mapping(bytes_);
}

ProcessRssBudget::MaterializationStamp
MemoryLimitPool::PendingAllocation::commit_mapping(size_t bytes) {
  assert(bytes <= bytes_);
  if (!bytes) return {};
  auto stamp = pool_.process_rss_.finish_reservation(bytes, true);
  bytes_ -= bytes;
  return stamp;
}

bool MemoryLimitPool::try_charge_fixed_with_wait(const size_t buffer_size,
                                                 std::atomic<size_t> *counter,
                                                 Backing backing,
                                                 size_t materialization) {
  if (try_charge_fixed(buffer_size, counter, backing, materialization)) {
    return true;
  }
  if (materialization &&
      !process_rss_.materialization_allowed(materialization)) {
    return false;
  }
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  const size_t fixed = fixed_used();
  if (fixed > capacity || buffer_size > capacity - fixed) {
    return false;
  }
  size_t waiting = workspace_waiting_size_.load(std::memory_order_relaxed);
  do {
    if (buffer_size > std::numeric_limits<size_t>::max() - waiting)
      return false;
  } while (!workspace_waiting_size_.compare_exchange_weak(
      waiting, waiting + buffer_size, std::memory_order_relaxed));
  auto release_priority = ScopeGuard::Make([this, buffer_size] {
    workspace_waiting_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    capacity_cv_.notify_all();
  });
  bg_cv_.notify_one();
  // Establish priority before another reclaim pass. A cache page may already
  // have won the preceding race, but cannot repeatedly consume the released
  // headroom while this mandatory consumer is waiting.
  if (try_charge_fixed(buffer_size, counter, backing, materialization))
    return true;
  // Dirty writeback and CLOCK aging can need more than one bounded pass.
  // A zero-page batch is not proof that the queue is unreclaimable: it may
  // only have cleared reference bits or demoted recently used entries.
  // Keep waiting while candidates remain, but never beyond 50 waits. Empty
  // queues with no capacity progress still fail after one idle wait.
  // Ordinary external-cache admission never enters this waiting path.
  size_t previous_available = available();
  for (size_t attempt = 0; attempt < 50; ++attempt) {
    (void)wait_for_available_impl(buffer_size, std::chrono::milliseconds(100),
                                  false);
    if (try_charge_fixed(buffer_size, counter, backing, materialization))
      return true;
    if (materialization &&
        !process_rss_.materialization_allowed(materialization))
      return false;
    const size_t current_available = available();
    if (current_available <= previous_available) {
      const auto queues = BlockEvictionQueue::get_instance().stats();
      if (std::all_of(queues.approximate_queue_sizes.begin(),
                      queues.approximate_queue_sizes.end(),
                      [](size_t depth) { return depth == 0; }))
        return false;
    }
    previous_available = current_available;
  }
  return false;
}

bool MemoryLimitPool::try_charge_fixed(const size_t buffer_size,
                                       std::atomic<size_t> *counter,
                                       Backing backing,
                                       size_t materialization) {
  if (buffer_size == 0) {
    return !materialization ||
           (backing == Backing::kPendingCopy
                ? process_rss_.try_reserve_materialization(materialization)
                : process_rss_.try_materialize(materialization));
  }
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  if (capacity == 0 || buffer_size > capacity) {
    high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  while (true) {
    if (try_reserve_allocation(buffer_size, backing, materialization)) {
      counter->fetch_add(buffer_size, std::memory_order_relaxed);
      bg_cv_.notify_one();
      return true;
    }

    // The measured gate can deny a materialization larger than the lifetime
    // charge. In that case there is no point in spinning on the smaller delta.
    if (materialization &&
        !process_rss_.materialization_allowed(materialization)) {
      return false;
    }

    // Logical admission may fail even with backing headroom while pages are
    // in flight. Do not spin on committed capacity alone in that case.
    const size_t reserved =
        std::max(used_size_.load(std::memory_order_relaxed),
                 committed_size_.load(std::memory_order_relaxed));
    size_t available = std::min(reserved >= capacity ? 0 : capacity - reserved,
                                process_rss_.available());
    if (available >= buffer_size) {
      continue;
    }
    if (trim_free_buffers(buffer_size - available) != 0) {
      process_rss_.refresh(true);
      continue;
    }

    // Reclaim until the reservation fits or eviction stops making progress.
    if (BlockEvictionQueue::get_instance().batch_recycle(256) == 0) {
      high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (process_rss_.limit()) {
      trim_free_buffers(buffer_size - available);
      process_rss_.refresh(true);
    }
  }
}

void MemoryLimitPool::release_buffer(char *buffer, const size_t buffer_size) {
  if (!buffer) {
    size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    (void)prev;
    assert(prev >= buffer_size);
    { std::lock_guard<std::mutex> lock(capacity_mutex_); }
    capacity_cv_.notify_one();
    return;
  }
  if (!is_cacheable_buffer_size(buffer_size)) {
    ailego_free(buffer);
    size_t committed_prev = release_committed(buffer_size);
    (void)committed_prev;
    assert(committed_prev >= buffer_size);
    size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    (void)prev;
    assert(prev >= buffer_size);
    { std::lock_guard<std::mutex> lock(capacity_mutex_); }
    capacity_cv_.notify_one();
    return;
  }
  // Owner retirement can return hundreds of MiB before the paced background
  // maintenance catches up. Keep the existing small reuse reserve, but do
  // not publish excess idle backing only to discard it several seconds later.
  // The returning page is exclusively owned here; cached/pinned pages are
  // never touched. Concurrent returns may overshoot the hint by one page per
  // returner. A discard failure keeps both the backing and its charge.
  const size_t reserve =
      process_rss_.pressure() ? 0 : reserve_margin(capacity());
  size_t committed = committed_size_.load(std::memory_order_relaxed);
  size_t used = used_size_.load(std::memory_order_relaxed);
  size_t released_retained = 0;
  if (committed > used && committed - used >= reserve &&
      retained_idle_bytes_.load(std::memory_order_relaxed) != 0) {
    // Idle mappings and free pages share one reserve. Prefer retiring an
    // exclusively idle mapping before discarding a page in an active reuse
    // cycle; otherwise old mappings can force a discard/refault on every
    // cache eviction. Unmapping still precedes any backing-budget refund.
    released_retained = trim_retained_blocks(buffer_size);
    committed = committed_size_.load(std::memory_order_relaxed);
    used = used_size_.load(std::memory_order_relaxed);
  }
  const bool discarded = committed > used && committed - used >= reserve &&
                         reclaim_slab_buffer(buffer);
  if (!discarded) {
    push_free_buffer(buffer, pick_shard());
  }
  // Publish reusable backing or physically discard it before releasing used.
  size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)prev;
  assert(prev >= buffer_size);
  { std::lock_guard<std::mutex> lock(capacity_mutex_); }
  if (released_retained != 0) {
    capacity_cv_.notify_all();
  } else {
    capacity_cv_.notify_one();
  }
}

void MemoryLimitPool::release_external(const size_t buffer_size) {
  release_fixed(buffer_size, &external_used_size_);
}

void MemoryLimitPool::release_unmapped_external(
    size_t bytes, ProcessRssBudget::MaterializationStamp stamp) {
  release_fixed(bytes, &external_used_size_, Backing::kLive,
                ProcessRssBudget::MappingProjection::Full(bytes, stamp));
}

void MemoryLimitPool::release_metadata(const size_t buffer_size) {
  release_fixed(buffer_size, &metadata_used_size_);
}

void MemoryLimitPool::release_unmapped_metadata(
    size_t bytes, ProcessRssBudget::MaterializationStamp stamp) {
  release_fixed(bytes, &metadata_used_size_, Backing::kLive,
                ProcessRssBudget::MappingProjection::Full(bytes, stamp));
}

void MemoryLimitPool::release_fixed(
    size_t buffer_size, std::atomic<size_t> *counter, Backing backing,
    ProcessRssBudget::MappingProjection projection) {
  if (buffer_size == 0) {
    return;
  }
  // Unconditional subtract: single RMW instead of a CAS loop.
  size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)prev;
  assert(prev >= buffer_size);
  size_t counter_prev =
      counter->fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)counter_prev;
  assert(counter_prev >= buffer_size);
  size_t committed_prev = release_committed(buffer_size, backing, projection);
  (void)committed_prev;
  assert(committed_prev >= buffer_size);
  { std::lock_guard<std::mutex> lock(capacity_mutex_); }
  capacity_cv_.notify_all();
}

bool MemoryLimitPool::is_full() {
  return used_size_.load(std::memory_order_relaxed) >=
         pool_size_.load(std::memory_order_relaxed);
}

size_t MemoryLimitPool::batch_acquire_buffers(size_t buffer_size, char **out,
                                              size_t count) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  if (count == 0 || buffer_size == 0) return 0;
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  const bool cacheable = is_cacheable_buffer_size(buffer_size);
  const size_t admission_limit = cacheable ? page_admission_limit() : capacity;
  process_rss_.refresh();
  if (cacheable && process_rss_.pressure()) return 0;
  size_t total_size = 0;
  size_t actual_count = count;
  size_t expected, desired;
  do {
    expected = used_size_.load(std::memory_order_relaxed);
    if (expected >= admission_limit) return 0;
    size_t avail = (admission_limit - expected) / buffer_size;
    if (avail == 0) return 0;
    if (avail < actual_count) actual_count = avail;
    total_size = actual_count * buffer_size;
    desired = expected + total_size;
  } while (!used_size_.compare_exchange_weak(
      expected, desired, std::memory_order_relaxed, std::memory_order_relaxed));

  size_t acquired = 0;
  size_t s = pick_shard();
  if (cacheable) {
    while (acquired < actual_count) {
      out[acquired] = pop_reusable_buffer(s, buffer_size);
      if (!out[acquired]) {
        break;
      }
      ++acquired;
    }
    alloc_from_freelist_.fetch_add(acquired, std::memory_order_relaxed);
  }

  while (acquired < actual_count) {
    if (!try_reserve_committed(buffer_size, Backing::kPending)) {
      trim_free_buffers(buffer_size);
      if (!try_reserve_committed(buffer_size, Backing::kPending)) {
        break;
      }
    }
    out[acquired] = cacheable ? acquire_slab_buffer()
                              : static_cast<char *>(ailego_aligned_malloc(
                                    buffer_size, kMinimumPageBytes));
    if (!out[acquired]) {
      release_committed(buffer_size, Backing::kPending);
      break;
    }
    if (!cacheable) process_rss_.finish_reservation(buffer_size, true);
    alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
    ++acquired;
  }
  if (acquired < actual_count) {
    release_buffer(nullptr, (actual_count - acquired) * buffer_size);
  }
  return acquired;
}

MemoryLimitPool::PoolStats MemoryLimitPool::stats() const {
  PoolStats s;
  s.pool_size = pool_size_.load(std::memory_order_relaxed);
  s.used = used_size_.load(std::memory_order_relaxed);
  s.committed = committed_size_.load(std::memory_order_relaxed);
  s.external_used = external_used_size_.load(std::memory_order_relaxed);
  s.metadata_used = metadata_used_size_.load(std::memory_order_relaxed);
  s.queue_metadata_used = queue_metadata_used();
  s.queue_allocated = queue_allocated();
  s.retained_block_bytes =
      retained_block_bytes_.load(std::memory_order_relaxed);
  s.retained_idle_bytes = retained_idle_bytes_.load(std::memory_order_relaxed);
  s.retained_block_reuses =
      retained_block_reuses_.load(std::memory_order_relaxed);
  const size_t fixed = s.external_used + s.metadata_used;
  s.page_used = s.used >= fixed ? s.used - fixed : 0;
  size_t free_buffers = 0;
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    free_buffers += free_shards_[i].count.load(std::memory_order_relaxed);
  }
  s.free_buffers = free_buffers;
  s.slab_count = slab_count_.load(std::memory_order_relaxed);
  s.slab_mapped_bytes = slab_mapped_bytes_.load(std::memory_order_relaxed);
  s.slab_header_bytes = s.slab_count * page_buffer_size();
  s.alloc_from_freelist = alloc_from_freelist_.load(std::memory_order_relaxed);
  s.alloc_from_slab = alloc_from_slab_.load(std::memory_order_relaxed);
  s.slab_reclaimed_pages =
      slab_reclaimed_pages_.load(std::memory_order_relaxed);
  s.bg_evict_rounds = bg_evict_rounds_.load(std::memory_order_relaxed);
  s.bg_evicted_buffers = bg_evicted_buffers_.load(std::memory_order_relaxed);
  s.bg_no_progress_sleeps =
      bg_no_progress_sleeps_.load(std::memory_order_relaxed);
  s.high_watermark_hits = high_watermark_hits_.load(std::memory_order_relaxed);
  s.capacity_waits = capacity_waits_.load(std::memory_order_relaxed);
  s.capacity_wait_timeouts =
      capacity_wait_timeouts_.load(std::memory_order_relaxed);
  return s;
}

void MemoryLimitPool::log_stats() const {
  PoolStats s = stats();
  LOG_INFO(
      "Shared cache stats: capacity=%llu used=%llu committed=%llu "
      "page_used=%llu external_used=%llu metadata_used=%llu "
      "queue_metadata_used=%llu queue_allocated=%llu "
      "free_buffers=%llu slab_count=%llu slab_mapped_bytes=%llu "
      "slab_header_bytes=%llu "
      "alloc_from_freelist=%llu alloc_from_slab=%llu "
      "slab_reclaimed_pages=%llu "
      "bg_evict_rounds=%llu bg_evicted_buffers=%llu "
      "bg_no_progress_sleeps=%llu "
      "high_watermark_hits=%llu capacity_waits=%llu "
      "capacity_wait_timeouts=%llu",
      static_cast<unsigned long long>(s.pool_size),
      static_cast<unsigned long long>(s.used),
      static_cast<unsigned long long>(s.committed),
      static_cast<unsigned long long>(s.page_used),
      static_cast<unsigned long long>(s.external_used),
      static_cast<unsigned long long>(s.metadata_used),
      static_cast<unsigned long long>(s.queue_metadata_used),
      static_cast<unsigned long long>(s.queue_allocated),
      static_cast<unsigned long long>(s.free_buffers),
      static_cast<unsigned long long>(s.slab_count),
      static_cast<unsigned long long>(s.slab_mapped_bytes),
      static_cast<unsigned long long>(s.slab_header_bytes),
      static_cast<unsigned long long>(s.alloc_from_freelist),
      static_cast<unsigned long long>(s.alloc_from_slab),
      static_cast<unsigned long long>(s.slab_reclaimed_pages),
      static_cast<unsigned long long>(s.bg_evict_rounds),
      static_cast<unsigned long long>(s.bg_evicted_buffers),
      static_cast<unsigned long long>(s.bg_no_progress_sleeps),
      static_cast<unsigned long long>(s.high_watermark_hits),
      static_cast<unsigned long long>(s.capacity_waits),
      static_cast<unsigned long long>(s.capacity_wait_timeouts));
}

}  // namespace ailego
}  // namespace zvec

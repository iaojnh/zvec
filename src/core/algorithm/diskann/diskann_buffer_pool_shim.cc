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

// Implementation of the DiskAnn <-> VecBufferPool bridge. Kept in a dedicated
// TU so it can include vector_page_table.h (which brings the
// <zvec/ailego/io/libaio_loader.h> definitions) without colliding with the
// DiskAnn reader's <ailego/io/libaio_loader.h> copy.
#include "diskann_buffer_pool_shim.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <zvec/ailego/buffer/vector_page_table.h>

namespace zvec {
namespace core {

int diskann_buffer_pool_read(ailego::VecBufferPool *pool,
                             const uint64_t *offsets, const uint64_t *lens,
                             void *const *bufs, size_t count) {
  if (pool == nullptr) {
    return kDiskAnnBufferPoolReadError;
  }
  if (count != 0 &&
      (offsets == nullptr || lens == nullptr || bufs == nullptr)) {
    return kDiskAnnBufferPoolInvalidArg;
  }

  // DiskANN does not include VecBufferPool's profiling types in its reader
  // translation unit because both stacks define libaio symbols. Install a
  // local profile here so diagnostic runs still collect physical read counts.
  // If a query-level profile is already bound, contribute to that instead.
  std::unique_ptr<ailego::BufferPoolIoProfile> local_profile;
  ailego::BufferPoolIoProfileBinding previous_binding;
  const bool owns_profile = pool->io_profile_enabled() &&
                            pool->current_thread_io_profile() == nullptr;
  if (owns_profile) {
    local_profile = std::make_unique<ailego::BufferPoolIoProfile>();
    previous_binding = pool->bind_thread_io_profile(local_profile.get());
  }
  struct ProfileGuard {
    ailego::VecBufferPool *pool;
    ailego::BufferPoolIoProfile *profile;
    ailego::BufferPoolIoProfileBinding previous;
    bool active;

    ~ProfileGuard() {
      if (!active) return;
      ailego::VecBufferPool::restore_thread_io_profile(previous);
      pool->merge_io_profile(*profile);
    }
  } profile_guard{pool, local_profile.get(), previous_binding, owns_profile};

  const size_t page_size = ailego::kVectorPageSize;

  // Expand every requested byte range into the backing pages it overlaps.
  // Segment offsets in a FileDumper container are not guaranteed to be page
  // aligned even though each DiskANN sector is 4KB.
  struct PageCopy {
    size_t request_index;
    size_t destination_offset;
    size_t page_offset;
    size_t size;
  };
  std::vector<ailego::block_id_t> page_ids;
  std::vector<PageCopy> mappings;
  page_ids.reserve(count);
  mappings.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    const uint64_t offset = offsets[i];
    const uint64_t len = lens[i];
    if (bufs[i] == nullptr || len == 0 ||
        offset > std::numeric_limits<size_t>::max() ||
        len > std::numeric_limits<size_t>::max() ||
        offset > std::numeric_limits<uint64_t>::max() - len) {
      return kDiskAnnBufferPoolInvalidArg;
    }
    size_t copied = 0;
    while (copied < static_cast<size_t>(len)) {
      const size_t absolute = static_cast<size_t>(offset) + copied;
      const size_t page_offset = absolute % page_size;
      const size_t copy_size =
          std::min(static_cast<size_t>(len) - copied, page_size - page_offset);
      page_ids.push_back(static_cast<ailego::block_id_t>(absolute / page_size));
      mappings.push_back(PageCopy{i, copied, page_offset, copy_size});
      copied += copy_size;
    }
  }

  if (page_ids.empty()) {
    return kDiskAnnBufferPoolOk;
  }

  std::vector<char *> pages(page_ids.size(), nullptr);
  if (!pool->acquire_pages(page_ids.data(), page_ids.size(), pages.data())) {
    return kDiskAnnBufferPoolReadError;
  }

  // Release every acquired pin exactly once on any exit path.
  struct PinGuard {
    ailego::VecBufferPool *pool;
    const ailego::block_id_t *ids;
    size_t count;
    ~PinGuard() {
      pool->release_pages(ids, count);
    }
  } guard{pool, page_ids.data(), page_ids.size()};

  for (size_t i = 0; i < pages.size(); ++i) {
    const auto &m = mappings[i];
    std::memcpy(
        static_cast<char *>(bufs[m.request_index]) + m.destination_offset,
        pages[i] + m.page_offset, m.size);
  }

  return kDiskAnnBufferPoolOk;
}

int diskann_buffer_pool_read_bypass(ailego::VecBufferPool *pool,
                                    const uint64_t *offsets,
                                    const uint64_t *lens, void *const *bufs,
                                    size_t count) {
  if (pool == nullptr) {
    return kDiskAnnBufferPoolReadError;
  }
  if (count != 0 &&
      (offsets == nullptr || lens == nullptr || bufs == nullptr)) {
    return kDiskAnnBufferPoolInvalidArg;
  }

  for (size_t i = 0; i < count; ++i) {
    if (bufs[i] == nullptr || lens[i] == 0 ||
        offsets[i] > std::numeric_limits<size_t>::max() ||
        lens[i] > std::numeric_limits<size_t>::max() ||
        offsets[i] > std::numeric_limits<uint64_t>::max() - lens[i]) {
      return kDiskAnnBufferPoolInvalidArg;
    }
    if (!pool->read_range_bypass(static_cast<size_t>(offsets[i]),
                                 static_cast<size_t>(lens[i]),
                                 static_cast<char *>(bufs[i]))) {
      return kDiskAnnBufferPoolReadError;
    }
  }
  return kDiskAnnBufferPoolOk;
}

DiskAnnBufferPoolOverlapStats diskann_buffer_pool_manage_overlap(
    ailego::VecBufferPool *pool, const uint64_t *page_ids, size_t count,
    bool evict) {
  DiskAnnBufferPoolOverlapStats stats;
  if (pool == nullptr || page_ids == nullptr || count == 0) {
    return stats;
  }

  std::vector<ailego::block_id_t> unique_pages;
  unique_pages.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    unique_pages.push_back(static_cast<ailego::block_id_t>(page_ids[i]));
  }
  std::sort(unique_pages.begin(), unique_pages.end());
  unique_pages.erase(std::unique(unique_pages.begin(), unique_pages.end()),
                     unique_pages.end());

  stats.unique_pages = unique_pages.size();
  for (ailego::block_id_t page_id : unique_pages) {
    if (!pool->is_page_resident(page_id)) {
      continue;
    }
    ++stats.resident_pages_before;
    if (evict && pool->evict_page(page_id)) {
      ++stats.evicted_pages;
    }
    if (pool->is_page_resident(page_id)) {
      ++stats.resident_pages_after;
    }
  }
  return stats;
}

}  // namespace core
}  // namespace zvec

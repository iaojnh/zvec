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
#include <cstring>
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

  const size_t page_size = ailego::kVectorPageSize;

  // Expand each sector request into one or more page ids. A DiskAnn sector is
  // 4KB, matching a single VecBufferPool page, so offset/len must be page
  // aligned.
  std::vector<ailego::block_id_t> page_ids;
  std::vector<std::pair<size_t, size_t>> mappings;  // (req_idx, page_idx)
  page_ids.reserve(count);
  mappings.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    const uint64_t offset = offsets[i];
    const uint64_t len = lens[i];
    if (bufs[i] == nullptr || len == 0 || offset % page_size != 0 ||
        len % page_size != 0) {
      return kDiskAnnBufferPoolInvalidArg;
    }
    const size_t page_count = static_cast<size_t>(len / page_size);
    for (size_t p = 0; p < page_count; ++p) {
      page_ids.push_back(
          static_cast<ailego::block_id_t>(offset / page_size + p));
      mappings.emplace_back(i, p);
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
    std::memcpy(static_cast<char *>(bufs[m.first]) + m.second * page_size,
                pages[i], page_size);
  }

  return kDiskAnnBufferPoolOk;
}

}  // namespace core
}  // namespace zvec

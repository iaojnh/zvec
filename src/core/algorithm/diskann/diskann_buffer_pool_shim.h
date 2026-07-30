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

#include <cstddef>
#include <cstdint>

namespace zvec {
namespace ailego {
class VecBufferPool;
}  // namespace ailego

namespace core {

// Result codes for diskann_buffer_pool_read. Kept as plain ints so this shim
// stays free of both the DiskAnn/libaio headers and the index framework.
enum DiskAnnBufferPoolReadStatus {
  kDiskAnnBufferPoolOk = 0,
  kDiskAnnBufferPoolInvalidArg = 1,
  kDiskAnnBufferPoolReadError = 2,
};

struct DiskAnnBufferPoolOverlapStats {
  uint64_t unique_pages{0};
  uint64_t resident_pages_before{0};
  uint64_t evicted_pages{0};
  uint64_t resident_pages_after{0};
};

// Satisfy a batch of aligned sector reads through a VecBufferPool paged cache.
// Each (offset[i], len[i]) must be page-aligned; every page is acquired in one
// batch (preserving the pool's batched miss / AIO path), copied into buf[i],
// then released. Duplicate page ids across requests are fine: each occurrence
// is pinned and released once. This lives in its own TU so it can include
// vector_page_table.h without colliding with the DiskAnn reader's libaio
// headers.
int diskann_buffer_pool_read(ailego::VecBufferPool *pool,
                             const uint64_t *offsets, const uint64_t *lens,
                             void *const *bufs, size_t count);

// Read directly from the pool's backing file without admitting or pinning
// pages. Static DiskANN cache construction uses this after reserving its own
// memory, so it cannot deadlock against that reservation for page buffers.
int diskann_buffer_pool_read_bypass(ailego::VecBufferPool *pool,
                                    const uint64_t *offsets,
                                    const uint64_t *lens, void *const *bufs,
                                    size_t count);

// Observe pages copied into DiskANN's static node cache and optionally evict
// their now-duplicated Buffer Pool copies. Duplicate page ids are accepted.
DiskAnnBufferPoolOverlapStats diskann_buffer_pool_manage_overlap(
    ailego::VecBufferPool *pool, const uint64_t *page_ids, size_t count,
    bool evict);

}  // namespace core
}  // namespace zvec

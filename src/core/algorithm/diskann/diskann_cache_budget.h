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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <zvec/core/framework/index_meta.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

/*! DiskANN static node-cache budget conversion.
 *
 * The two large payload allocations are exact: one original vector and
 * (max_degree + 1) neighbor slots per cached node. The search path also keeps
 * two std::map entries per node. Their allocator-specific node size is not
 * portable, so the estimate accounts for each value_type plus four pointers
 * of tree bookkeeping. Benchmark RSS remains the source of truth.
 */
class DiskAnnCacheBudget {
 public:
  using CoordCacheValue = std::pair<const diskann_id_t, void *>;
  using NeighborValue = std::pair<uint32_t, diskann_id_t *>;
  using NeighborCacheValue = std::pair<const diskann_id_t, NeighborValue>;

  static uint64_t PayloadBytesPerNode(const IndexMeta &meta,
                                      uint32_t max_degree) {
    return static_cast<uint64_t>(meta.element_size()) +
           (static_cast<uint64_t>(max_degree) + 1) * sizeof(diskann_id_t);
  }

  static uint64_t EstimatedBytesPerNode(const IndexMeta &meta,
                                        uint32_t max_degree) {
    constexpr uint64_t kTreeLinksPerEntry = 4;
    const uint64_t map_bookkeeping = sizeof(CoordCacheValue) +
                                     sizeof(NeighborCacheValue) +
                                     2 * kTreeLinksPerEntry * sizeof(void *);
    return PayloadBytesPerNode(meta, max_degree) + map_bookkeeping;
  }

  static uint32_t ResolveNodeCount(uint64_t budget_bytes, uint64_t doc_count,
                                   uint64_t estimated_bytes_per_node) {
    if (budget_bytes == 0 || doc_count == 0 || estimated_bytes_per_node == 0) {
      return 0;
    }

    const uint64_t budget_nodes = budget_bytes / estimated_bytes_per_node;
    const uint64_t resolved =
        std::min({budget_nodes, doc_count,
                  static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    return static_cast<uint32_t>(resolved);
  }
};

}  // namespace core
}  // namespace zvec

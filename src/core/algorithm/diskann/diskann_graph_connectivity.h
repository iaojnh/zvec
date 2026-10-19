// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <limits>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include "diskann_entity.h"

namespace zvec::core {

// Build-only workspace; all graph writers must have joined. The first-discovery
// edges form a protected spanning tree of the currently reachable component.
// Non-tree edges can be replaced without disconnecting any reached vertex.
class DiskAnnGraphConnectivity {
 public:
  DiskAnnGraphConnectivity(const DiskAnnEntity &entity, uint32_t max_degree)
      : entity_(entity), max_degree_(max_degree), count_(entity.doc_cnt()) {
    if (count_ >
        std::numeric_limits<size_t>::max() / (2 * sizeof(diskann_id_t)))
      throw std::bad_alloc();
    memory_.resize(static_cast<size_t>(count_) * 2 * sizeof(diskann_id_t));
    parents_ = reinterpret_cast<diskann_id_t *>(memory_.data());
    queue_ = parents_ + count_;
    reset();
  }

  void reset() {
    std::fill_n(parents_, count_, kInvalidId);
    head_ = 0;
    reached_ = 0;
  }

  int visit(diskann_id_t root, diskann_id_t parent) {
    if (root >= count_ || parent >= count_ || contains(root))
      return IndexError_InvalidArgument;
    parents_[root] = parent;
    queue_[reached_++] = root;
    ailego::BudgetedVector<diskann_id_t> neighbors;
    while (head_ < reached_) {
      const diskann_id_t node = queue_[head_++];
      const int ret = entity_.read_neighbors(node, &neighbors);
      if (ret != 0) return ret;
      if (neighbors.size() > max_degree_) return IndexError_ReadData;
      for (diskann_id_t neighbor : neighbors) {
        if (neighbor >= count_) return IndexError_ReadData;
        if (contains(neighbor)) continue;
        parents_[neighbor] = node;
        queue_[reached_++] = neighbor;
      }
    }
    return 0;
  }

  bool contains(diskann_id_t id) const {
    return parents_[id] != kInvalidId;
  }
  size_t reached() const {
    return reached_;
  }
  diskann_id_t at(size_t i) const {
    return queue_[i];
  }

  // The row has already been validated by visit(). Prefer appending; otherwise
  // replace the last unprotected edge, leaving all spanning-tree edges intact.
  size_t link_slot(
      diskann_id_t owner,
      const ailego::BudgetedVector<diskann_id_t> &neighbors) const {
    if (neighbors.size() < max_degree_) return neighbors.size();
    for (size_t i = neighbors.size(); i != 0; --i) {
      const diskann_id_t child = neighbors[i - 1];
      if (child == owner || parents_[child] != owner) return i - 1;
    }
    return std::numeric_limits<size_t>::max();
  }

 private:
  const DiskAnnEntity &entity_;
  uint32_t max_degree_;
  uint64_t count_;
  ailego::BudgetedMappedBuffer memory_;
  diskann_id_t *parents_{nullptr};
  diskann_id_t *queue_{nullptr};
  size_t head_{0};
  size_t reached_{0};
};

}  // namespace zvec::core

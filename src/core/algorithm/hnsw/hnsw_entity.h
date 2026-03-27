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

#include <string.h>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/container/heap.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

using node_id_t = uint32_t;
using key_t = uint64_t;
using level_t = int32_t;
using dist_t = float;
using TopkHeap = ailego::KeyValueHeap<node_id_t, dist_t>;
using CandidateHeap =
    ailego::KeyValueHeap<node_id_t, dist_t, std::greater<dist_t>>;
constexpr node_id_t kInvalidNodeId = static_cast<node_id_t>(-1);
constexpr key_t kInvalidKey = static_cast<key_t>(-1);
class DistCalculator;

struct GraphHeader {
  uint32_t size;
  uint32_t version;
  uint32_t graph_type;
  uint32_t doc_count;
  uint32_t vector_size;
  uint32_t node_size;
  uint32_t l0_neighbor_count;
  uint32_t prune_type;
  uint32_t prune_neighbor_count;
  uint32_t ef_construction;
  uint32_t options;
  uint32_t min_neighbor_count;
  uint8_t reserved_[4080];
};

static_assert(sizeof(GraphHeader) % 32 == 0,
              "GraphHeader must be aligned with 32 bytes");

//! Hnsw upper neighbor header
struct HnswHeader {
  uint32_t size;      // header size
  uint32_t revision;  // current total docs of the graph
  uint32_t upper_neighbor_count;
  uint32_t ef_construction;
  uint32_t scaling_factor;
  uint32_t max_level;
  uint32_t entry_point;
  uint32_t options;
  uint8_t reserved_[30];
};

static_assert(sizeof(HnswHeader) % 32 == 0,
              "GraphHeader must be aligned with 32 bytes");

//! Hnsw common header and upper neighbor header
struct HNSWHeader {
  HNSWHeader() {
    clear();
  }

  HNSWHeader(const HNSWHeader &header) {
    memcpy(this, &header, sizeof(header));
  }

  HNSWHeader &operator=(const HNSWHeader &header) {
    memcpy(this, &header, sizeof(header));
    return *this;
  }

  //! Reset state to zero, and the params is untouched
  void inline reset() {
    graph.doc_count = 0U;
    hnsw.entry_point = kInvalidNodeId;
    hnsw.max_level = 0;
  }

  //! Clear all fields to init value
  void inline clear() {
    memset(this, 0, sizeof(HNSWHeader));
    hnsw.entry_point = kInvalidNodeId;
    graph.size = sizeof(GraphHeader);
    hnsw.size = sizeof(HnswHeader);
  }

  size_t l0_neighbor_cnt() const {
    return graph.l0_neighbor_count;
  }

  size_t upper_neighbor_cnt() const {
    return hnsw.upper_neighbor_count;
  }

  size_t vector_size() const {
    return graph.vector_size;
  }

  size_t ef_construction() const {
    return graph.ef_construction;
  }

  size_t scaling_factor() const {
    return hnsw.scaling_factor;
  }

  size_t neighbor_prune_cnt() const {
    return graph.prune_neighbor_count;
  }

  node_id_t entry_point() const {
    return hnsw.entry_point;
  }

  node_id_t doc_cnt() const {
    return graph.doc_count;
  }

  GraphHeader graph;
  HnswHeader hnsw;
};

struct NeighborsHeader {
  uint32_t neighbor_cnt;
  node_id_t neighbors[0];
};

struct Neighbors {
  Neighbors() : cnt{0}, data{nullptr} {}

  Neighbors(uint32_t cnt_in, const node_id_t *data_in)
      : cnt{cnt_in}, data{data_in} {}

  Neighbors(const IndexStorage::MemoryBlock &mem_block)
      : neighbor_block{mem_block} {
    auto hd = reinterpret_cast<const NeighborsHeader *>(neighbor_block.data());
    cnt = hd->neighbor_cnt;
    data = hd->neighbors;
  }

  size_t size(void) const {
    return cnt;
  }

  const node_id_t &operator[](size_t idx) const {
    return data[idx];
  }

  uint32_t cnt;
  const node_id_t *data;
  IndexStorage::MemoryBlock neighbor_block;
};

//! level 0 neighbors offset
struct GraphNeighborMeta {
  GraphNeighborMeta(size_t o, size_t cnt) : offset(o), neighbor_cnt(cnt) {}

  uint64_t offset : 48;
  uint64_t neighbor_cnt : 16;
};

//! hnsw upper neighbors meta
struct HnswNeighborMeta {
  HnswNeighborMeta(size_t o, size_t l) : offset(o), level(l) {}

  uint64_t offset : 48;  // offset = idx * upper neighors size
  uint64_t level : 16;
};

}  // namespace core
}  // namespace zvec

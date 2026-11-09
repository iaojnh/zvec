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
#include "tests/core/algorithm/graph_dump_test_helper.h"

namespace zvec::core::testing {
template <typename Entity,
          typename VectorBlocks =
              ailego::BudgetedVector<IndexStorage::MemoryBlock>>
class GeneratedHnsw : public Entity {
 public:
  explicit GeneratedHnsw(uint32_t count) {
    this->mutable_header()->graph.doc_count = count;
    this->mutable_header()->graph.vector_size = 32;
    this->set_l0_neighbor_cnt(2);
    this->set_upper_neighbor_cnt(2);
  }
  key_t get_key(node_id_t id) const override {
    return 100 - id % 100;
  }
  const void *get_vector(node_id_t) const override {
    return row_.data();
  }
  int get_vector(node_id_t id,
                 IndexStorage::MemoryBlock &block) const override {
    block = IndexStorage::MemoryBlock(const_cast<void *>(get_vector(id)));
    return 0;
  }
  int get_vector(const node_id_t *, uint32_t, const void **) const override {
    return IndexError_NotImplemented;
  }
  int get_vector(const node_id_t *, uint32_t, VectorBlocks &) const override {
    return IndexError_NotImplemented;
  }
  const Neighbors get_neighbors(level_t level, node_id_t id) const override {
    neighbors_ = {id + static_cast<node_id_t>(level), id + 1};
    return Neighbors(1 + id % 2, neighbors_.data());
  }
  int64_t export_segments(const IndexDumper::Pointer &dumper) const {
    return this->dump_segments(dumper, [&](node_id_t id) -> level_t {
      return fail_level ? IndexError_ReadData : static_cast<level_t>(id % 3);
    });
  }
  bool fail_level{false};

 private:
  std::array<float, 8> row_{};
  mutable std::array<node_id_t, 2> neighbors_{};
};

template <typename Entity, typename VectorBlocks = ailego::BudgetedVector<
                               IndexStorage::MemoryBlock>>
void CheckGraphDumpReadError() {
  GeneratedHnsw<Entity, VectorBlocks> entity(3);
  entity.fail_level = true;
  auto capture = std::make_shared<testing::GraphDumpCapture>();
  EXPECT_EQ(IndexError_ReadData, entity.export_segments(capture));
  EXPECT_EQ(0u, capture->segments.count(Entity::kHnswOffsetsSegmentId));
}
}  // namespace zvec::core::testing

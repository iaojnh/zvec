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
#include <numeric>
#include "tests/core/algorithm/graph_dump_test_helper.h"

namespace zvec::core::testing {
template <typename Entity>
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
  int get_vector(const node_id_t *, uint32_t,
                 std::vector<IndexStorage::MemoryBlock> &) const override {
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

template <typename Entity>
void CheckStreamingGraphDump() {
  for (uint32_t count : {0u, 1u, 8193u}) {
    SCOPED_TRACE(count);
    GeneratedHnsw<Entity> entity(count);
    auto capture = std::make_shared<testing::GraphDumpCapture>();
    ASSERT_GE(entity.export_segments(capture), 0);
    EXPECT_EQ(9u, capture->segments.size());
    EXPECT_LE(capture->max_write, 64u * 1024);
    auto graph =
        capture->records<GraphNeighborMeta>(Entity::kGraphOffsetsSegmentId);
    auto upper =
        capture->records<HnswNeighborMeta>(Entity::kHnswOffsetsSegmentId);
    auto neighbors =
        capture->records<node_id_t>(Entity::kGraphNeighborsSegmentId);
    auto upper_neighbors =
        capture->records<node_id_t>(Entity::kHnswNeighborsSegmentId);
    auto keys = capture->records<key_t>(Entity::kGraphKeysSegmentId);
    auto mapping = capture->records<node_id_t>(Entity::kGraphMappingSegmentId);
    ASSERT_EQ(count, graph.size());
    ASSERT_EQ(count, upper.size());
    ASSERT_EQ(count, keys.size());
    std::vector<node_id_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](node_id_t a, node_id_t b) {
      return entity.get_key(a) != entity.get_key(b)
                 ? entity.get_key(a) < entity.get_key(b)
                 : a < b;
    });
    EXPECT_EQ(order, mapping);
    size_t graph_pos = 0, upper_pos = 0;
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_EQ(entity.get_key(i), keys[i]);
      EXPECT_EQ(graph_pos * sizeof(node_id_t), graph[i].offset);
      EXPECT_EQ(1u + i % 2, graph[i].neighbor_cnt);
      for (uint32_t j = 0; j < 1 + i % 2; ++j)
        EXPECT_EQ(i + j, neighbors.at(graph_pos++));
      EXPECT_EQ(i % 3, upper[i].level);
      EXPECT_EQ(i % 3 ? upper_pos * sizeof(node_id_t) : 0, upper[i].offset);
      for (uint32_t level = 1; level <= i % 3; ++level) {
        EXPECT_EQ(1u + i % 2, upper_neighbors.at(upper_pos++));
        EXPECT_EQ(i + level, upper_neighbors.at(upper_pos++));
        EXPECT_EQ(i % 2 ? i + 1 : 0, upper_neighbors.at(upper_pos++));
      }
    }
    EXPECT_EQ(neighbors.size(), graph_pos);
    EXPECT_EQ(upper_neighbors.size(), upper_pos);
  }
}

template <typename Entity>
void CheckGraphDumpReadError() {
  GeneratedHnsw<Entity> entity(3);
  entity.fail_level = true;
  auto capture = std::make_shared<testing::GraphDumpCapture>();
  EXPECT_EQ(IndexError_ReadData, entity.export_segments(capture));
  EXPECT_EQ(0u, capture->segments.count(Entity::kHnswOffsetsSegmentId));
}
}  // namespace zvec::core::testing

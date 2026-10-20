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
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "hnsw_sparse_streamer_entity.h"

namespace zvec {
namespace core {

TEST(HnswSparsePortableLayout, RebindAndGrowUsingPersistedGeometry) {
  const std::string path = "hnsw_sparse_portable_layout.index";
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  IndexStreamer::Stats stats;
  std::vector<uint32_t> indices(128);
  for (uint32_t i = 0; i < indices.size(); ++i) indices[i] = i;
  std::vector<float> values(128, 1.0f);
  std::string vector(reinterpret_cast<const char *>(indices.data()),
                     indices.size() * sizeof(uint32_t));
  vector.append(reinterpret_cast<const char *>(values.data()),
                values.size() * sizeof(float));
  for (uint32_t round = 0; round < 3; ++round) {
    ASSERT_EQ(0, storage->open(path, round == 0));
    HnswSparseStreamerEntity entity(stats);
    entity.set_chunk_size(round == 1 ? 65536 : 4096);
    entity.set_l0_neighbor_cnt(36);
    entity.set_upper_neighbor_cnt(36);
    // Every node in this fixture has an upper level.
    entity.set_scaling_factor(2);
    entity.set_ef_construction(100);
    entity.set_sparse_meta_size(sizeof(uint64_t) + sizeof(uint32_t));
    entity.set_sparse_unit_size(sizeof(uint32_t) + sizeof(float));
    ASSERT_EQ(0, entity.init(128 * 1024 * 1024, 10000));
    ASSERT_EQ(0, entity.open(storage, false));
    const node_id_t before = round * 200;
    EXPECT_EQ(before, entity.doc_cnt());
    for (node_id_t i = 0; i < before; ++i) {
      EXPECT_EQ(i, entity.get_key(i));
      const auto neighbors = entity.get_neighbors(1, i);
      ASSERT_EQ(1u, neighbors.size());
      EXPECT_EQ(i, neighbors[0]);
    }
    for (node_id_t i = before; i < before + 200; ++i) {
      node_id_t id = kInvalidNodeId;
      ASSERT_EQ(0, entity.add_vector(1, i, vector, 128, &id));
      ASSERT_EQ(i, id);
      ASSERT_EQ(0, entity.update_neighbors(1, id, {{id, 0.0f}}));
    }
    entity.update_ep_and_level(0, 1);
    ASSERT_EQ(0, entity.close());
    ASSERT_EQ(0, storage->close());
  }
  EXPECT_TRUE(ailego::File::Delete(path));
}

}  // namespace core
}  // namespace zvec

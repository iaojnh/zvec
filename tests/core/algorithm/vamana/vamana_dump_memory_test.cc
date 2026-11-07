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

#include <array>
#include "tests/core/algorithm/graph_dump_test_helper.h"
#include "vamana_entity.h"

namespace zvec::core {
namespace {
class GeneratedVamana : public VamanaEntity {
 public:
  explicit GeneratedVamana(uint32_t count, bool distances)
      : distances_(distances) {
    *mutable_doc_cnt() = count;
    set_max_degree(3);
    set_vector_size(7);
  }
  key_t get_key(node_id_t id) const override {
    return 1000 - id;
  }
  const void *get_vector(node_id_t id) const override {
    row_.fill(static_cast<uint8_t>(id));
    return row_.data();
  }
  const void *get_extra_values(node_id_t) const override {
    return nullptr;
  }
  int get_vector(node_id_t id,
                 IndexStorage::MemoryBlock &block) const override {
    if (fail_read) return IndexError_ReadData;
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
  const Neighbors get_neighbors(node_id_t id) const override {
    neighbors_ = {id, id + 1, id + 2};
    return Neighbors(id % 4, neighbors_.data());
  }
  bool dist_storage_loaded() const override {
    return distances_;
  }
  const dist_t *get_neighbor_dists(node_id_t id) const override {
    if (id % 2) return nullptr;
    dists_.fill(static_cast<float>(id));
    return dists_.data();
  }
  int64_t export_segments(const IndexDumper::Pointer &dumper) const {
    return dump_segments(dumper);
  }
  bool fail_read{false};

 private:
  bool distances_;
  mutable std::array<uint8_t, 7> row_{};
  mutable std::array<node_id_t, 3> neighbors_{};
  mutable std::array<float, 3> dists_{};
};

TEST(VamanaDumpMemoryTest, StreamingPreservesRecordsPaddingAndCrc) {
  for (uint32_t count : {0u, 1u, 10001u}) {
    for (bool distances : {false, true}) {
      SCOPED_TRACE(::testing::Message() << count << ':' << distances);
      GeneratedVamana entity(count, distances);
      auto capture = std::make_shared<testing::GraphDumpCapture>();
      ASSERT_GE(entity.export_segments(capture), 0);
      EXPECT_LE(capture->max_write, 64u * 1024);
      EXPECT_EQ(distances ? 5u : 4u, capture->segments.size());
      const auto &features =
          capture->segments.at(VamanaEntity::kGraphFeaturesSegmentId);
      ASSERT_EQ(count * 7u, features.size());
      auto keys = capture->records<key_t>(VamanaEntity::kGraphMappingSegmentId);
      auto neighbors =
          capture->records<node_id_t>(VamanaEntity::kGraphNeighborsSegmentId);
      ASSERT_EQ(count, keys.size());
      ASSERT_EQ(count * 4u, neighbors.size());
      for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(std::string(7, static_cast<char>(i)),
                  features.substr(i * 7, 7));
        EXPECT_EQ(entity.get_key(i), keys[i]);
        EXPECT_EQ(i % 4, neighbors[i * 4]);
        for (uint32_t j = 0; j < 3; ++j)
          EXPECT_EQ(j < i % 4 ? i + j : 0, neighbors[i * 4 + j + 1]);
      }
      if (distances) {
        auto dists =
            capture->records<float>(VamanaEntity::kGraphNeighborDistsSegmentId);
        ASSERT_EQ(count * 3u, dists.size());
        for (uint32_t i = 0; i < count; ++i)
          for (size_t j = 0; j < 3; ++j)
            EXPECT_EQ(i % 2 ? 0.0f : static_cast<float>(i), dists[i * 3 + j]);
      }
    }
  }
}

TEST(VamanaDumpMemoryTest, ReadAndWriteErrorsPropagate) {
  GeneratedVamana entity(3, true);
  auto capture = std::make_shared<testing::GraphDumpCapture>();
  entity.fail_read = true;
  EXPECT_EQ(IndexError_ReadData, entity.export_segments(capture));
  capture = std::make_shared<testing::GraphDumpCapture>();
  entity.fail_read = false;
  capture->fail_write = true;
  EXPECT_EQ(IndexError_WriteData, entity.export_segments(capture));
}
}  // namespace
}  // namespace zvec::core

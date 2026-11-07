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

#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "utility/sparse_utility.h"
#include "hnsw_sparse_context.h"
#include "hnsw_sparse_streamer_entity.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
TEST(HnswSparseDirectIdsTest, HolesUpdatesCloneAndReopenWithoutKeyMap) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  for (const auto *mode : {"MMapFileStorage", "BufferStorage"}) {
    ASSERT_EQ(0, pool.init(16 * 1024 * 1024));
    const std::string path = std::string("sparse_direct_ids_") + mode;
    ASSERT_FALSE(std::filesystem::exists(path));
    AILEGO_DEFER([&] { std::filesystem::remove(path); });
    for (int round = 0; round < 2; ++round) {
      auto storage = IndexFactory::CreateStorage(mode);
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(path, true));
      IndexStreamer::Stats stats;
      HnswSparseStreamerEntity entity(stats);
      entity.set_sparse_meta_size(HnswSparseEntity::kSparseMetaSize);
      entity.set_sparse_unit_size(sizeof(float));
      entity.set_l0_neighbor_cnt(16);
      entity.set_upper_neighbor_cnt(8);
      entity.set_scaling_factor(16);
      entity.set_chunk_size(64 * 1024);
      entity.set_get_vector(true);
      entity.set_use_key_info_map(false);
      ASSERT_EQ(0, entity.init(64 * 1024 * 1024, 1024));
      ASSERT_EQ(0, entity.open(storage, false));
      const uint32_t indices[]{1, 4, 7};
      const float values[]{0.5f, 1.0f, 2.0f};
      std::string row;
      SparseUtility::TransSparseFormat(3, indices, values, sizeof(float), row);
      if (!round) {
        for (node_id_t id = 3; id < 131; ++id)
          ASSERT_EQ(0, entity.add_vector_with_id(0, id, row, 3));
        entity.update_ep_and_level(3, 0);
      }
      EXPECT_EQ(kInvalidNodeId, entity.get_id(0));
      EXPECT_EQ(kInvalidNodeId, entity.get_id(uint64_t{1} << 32));
      EXPECT_EQ(kInvalidNodeId, entity.get_id(131));
      EXPECT_EQ(130u, entity.get_id(130));
      ASSERT_EQ(0, entity.add_vector_with_id(0, 1, row, 3));
      EXPECT_EQ(1u, entity.get_id(1));
      auto clone =
          std::dynamic_pointer_cast<HnswSparseStreamerEntity>(entity.clone());
      ASSERT_NE(nullptr, clone);
      EXPECT_EQ(130u, clone->get_id(130));
      uint32_t nnz = 0;
      std::string got_indices, got_values;
      ASSERT_EQ(0, clone->get_sparse_vector_by_key(130, &nnz, &got_indices,
                                                   &got_values));
      EXPECT_EQ(3u, nnz);
      EXPECT_EQ(
          std::string(reinterpret_cast<const char *>(indices), sizeof(indices)),
          got_indices);
      EXPECT_EQ(
          std::string(reinterpret_cast<const char *>(values), sizeof(values)),
          got_values);
      clone.reset();
      ASSERT_EQ(0, entity.flush(0));
      ASSERT_EQ(0, entity.close());
      ASSERT_EQ(0, storage->close());
    }
  }
}

TEST(HnswSparseDirectIdsTest, StreamerOptionAndCompactExactContext) {
  const std::string path = "sparse_direct_streamer.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE, true);
  params.set(PARAM_HNSW_SPARSE_STREAMER_USE_ID_MAP, false);
  params.set(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, 8);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, 16);
  params.set(PARAM_HNSW_SPARSE_STREAMER_EF, 128);
  params.set(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD, 0u);
  for (int round = 0; round < 2; ++round) {
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    auto streamer = IndexFactory::CreateStreamer("HnswSparseStreamer");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, !round));
    ASSERT_EQ(0, streamer->init(meta, params));
    ASSERT_EQ(0, streamer->open(storage));
    auto context = streamer->create_context();
    auto *sparse = dynamic_cast<HnswSparseContext *>(context.get());
    ASSERT_NE(nullptr, sparse);
    EXPECT_EQ(VisitFilter::EpochBitMap, sparse->visit_filter().get_mode());
    IndexQueryMeta query_meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
    const uint32_t indices[]{1, 4, 7};
    float values[]{1, 1, 1};
    if (!round)
      for (uint32_t id = 0; id < 128; ++id) {
        values[0] = 1.0f + static_cast<float>(id);
        ASSERT_EQ(0, streamer->add_with_id_impl(id, 3, indices, values,
                                                query_meta, context));
      }
    context->set_topk(5);
    std::vector<uint64_t> reference;
    for (auto mode : {VisitFilter::EpochBitMap, VisitFilter::ByteMap}) {
      sparse->set_filter_mode(mode);
      ASSERT_EQ(0, sparse->init(HnswSparseContext::kSparseStreamerContext));
      ASSERT_EQ(0,
                streamer->search_impl(3, indices, values, query_meta, context));
      std::vector<uint64_t> keys;
      for (const auto &doc : context->result()) keys.push_back(doc.key());
      ASSERT_EQ(5u, keys.size());
      if (reference.empty())
        reference = keys;
      else
        EXPECT_EQ(reference, keys);
    }
    ASSERT_EQ(0, streamer->flush(0));
    context.reset();
    ASSERT_EQ(0, streamer->close());
    ASSERT_EQ(0, storage->close());
  }
}
}  // namespace
}  // namespace zvec::core

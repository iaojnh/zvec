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
#include <filesystem>
#include <numeric>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "tests/core/algorithm/graph_dump_test_helper.h"
#include "utility/sparse_utility.h"
#include "hnsw_sparse_builder.h"
#include "hnsw_sparse_dist_calculator.h"
#include "hnsw_sparse_params.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include "hnsw_sparse_context.h"
#include "hnsw_sparse_streamer_entity.h"
#include <zvec/core/framework/index_factory.h>
#include "hnsw_sparse_index_hash.h"

namespace zvec::core {
namespace {
class SparseDump : public testing::GraphDumpCapture {
 public:
  int append(const std::string &id, size_t size, size_t pad,
             uint32_t crc) override {
    // Sparse variable-length vector segments need not be padded to 32 bytes.
    EXPECT_EQ(size + pad, pending.size());
    EXPECT_EQ(ailego::Crc32c::Hash(pending.data(), size), crc);
    EXPECT_TRUE(segments.emplace(id, pending.substr(0, size)).second);
    pending.clear();
    return 0;
  }
};

class HnswSparseBufferedBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    old_capacity_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(0));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_capacity_));
  }
  static void configure(HnswSparseBuilderEntity &entity, size_t unit = 4) {
    entity.set_sparse_meta_size(HnswSparseEntity::kSparseMetaSize);
    entity.set_sparse_unit_size(unit);
    entity.set_l0_neighbor_cnt(64);
    entity.set_upper_neighbor_cnt(32);
    ASSERT_EQ(0, entity.init());
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t old_capacity_{0};
};

TEST_F(HnswSparseBufferedBuilderTest,
       EvictsCorpusAndPreservesDumpAndOwnedRows) {
  // Keep the key-sort workspace below 64 KiB, so this test's 4 MiB budget
  // isolates corpus paging rather than a second writable page table at dump.
  static constexpr size_t count = 3072, nnz = 384;
  auto reference = std::make_shared<SparseDump>();
  for (bool buffered : {false, true}) {
    ASSERT_EQ(0, pool_.init(buffered ? 4 * 1024 * 1024 : 0));
    HnswSparseBuilderEntity entity;
    configure(entity);
    ASSERT_EQ(buffered, entity.buffered());
    // Leave the buffered size unknown to exercise incremental growth.
    if (!buffered) ASSERT_EQ(0, entity.reserve_space(count, count * nnz));
    std::array<uint32_t, nnz> indices;
    std::iota(indices.begin(), indices.end(), 0);
    std::array<float, nnz> values;
    for (size_t i = 0; i < count; ++i) {
      values.fill(static_cast<float>(i) + 0.25f);
      node_id_t id;
      ASSERT_EQ(0, entity.add_vector(i % 3, i + 100, nnz, indices.data(),
                                     values.data(), &id));
      ASSERT_EQ(i, id);
    }
    IndexStorage::MemoryBlock held;
    ASSERT_EQ(0, entity.get_sparse_data(0, held));
    values.fill(0.25f);
    std::string expected;
    SparseUtility::TransSparseFormat(nnz, indices.data(), values.data(), 4,
                                     expected);
    // Reverse traversal crosses both page and 1 MiB chunk boundaries, and
    // forces reuse of a cache much smaller than the complete arrays.
    for (size_t i = count; i-- > 0;) {
      EXPECT_EQ(i + 100, entity.get_key(i));
      EXPECT_EQ(i % 3, static_cast<size_t>(entity.get_level(i)));
      IndexStorage::MemoryBlock data;
      ASSERT_EQ(0, entity.get_sparse_data(i, data));
      values.fill(static_cast<float>(i) + 0.25f);
      std::string row;
      SparseUtility::TransSparseFormat(nnz, indices.data(), values.data(), 4,
                                       row);
      ASSERT_EQ(0, std::memcmp(row.data(), data.data(), row.size()));
      for (level_t level = 0; level <= entity.get_level(i); ++level) {
        entity.add_neighbor(level, i, 0, (i + 1) % count);
        auto old = entity.get_neighbors(level, i);
        ASSERT_EQ(1u, old.size());
        ASSERT_EQ(0, entity.update_neighbors(level, i, {{(i + 2) % count, 0}}));
        // Buffered reads own their row; later writes cannot overwrite it.
        if (buffered) EXPECT_EQ((i + 1) % count, old[0]);
        EXPECT_EQ((i + 2) % count, entity.get_neighbors(level, i)[0]);
      }
    }
    ASSERT_EQ(0, entity.status());
    EXPECT_EQ(0, std::memcmp(expected.data(), held.data(), expected.size()));
    entity.update_ep_and_level(2, 2);
    auto dump = buffered ? std::make_shared<SparseDump>() : reference;
    ASSERT_EQ(0, entity.dump(dump));
    if (buffered) {
      EXPECT_EQ(reference->segments, dump->segments);
      EXPECT_GT(pool_.used(), 0u);
      EXPECT_LE(pool_.used(), pool_.capacity());
    }
    ASSERT_EQ(0, entity.cleanup());
    if (buffered) {
      // The escaped row, unlike dump's local scratch, still owns its bytes.
      EXPECT_GT(pool_.used(), 0u);
      EXPECT_EQ(0, std::memcmp(expected.data(), held.data(), expected.size()));
    }
    held.reset();
    EXPECT_EQ(0u, pool_.used());
  }
}

TEST_F(HnswSparseBufferedBuilderTest, DistancesKeepBothOperandsAndQueryAlive) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  HnswSparseBuilderEntity entity;
  configure(entity);
  const uint32_t index = 7;
  for (float value : {2.0f, 3.0f, 5.0f}) {
    node_id_t id;
    ASSERT_EQ(0, entity.add_vector(0, 0, 1, &index, &value, &id));
  }
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  auto metric = IndexFactory::CreateMetric("InnerProductSparse");
  ASSERT_NE(nullptr, metric);
  ASSERT_EQ(0, metric->init(meta, {}));
  IndexStorage::MemoryBlock query;
  ASSERT_EQ(0, entity.get_sparse_data(2, query));
  HnswSparseDistCalculator dc(&entity, metric, query.data());
  EXPECT_FLOAT_EQ(-6.0f, dc.dist(node_id_t{0}, node_id_t{1}));
  EXPECT_FLOAT_EQ(-10.0f, dc.dist(node_id_t{0}));
  EXPECT_FLOAT_EQ(-15.0f, dc.dist(node_id_t{1}));
  EXPECT_FALSE(dc.error());
  const node_id_t ids[]{0, 1};
  std::vector<IndexStorage::MemoryBlock> batch;
  ASSERT_EQ(0, entity.get_vector_metas(ids, 2, batch));
  EXPECT_FLOAT_EQ(-10.0f, dc.dist(batch[0].data()));
  EXPECT_FLOAT_EQ(-15.0f, dc.dist(batch[1].data()));
}

TEST_F(HnswSparseBufferedBuilderTest, CleanupResetsPayloadOffsetInBothModes) {
  for (bool buffered : {false, true}) {
    ASSERT_EQ(0, pool_.init(buffered ? 4 * 1024 * 1024 : 0));
    HnswSparseBuilderEntity entity;
    for (float value : {1.0f, 9.0f}) {
      configure(entity);
      const uint32_t index = 2;
      node_id_t id;
      ASSERT_EQ(0, entity.add_vector(1, 42, 1, &index, &value, &id));
      ASSERT_EQ(0u, id);
      IndexStorage::MemoryBlock meta, data;
      ASSERT_EQ(0, entity.get_vector_meta(0, meta));
      EXPECT_EQ(0u, *static_cast<const uint64_t *>(meta.data()));
      ASSERT_EQ(0, entity.get_sparse_data(0, data));
      std::string expected;
      SparseUtility::TransSparseFormat(1, &index, &value, 4, expected);
      EXPECT_EQ(0, std::memcmp(expected.data(), data.data(), expected.size()));
      ASSERT_EQ(0, entity.cleanup());
    }
  }
}

TEST_F(HnswSparseBufferedBuilderTest, Fp16AndEmptyRowsRoundTrip) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  HnswSparseBuilderEntity entity;
  configure(entity, 2);
  const uint32_t indices[]{1, 17, 8193};
  const ailego::Float16 values[]{1.0f, -2.0f, 3.5f};
  node_id_t id;
  ASSERT_EQ(0, entity.add_vector(0, 10, 0, nullptr, nullptr, &id));
  ASSERT_EQ(0, entity.add_vector(2, 11, 3, indices, values, &id));
  for (node_id_t i = 0; i < 2; ++i) {
    std::string expected;
    SparseUtility::TransSparseFormat(i ? 3 : 0, indices, values, 2, expected);
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, entity.get_sparse_data(i, block));
    EXPECT_EQ(0, std::memcmp(expected.data(), block.data(), expected.size()));
  }
  EXPECT_EQ(IndexError_InvalidValue,
            entity.add_vector(0, 12, HnswSparseEntity::kSparseMaxDimSize + 1,
                              indices, values, &id));
  EXPECT_EQ(2u, entity.doc_cnt());
  EXPECT_EQ(0, entity.status());
}

TEST_F(HnswSparseBufferedBuilderTest, TinyBudgetFailsWithoutHeapFallback) {
  ASSERT_EQ(0, pool_.init(1));
  HnswSparseBuilderEntity entity;
  entity.set_sparse_meta_size(HnswSparseEntity::kSparseMetaSize);
  entity.set_sparse_unit_size(4);
  EXPECT_NE(0, entity.init());
  EXPECT_NE(0, entity.status());
  EXPECT_EQ(0u, pool_.used());
}

TEST_F(HnswSparseBufferedBuilderTest,
       ReadFailurePoisonsDumpAndFurtherIngestion) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  HnswSparseBuilderEntity entity;
  configure(entity);
  const uint32_t index = 2;
  const float value = 1;
  node_id_t id;
  ASSERT_EQ(0, entity.add_vector(0, 1, 1, &index, &value, &id));
  IndexStorage::MemoryBlock block;
  EXPECT_NE(0, entity.get_sparse_data(UINT64_MAX, 4, block));
  EXPECT_EQ(nullptr, block.data());
  auto dump = std::make_shared<SparseDump>();
  EXPECT_EQ(entity.status(), entity.dump(dump));
  EXPECT_TRUE(dump->segments.empty());
  EXPECT_EQ(entity.status(), entity.add_vector(0, 2, 1, &index, &value, &id));
  EXPECT_EQ(1u, entity.doc_cnt());
}

TEST_F(HnswSparseBufferedBuilderTest, ParallelPublicCsrBuildDumpsAndReopens) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  const std::string path = "sparse_buffered_builder_test.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  constexpr size_t count = 256, nnz = 16;
  std::vector<uint64_t> keys(count), offsets(count + 1);
  std::vector<uint32_t> indices(count * nnz);
  std::vector<float> values(count * nnz);
  for (size_t i = 0; i < count; ++i) {
    keys[i] = 1000 + i;
    offsets[i] = i * nnz;
    for (size_t j = 0; j < nnz; ++j) {
      indices[i * nnz + j] = j;
      values[i * nnz + j] = static_cast<float>((i + j) % 13) / 13;
    }
  }
  offsets[count] = count * nnz;
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params());
  ailego::Params params;
  params.set(PARAM_HNSW_SPARSE_BUILDER_THREAD_COUNT, 4);
  params.set(PARAM_HNSW_SPARSE_BUILDER_MAX_NEIGHBOR_COUNT, 8);
  params.set(PARAM_HNSW_SPARSE_BUILDER_EFCONSTRUCTION, 16);
  auto builder = IndexFactory::CreateBuilder("HnswSparseBuilder");
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->build(IndexThreads::Pointer{}, count, keys.data(),
                              offsets.data(), indices.data(), values.data()));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  ASSERT_EQ(count, builder->stats().built_count());
  ASSERT_EQ(0, builder->cleanup());
  EXPECT_EQ(0u, pool_.used());
  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, container->init({}));
  ASSERT_EQ(0, container->open(path, false));
  auto searcher = IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_EQ(0, searcher->init({}));
  ASSERT_EQ(0, searcher->load(container, nullptr));
  auto ctx = searcher->create_context();
  ASSERT_NE(nullptr, ctx);
  ctx->set_topk(10);
  ASSERT_EQ(0, searcher->search_impl(nnz, indices.data(), values.data(),
                                     IndexQueryMeta(IndexMeta::DT_FP32), ctx));
  ASSERT_EQ(10u, ctx->result().size());
}
}  // namespace
}  // namespace zvec::core

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

namespace zvec::core {
TEST(HnswSparsePagedHashTest,
     ReadsLegacyLayoutAndPersistsUpdatesUnderPressure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const auto old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "sparse_paged_hash_test.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(old));
    std::filesystem::remove(path);
  });
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  IndexStreamer::Stats stats;
  auto broker = std::make_shared<SparseChunkBroker>(stats);
  ASSERT_EQ(0, broker->open(storage, 64 * 1024 * 1024, 2 * 1024 * 1024, false));
  using Hash = HnswSparseIndexHashMap<uint32_t, uint32_t>;
  static constexpr size_t bytes = 2 * 1024 * 1024;
  // Write the old on-disk layout directly, including a wrapped collision.
  auto allocated = broker->alloc_chunk(
      SparseChunkBroker::CHUNK_TYPE_NEIGHBOR_INDEX, 0, bytes);
  ASSERT_EQ(0, allocated.first);
  auto segment = allocated.second;
  ASSERT_EQ(bytes, segment->resize(bytes));
  const std::array<char, 4096> zeros{};
  for (size_t offset = 0; offset < bytes; offset += zeros.size())
    ASSERT_EQ(zeros.size(), segment->write(offset, zeros.data(), zeros.size()));
  const uint32_t slots = bytes / sizeof(Hash::Item);
  const Hash::Item first{slots - 1, 123}, wrapped{2 * slots - 1, 456};
  ASSERT_EQ(sizeof(first),
            segment->write(bytes - sizeof(first), &first, sizeof(first)));
  ASSERT_EQ(sizeof(wrapped), segment->write(0, &wrapped, sizeof(wrapped)));
  Hash hash;
  ASSERT_EQ(0, hash.init(broker, bytes, 4, 4 * slots, 2.0f));
  auto saved = hash.find(slots - 1);
  ASSERT_TRUE(saved.has_value());
  EXPECT_EQ(123u, saved->second);
  ASSERT_TRUE(hash.find(2 * slots - 1).has_value());
  EXPECT_EQ(456u, hash.find(2 * slots - 1)->second);
  ASSERT_TRUE(hash.insert(slots - 1, 789));
  EXPECT_EQ(123u, saved->second);
  ASSERT_EQ(0, hash.cleanup());
  segment.reset();
  allocated.second.reset();
  ASSERT_EQ(0, broker->flush(0));
  ASSERT_EQ(0, broker->close());
  broker.reset();
  ASSERT_EQ(0, storage->close());
  ASSERT_EQ(0, storage->open(path, false));
  broker = std::make_shared<SparseChunkBroker>(stats);
  ASSERT_EQ(0, broker->open(storage, 64 * 1024 * 1024, bytes, false));
  ASSERT_EQ(0, hash.init(broker, bytes, 4, 4 * slots, 2.0f));
  ASSERT_TRUE(hash.find(slots - 1).has_value());
  EXPECT_EQ(789u, hash.find(slots - 1)->second);
  EXPECT_EQ(456u, hash.find(2 * slots - 1)->second);
  EXPECT_LE(pool.used(), pool.capacity());
  ASSERT_EQ(0, hash.cleanup());
  ASSERT_EQ(0, broker->close());
  broker.reset();
  ASSERT_EQ(0, storage->close());
}
}  // namespace zvec::core

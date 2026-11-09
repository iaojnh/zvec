// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "diskann_graph_connectivity.h"
#include <algorithm>
#include <array>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "utility/temporary_buffer_storage.h"
#include "diskann_algorithm.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

class ConnectivityEntity : public DiskAnnEntity {
 public:
  explicit ConnectivityEntity(size_t count) : vectors(count), graph(count) {
    meta_header_.doc_cnt = count;
    for (size_t i = 0; i < count; ++i) vectors[i].fill(static_cast<float>(i));
  }
  diskann_id_t get_id(diskann_key_t key) const override {
    return static_cast<diskann_id_t>(key);
  }
  diskann_key_t get_key(diskann_id_t id) const override {
    return id;
  }
  int read_vector(diskann_id_t id,
                  IndexStorage::MemoryBlock &block) const override {
    ++vector_reads;
    if (id == failed_vector) return IndexError_IO;
    block = IndexStorage::MemoryBlock::MakeBorrowedView(
        const_cast<float *>(vectors.at(id).data()));
    return 0;
  }
  bool prefer_vector_batch() const override {
    return batch_vectors;
  }
  int read_vectors(const diskann_id_t *ids, IndexStorage::MemoryBlock *blocks,
                   size_t count) const override {
    ++vector_batches;
    if (batch_error) return batch_error;
    if (vector_batches == fail_batch_at) {
      if (count) {
        const int ret = read_vector(ids[0], blocks[0]);
        if (ret != 0) return ret;
      }
      return IndexError_NoMemory;
    }
    return DiskAnnEntity::read_vectors(ids, blocks, count);
  }
  int read_neighbors(diskann_id_t id,
                     ailego::BudgetedVector<diskann_id_t> *out) const override {
    if (id == failed_read) return IndexError_IO;
    *out = graph.at(id);
    return 0;
  }
  int set_neighbors(
      diskann_id_t id,
      const ailego::BudgetedVector<diskann_id_t> &value) override {
    if (fail_write) return IndexError_WriteData;
    graph.at(id) = value;
    ++writes;
    return 0;
  }
  int add_neighbor(diskann_id_t id, diskann_id_t neighbor) override {
    graph.at(id).push_back(neighbor);
    ++writes;
    return 0;
  }
  GraphQuantization graph_quantization() const override {
    return guidance;
  }

  std::vector<std::array<float, 4>> vectors;
  std::vector<ailego::BudgetedVector<diskann_id_t>> graph;
  diskann_id_t failed_vector{kInvalidId};
  diskann_id_t failed_read{kInvalidId};
  bool fail_write{false};
  size_t writes{0};
  GraphQuantization guidance;
  bool batch_vectors{false};
  int batch_error{0};
  mutable size_t vector_reads{0};
  mutable size_t vector_batches{0};
  size_t fail_batch_at{0};
};

class ReadAheadEntity : public ConnectivityEntity {
 public:
  using ConnectivityEntity::ConnectivityEntity;
  class Reader : public VectorReader {
   public:
    explicit Reader(const ReadAheadEntity &entity) : entity_(entity) {}
    ~Reader() override {
      if (entity_.active) ++entity_.cancelled;
      entity_.active = false;
    }
    int submit(const diskann_id_t *ids, size_t count) override {
      if (entity_.submit_error) return entity_.submit_error;
      std::copy_n(ids, count, ids_.data());
      count_ = count;
      entity_.active = true;
      return 0;
    }
    int finish(IndexStorage::MemoryBlock *blocks, size_t count) override {
      EXPECT_TRUE(entity_.active);
      EXPECT_EQ(count_, count);
      entity_.active = false;
      if (entity_.finish_error) return entity_.finish_error;
      return entity_.read_vectors(ids_.data(), blocks, count);
    }

   private:
    const ReadAheadEntity &entity_;
    std::array<diskann_id_t, kVectorReadBatchSize> ids_;
    size_t count_{0};
  };
  std::unique_ptr<VectorReader> create_vector_reader() const override {
    return std::make_unique<Reader>(*this);
  }
  mutable bool active{false};
  mutable size_t cancelled{0};
  int submit_error{0}, finish_error{0};
};

class DiskAnnGraphConnectivityTest : public testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    old_capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(32u * 1024u * 1024u));
  }
  void TearDown() override {
    context_.reset();
    entity_.reset();
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(old_capacity_));
  }
  void initialize(size_t count, uint32_t degree, uint32_t list_size = 8) {
    IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    auto metric = IndexFactory::CreateMetric(meta.metric_name());
    ASSERT_NE(nullptr, metric);
    ASSERT_EQ(0, metric->init(meta, ailego::Params()));
    entity_ = std::make_shared<ConnectivityEntity>(count);
    context_ = std::make_unique<DiskAnnContext>(meta, metric, entity_);
    ASSERT_EQ(0, context_->init(DiskAnnContext::kBuilderContext, degree, 0,
                                meta.element_size(), false));
    context_->set_list_size(list_size);
  }
  void expect_reachable(uint32_t degree) const {
    // Independent traversal, not the repair's workspace or counters.
    std::vector<bool> seen(entity_->doc_cnt(), false);
    std::vector<diskann_id_t> queue{0};
    seen[0] = true;
    for (size_t i = 0; i < queue.size(); ++i) {
      const auto &row = entity_->graph[queue[i]];
      EXPECT_LE(row.size(), degree);
      for (diskann_id_t next : row) {
        ASSERT_LT(next, seen.size());
        if (!seen[next]) {
          seen[next] = true;
          queue.push_back(next);
        }
      }
    }
    EXPECT_EQ(entity_->doc_cnt(), queue.size());
  }

  size_t old_capacity_{0};
  std::shared_ptr<ConnectivityEntity> entity_;
  DiskAnnContext::Pointer context_;
};

TEST_F(DiskAnnGraphConnectivityTest,
       RepairsIsolatedVerticesWithoutBreakingTree) {
  initialize(6, 2);
  entity_->graph = {{1, 2}, {0, 2}, {0, 1}, {0, 1}, {0, 1}, {0, 1}};
  DiskAnnAlgorithm algorithm(*entity_, 2);
  size_t added = 0;
  ASSERT_EQ(0, algorithm.repair_connectivity(context_.get(), &added));
  EXPECT_EQ(3u, added);
  EXPECT_EQ((ailego::BudgetedVector<diskann_id_t>{1, 2}), entity_->graph[0]);
  expect_reachable(2);
  const auto repaired = entity_->graph;
  ASSERT_EQ(0, algorithm.repair_connectivity(context_.get(), &added));
  EXPECT_EQ(0u, added);
  EXPECT_EQ(repaired, entity_->graph);
}

TEST_F(DiskAnnGraphConnectivityTest,
       RepairsDisconnectedCycleWithNoZeroIndegree) {
  initialize(4, 1);
  entity_->graph = {{1}, {0}, {3}, {2}};
  DiskAnnAlgorithm algorithm(*entity_, 1);
  size_t added = 0;
  ASSERT_EQ(0, algorithm.repair_connectivity(context_.get(), &added));
  EXPECT_EQ(1u, added);
  EXPECT_EQ((ailego::BudgetedVector<diskann_id_t>{1}), entity_->graph[0]);
  EXPECT_EQ((ailego::BudgetedVector<diskann_id_t>{2}), entity_->graph[1]);
  expect_reachable(1);
}

TEST_F(DiskAnnGraphConnectivityTest, PropagatesReadVectorAndWriteFailures) {
  initialize(4, 1);
  entity_->graph = {{1}, {0}, {3}, {2}};
  const auto original = entity_->graph;
  DiskAnnAlgorithm algorithm(*entity_, 1);
  entity_->failed_read = 0;
  EXPECT_EQ(IndexError_IO, algorithm.repair_connectivity(context_.get()));
  entity_->failed_read = kInvalidId;
  entity_->failed_vector = 2;
  EXPECT_EQ(IndexError_IO, algorithm.repair_connectivity(context_.get()));
  entity_->failed_vector = kInvalidId;
  entity_->fail_write = true;
  EXPECT_EQ(IndexError_WriteData,
            algorithm.repair_connectivity(context_.get()));
  EXPECT_EQ(original, entity_->graph);
  entity_->fail_write = false;
  EXPECT_EQ(0, algorithm.repair_connectivity(context_.get()));
  expect_reachable(1);
}

TEST_F(DiskAnnGraphConnectivityTest, AdmitsWorkspaceBeforeMutationAndCanRetry) {
  initialize(4, 1);
  entity_->graph = {{1}, {0}, {3}, {2}};
  const auto original = entity_->graph;
  DiskAnnAlgorithm algorithm(*entity_, 1);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t charge = pool.available();
  ASSERT_TRUE(pool.try_charge_external(charge));
  const int ret = algorithm.repair_connectivity(context_.get());
  pool.release_external(charge);
  EXPECT_EQ(IndexError_NoMemory, ret);
  EXPECT_EQ(original, entity_->graph);
  EXPECT_EQ(0u, entity_->writes);
  ASSERT_EQ(0, algorithm.repair_connectivity(context_.get()));
  expect_reachable(1);
}

TEST_F(DiskAnnGraphConnectivityTest,
       QuantizedTraversalRestoresExactPruneDistancesAndReadErrors) {
  initialize(128, 8, 128);
  for (diskann_id_t id = 1; id < 128; ++id) entity_->graph[0].push_back(id);
  const auto original = entity_->graph;
  DiskAnnAlgorithm algorithm(*entity_, 8);
  ASSERT_EQ(0, algorithm.add_node(65, context_.get()));
  const auto expected = entity_->graph;

  IndexMeta meta(IndexMeta::DT_FP32, 4);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto quantizer = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ailego::Params params;
  params.set("num_chunk", uint32_t{2});
  params.set("thread_count", uint32_t{1});
  params.set("precompute_sdc", false);
  ASSERT_EQ(0, quantizer->init(meta, params));
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(4);
  for (size_t id = 0; id < 128; ++id)
    ASSERT_TRUE(holder->emplace(id, ailego::NumericalVector<float>(4, id)));
  ASSERT_EQ(0, quantizer->train(holder));
  std::vector<uint8_t> codes(128 * 2);
  // Deliberately give traversal wrong scores. Its beam still visits this whole
  // star, so exact refinement must recover the original pruning and graph.
  for (size_t id = 0; id < 128; ++id)
    quantizer->quantize_data(entity_->vectors[127 - id].data(),
                             codes.data() + id * 2);
  turbo::DistanceImpl approximate;
  ASSERT_EQ(0, quantizer->prepare_distance(entity_->vectors[65].data(),
                                           &approximate));
  ASSERT_GT(approximate(codes.data() + 65 * 2), 0.0f);

  for (const bool paged : {false, true}) {
    SCOPED_TRACE(paged);
    TemporaryBufferStorage::Pointer code_storage;
    context_->clear();
    entity_->graph = original;
    entity_->failed_vector = kInvalidId;
    entity_->guidance = {quantizer.get(), codes.data(), 2};
    if (paged) {
      std::string prefix;
      ASSERT_EQ(0, TemporaryBufferStorage::DefaultPrefix(
                       "zvec-diskann-graph-codes-test", &prefix));
      ASSERT_EQ(0, TemporaryBufferStorage::Create(prefix, codes.size(),
                                                  &code_storage));
      ASSERT_EQ(0, code_storage->write(0, codes.data(), codes.size()));
      entity_->guidance.codes = nullptr;
      entity_->guidance.paged_codes = code_storage->storage()->get("data");
      ASSERT_NE(nullptr, entity_->guidance.paged_codes);
    }
    ASSERT_EQ(0, algorithm.add_node(65, context_.get()));
    EXPECT_EQ(expected, entity_->graph);
    ASSERT_EQ(127u, context_->expanded_nodes().size());
    for (const auto &candidate : context_->expanded_nodes()) {
      const float difference = static_cast<float>(candidate.id) - 65.0f;
      EXPECT_FLOAT_EQ(4.0f * difference * difference, candidate.distance);
    }

    context_->clear();
    entity_->graph = original;
    entity_->failed_vector = 64;
    entity_->writes = 0;
    EXPECT_EQ(IndexError_IO, algorithm.add_node(65, context_.get()));
    EXPECT_EQ(original, entity_->graph);
    EXPECT_EQ(0u, entity_->writes);
    context_->clear();
    entity_->guidance = {};
  }
}

TEST_F(DiskAnnGraphConnectivityTest,
       ReadAheadCancelsOnConsumerExceptionAndPropagatesReadFailure) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto metric = IndexFactory::CreateMetric(meta.metric_name());
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  ReadAheadEntity entity(100);
  DistCalculator dc(&entity, meta, metric);
  std::array<diskann_id_t, 64> ids;
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = i;
  EXPECT_THROW(dc.visit_vector_batches(
                   ids.data(), ids.size(),
                   [](const diskann_id_t *, IndexStorage::MemoryBlock *,
                      size_t) -> int { throw std::bad_alloc(); }),
               std::bad_alloc);
  EXPECT_FALSE(entity.active);
  EXPECT_EQ(1u, entity.cancelled);
  entity.finish_error = IndexError_IO;
  EXPECT_EQ(IndexError_IO,
            dc.visit_vector_batches(
                ids.data(), ids.size(),
                [](const diskann_id_t *, IndexStorage::MemoryBlock *, size_t) {
                  return 0;
                }));
  EXPECT_FALSE(entity.active);
}

}  // namespace
}  // namespace zvec::core

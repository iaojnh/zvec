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

TEST_F(DiskAnnGraphConnectivityTest, ConnectedGraphIsUnchangedAndIdempotent) {
  initialize(4, 2);
  entity_->graph = {{1, 2}, {2, 3}, {0}, {1}};
  const auto original = entity_->graph;
  DiskAnnAlgorithm algorithm(*entity_, 2);
  size_t added = 99;
  ASSERT_EQ(0, algorithm.repair_connectivity(context_.get(), &added));
  EXPECT_EQ(0u, added);
  EXPECT_EQ(original, entity_->graph);
  EXPECT_EQ(0u, entity_->writes);
  EXPECT_EQ(0, algorithm.repair_connectivity(context_.get()));
  expect_reachable(2);
}

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

TEST_F(DiskAnnGraphConnectivityTest, FallsBackWhenBeamOnlyFindsFullTreeRow) {
  initialize(4, 2, 1);
  entity_->vectors[0].fill(0);
  entity_->vectors[1].fill(100);
  entity_->vectors[2].fill(200);
  entity_->vectors[3].fill(0.1F);
  entity_->graph = {{1, 2}, {0, 2}, {0, 1}, {0}};
  DiskAnnAlgorithm algorithm(*entity_, 2);
  ASSERT_EQ(0, algorithm.repair_connectivity(context_.get()));
  EXPECT_EQ((ailego::BudgetedVector<diskann_id_t>{1, 2}), entity_->graph[0]);
  EXPECT_EQ((ailego::BudgetedVector<diskann_id_t>{0, 3}), entity_->graph[1]);
  expect_reachable(2);
}

TEST_F(DiskAnnGraphConnectivityTest, AppendsWhenRowHasRoomAndHandlesSingleton) {
  initialize(4, 2);
  DiskAnnAlgorithm algorithm(*entity_, 2);
  ASSERT_EQ(0, algorithm.repair_connectivity(context_.get()));
  expect_reachable(2);
  initialize(1, 1);
  DiskAnnAlgorithm singleton(*entity_, 1);
  EXPECT_EQ(0, singleton.repair_connectivity(context_.get()));
  EXPECT_EQ(0u, entity_->writes);
  ConnectivityEntity empty(0);
  DiskAnnAlgorithm empty_algorithm(empty, 1);
  EXPECT_EQ(0, empty_algorithm.repair_connectivity(nullptr));
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

TEST_F(DiskAnnGraphConnectivityTest,
       RejectsMalformedGraphAndInvalidParameters) {
  initialize(2, 1);
  DiskAnnAlgorithm algorithm(*entity_, 1);
  entity_->graph = {{2}, {0}};
  EXPECT_EQ(IndexError_ReadData, algorithm.repair_connectivity(context_.get()));
  entity_->graph = {{0, 1}, {0}};
  EXPECT_EQ(IndexError_ReadData, algorithm.repair_connectivity(context_.get()));
  *entity_->mutable_medoid() = 2;
  EXPECT_EQ(IndexError_InvalidArgument,
            algorithm.repair_connectivity(context_.get()));
  *entity_->mutable_medoid() = 0;
  context_->set_list_size(0);
  EXPECT_EQ(IndexError_InvalidArgument,
            algorithm.repair_connectivity(context_.get()));
  EXPECT_EQ(IndexError_InvalidArgument, algorithm.repair_connectivity(nullptr));
  EXPECT_EQ(0u, entity_->writes);
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

TEST_F(DiskAnnGraphConnectivityTest, SeedGraphHasDistinctBoundedNonSelfEdges) {
  initialize(64, 8);
  DiskAnnAlgorithm algorithm(*entity_, 8);
  ASSERT_EQ(0, algorithm.initialize_graph());
  for (size_t id = 0; id < entity_->graph.size(); ++id) {
    const auto &neighbors = entity_->graph[id];
    ASSERT_EQ(8u, neighbors.size());
    std::array<bool, 64> seen{};
    for (const auto neighbor : neighbors) {
      ASSERT_LT(neighbor, seen.size());
      EXPECT_NE(id, neighbor);
      EXPECT_FALSE(seen[neighbor]);
      seen[neighbor] = true;
    }
  }
}

TEST_F(DiskAnnGraphConnectivityTest, SeedGraphHandlesSingleton) {
  initialize(1, 8);
  DiskAnnAlgorithm algorithm(*entity_, 8);
  ASSERT_EQ(0, algorithm.initialize_graph());
  EXPECT_TRUE(entity_->graph[0].empty());
}

TEST_F(DiskAnnGraphConnectivityTest, SeedGraphPropagatesWriteFailure) {
  initialize(4, 2);
  entity_->fail_write = true;
  DiskAnnAlgorithm algorithm(*entity_, 2);
  EXPECT_EQ(IndexError_WriteData, algorithm.initialize_graph());
  EXPECT_EQ(0u, entity_->writes);
}

TEST_F(DiskAnnGraphConnectivityTest, SeedGraphAdmitsWorkspaceBeforeMutation) {
  initialize(4, 2);
  DiskAnnAlgorithm algorithm(*entity_, 2);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t charge = pool.available();
  ASSERT_TRUE(pool.try_charge_external(charge));
  const int ret = algorithm.initialize_graph();
  pool.release_external(charge);
  EXPECT_EQ(IndexError_NoMemory, ret);
  EXPECT_EQ(0u, entity_->writes);
  ASSERT_EQ(0, algorithm.initialize_graph());
  EXPECT_EQ(4u, entity_->writes);
}

TEST_F(DiskAnnGraphConnectivityTest, InsertionPreservesUnsearchedExistingEdge) {
  initialize(4, 2, 4);
  entity_->vectors[1].fill(10.0f);
  entity_->vectors[3].fill(10.1f);
  entity_->graph[1] = {3};
  // The medoid cannot reach node 3 yet. It is a valid reciprocal edge added
  // before node 1's own insertion, and must participate in robust pruning.
  DiskAnnAlgorithm algorithm(*entity_, 2);
  ASSERT_EQ(0, algorithm.add_node(1, context_.get()));
  const auto &neighbors = entity_->graph[1];
  EXPECT_NE(neighbors.end(), std::find(neighbors.begin(), neighbors.end(), 3));
  EXPECT_LE(neighbors.size(), 2u);
}

TEST_F(DiskAnnGraphConnectivityTest, WorkspaceIsChargedAndReleasedWithMapping) {
  initialize(4, 1);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t before = pool.external_used();
  {
    DiskAnnGraphConnectivity workspace(*entity_, 1);
    EXPECT_EQ(before + ailego::BudgetedMappedBuffer::mapping_size(4 * 8),
              pool.external_used());
  }
  EXPECT_EQ(before, pool.external_used());
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
       BufferedPruningReusesCandidateVectorsAndPreservesNeighbors) {
  initialize(65, 8);
  for (size_t id = 0; id < entity_->vectors.size(); ++id) {
    for (size_t d = 0; d < 4; ++d) {
      entity_->vectors[id][d] = static_cast<float>((id * (17 + d * 13)) % 101);
    }
  }
  for (diskann_id_t id = 1; id < 65; ++id) entity_->graph[0].push_back(id);
  const auto original = entity_->graph[0];
  DiskAnnAlgorithm algorithm(*entity_, 8);
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  const auto expected = entity_->graph[0];
  const auto scalar_reads = entity_->vector_reads;
  const auto scalar_comparisons = context_->dist_calculator().compare_cnt();
  ASSERT_FALSE(expected.empty());
  entity_->graph[0] = original;
  entity_->vector_reads = 0;
  entity_->batch_vectors = true;
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  EXPECT_EQ(expected, entity_->graph[0]);
  EXPECT_EQ(scalar_comparisons, context_->dist_calculator().compare_cnt());
  EXPECT_EQ(original.size() + 1u, entity_->vector_reads);
  EXPECT_LT(entity_->vector_reads, scalar_reads);
  EXPECT_EQ(2u, entity_->vector_batches);

  // Fail after retaining the first batch and partially filling the second.
  entity_->graph[0] = original;
  entity_->fail_batch_at = entity_->vector_batches + 2;
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  EXPECT_EQ(expected, entity_->graph[0]);
  EXPECT_EQ(scalar_comparisons, context_->dist_calculator().compare_cnt());
  entity_->fail_batch_at = 0;

  entity_->graph[0] = original;
  entity_->batch_error = IndexError_NoMemory;
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  EXPECT_EQ(expected, entity_->graph[0]);
  EXPECT_EQ(scalar_comparisons, context_->dist_calculator().compare_cnt());

  entity_->graph[0] = original;
  entity_->batch_error = 0;
  auto &budget = ailego::PoolPayloadBudget::get_instance();
  constexpr size_t remaining = 2048;
  ASSERT_GT(original.size() * sizeof(IndexStorage::MemoryBlock), remaining);
  ASSERT_GT(budget.available(), remaining);
  const auto charge = budget.available() - remaining;
  ASSERT_TRUE(budget.try_charge_external(charge));
  const auto batches_before = entity_->vector_batches;
  const int ret = algorithm.prune_node(0, context_.get());
  budget.release_external(charge);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(expected, entity_->graph[0]);
  EXPECT_GT(entity_->vector_batches, batches_before);
  EXPECT_EQ(scalar_comparisons, context_->dist_calculator().compare_cnt());
}

TEST_F(DiskAnnGraphConnectivityTest, BufferedPruningPropagatesReadFailure) {
  initialize(12, 2);
  entity_->graph[0] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const auto original = entity_->graph;
  entity_->batch_vectors = true;
  entity_->batch_error = IndexError_IO;
  DiskAnnAlgorithm algorithm(*entity_, 2);
  EXPECT_EQ(IndexError_IO, algorithm.prune_node(0, context_.get()));
  EXPECT_EQ(original, entity_->graph);
  EXPECT_EQ(0u, entity_->writes);
}

TEST_F(DiskAnnGraphConnectivityTest,
       CenterBatchPreservesDistanceOrientationAndFallsBackOnMemoryPressure) {
  initialize(40, 8);
  auto &dc = context_->dist_calculator();
  dc.update_distance([](const void *lhs, const void *rhs, size_t, float *out) {
    *out = 3.0f * static_cast<const float *>(lhs)[0] +
           static_cast<const float *>(rhs)[0];
  });
  const std::array<diskann_id_t, 5> ids{{1, 7, 13, 2, 39}};
  std::array<dist_t, 5> expected, actual;
  for (size_t i = 0; i < ids.size(); ++i) expected[i] = dc.dist(3, ids[i]);
  dc.clear();
  dc.dist_batch(3, ids.data(), ids.size(), actual.data());
  ASSERT_FALSE(dc.error());
  EXPECT_EQ(expected, actual);
  EXPECT_EQ(ids.size(), dc.compare_cnt());
  dc.clear();
  entity_->batch_error = IndexError_NoMemory;
  dc.dist_batch(3, ids.data(), ids.size(), actual.data());
  ASSERT_FALSE(dc.error());
  EXPECT_EQ(expected, actual);
  EXPECT_EQ(ids.size(), dc.compare_cnt());
  dc.clear();
  entity_->batch_error = IndexError_IO;
  dc.dist_batch(3, ids.data(), ids.size(), actual.data());
  EXPECT_EQ(IndexError_IO, dc.error_code());
}

TEST_F(DiskAnnGraphConnectivityTest, BufferedSelfOnlyPruningNeedsNoVectorRead) {
  initialize(4, 1);
  entity_->graph[0] = {0, 0, 0};
  entity_->batch_vectors = true;
  entity_->failed_vector = 0;
  DiskAnnAlgorithm algorithm(*entity_, 1);
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  EXPECT_TRUE(entity_->graph[0].empty());
  EXPECT_EQ(0u, entity_->vector_reads);
}

TEST_F(DiskAnnGraphConnectivityTest,
       OptionalCandidatePinsYieldToSmallWorkspace) {
  initialize(65, 8);
  for (diskann_id_t id = 1; id < 65; ++id) entity_->graph[0].push_back(id);
  const auto original = entity_->graph[0];
  DiskAnnAlgorithm algorithm(*entity_, 8);
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  const auto expected = entity_->graph[0];
  const auto comparisons = context_->dist_calculator().compare_cnt();
  entity_->batch_vectors = true;
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  for (size_t remaining = 2048; remaining <= 8192; remaining += 256) {
    SCOPED_TRACE(remaining);
    entity_->graph[0] = original;
    ailego::BudgetedVector<float>().swap(context_->occlude_factor());
    ASSERT_GT(pool.available(), remaining);
    const auto charge = pool.available() - remaining;
    ASSERT_TRUE(pool.try_charge_external(charge));
    const int ret = algorithm.prune_node(0, context_.get());
    pool.release_external(charge);
    ASSERT_EQ(0, ret);
    EXPECT_EQ(expected, entity_->graph[0]);
    EXPECT_EQ(comparisons, context_->dist_calculator().compare_cnt());
  }
}

TEST_F(DiskAnnGraphConnectivityTest,
       BufferedPruningKeepsDuplicateAndSelfFilteringAcrossBatchBoundaries) {
  initialize(80, 8);
  for (diskann_id_t id = 1; id < 80; ++id) {
    entity_->graph[0].push_back(id);
    if (id % 7 == 0) entity_->graph[0].push_back(id);
    if (id % 11 == 0) entity_->graph[0].push_back(0);
  }
  const auto original = entity_->graph[0];
  DiskAnnAlgorithm algorithm(*entity_, 8);
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  const auto expected = entity_->graph[0];
  const auto comparisons = context_->dist_calculator().compare_cnt();
  entity_->graph[0] = original;
  entity_->batch_vectors = true;
  ASSERT_EQ(0, algorithm.prune_node(0, context_.get()));
  EXPECT_EQ(expected, entity_->graph[0]);
  EXPECT_EQ(comparisons, context_->dist_calculator().compare_cnt());
}

TEST_F(DiskAnnGraphConnectivityTest,
       ReadAheadOverlapsNextBatchAndPreservesScoresAndComparisonOrder) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto metric = IndexFactory::CreateMetric(meta.metric_name());
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  ReadAheadEntity entity(100);
  DistCalculator dc(&entity, meta, metric);
  dc.reset_query(entity.vectors[0].data());
  std::array<diskann_id_t, 96> ids;
  std::array<float, 96> expected, actual;
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i] = (i * 37) % 99 + 1;
    expected[i] = dc.dist(ids[i]);
  }
  std::vector<bool> overlapped;
  const auto run = [&] {
    dc.clear();
    size_t offset = 0;
    const int ret = dc.visit_vector_batches(
        ids.data(), ids.size(),
        [&](const diskann_id_t *batch, IndexStorage::MemoryBlock *blocks,
            size_t count) {
          overlapped.push_back(entity.active);
          for (size_t i = 0; i < count; ++i) {
            EXPECT_EQ(ids[offset], batch[i]);
            actual[offset++] = dc.dist(blocks[i].data());
          }
          return 0;
        });
    EXPECT_EQ(0, ret);
    EXPECT_EQ(expected, actual);
    EXPECT_EQ(ids.size(), dc.compare_cnt());
  };
  run();
  EXPECT_EQ((std::vector<bool>{true, true, false}), overlapped);
  entity.submit_error = IndexError_NoMemory;
  overlapped.clear();
  run();
  EXPECT_FALSE(entity.active);
  EXPECT_EQ(0u, entity.cancelled);
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

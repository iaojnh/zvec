// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "diskann_graph_connectivity.h"
#include <array>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
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
    if (id == failed_vector) return IndexError_IO;
    block = IndexStorage::MemoryBlock::MakeBorrowedView(
        const_cast<float *>(vectors.at(id).data()));
    return 0;
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

  std::vector<std::array<float, 4>> vectors;
  std::vector<ailego::BudgetedVector<diskann_id_t>> graph;
  diskann_id_t failed_vector{kInvalidId};
  diskann_id_t failed_read{kInvalidId};
  bool fail_write{false};
  size_t writes{0};
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

}  // namespace
}  // namespace zvec::core

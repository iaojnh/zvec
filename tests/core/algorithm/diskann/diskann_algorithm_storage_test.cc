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

#include <algorithm>
#include <array>
#include <memory>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "ailego/pattern/defer.h"
#include "diskann_algorithm.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

TEST(DiskAnnPruneWorkspace, BoundedHistoryMatchesFullSortAndTruncate) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(32 * 1024));
  std::vector<Neighbor> input;
  for (uint32_t i = 0; i < 10000; ++i)
    input.emplace_back(i, static_cast<float>(i % 97));
  // Duplicate IDs, distance ties, and the query itself are preserved/excluded
  // in exactly the same way as the old full-history collection.
  input.emplace_back(42, 42);
  input.emplace_back(123, 0);
  std::mt19937 random(31);
  for (int order = 0; order < 3; ++order) {
    if (order == 1) std::reverse(input.begin(), input.end());
    if (order == 2) std::shuffle(input.begin(), input.end(), random);
    std::vector<Neighbor> expected;
    PruneCandidates actual;
    for (const auto &candidate : input) {
      if (candidate.id != 123) expected.push_back(candidate);
      CollectPruneCandidate(actual, candidate, 123, 750);
      ASSERT_LE(actual.size(), 750u);
    }
    std::sort(expected.begin(), expected.end());
    expected.resize(750);
    std::sort(actual.begin(), actual.end());
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(expected[i].id, actual[i].id);
      EXPECT_EQ(expected[i].distance, actual[i].distance);
    }
    EXPECT_LT(pool.used(), 16u * 1024u);
  }
}

TEST(DiskAnnPruneWorkspace, RejectedHistoryGrowthLeavesReusableCandidates) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(128));
  PruneCandidates candidates;
  EXPECT_THROW(
      {
        for (uint32_t i = 0; i < 100; ++i)
          CollectPruneCandidate(candidates, Neighbor(i, float(i)), kInvalidId,
                                750);
      },
      std::bad_alloc);
  EXPECT_TRUE(std::is_heap(candidates.begin(), candidates.end()));
  EXPECT_LE(pool.used(), pool.capacity());
  PruneCandidates().swap(candidates);
  EXPECT_EQ(0u, pool.used());
  EXPECT_NO_THROW(
      CollectPruneCandidate(candidates, Neighbor(1, 0), kInvalidId, 1));
}

TEST(DiskAnnEncodeWorkspace, AdaptsBatchAndRejectsBeforeAllocation) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(8192));
  {
    DiskAnnEncodeBuffer buffer;
    ASSERT_EQ(0, buffer.init(1000000, 3072, 384));
    EXPECT_EQ(1u, buffer.rows());
    EXPECT_EQ(buffer.input() + 3072, buffer.codes());
    EXPECT_GE(pool.used(), 3456u);
    EXPECT_LE(pool.used(), pool.capacity());
    std::memset(buffer.input(), 1, 3072);
    std::memset(buffer.codes(), 2, 384);
    EXPECT_EQ(1, buffer.input()[3071]);
    EXPECT_EQ(2, buffer.codes()[383]);
  }
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(16));
  DiskAnnEncodeBuffer buffer;
  EXPECT_EQ(IndexError_NoMemory, buffer.init(1000000, 3072, 384));
  EXPECT_EQ(0u, buffer.rows());
  EXPECT_EQ(0u, pool.used());
  EXPECT_EQ(IndexError_InvalidArgument, buffer.init(1, SIZE_MAX, 384));
  ASSERT_EQ(0, pool.init(64 * 1024));
  EXPECT_EQ(0, buffer.init(1000000, 3072, 384));
  EXPECT_GT(buffer.rows(), 1u);
  EXPECT_LE(pool.used(), pool.capacity());
}

// A paged entity must not expose an unpinned get_vector pointer. Every read
// returns a fresh owned vector, allowing the tests to observe its lifetime.
class TransientEntity : public DiskAnnEntity {
 public:
  TransientEntity() {
    meta_header_.doc_cnt = 4;
    meta_header_.medoid = 0;
    for (size_t i = 0; i < vectors.size(); ++i) {
      vectors[i].fill(static_cast<float>(i));
    }
  }

  diskann_id_t get_id(diskann_key_t key) const override {
    return static_cast<diskann_id_t>(key);
  }

  diskann_key_t get_key(diskann_id_t id) const override {
    return id;
  }

  int read_vector(diskann_id_t id,
                  IndexStorage::MemoryBlock &block) const override {
    block = IndexStorage::MemoryBlock();
    if (id >= vectors.size()) {
      return IndexError_OutOfRange;
    }
    const size_t reads = vector_reads[id]++;
    if (id == failed_vector && reads >= failed_vector_after) {
      return IndexError_IO;
    }
    auto *copy = new std::array<float, 4>(vectors[id]);
    ++live_vectors;
    peak_live_vectors = std::max(peak_live_vectors, live_vectors);
    std::shared_ptr<void> owner(copy, [this](void *ptr) {
      delete static_cast<std::array<float, 4> *>(ptr);
      --live_vectors;
    });
    block = IndexStorage::MemoryBlock::MakeSharedView(copy->data(), owner);
    return 0;
  }

  int read_neighbors(
      diskann_id_t id,
      ailego::BudgetedVector<diskann_id_t> *result) const override {
    if (id == failed_neighbors) {
      return IndexError_IO;
    }
    *result = neighbors.at(id);
    return 0;
  }

  int set_neighbors(
      diskann_id_t id,
      const ailego::BudgetedVector<diskann_id_t> &result) override {
    if (id == failed_set) {
      return IndexError_WriteData;
    }
    neighbors.at(id) = result;
    return 0;
  }

  int add_neighbor(diskann_id_t id, diskann_id_t neighbor) override {
    if (id == failed_add) {
      return IndexError_NoBuffer;
    }
    neighbors.at(id).push_back(neighbor);
    return 0;
  }

  std::array<std::array<float, 4>, 4> vectors{};
  std::array<ailego::BudgetedVector<diskann_id_t>, 4> neighbors{};
  diskann_id_t failed_vector{kInvalidId};
  diskann_id_t failed_neighbors{kInvalidId};
  diskann_id_t failed_set{kInvalidId};
  diskann_id_t failed_add{kInvalidId};
  size_t failed_vector_after{0};
  mutable std::array<size_t, 4> vector_reads{};
  mutable size_t live_vectors{0};
  mutable size_t peak_live_vectors{0};
};

class DiskAnnAlgorithmStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    meta_.set_meta(IndexMeta::DataType::DT_FP32, 4);
    meta_.set_metric("SquaredEuclidean", 0, ailego::Params());
    metric_ = IndexFactory::CreateMetric(meta_.metric_name());
    ASSERT_NE(metric_, nullptr);
    ASSERT_EQ(metric_->init(meta_, ailego::Params()), 0);
    entity_ = std::make_shared<TransientEntity>();
    context_ = std::make_unique<DiskAnnContext>(meta_, metric_, entity_);
    ASSERT_EQ(context_->init(DiskAnnContext::kBuilderContext, 2, 0,
                             meta_.element_size(), false),
              0);
    context_->set_list_size(4);
  }

  IndexMeta meta_;
  IndexMetric::Pointer metric_;
  std::shared_ptr<TransientEntity> entity_;
  DiskAnnContext::Pointer context_;
};

TEST(DiskAnnPruneWorkspace, LocksScaleDownAndStayChargedUntilDestruction) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  TransientEntity entity;
  ASSERT_EQ(0, pool.init(2 * sizeof(std::mutex) + alignof(std::mutex) - 1 +
                         sizeof(size_t)));
  {
    DiskAnnAlgorithm algorithm(entity, 2, 128);
    EXPECT_EQ(2u, algorithm.lock_count());
    EXPECT_GE(pool.used(), 2 * sizeof(std::mutex));
    EXPECT_LE(pool.used(), pool.capacity());
  }
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1));
  EXPECT_THROW(DiskAnnAlgorithm(entity, 2, 1), std::bad_alloc);
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(4096));
  DiskAnnAlgorithm retry(entity, 2, 1);
  EXPECT_EQ(4u, retry.lock_count());
}

TEST_F(DiskAnnAlgorithmStorageTest, DistancePinsBothVectorsUntilComparison) {
  auto &dc = context_->dist_calculator();
  EXPECT_FLOAT_EQ(dc.dist(diskann_id_t{0}, diskann_id_t{2}), 16.0f);
  EXPECT_FALSE(dc.error());
  EXPECT_EQ(entity_->peak_live_vectors, 2u);
  EXPECT_EQ(entity_->live_vectors, 0u);
}

TEST_F(DiskAnnAlgorithmStorageTest,
       DistanceReleasesLeftVectorOnRightReadError) {
  entity_->failed_vector = 2;
  auto &dc = context_->dist_calculator();
  dc.dist(diskann_id_t{0}, diskann_id_t{2});
  EXPECT_EQ(dc.error_code(), IndexError_IO);
  EXPECT_EQ(entity_->live_vectors, 0u);
}

TEST_F(DiskAnnAlgorithmStorageTest, QueryUsesContextOwnedCopy) {
  std::array<float, 4> query{{1, 1, 1, 1}};
  context_->reset_query(query.data());
  query.fill(99);
  EXPECT_FLOAT_EQ(context_->dist_calculator().dist(diskann_id_t{0}), 4.0f);
}

TEST_F(DiskAnnAlgorithmStorageTest, EmptyAdjacencyStillAddsNode) {
  DiskAnnAlgorithm algorithm(*entity_, 2);
  EXPECT_EQ(algorithm.add_node(1, context_.get()), 0);
  EXPECT_EQ(entity_->neighbors[1], ailego::BudgetedVector<diskann_id_t>({0}));
  EXPECT_EQ(entity_->neighbors[0], ailego::BudgetedVector<diskann_id_t>({1}));
  EXPECT_EQ(entity_->live_vectors, 0u);
}

TEST_F(DiskAnnAlgorithmStorageTest, AddPropagatesQueryAndTraversalReadErrors) {
  DiskAnnAlgorithm algorithm(*entity_, 2);
  entity_->failed_vector = 1;
  EXPECT_EQ(algorithm.add_node(1, context_.get()), IndexError_IO);

  entity_->failed_vector = 0;
  EXPECT_EQ(algorithm.add_node(1, context_.get()), IndexError_IO);

  entity_->failed_vector = kInvalidId;
  entity_->failed_neighbors = 0;
  context_->clear();
  EXPECT_EQ(algorithm.add_node(1, context_.get()), IndexError_IO);
  EXPECT_EQ(entity_->live_vectors, 0u);
}

TEST_F(DiskAnnAlgorithmStorageTest, AddPropagatesForwardAndReverseWriteErrors) {
  DiskAnnAlgorithm algorithm(*entity_, 2);
  entity_->failed_set = 1;
  EXPECT_EQ(algorithm.add_node(1, context_.get()), IndexError_WriteData);

  entity_->failed_set = kInvalidId;
  entity_->failed_add = 0;
  context_->clear();
  EXPECT_EQ(algorithm.add_node(1, context_.get()), IndexError_NoBuffer);
}

TEST_F(DiskAnnAlgorithmStorageTest, PrunePropagatesReadAndWriteErrors) {
  DiskAnnAlgorithm algorithm(*entity_, 1);
  entity_->neighbors[0] = {1, 2, 3};
  entity_->failed_neighbors = 0;
  EXPECT_EQ(algorithm.prune_node(0, context_.get()), IndexError_IO);

  entity_->failed_neighbors = kInvalidId;
  entity_->failed_vector = 2;
  EXPECT_EQ(algorithm.prune_node(0, context_.get()), IndexError_IO);

  entity_->failed_vector = kInvalidId;
  entity_->failed_set = 0;
  EXPECT_EQ(algorithm.prune_node(0, context_.get()), IndexError_WriteData);
  EXPECT_EQ(entity_->live_vectors, 0u);
}

TEST_F(DiskAnnAlgorithmStorageTest, PrunePropagatesOcclusionReadError) {
  DiskAnnAlgorithm algorithm(*entity_, 1);
  entity_->neighbors[0] = {1, 2, 3};
  entity_->failed_vector = 2;
  // Candidate-distance calculation reads vector 2 once successfully. The
  // second read happens in occlude_list when comparing candidates directly.
  entity_->failed_vector_after = 1;
  EXPECT_EQ(algorithm.prune_node(0, context_.get()), IndexError_IO);
  EXPECT_EQ(entity_->neighbors[0],
            ailego::BudgetedVector<diskann_id_t>({1, 2, 3}));
  EXPECT_EQ(entity_->live_vectors, 0u);
}

TEST_F(DiskAnnAlgorithmStorageTest, AddPropagatesReversePruneWriteError) {
  DiskAnnAlgorithm algorithm(*entity_, 1);
  entity_->neighbors[0] = {2};
  entity_->failed_set = 0;
  EXPECT_EQ(algorithm.add_node(1, context_.get()), IndexError_WriteData);
  EXPECT_EQ(entity_->live_vectors, 0u);
}

class LegacyEntity : public TransientEntity {
 public:
  const void *get_vector(diskann_id_t id) const override {
    return id < vectors.size() ? vectors[id].data() : nullptr;
  }

  std::pair<uint32_t, const diskann_id_t *> get_neighbors(
      diskann_id_t id) const override {
    if (id == kInvalidId) {
      return {1, nullptr};
    }
    return {static_cast<uint32_t>(neighbors.at(id).size()),
            neighbors.at(id).data()};
  }
};

TEST(DiskAnnEntityStorageTest, LegacyViewsValidateNullAndAllowEmptyAdjacency) {
  LegacyEntity entity;
  IndexStorage::MemoryBlock block;
  EXPECT_EQ(entity.DiskAnnEntity::read_vector(0, block), 0);
  EXPECT_EQ(block.data(), entity.vectors[0].data());
  EXPECT_EQ(entity.DiskAnnEntity::read_vector(kInvalidId, block),
            IndexError_ReadData);
  EXPECT_EQ(block.data(), nullptr);

  ailego::BudgetedVector<diskann_id_t> neighbors{3};
  EXPECT_EQ(entity.DiskAnnEntity::read_neighbors(0, &neighbors), 0);
  EXPECT_TRUE(neighbors.empty());
  EXPECT_EQ(entity.DiskAnnEntity::read_neighbors(kInvalidId, &neighbors),
            IndexError_ReadData);
  EXPECT_EQ(entity.DiskAnnEntity::read_neighbors(0, nullptr),
            IndexError_InvalidArgument);
}

}  // namespace
}  // namespace zvec::core

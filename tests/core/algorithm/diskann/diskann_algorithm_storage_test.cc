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

}  // namespace
}  // namespace zvec::core

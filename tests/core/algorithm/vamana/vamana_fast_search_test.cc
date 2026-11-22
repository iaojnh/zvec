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
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "tests/test_util.h"
#include "vamana_algorithm.h"

namespace zvec::core {
namespace {

// Exercise the actual contiguous algorithm with an exact hand-written graph.
// Only distance calls are observed: no test branches in the production loop.
class VamanaFastSearchTest : public testing::Test {
 protected:
  void SetUp() override {
    test_util::RemoveTestPath(path_);
  }

  void TearDown() override {
    context_.reset();
    algorithm_.reset();
    if (entity_) entity_->close();
    entity_.reset();
    if (storage_) storage_->close();
    storage_.reset();
    test_util::RemoveTestPath(path_);
  }

  void create_graph(const std::vector<float> &values,
                    const std::vector<std::vector<node_id_t>> &rows) {
    ASSERT_EQ(values.size(), rows.size());
    storage_ = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(storage_);
    ASSERT_EQ(0, storage_->init(ailego::Params()));
    ASSERT_EQ(0, storage_->open(path_, true));
    entity_ = std::make_shared<VamanaContiguousStreamerEntity>(stats_);
    entity_->set_vector_size(sizeof(float) * kDimension);
    entity_->set_max_degree(8);
    entity_->set_search_list_size(128);
    entity_->set_max_occlusion_size(128);
    ASSERT_EQ(0, entity_->init(values.size()));
    ASSERT_EQ(0, entity_->open(storage_, 0, false));
    for (node_id_t i = 0; i < values.size(); ++i) {
      std::array<float, kDimension> vector{};
      vector[0] = values[i];
      node_id_t id = kInvalidNodeId;
      ASSERT_EQ(0, entity_->add_vector(i, vector.data(), &id));
      ASSERT_EQ(i, id);
    }
    for (node_id_t i = 0; i < rows.size(); ++i) {
      std::vector<std::pair<node_id_t, dist_t>> neighbors;
      for (node_id_t id : rows[i]) neighbors.emplace_back(id, 0.0f);
      ASSERT_EQ(0, entity_->update_neighbors(i, neighbors));
    }
    entity_->update_entry_point(0);
    ASSERT_EQ(0, entity_->build_contiguous_memory());
    ASSERT_TRUE(entity_->is_contiguous());
    for (node_id_t i = 0; i < values.size(); ++i) {
      ids_[entity_->get_vector_ptr(i)] = i;
      // Entry-point distance uses the canonical (mmap) accessor.
      ids_[entity_->get_vector(i)] = i;
    }
    metric_ = IndexFactory::CreateMetric("SquaredEuclidean");
    ASSERT_TRUE(metric_);
    ASSERT_EQ(0,
              metric_->init(IndexMeta(IndexMeta::DataType::DT_FP32, kDimension),
                            ailego::Params()));
    algorithm_ =
        std::make_unique<VamanaAlgorithm<VamanaContiguousStreamerEntity>>(
            *entity_);
    make_context();
  }

  void make_context(std::optional<VisitFilter::Mode> mode = std::nullopt) {
    context_ = std::make_unique<VamanaContext>(kDimension, metric_, entity_);
    if (mode) context_->set_filter_mode(*mode);
    context_->set_max_scan_num(10000);
    ASSERT_EQ(0, context_->init(VamanaContext::kSearcherContext));
    const auto batch = metric_->batch_distance();
    context_->update_dist_calculator_distance(
        metric_->distance(),
        [this, batch](const void **vectors, const void *query, size_t count,
                      size_t dim, float *out, const void **extra) {
          std::vector<node_id_t> row;
          for (size_t i = 0; i < count; ++i) row.push_back(ids_.at(vectors[i]));
          evaluated_.push_back(std::move(row));
          batch(vectors, query, count, dim, out, extra);
        });
  }

  void search(float value, uint32_t capacity) {
    ASSERT_TRUE(context_);
    evaluated_.clear();
    context_->clear();
    context_->set_ef(capacity);
    context_->set_topk(capacity);
    std::array<float, kDimension> query{};
    query[0] = value;
    context_->reset_query(query.data());
    ASSERT_EQ(0, algorithm_->search(context_.get()));
    ASSERT_FALSE(context_->error());
    context_->topk_to_result();
  }

  std::vector<std::pair<uint64_t, float>> results() const {
    std::vector<std::pair<uint64_t, float>> result;
    for (const auto &doc : context_->result()) {
      result.emplace_back(doc.key(), doc.score());
    }
    return result;
  }

  static constexpr size_t kDimension = 16;
  const std::string path_{"vamana_fast_search_test.index"};
  IndexStreamer::Stats stats_;
  IndexStorage::Pointer storage_;
  std::shared_ptr<VamanaContiguousStreamerEntity> entity_;
  IndexMetric::Pointer metric_;
  std::unique_ptr<VamanaAlgorithm<VamanaContiguousStreamerEntity>> algorithm_;
  VamanaContext::Pointer context_;
  std::unordered_map<const void *, node_id_t> ids_;
  std::vector<std::vector<node_id_t>> evaluated_;
};

TEST(VamanaReadFailureTest, QueriesFailAndConstructionKeepsReadableCandidates) {
  class FaultingChunk : public IndexSegmentStorage::Segment {
   public:
    FaultingChunk(const Chunk::Pointer &chunk, size_t adjacency, size_t vector)
        : IndexSegmentStorage::Segment(chunk, 0, chunk->data_size(),
                                       chunk->padding_size(), 0),
          chunk_(chunk), adjacency_(adjacency), vector_(vector) {}

    size_t read(size_t offset, IndexStorage::MemoryBlock &block,
                size_t size) override {
      if ((offset == adjacency_ && adjacency_reads_until_failure &&
           --adjacency_reads_until_failure == 0) ||
          (offset == vector_ && std::exchange(fail_neighbor_vector, false))) {
        block.reset();
        return 0;
      }
      return chunk_->read(offset, block, size);
    }

    size_t write(size_t offset, const void *data, size_t size) override {
      return chunk_->write(offset, data, size);
    }

    int read_error() const override {
      return IndexError_NoMemory;
    }

    int adjacency_reads_until_failure{0};
    bool fail_neighbor_vector{false};

   private:
    Chunk::Pointer chunk_;
    size_t adjacency_;
    size_t vector_;
  };

  class FaultingEntity : public VamanaBufferPoolStreamerEntity {
   public:
    using VamanaBufferPoolStreamerEntity::VamanaBufferPoolStreamerEntity;

    std::shared_ptr<FaultingChunk> inject_read_faults() {
      auto faulting_chunk = std::make_shared<FaultingChunk>(
          node_chunks_.front(), vector_size() + sizeof(zvec::core::key_t),
          node_size());
      node_chunks_.front() = faulting_chunk;
      return faulting_chunk;
    }
  };

  constexpr uint32_t dimension = 16;
  const std::string path = "vamana_read_failure_test.index";
  test_util::RemoveTestPath(path);
  const auto cleanup = ailego::ScopeGuard::Make(
      [&] { test_util::RemoveTestPath(path); });
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  IndexStreamer::Stats stats;
  auto entity = std::make_shared<FaultingEntity>(stats);
  entity->set_vector_size(sizeof(float) * dimension);
  entity->set_max_degree(8);
  entity->set_search_list_size(32);
  ASSERT_EQ(0, entity->init(3));
  ASSERT_EQ(0, entity->open(storage, 0, false));

  std::array<float, dimension> vector{};
  for (node_id_t i = 0; i < 3; ++i) {
    vector[0] = static_cast<float>(i);
    node_id_t id = kInvalidNodeId;
    ASSERT_EQ(0, entity->add_vector(i, vector.data(), &id));
    ASSERT_EQ(i, id);
  }
  ASSERT_EQ(0, entity->update_neighbors(0, {{1, 1.0f}}));
  entity->update_entry_point(0);
  auto faults = entity->inject_read_faults();

  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);
  ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DT_FP32, dimension), {}));
  VamanaAlgorithm<VamanaBufferPoolStreamerEntity> algorithm(*entity);
  VamanaContext context(dimension, metric, entity);
  ASSERT_EQ(0, context.init(VamanaContext::kStreamerContext));
  context.set_max_scan_num(1000);
  context.set_topk(3);
  context.reset_query(vector.data());
  faults->adjacency_reads_until_failure = 1;
  EXPECT_EQ(IndexError_NoMemory, algorithm.search(&context));

  context.clear();
  faults->fail_neighbor_vector = true;
  faults->adjacency_reads_until_failure = 2;
  ASSERT_EQ(0, algorithm.add_node(2, &context));
  EXPECT_FALSE(faults->fail_neighbor_vector);
  EXPECT_EQ(0, faults->adjacency_reads_until_failure);
  const auto existing = entity->get_neighbors(0);
  ASSERT_EQ(1U, existing.size());
  EXPECT_EQ(1U, existing[0]);
  const auto inserted = entity->get_neighbors(2);
  ASSERT_EQ(1U, inserted.size());
  EXPECT_EQ(0U, inserted[0]);

  context.clear();
  context.resize_results(1);
  context.set_fetch_vector(true);
  vector[0] = 0.0f;
  context.reset_query(vector.data());
  ASSERT_EQ(0, algorithm.search(&context));

  faults->fail_neighbor_vector = true;
  EXPECT_EQ(IndexError_NoMemory, context.topk_to_result(0));
  EXPECT_FALSE(faults->fail_neighbor_vector);
  EXPECT_TRUE(context.result().empty());

  ASSERT_EQ(0, context.topk_to_result(0));
  ASSERT_GE(context.result().size(), 2u);
  for (const auto &doc : context.result()) {
    EXPECT_NE(nullptr, doc.vector());
  }
  ASSERT_EQ(0, entity->close());
  ASSERT_EQ(0, storage->close());
}

TEST_F(VamanaFastSearchTest, ContextDefaultsToBitmapAndAllowsExplicitOverride) {
  create_graph({10, 2}, {{1}, {0}});
  EXPECT_EQ(VisitFilter::BitMap, context_->visit_filter().get_mode());
  for (auto type :
       {VamanaContext::kBuilderContext, VamanaContext::kSearcherContext,
        VamanaContext::kStreamerContext}) {
    SCOPED_TRACE(type);
    VamanaContext context(kDimension, metric_, entity_);
    ASSERT_EQ(0, context.init(type));
    auto &visit = context.visit_filter();
    EXPECT_EQ(VisitFilter::BitMap, visit.get_mode());
    EXPECT_FALSE(visit.visited(1));
    visit.set_visited(1);
    EXPECT_TRUE(visit.visited(1));
    visit.clear();
    EXPECT_FALSE(visit.visited(1));

    VamanaContext explicit_context(kDimension, metric_, entity_);
    explicit_context.set_filter_mode(VisitFilter::ByteMap);
    ASSERT_EQ(0, explicit_context.init(type));
    EXPECT_EQ(VisitFilter::ByteMap, explicit_context.visit_filter().get_mode());
  }
}

TEST_F(VamanaFastSearchTest, LocalOptimumReusesRowThenPoolFindsBetterPoint) {
  create_graph({10, 2, 3, 1}, {{1, 2}, {0, 2}, {3}, {}});
  search(0, 4);
  const std::vector<std::vector<node_id_t>> expected = {
      {0}, {1, 2}, {0, 2}, {3}};
  EXPECT_EQ(expected, evaluated_);
  EXPECT_EQ(6U, context_->get_scan_num());
  const std::vector<std::pair<uint64_t, float>> expected_results = {
      {3, 1}, {1, 4}, {2, 9}, {0, 100}};
  EXPECT_EQ(expected_results, results());
}

TEST_F(VamanaFastSearchTest, HundredStepCapRecomputesLandingRow) {
  constexpr node_id_t kCount = 104;
  std::vector<float> values(kCount);
  std::vector<std::vector<node_id_t>> rows(kCount);
  for (node_id_t i = 0; i < kCount; ++i) {
    values[i] = static_cast<float>(kCount - i);
    if (i > 0) rows[i].push_back(i - 1);
    if (i + 1 < kCount) rows[i].push_back(i + 1);
  }
  create_graph(values, rows);
  search(0, 1);
  ASSERT_EQ(104U, evaluated_.size());
  EXPECT_EQ((std::vector<node_id_t>{98, 100}), evaluated_[100]);
  // Depth 100 stops at node 100, while the cached row belongs to node 99.
  EXPECT_EQ((std::vector<node_id_t>{99, 101}), evaluated_[101]);
  // The pool phase skips the already visited backwards edges.
  EXPECT_EQ((std::vector<node_id_t>{102}), evaluated_[102]);
  EXPECT_EQ((std::vector<node_id_t>{103}), evaluated_[103]);
  EXPECT_EQ(204U, context_->get_scan_num());
  EXPECT_EQ((std::vector<std::pair<uint64_t, float>>{{103, 1}}), results());
}

TEST_F(VamanaFastSearchTest, EqualDistanceStopsDescentWithoutLosingNeighbors) {
  create_graph({2, 2, 1}, {{1}, {2}, {}});
  search(0, 3);
  EXPECT_EQ(3U, context_->get_scan_num());
  ASSERT_EQ(3U, results().size());
  EXPECT_EQ(2U, results()[0].first);
  EXPECT_EQ(1.0f, results()[0].second);
}

TEST_F(VamanaFastSearchTest, DuplicateNeighborsAndReusedContextMatchFresh) {
  create_graph({10, 2, 3, 1}, {{1, 1, 2}, {0, 2, 2, 1}, {3, 3}, {1, 1}});
  search(0, 4);
  const auto expected = results();
  ASSERT_EQ(4U, expected.size());
  EXPECT_EQ(9U, context_->get_scan_num());
  const auto first_trace = evaluated_;
  for (uint32_t capacity : {1U, 4U, 2U, 4U}) {
    search(12, capacity);
    const auto reused_result = results();
    const auto reused_trace = evaluated_;
    auto reused_context = std::move(context_);
    make_context();
    search(12, capacity);
    EXPECT_EQ(reused_result, results());
    EXPECT_EQ(reused_trace, evaluated_);
    context_ = std::move(reused_context);
    search(0, 4);
    EXPECT_EQ(expected, results());
    EXPECT_EQ(first_trace, evaluated_);
  }
}

TEST_F(VamanaFastSearchTest, EmptyNeighborRowKeepsTheEntryPoint) {
  create_graph({0}, {{}});
  search(0, 1);
  EXPECT_EQ((std::vector<std::vector<node_id_t>>{{0}}), evaluated_);
  EXPECT_EQ((std::vector<std::pair<uint64_t, float>>{{0, 0}}), results());
}

TEST_F(VamanaFastSearchTest, ExactVisitModesPreserveResultsAndSearchTrace) {
  create_graph({10, 2, 3, 1}, {{1, 1, 2}, {0, 2, 2, 1}, {3, 3}, {1, 1}});
  for (float query : {0.0f, 12.0f}) {
    for (uint32_t capacity : {1U, 4U}) {
      make_context(VisitFilter::ByteMap);
      search(query, capacity);
      const auto expected = results();
      const auto expected_trace = evaluated_;
      make_context(VisitFilter::BitMap);
      search(query, capacity);
      EXPECT_EQ(expected, results());
      EXPECT_EQ(expected_trace, evaluated_);
    }
  }
}

TEST_F(VamanaFastSearchTest, InvalidVisitFilterFailsSearchExplicitly) {
  create_graph({10, 2, 3, 1}, {{1, 2}, {0, 2}, {3}, {}});
  make_context(VisitFilter::Default);
  context_->set_topk(4);
  context_->set_ef(4);
  std::array<float, kDimension> query{};
  context_->reset_query(query.data());
  EXPECT_EQ(IndexError_Runtime, algorithm_->search(context_.get()));
  EXPECT_EQ(0U, context_->search_heap().size());
  EXPECT_TRUE(evaluated_.empty());
}

}  // namespace
}  // namespace zvec::core

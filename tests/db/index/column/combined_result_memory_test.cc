// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "ailego/pattern/defer.h"
#include "db/index/column/vector_column/combined_vector_column_indexer.h"

namespace zvec {
namespace {
class FixedDecodedIndexer : public VectorColumnIndexer {
 public:
  explicit FixedDecodedIndexer(float score) : score_(score) {}
  Result<IndexResults::Ptr> search(
      const vector_column_params::VectorData &,
      const vector_column_params::QueryParams &params) override {
    core_interface::SearchResult result;
    result.reserve_reverted_vector(4096);
    result.reserve_reverted_array(1, sizeof(std::string));
    if (params.group_by) {
      result.reserve_reverted_array(1, sizeof(std::vector<std::string>));
      core::GroupIndexDocument group;
      group.set_group_id("same-group");
      group.mutable_docs()->emplace_back(0, score_);
      result.group_doc_list_.push_back(std::move(group));
      result.group_reverted_vector_list_.push_back({std::string(4096, 'v')});
      IndexResults::Ptr out = std::make_shared<GroupVectorIndexResults>(
          std::move(result.group_doc_list_),
          std::move(result.group_reverted_vector_list_),
          std::move(result.group_reverted_sparse_values_list_),
          result.take_reverted_memory());
      return out;
    }
    result.doc_list_.emplace_back(0, score_);
    result.reverted_vector_list_.emplace_back(4096, 'v');
    IndexResults::Ptr out = std::make_shared<VectorIndexResults>(
        false, std::move(result.doc_list_),
        std::move(result.reverted_vector_list_),
        std::move(result.reverted_sparse_values_list_),
        result.take_reverted_memory());
    return out;
  }

 private:
  float score_;
};

TEST(CombinedResultMemory, ReservationSurvivesBlockHandoffAndShrinksAfterTopk) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(65536));
  for (bool grouped : {false, true}) {
    SCOPED_TRACE(grouped);
    auto a = std::make_shared<FixedDecodedIndexer>(1);
    auto b = std::make_shared<FixedDecodedIndexer>(2);
    FieldSchema schema("test", DataType::VECTOR_FP32, 1024u, false);
    std::vector<BlockMeta> blocks{
        BlockMeta(0, BlockType::VECTOR_INDEX, 0, 0, 1, {"test"}),
        BlockMeta(1, BlockType::VECTOR_INDEX, 1, 1, 1, {"test"})};
    CombinedVectorColumnIndexer combined({a, b}, {}, schema, SegmentMeta{},
                                         blocks, MetricType::IP);
    vector_column_params::QueryParams params;
    params.topk = 1;
    if (grouped)
      params.group_by = std::make_unique<vector_column_params::GroupByParams>(
          1, 1, [](uint64_t) { return std::string("same-group"); });
    auto result = combined.search(vector_column_params::VectorData{}, params);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(1u, result.value()->count());
    EXPECT_GE(pool.used(), 4096u);
    EXPECT_LT(pool.used(), 8192u);  // No longer reserves both block payloads.
    auto iter = result.value()->create_iterator();
    ASSERT_TRUE(iter->valid());
    EXPECT_EQ(1u, iter->doc_id());
    auto vector =
        std::get<vector_column_params::DenseVector>(iter->vector().vector);
    EXPECT_EQ(std::string(4096, 'v'),
              std::string(static_cast<const char *>(vector.data), 4096));
    iter.reset();
    result.value().reset();
    EXPECT_EQ(0u, pool.used());
  }
}

TEST(CombinedResultMemory, BudgetFailureReleasesPartialResults) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(32));
  auto indexer = std::make_shared<FixedDecodedIndexer>(1);
  FieldSchema schema("test", DataType::VECTOR_FP32, 1024u, false);
  std::vector<BlockMeta> blocks{
      BlockMeta(0, BlockType::VECTOR_INDEX, 0, 0, 1, {"test"})};
  CombinedVectorColumnIndexer combined({indexer}, {}, schema, SegmentMeta{},
                                       blocks, MetricType::IP);
  vector_column_params::QueryParams params;
  auto result = combined.search(vector_column_params::VectorData{}, params);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, result.error().code());
  EXPECT_EQ(0u, pool.used());
}

TEST(CombinedResultMemory, ManyBlocksFitOneResultBudget) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(65536));
  std::vector<VectorColumnIndexer::Ptr> indexers;
  std::vector<BlockMeta> blocks;
  for (size_t i = 0; i < 64; ++i) {
    indexers.push_back(std::make_shared<FixedDecodedIndexer>(float(i)));
    blocks.emplace_back(i, BlockType::VECTOR_INDEX, i, i, 1,
                        std::vector<std::string>{"test"});
  }
  FieldSchema schema("test", DataType::VECTOR_FP32, 1024u, false);
  for (auto metric : {MetricType::IP, MetricType::L2, MetricType::COSINE}) {
    for (bool grouped : {false, true}) {
      CombinedVectorColumnIndexer combined(indexers, {}, schema, SegmentMeta{},
                                           blocks, metric);
      vector_column_params::QueryParams params;
      params.topk = 1;
      if (grouped)
        params.group_by = std::make_unique<vector_column_params::GroupByParams>(
            1, 1, [](uint64_t) { return std::string("same-group"); });
      auto result = combined.search(vector_column_params::VectorData{}, params);
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(1u, result.value()->count());
      auto iter = result.value()->create_iterator();
      EXPECT_EQ(metric == MetricType::IP ? 63u : 0u, iter->doc_id());
      EXPECT_LT(pool.used(), 8192u);
    }
  }
}

class KeyViewIndexer : public VectorColumnIndexer {
 public:
  Result<IndexResults::Ptr> search(
      const vector_column_params::VectorData &,
      const vector_column_params::QueryParams &params) override {
    EXPECT_TRUE(params.bf_pks.empty());
    EXPECT_NE(nullptr, params.bf_pks_view);
    if (params.bf_pks_view) {
      EXPECT_EQ((std::vector<uint64_t>{9, 0}), *params.bf_pks_view);
      EXPECT_GE(ailego::MemoryLimitPool::get_instance().used(), 16u);
    }
    ++calls;
    IndexResults::Ptr result = std::make_shared<VectorIndexResults>(
        false, core::IndexDocumentList{}, std::vector<std::string>{},
        std::vector<std::string>{});
    return result;
  }
  size_t calls{0};
};

TEST(CombinedResultMemory, CandidateKeysAreTranslatedOneBlockAtATime) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(1024));
  auto indexer = std::make_shared<KeyViewIndexer>();
  std::vector<VectorColumnIndexer::Ptr> indexers(128, indexer);
  std::vector<BlockMeta> blocks;
  vector_column_params::QueryParams params;
  params.bf_pks.resize(1);
  for (size_t i = 0; i < indexers.size(); ++i) {
    blocks.emplace_back(i, BlockType::VECTOR_INDEX, i * 10, i * 10 + 9, 10,
                        std::vector<std::string>{"test"});
    params.bf_pks[0].push_back(i * 10 + 9);
    params.bf_pks[0].push_back(i * 10);
  }
  const auto original = params.bf_pks;
  FieldSchema schema("test", DataType::VECTOR_FP32, 4u, false);
  CombinedVectorColumnIndexer combined(indexers, {}, schema, SegmentMeta{},
                                       blocks, MetricType::IP);
  auto result = combined.search(vector_column_params::VectorData{}, params);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(128u, indexer->calls);
  EXPECT_EQ(original, params.bf_pks);
  EXPECT_EQ(0u, pool.used());
}

class ReenteringGroupIndexer : public VectorColumnIndexer {
 public:
  explicit ReenteringGroupIndexer(size_t block) : block_(block) {}
  Result<IndexResults::Ptr> search(
      const vector_column_params::VectorData &,
      const vector_column_params::QueryParams &) override {
    core::GroupIndexDocument group;
    group.set_group_id(block_ == 1 ? "b" : "a");
    group.mutable_docs()->emplace_back(0, block_ == 0   ? 5
                                          : block_ == 1 ? 8
                                                        : 10);
    if (block_ != 2) group.mutable_docs()->emplace_back(1, block_ == 0 ? 4 : 7);
    core::IndexGroupDocumentList groups;
    groups.push_back(std::move(group));
    IndexResults::Ptr result =
        std::make_shared<GroupVectorIndexResults>(std::move(groups));
    return result;
  }

 private:
  size_t block_;
};

TEST(CombinedResultMemory, ReenteringGroupKeepsItsEarlierRunnerUp) {
  FieldSchema schema("test", DataType::VECTOR_FP32, 4u, false);
  std::vector<VectorColumnIndexer::Ptr> indexers;
  std::vector<BlockMeta> blocks;
  for (size_t i = 0; i < 3; ++i) {
    indexers.push_back(std::make_shared<ReenteringGroupIndexer>(i));
    blocks.emplace_back(i, BlockType::VECTOR_INDEX, i * 2, i * 2 + 1, 2,
                        std::vector<std::string>{"test"});
  }
  CombinedVectorColumnIndexer combined(indexers, {}, schema, SegmentMeta{},
                                       blocks, MetricType::IP);
  vector_column_params::QueryParams params;
  params.topk = 2;
  params.group_by = std::make_unique<vector_column_params::GroupByParams>(
      2, 1, [](uint64_t) { return std::string{}; });
  auto result = combined.search(vector_column_params::VectorData{}, params);
  ASSERT_TRUE(result.has_value());
  auto *groups = dynamic_cast<GroupVectorIndexResults *>(result.value().get());
  ASSERT_NE(nullptr, groups);
  ASSERT_EQ(1u, groups->groups().size());
  const auto &group = groups->groups()[0];
  ASSERT_EQ("a", group.group_id());
  ASSERT_EQ(2u, group.docs().size());
  EXPECT_EQ(10, group.docs()[0].score());
  EXPECT_EQ(5, group.docs()[1].score());
}
}  // namespace
}  // namespace zvec

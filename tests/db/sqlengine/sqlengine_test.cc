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

#include "db/sqlengine/sqlengine.h"
#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include "zvec/db//schema.h"
#include "zvec/db/query_params.h"
#include "zvec/db/type.h"
#include "mock_segment.h"

namespace zvec::sqlengine {

class SqlEngineTest : public testing::Test {
 public:
  void SetUp() override {
    auto vector_params = std::make_shared<FlatIndexParams>(MetricType::IP);
    schema_ = std::make_shared<CollectionSchema>(
        "test_collection",
        std::vector<FieldSchema::Ptr>{
            std::make_shared<FieldSchema>("id", DataType::INT32, false, 0,
                                          nullptr),
            std::make_shared<FieldSchema>(
                "name", DataType::STRING, false, 0,  // nullptr
                std::make_shared<InvertIndexParams>(false)),
            std::make_shared<FieldSchema>("age", DataType::INT64, false, 0,
                                          nullptr),
            std::make_shared<FieldSchema>("score", DataType::DOUBLE, false, 0,
                                          nullptr),
            std::make_shared<FieldSchema>("tag_list", DataType::ARRAY_INT32,
                                          false, 0, nullptr),
            std::make_shared<FieldSchema>("vector",
                                          DataType::SPARSE_VECTOR_FP32, false,
                                          4, vector_params),
        });
  }

 protected:
  CollectionSchema::Ptr schema_;
};

TEST_F(SqlEngineTest, VectorMemoryFailureSurvivesArrowExecution) {
  class FailingIndexer : public MockVectorIndexer {
   public:
    StatusCode code{StatusCode::RESOURCE_EXHAUSTED};
    Result<IndexResults::Ptr> search(
        const vector_column_params::VectorData &data,
        const vector_column_params::QueryParams &params) override {
      if (code == StatusCode::OK)
        return MockVectorIndexer::search(data, params);
      return tl::make_unexpected(Status(code, "vector admission test failure"));
    }
  };
  class FailingSegment : public MockSegment {
   public:
    std::shared_ptr<FailingIndexer> indexer =
        std::make_shared<FailingIndexer>();
    CombinedVectorColumnIndexer::Ptr get_combined_vector_indexer(
        const std::string &) const override {
      return indexer;
    }
    CombinedVectorColumnIndexer::Ptr get_quant_combined_vector_indexer(
        const std::string &) const override {
      return indexer;
    }
  };
  auto segment = std::make_shared<FailingSegment>();
  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  SearchQuery query;
  query.topk_ = 1;
  query.output_fields_ = {"id"};
  query.target_.set_sparse_vector("[0, 1, 2, 3]", "[0.1, 0.2, 0.3, 0.4]");
  query.target_.field_name_ = "vector";
  query.target_.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  for (auto code :
       {StatusCode::RESOURCE_EXHAUSTED, StatusCode::INTERNAL_ERROR}) {
    segment->indexer->code = code;
    auto result = engine->execute(schema_, query, {segment});
    ASSERT_FALSE(result);
    EXPECT_EQ(code, result.error().code());
    EXPECT_NE(std::string::npos,
              result.error().message().find("vector admission test failure"));
  }
}

TEST_F(SqlEngineTest, RecallMaterializationPreservesMemoryAndReadFailures) {
  class FailingSegment : public MockSegment {
   public:
    StatusCode code{StatusCode::OK};
    TablePtr fetch(const std::vector<std::string> &columns,
                   const std::vector<int> &indices) const override {
      if (!indices.empty()) {
        if (code == StatusCode::RESOURCE_EXHAUSTED) throw std::bad_alloc();
        if (code == StatusCode::INTERNAL_ERROR) return nullptr;
      }
      return MockSegment::fetch(columns, indices);
    }
  };
  auto segment = std::make_shared<FailingSegment>();
  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  for (bool vector_query : {false, true}) {
    SCOPED_TRACE(vector_query);
    SearchQuery query;
    query.topk_ = 1;
    query.output_fields_ = {"id"};
    if (vector_query) {
      query.target_.set_sparse_vector("[0, 1, 2, 3]", "[0.1, 0.2, 0.3, 0.4]");
      query.target_.field_name_ = "vector";
      query.target_.query_params_ =
          std::make_shared<QueryParams>(IndexType::FLAT);
    } else {
      query.filter_ = "name is not null";
    }
    for (auto code :
         {StatusCode::RESOURCE_EXHAUSTED, StatusCode::INTERNAL_ERROR}) {
      segment->code = code;
      auto result = engine->execute(schema_, query, {segment});
      ASSERT_FALSE(result);
      EXPECT_EQ(code, result.error().code());
      EXPECT_NE(std::string::npos,
                result.error().message().find("Segment forward fetch"));
    }
    segment->code = StatusCode::OK;
    EXPECT_TRUE(engine->execute(schema_, query, {segment}));
  }
}

TEST_F(SqlEngineTest, Forward) {
  std::vector<Segment::Ptr> segments = {std::make_shared<MockSegment>()};
  SearchQuery query;
  query.output_fields_ = {"id", "name", "age", "tag_list"};
  query.topk_ = 11;
  // query.filter_ = "id > 3 and score < 0.1";
  // query.filter_ = "name like 'name_2%'";
  // query.filter_ = "name not in ('name_2','name_4')";
  // query.filter_ = "tag_list contain_all (1,2,3,4)";
  query.filter_ = "tag_list is null";
  if (const char *env_var = std::getenv("FILTER"); env_var != nullptr) {
    query.filter_ = env_var;
  }
  query.include_vector_ = true;

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(schema_, query, segments);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  EXPECT_TRUE(ret.has_value());
}

TEST_F(SqlEngineTest, Vector) {
  std::vector<Segment::Ptr> segments = {std::make_shared<MockSegment>()};
  SearchQuery query;
  query.output_fields_ = {"id", "name", "score"};
  query.topk_ = 11;
  query.filter_ = "id > 3 and score < 0.1";
  if (const char *env_var = std::getenv("FILTER"); env_var != nullptr) {
    query.filter_ = env_var;
  }
  // query.target_.set_vector("[0.1, 0.2, 0.3, 0.4]");
  query.target_.set_sparse_vector("[0, 1, 2, 3]", "[0.1, 0.2, 0.3, 0.4]");
  query.target_.field_name_ = "vector";
  query.include_vector_ = true;
  query.target_.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.target_.query_params_->set_radius(0.8F);


  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(schema_, query, segments);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  EXPECT_TRUE(ret.has_value());
}

TEST_F(SqlEngineTest, Invert) {
  std::vector<Segment::Ptr> segments = {std::make_shared<MockSegment>()};
  SearchQuery query;
  query.output_fields_ = {"id", "age", "score"};
  query.topk_ = 11;
  // query.filter_ = "name = 'test_name'";
  query.filter_ = "name is not null";
  if (const char *env_var = std::getenv("FILTER"); env_var != nullptr) {
    query.filter_ = env_var;
  }
  query.include_vector_ = true;

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(schema_, query, segments);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  EXPECT_TRUE(ret.has_value());
}

TEST_F(SqlEngineTest, MultiSegments) {
  std::vector<Segment::Ptr> segments = {std::make_shared<MockSegment>(),
                                        std::make_shared<MockSegment>()};
  SearchQuery query;
  query.output_fields_ = {"id", "name", "age", "score"};
  query.topk_ = 11;
  query.target_.set_vector("[0.1, 0.2, 0.3, 0.4]");
  query.target_.field_name_ = "vector";
  // query.filter_ = "name = 'test_name'";
  if (const char *env_var = std::getenv("FILTER"); env_var != nullptr) {
    query.filter_ = env_var;
  }


  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(schema_, query, segments);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  EXPECT_TRUE(ret.has_value());
}

TEST_F(SqlEngineTest, GroupBy) {
  std::vector<Segment::Ptr> segments = {std::make_shared<MockSegment>()};
  GroupByVectorQuery query;
  query.group_by_field_name_ = "name";
  query.group_count_ = 3;
  query.topk_per_group_ = 2;
  query.output_fields_ = {"id", "name", "score"};
  query.filter_ = "id > 3 and score < 0.1";
  if (const char *env_var = std::getenv("FILTER"); env_var != nullptr) {
    query.filter_ = env_var;
  }
  // query.target_.set_vector("[0.1, 0.2, 0.3, 0.4]");
  query.target_.set_sparse_vector("[0, 1, 2, 3]", "[0.1, 0.2, 0.3, 0.4]");
  query.target_.field_name_ = "vector";
  query.include_vector_ = true;
  query.target_.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.target_.query_params_->set_radius(0.8F);


  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute_group_by(schema_, query, segments);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  EXPECT_TRUE(ret.has_value());
}

}  // namespace zvec::sqlengine

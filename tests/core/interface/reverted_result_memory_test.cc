// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "ailego/pattern/defer.h"
#include "db/index/column/vector_column/vector_index_results.h"
#include "tests/test_util.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core_interface {
namespace {
void Populate(SearchResult *result, bool grouped) {
  result->reserve_reverted_vector(4096);
  result->reserve_reverted_array(1, sizeof(std::string));
  if (grouped) {
    result->reserve_reverted_array(1, sizeof(std::vector<std::string>));
    result->group_reverted_sparse_values_list_.push_back(
        {std::string(4096, 'v')});
  } else {
    result->reverted_vector_list_.emplace_back(4096, 'v');
  }
}

TEST(RevertedResultMemory, CopiesChargeAndMovesTransferUntilLastOwner) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(65536));
  SearchResult result;
  Populate(&result, false);
  const size_t charged = pool.used();
  ASSERT_GT(charged, 4096u);
  SearchResult copied(result);
  EXPECT_EQ(charged * 2, pool.used());
  SearchResult moved(std::move(result));
  EXPECT_EQ(0u, result.reverted_memory_bytes());
  EXPECT_EQ(charged * 2, pool.used());
  copied = std::move(moved);
  EXPECT_EQ(charged, pool.used());
  auto db_result = std::make_shared<VectorIndexResults>(
      false, std::move(copied.doc_list_),
      std::move(copied.reverted_vector_list_),
      std::move(copied.reverted_sparse_values_list_),
      copied.take_reverted_memory());
  copied.clear();
  EXPECT_EQ(charged, pool.used());
  EXPECT_EQ(std::string(4096, 'v'), db_result->reverted_vector_list().front());
  auto retained = db_result;
  db_result.reset();
  EXPECT_EQ(charged, pool.used());
  retained.reset();
  EXPECT_EQ(0u, pool.used());
}

TEST(RevertedResultMemory,
     GroupedDbCopyRejectsPressureWithoutReleasingOriginal) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(16384));
  SearchResult result;
  Populate(&result, true);
  const size_t charged = pool.used();
  GroupVectorIndexResults db_result(
      std::move(result.group_doc_list_),
      std::move(result.group_reverted_vector_list_),
      std::move(result.group_reverted_sparse_values_list_),
      result.take_reverted_memory());
  result.clear();
  EXPECT_THROW({ GroupVectorIndexResults copy(db_result); }, std::bad_alloc);
  EXPECT_EQ(charged, pool.used());
  EXPECT_EQ(std::string(4096, 'v'),
            db_result.reverted_sparse_values_list()[0][0]);
}

TEST(RevertedResultMemory, CopyAssignmentFailurePreservesBothResults) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(24576));
  SearchResult a, b;
  Populate(&a, false);
  Populate(&b, false);
  const size_t before = pool.used();
  EXPECT_THROW(b = a, std::bad_alloc);
  EXPECT_EQ(before, pool.used());
  EXPECT_EQ(a.reverted_vector_list_, b.reverted_vector_list_);
  b.clear();
  EXPECT_NO_THROW(b = a);
}

TEST(RevertedResultMemory, PublicQuantizedSearchReplacesOldDecodedPayload) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  // The default mmap metadata region alone is 1 MiB plus header/footer and
  // page alignment. Leave setup headroom; the failure check below explicitly
  // reserves all remaining capacity except one byte.
  ASSERT_EQ(0, pool.init(2 * 1024 * 1024));
  const std::string path = "reverted_result_memory_test.index";
  test_util::RemoveTestFiles(path);
  AILEGO_DEFER([&] { test_util::RemoveTestFiles(path); });
  auto params = FlatIndexParamBuilder()
                    .with_metric_type(MetricType::kL2sq)
                    .with_data_type(DataType::DT_FP32)
                    .with_dimension(64)
                    .with_quantizer_param(QuantizerParam{QuantizerType::kFP16})
                    .build();
  auto index = IndexFactory::CreateAndInitIndex(*params);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  AILEGO_DEFER([&] { index->close(); });
  std::vector<float> row(64, 1.0f);
  VectorData data;
  data.vector = DenseVector{row.data()};
  for (uint64_t i = 0; i < 32; ++i) ASSERT_EQ(0, index->add(data, i));
  ASSERT_EQ(0, index->train());
  auto query =
      FlatQueryParamBuilder().with_topk(16).with_fetch_vector(true).build();
  SearchResult result;
  ASSERT_EQ(0, index->search(data, query, &result));
  ASSERT_EQ(16u, result.reverted_vector_list_.size());
  const size_t charged = result.reverted_memory_bytes();
  EXPECT_GE(charged, 16 * 64 * sizeof(float));
  VectorData borrowed_query;
  borrowed_query.vector = DenseVector{result.reverted_vector_list_[0].data()};
  ASSERT_EQ(0, index->search(borrowed_query, query, &result));
  EXPECT_EQ(16u, result.reverted_vector_list_.size());
  EXPECT_EQ(charged, result.reverted_memory_bytes());
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(row.data()),
                        row.size() * sizeof(float)),
            result.reverted_vector_list_[0]);
  result.clear();
  const size_t held = pool.capacity() - pool.used() - 1;
  ASSERT_TRUE(pool.try_charge_external(held));
  AILEGO_DEFER([&] { pool.release_external(held); });
  EXPECT_NE(0, index->search(data, query, &result));
  EXPECT_TRUE(result.doc_list_.empty());
  EXPECT_TRUE(result.reverted_vector_list_.empty());
  EXPECT_EQ(0u, result.reverted_memory_bytes());
}
}  // namespace
}  // namespace zvec::core_interface

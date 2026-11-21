// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/sqlengine/planner/fts_rank_batch.h"
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "ailego/pattern/scope_guard.h"
#include "db/common/constants.h"
#include "db/index/storage/parquet_memory_pool.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::sqlengine {
namespace {

class FtsRankBatchTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_capacity_ = budget_.capacity();
    ASSERT_EQ(0u, budget_.used());
    ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, budget_.used());
    EXPECT_EQ(0, budget_.init(previous_capacity_));
  }
  ailego::PoolPayloadBudget &budget_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_capacity_{0};
  std::shared_ptr<arrow::Schema> schema_ = arrow::schema(
      {arrow::field(LOCAL_ROW_ID, arrow::int32()),
       arrow::field(FIELD_SCORE, arrow::float32())});
};

TEST_F(FtsRankBatchTest, OneAllocationRetainsBothColumnsUntilLastConsumer) {
  detail::ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  std::shared_ptr<arrow::Array> rows, scores;
  {
    FtsIndexResults results({{7, 9.5f}, {11, 3.f}, {29, 0.25f}});
    auto batch = MakeFtsRankBatch(results, 0, 1024, schema_, &pool);
    ASSERT_TRUE(batch.ok()) << batch.status();
    EXPECT_TRUE((*batch)->ValidateFull().ok());
    EXPECT_EQ(3, (*batch)->num_rows());
    EXPECT_EQ(1, pool.num_allocations());
    rows = (*batch)->column(0);
    scores = (*batch)->column(1);
    EXPECT_EQ(rows->data()->buffers[1]->parent(),
              scores->data()->buffers[1]->parent());
  }
  // The producing batch and FTS result are gone; each column still owns data.
  EXPECT_EQ(11, static_cast<arrow::Int32Array &>(*rows).Value(1));
  EXPECT_FLOAT_EQ(0.25f, static_cast<arrow::FloatArray &>(*scores).Value(2));
  const auto charge = pool.charged_bytes();
  EXPECT_GT(charge, 0u);
  rows.reset();
  EXPECT_EQ(charge, pool.charged_bytes());
  scores.reset();
  EXPECT_EQ(0u, pool.charged_bytes());
}

TEST_F(FtsRankBatchTest, BatchOffsetsPreserveRowsAndScores) {
  detail::ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  FtsIndexResults results({{7, 9.5f}, {11, 3.f}, {29, 0.25f}});
  auto batch = MakeFtsRankBatch(results, 2, 2, schema_, &pool);
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_TRUE((*batch)->ValidateFull().ok());
  EXPECT_EQ(1, (*batch)->num_rows());
  EXPECT_EQ(29, static_cast<arrow::Int32Array &>(*(*batch)->column(0)).Value(0));
  EXPECT_FLOAT_EQ(
      0.25f, static_cast<arrow::FloatArray &>(*(*batch)->column(1)).Value(0));
  EXPECT_TRUE(MakeFtsRankBatch(results, 3, 2, schema_, &pool).status().IsInvalid());
  EXPECT_TRUE(MakeFtsRankBatch(results, 0, 0, schema_, &pool).status().IsInvalid());
}

TEST_F(FtsRankBatchTest, RefusedBackingDoesNotConsumeBudget) {
  detail::ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  FtsIndexResults results({{7, 9.5f}});
  auto &native = ailego::MemoryLimitPool::get_instance();
  const auto occupied = native.capacity() - native.used();
  ASSERT_TRUE(native.try_charge_external(occupied));
  auto release = ailego::ScopeGuard::Make(
      [&] { native.release_external(occupied); });
  auto batch = MakeFtsRankBatch(results, 0, 1, schema_, &pool);
  EXPECT_TRUE(batch.status().IsOutOfMemory()) << batch.status();
  EXPECT_EQ(native.capacity(), native.used());
  EXPECT_EQ(0u, pool.charged_bytes());
}

}  // namespace
}  // namespace zvec::sqlengine

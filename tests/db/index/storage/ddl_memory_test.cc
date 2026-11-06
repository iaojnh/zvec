// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <arrow/compute/initialize.h>
#include <gtest/gtest.h>
#include "db/index/storage/store_helper.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include "../../common/memory_budget_test_helper.h"

namespace zvec {
namespace {

class RejectingDdlPool : public arrow::ProxyMemoryPool {
 public:
  RejectingDdlPool() : arrow::ProxyMemoryPool(arrow::default_memory_pool()) {}
  arrow::Status Allocate(int64_t, int64_t, uint8_t **) override {
    return arrow::Status::OutOfMemory("DDL allocation rejected");
  }
};

TEST(DdlMemoryTest, ComputeCastAndScalarExpansionUseSuppliedPool) {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  auto data = arrow::MakeArrayFromScalar(arrow::Int64Scalar(7), 4096);
  ASSERT_TRUE(data.ok());
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("source", arrow::int64())}), {*data});
  const std::vector<arrow::compute::Expression> expressions{
      arrow::compute::field_ref("source"), arrow::compute::literal(3),
      arrow::compute::call("add", {arrow::compute::field_ref("source"),
                                   arrow::compute::literal(2)})};
  RejectingDdlPool rejected;
  for (const auto &expr : expressions) {
    auto result = EvaluateExpressionOnTable(table, "converted", expr,
                                            arrow::float64(), &rejected);
    ASSERT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsOutOfMemory()) << result.status();
    arrow::ProxyMemoryPool tracked(arrow::default_memory_pool());
    {
      auto retried = EvaluateExpressionOnTable(table, "converted", expr,
                                               arrow::float64(), &tracked);
      ASSERT_TRUE(retried.ok()) << retried.status();
      EXPECT_EQ(table->num_rows(), (*retried)->num_rows());
      EXPECT_GT(tracked.bytes_allocated(), 0);
    }
    EXPECT_EQ(0, tracked.bytes_allocated());
  }
}

TEST(DdlMemoryTest, BlockReadsUseManagedBuffersAndReturnOutOfMemory) {
  constexpr size_t kMiB = 1024 * 1024;
  ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(64 * kMiB));
  auto root =
      std::filesystem::temp_directory_path() /
      ("zvec_ddl_budget_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto path = FileHelper::MakeForwardBlockPath(root.string(), 0, 1, true);
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  constexpr int64_t kRows = 128 * 1024;
  auto data = arrow::MakeArrayFromScalar(arrow::Int64Scalar(7), kRows);
  ASSERT_TRUE(data.ok());
  auto schema = arrow::schema({arrow::field("source", arrow::int64())});
  auto input = arrow::Table::Make(schema, {*data});
  {
    auto writer = ChunkedFileWriter::Open(path, schema, FileFormat::PARQUET);
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->write(*input).ok());
    ASSERT_TRUE(writer->close().ok());
  }
  const std::vector<BlockMeta> blocks{
      BlockMeta(1, BlockType::SCALAR, 0, kRows - 1, kRows, {"source"})};
  const auto memory = detail::GetParquetMemoryPool();
  const auto before = memory->bytes_allocated();
  {
    auto result = ReadBlocksAsTable(blocks, root.string(), 0, true);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(kRows, (*result)->num_rows());
    EXPECT_GT(memory->bytes_allocated(), before);
  }
  EXPECT_EQ(before, memory->bytes_allocated());
  const auto remaining = pool.capacity() - pool.used();
  ASSERT_TRUE(pool.try_charge_external(remaining));
  auto denied = ReadBlocksAsTable(blocks, root.string(), 0, true);
  pool.release_external(remaining);
  ASSERT_FALSE(denied.ok());
  EXPECT_TRUE(denied.status().IsOutOfMemory()) << denied.status();
  {
    auto recovered = ReadBlocksAsTable(blocks, root.string(), 0, true);
    ASSERT_TRUE(recovered.ok()) << recovered.status();
  }
  EXPECT_EQ(before, memory->bytes_allocated());
  std::filesystem::remove_all(root);
  EXPECT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
}

}  // namespace
}  // namespace zvec

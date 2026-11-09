// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <utility>
#include <arrow/api.h>
#include <arrow/compute/initialize.h>
#include <arrow/io/api.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include "ailego/pattern/defer.h"
#include "db/common/constants.h"
#include "db/common/global_resource.h"
#include "db/index/segment/filtering_reader.h"
#include "db/index/storage/parquet_memory_pool.h"
#include "db/sqlengine/planner/segment_node.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::sqlengine {
namespace {
using Batch = std::optional<arrow::compute::ExecBatch>;
PlanInfo::Ptr make_source(const std::shared_ptr<std::atomic<int>> &calls,
                          const std::shared_ptr<int> &lease, bool fail,
                          arrow::Future<Batch> gate = {}) {
  auto schema = arrow::schema({arrow::field("row", arrow::int32())});
  arrow::Int32Builder builder;
  EXPECT_TRUE(builder.AppendValues(std::vector<int32_t>(1024, 42)).ok());
  auto batch =
      arrow::RecordBatch::Make(schema, 1024, {builder.Finish().ValueOrDie()});
  arrow::AsyncGenerator<Batch> generator = [calls, lease, fail, batch, gate] {
    const int call = calls->fetch_add(1);
    if (call && gate.is_valid()) return gate;
    if (call == 200) return arrow::Future<Batch>::MakeFinished(std::nullopt);
    if (fail)
      return arrow::Future<Batch>::MakeFinished(
          arrow::Status::IOError("injected read failure"));
    return arrow::Future<Batch>::MakeFinished(
        arrow::compute::ExecBatch(*batch));
  };
  return std::make_shared<PlanInfo>(
      arrow::acero::Declaration{
          "source",
          arrow::acero::SourceNodeOptions{
              schema, generator, arrow::compute::Ordering::Implicit()}},
      schema);
}

bool expires(const std::weak_ptr<int> &lease) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!lease.expired() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return lease.expired();
}

TEST(SegmentStreamTest, AsyncCancellationReleasesSourceWithOneCpuWorker) {
  const auto capacity = arrow::GetCpuThreadPoolCapacity();
  ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(arrow::SetCpuThreadPoolCapacity(capacity).ok()); });
  auto calls = std::make_shared<std::atomic<int>>(0);
  auto gate = arrow::Future<Batch>::Make();
  auto unblock = ailego::ScopeGuard::Make([&] {
    if (!gate.is_finished()) gate.MarkFinished(Batch{});
  });
  std::weak_ptr<int> weak;
  {
    auto lease = std::make_shared<int>(1);
    weak = lease;
    auto plan = make_source(calls, lease, false, gate);
    auto stream = plan->execute_to_generator(8192);
    ASSERT_TRUE(stream);
    ASSERT_TRUE((*stream)().result().ok());
  }
  // Cancellation must not synchronously wait for the pending source read.
  // Complete it after releasing the consumer, then check the plan's lifetime.
  gate.MarkFinished(Batch{});
  EXPECT_TRUE(expires(weak));
  EXPECT_LE(calls->load(), 2);
}

TEST(SegmentStreamTest, ErrorIsPropagatedAndSourceIsReleased) {
  auto calls = std::make_shared<std::atomic<int>>(0);
  std::weak_ptr<int> weak;
  {
    auto lease = std::make_shared<int>(1);
    weak = lease;
    auto plan = make_source(calls, lease, true);
    auto stream = plan->execute_to_generator(8192);
    ASSERT_TRUE(stream);
    auto status = (*stream)().status();
    EXPECT_TRUE(status.IsIOError()) << status.ToString();
    EXPECT_NE(std::string::npos,
              status.ToString().find("injected read failure"));
  }
  EXPECT_TRUE(expires(weak));
}

TEST(SegmentStreamTest,
     BufferPlanAccountsKernelOutputAndDeniesBeforeAllocation) {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const auto capacity = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] { pool.init(capacity); });
  pool.init(1024 * 1024);
  auto *arrow_pool = detail::GetParquetMemoryPool().get();
  const auto baseline = arrow_pool->charged_bytes();
  auto schema = arrow::schema({arrow::field("row", arrow::int32())});
  arrow::Int32Builder builder;
  ASSERT_TRUE(builder.AppendValues(std::vector<int32_t>(4096, 42)).ok());
  auto input =
      arrow::RecordBatch::Make(schema, 4096, {builder.Finish().ValueOrDie()});
  auto run = [&](bool budgeted) {
    auto delivered = std::make_shared<bool>(false);
    arrow::AsyncGenerator<Batch> source = [input, delivered] {
      if (std::exchange(*delivered, true))
        return arrow::Future<Batch>::MakeFinished(Batch{});
      return arrow::Future<Batch>::MakeFinished(
          arrow::compute::ExecBatch(*input));
    };
    PlanInfo plan(
        arrow::acero::Declaration::Sequence(
            {{"source", arrow::acero::SourceNodeOptions{schema, source}},
             {"project",
              arrow::acero::ProjectNodeOptions{
                  {arrow::compute::call("add",
                                        {arrow::compute::field_ref("row"),
                                         arrow::compute::literal(1)})},
                  {"row"}}}}),
        schema, budgeted);
    return plan.execute_to_reader();
  };
  {
    auto reader = run(true);
    ASSERT_TRUE(reader) << reader.error().message();
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_TRUE((*reader)->ReadNext(&batch).ok());
    ASSERT_NE(nullptr, batch);
    EXPECT_GT(arrow_pool->charged_bytes(), baseline);
    EXPECT_EQ(43, std::static_pointer_cast<arrow::Int32Array>(batch->column(0))
                      ->Value(0));
    ASSERT_TRUE((*reader)->Close().ok());
  }
  EXPECT_EQ(baseline, arrow_pool->charged_bytes());
  ASSERT_TRUE(pool.try_charge_external(pool.capacity()));
  auto release =
      ailego::ScopeGuard::Make([&] { pool.release_external(pool.capacity()); });
  {
    auto reader = run(true);
    ASSERT_TRUE(reader);
    std::shared_ptr<arrow::RecordBatch> batch;
    EXPECT_TRUE((*reader)->ReadNext(&batch).IsOutOfMemory());
    EXPECT_EQ(baseline, arrow_pool->charged_bytes());
  }
  // Mmap plans preserve the default allocator even under Buffer pressure.
  auto reader = run(false);
  ASSERT_TRUE(reader);
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE((*reader)->ReadNext(&batch).ok());
  ASSERT_NE(nullptr, batch);
}

}  // namespace
}  // namespace zvec::sqlengine

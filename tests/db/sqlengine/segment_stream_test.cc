// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <thread>
#include <arrow/api.h>
#include <arrow/compute/initialize.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "ailego/pattern/defer.h"
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

TEST(SegmentStreamTest, ReadySourceHonorsPressureBeforePullingMore) {
  const auto capacity = arrow::GetCpuThreadPoolCapacity();
  ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(arrow::SetCpuThreadPoolCapacity(capacity).ok()); });
  auto calls = std::make_shared<std::atomic<int>>(0);
  auto plan = make_source(calls, std::make_shared<int>(1), false);
  auto stream = plan->execute_to_generator(8192);
  ASSERT_TRUE(stream);
  auto barrier = arrow::internal::GetCpuThreadPool()->Submit([] {});
  ASSERT_TRUE(barrier.ok());
  barrier->Wait();
  // Each source batch is 4 KiB. The one crossing the 8 KiB watermark is
  // allowed, but no subsequent source read or delivery task may be queued.
  EXPECT_LE(calls->load(), 3);
  EXPECT_GT(calls->load(), 0);
  size_t rows = 0;
  for (;;) {
    auto batch = (*stream)().result();
    ASSERT_TRUE(batch.ok()) << batch.status().ToString();
    if (!*batch) break;
    rows += (*batch)->length;
  }
  EXPECT_EQ(200u * 1024, rows);
  EXPECT_EQ(201, calls->load());
}

TEST(SegmentStreamTest, ReaderSourceDoesNotPrefetchPastPressure) {
  class Reader : public arrow::RecordBatchReader {
   public:
    explicit Reader(std::shared_ptr<std::atomic<int>> calls)
        : calls_(std::move(calls)) {
      arrow::Int32Builder builder;
      EXPECT_TRUE(builder.AppendValues(std::vector<int32_t>(1024, 7)).ok());
      batch_ = arrow::RecordBatch::Make(schema(), 1024,
                                        {builder.Finish().ValueOrDie()});
    }
    std::shared_ptr<arrow::Schema> schema() const override {
      return arrow::schema({arrow::field("row", arrow::int32())});
    }
    arrow::Status ReadNext(
        std::shared_ptr<arrow::RecordBatch> *batch) override {
      *batch = calls_->fetch_add(1) < 200 ? batch_ : nullptr;
      return arrow::Status::OK();
    }

   private:
    std::shared_ptr<std::atomic<int>> calls_;
    std::shared_ptr<arrow::RecordBatch> batch_;
  };
  auto calls = std::make_shared<std::atomic<int>>(0);
  auto source = std::make_shared<Reader>(calls);
  PlanInfo plan({"record_batch_reader_source",
                 arrow::acero::RecordBatchReaderSourceNodeOptions{source}},
                source->schema());
  auto reader = plan.execute_to_reader(8192);
  ASSERT_TRUE(reader);
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE((*reader)->ReadNext(&batch).ok());
  ASSERT_NE(nullptr, batch);
  EXPECT_EQ(1024, batch->num_rows());
  ASSERT_TRUE((*reader)->Close().ok());
  EXPECT_LE(calls->load(), 4);
}

TEST(SegmentStreamTest, DistinctBuffersAreBoundedAndReleased) {
  arrow::ProxyMemoryPool memory(arrow::default_memory_pool());
  constexpr int64_t kRows = arrow::acero::ExecPlan::kMaxBatchSize;
  constexpr int64_t kBytes = kRows * sizeof(int32_t);
  auto calls = std::make_shared<std::atomic<int>>(0);
  auto schema = arrow::schema({arrow::field("row", arrow::int32())});
  std::weak_ptr<int> weak;
  {
    auto lease = std::make_shared<int>(1);
    weak = lease;
    arrow::AsyncGenerator<Batch> generator =
        [calls, schema, lease, &memory]() -> arrow::Future<Batch> {
      if (calls->fetch_add(1) == 200) return Batch{};
      const int64_t batch_rows = arrow::acero::ExecPlan::kMaxBatchSize;
      arrow::Int32Builder builder(&memory);
      ARROW_RETURN_NOT_OK(builder.Resize(batch_rows));
      for (int64_t i = 0; i < batch_rows; ++i)
        builder.UnsafeAppend(static_cast<int32_t>(i));
      ARROW_ASSIGN_OR_RAISE(auto values, builder.Finish());
      return Batch{arrow::compute::ExecBatch(
          *arrow::RecordBatch::Make(schema, batch_rows, {values}))};
    };
    PlanInfo plan(
        {"source",
         arrow::acero::SourceNodeOptions{schema, generator,
                                         arrow::compute::Ordering::Implicit()}},
        schema);
    auto stream = plan.execute_to_generator(2 * kBytes);
    ASSERT_TRUE(stream);
    int64_t rows = 0;
    for (;;) {
      auto batch = (*stream)().result();
      ASSERT_TRUE(batch.ok()) << batch.status().ToString();
      if (!*batch) break;
      rows += (*batch)->length;
    }
    EXPECT_EQ(200 * kRows, rows);
    // Up to three sink batches (including the watermark-crossing batch),
    // one consumer-held batch and one in-progress builder. The builder also
    // owns a validity bitmap until Finish(), even though these rows are all
    // valid. Do not confuse that transient bitmap with another data batch.
    constexpr int64_t kValidityBytes = (kRows + 7) / 8;
    EXPECT_LE(memory.max_memory(), 5 * kBytes + kValidityBytes);
  }
  EXPECT_TRUE(expires(weak));
  EXPECT_EQ(0, memory.bytes_allocated());
}

TEST(SegmentStreamTest, OversizedSourceBatchesRetainArrowSliceContract) {
  const int64_t rows = 2 * arrow::acero::ExecPlan::kMaxBatchSize + 7;
  auto schema = arrow::schema({arrow::field("row", arrow::int32())});
  arrow::Int32Builder builder;
  ASSERT_TRUE(builder.AppendValues(std::vector<int32_t>(rows, 42)).ok());
  auto batch =
      arrow::RecordBatch::Make(schema, rows, {builder.Finish().ValueOrDie()});
  arrow::AsyncGenerator<Batch> generator = [batch, first = true]() mutable {
    if (!first) return arrow::Future<Batch>::MakeFinished(Batch{});
    first = false;
    return arrow::Future<Batch>::MakeFinished(
        arrow::compute::ExecBatch(*batch));
  };
  PlanInfo plan({"source",
                 arrow::acero::SourceNodeOptions{
                     schema, generator, arrow::compute::Ordering::Implicit()}},
                schema);
  auto stream = plan.execute_to_generator(8192);
  ASSERT_TRUE(stream);
  int64_t actual_rows = 0;
  for (int64_t i = 0; i < 3; ++i) {
    auto result = (*stream)().result();
    ASSERT_TRUE(result.ok());
    ASSERT_TRUE(result->has_value());
    EXPECT_LE((*result)->length, arrow::acero::ExecPlan::kMaxBatchSize);
    EXPECT_EQ(i, (*result)->index);
    actual_rows += (*result)->length;
  }
  auto end = (*stream)().result();
  ASSERT_TRUE(end.ok());
  EXPECT_FALSE(end->has_value());
  EXPECT_EQ(rows, actual_rows);
}

TEST(SegmentStreamTest, EmptySegmentsCompleteWithOneCpuWorker) {
  const auto capacity = arrow::GetCpuThreadPoolCapacity();
  ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(arrow::SetCpuThreadPoolCapacity(capacity).ok()); });
  auto schema = arrow::schema({arrow::field("row", arrow::int32())});
  arrow::AsyncGenerator<Batch> empty = [] {
    return arrow::Future<Batch>::MakeFinished(Batch{});
  };
  std::vector<PlanInfo::Ptr> plans;
  for (size_t i = 0; i < 3; ++i)
    plans.push_back(std::make_shared<PlanInfo>(
        arrow::acero::Declaration{
            "source", arrow::acero::SourceNodeOptions{schema, empty}},
        schema));
  auto node = std::make_shared<SegmentNode>(std::move(plans), nullptr);
  auto result = node->gen()().result();
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_FALSE(result->has_value());
}

TEST(SegmentStreamTest, ReadsSegmentsLazilyWithOneCpuWorker) {
  const auto capacity = arrow::GetCpuThreadPoolCapacity();
  ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(arrow::SetCpuThreadPoolCapacity(capacity).ok()); });
  auto first = std::make_shared<std::atomic<int>>(0);
  auto last = std::make_shared<std::atomic<int>>(0);
  auto lease = std::make_shared<int>(1);
  auto gate = arrow::Future<Batch>::Make();
  auto unblock = ailego::ScopeGuard::Make([&] {
    if (!gate.is_finished()) gate.MarkFinished(Batch{});
  });
  {
    auto node = std::make_shared<SegmentNode>(
        std::vector<PlanInfo::Ptr>{make_source(first, lease, false),
                                   make_source(last, lease, false, gate)},
        nullptr);
    EXPECT_EQ(0, first->load());
    EXPECT_EQ(0, last->load());
    auto generator = node->gen();
    auto result = generator().result();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->has_value());
    EXPECT_EQ(1024, result->value().length);
  }
  gate.MarkFinished(Batch{});
  EXPECT_EQ(0, first->load());
  EXPECT_GT(last->load(), 0);
  EXPECT_LE(last->load(), 2);
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

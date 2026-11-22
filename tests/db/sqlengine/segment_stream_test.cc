// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <utility>
#include <arrow/api.h>
#include <arrow/compute/api.h>
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
#include "db/index/storage/resident_record_batch_reader.h"
#include "db/sqlengine/planner/ops/fetch_vector_op.h"
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

TEST(SegmentStreamTest, FetchVectorFinishPropagatesOutOfMemory) {
  class RejectShrinkPool : public arrow::ProxyMemoryPool {
   public:
    RejectShrinkPool() : ProxyMemoryPool(arrow::default_memory_pool()) {}

    arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                             int64_t alignment, uint8_t **ptr) override {
      if (reject_shrink && new_size > 0 && new_size < old_size) {
        return arrow::Status::OutOfMemory("injected vector buffer shrink");
      }
      return ProxyMemoryPool::Reallocate(old_size, new_size, alignment, ptr);
    }

    bool reject_shrink = true;
  };

  class TestIndexer : public CombinedVectorColumnIndexer {
   public:
    explicit TestIndexer(bool dense) : dense_(dense) {}

    Result<vector_column_params::VectorDataBuffer> fetch(uint32_t) const override {
      std::string bytes(128, '\0');
      if (dense_) {
        return vector_column_params::VectorDataBuffer{
            vector_column_params::DenseVectorBuffer(std::move(bytes))};
      }
      return vector_column_params::VectorDataBuffer{
          vector_column_params::SparseVectorBuffer{bytes, bytes}};
    }

   private:
    bool dense_;
  };

  const uint64_t row_ids[] = {0, 1, 2};
  arrow::UInt64Builder row_builder;
  ASSERT_TRUE(row_builder.AppendValues(row_ids, 3).ok());
  auto rows = row_builder.Finish().ValueOrDie();
  cp::ExecBatch input({rows}, 3);
  for (bool dense : {true, false}) {
    SCOPED_TRACE(dense);
    RejectShrinkPool pool;
    cp::ExecContext exec_context(&pool);
    cp::KernelContext kernel_context(&exec_context);
    FetchVectorOp::Options options(std::make_shared<TestIndexer>(dense), dense);
    FetchVectorOp::FetchVectorState state(&options);
    kernel_context.SetState(&state);
    cp::ExecResult output;
    auto status = FetchVectorOp::FetchVectorFunction(
        &kernel_context, cp::ExecSpan(input), &output);
    EXPECT_TRUE(status.IsOutOfMemory()) << status.ToString();
    EXPECT_NE(status.message().find("Finish vector builder failed"),
              std::string::npos);
    EXPECT_EQ(0, pool.bytes_allocated());

    pool.reject_shrink = false;
    status = FetchVectorOp::FetchVectorFunction(
        &kernel_context, cp::ExecSpan(input), &output);
    EXPECT_TRUE(status.ok()) << status.ToString();
  }
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

TEST(SegmentStreamTest, DeliveryErrorWaitsForOutstandingPrefetch) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const auto capacity = pool.capacity();
  ASSERT_EQ(0, pool.init(128UL << 20));
  auto restore_pool = ailego::ScopeGuard::Make([&] { pool.init(capacity); });
  auto *cpu = GlobalResource::Instance().arrow_query_thread_pool();
  ASSERT_NE(nullptr, cpu);
  const auto cpu_count = cpu->GetCapacity();
  ASSERT_TRUE(cpu->SetCapacity(2).ok());
  auto restore_cpu = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(cpu->SetCapacity(cpu_count).ok()); });

  class RejectDelivery : public arrow::acero::ExecNode {
   public:
    RejectDelivery(arrow::acero::ExecPlan *plan, ExecNode *input)
        : ExecNode(plan, {input}, {"input"}, input->output_schema()) {}
    const char *kind_name() const override {
      return "RejectDelivery";
    }
    const arrow::compute::Ordering &ordering() const override {
      return inputs_[0]->ordering();
    }
    arrow::Status InputReceived(ExecNode *, arrow::compute::ExecBatch) override {
      return arrow::Status::IOError("injected delivery failure");
    }
    arrow::Status InputFinished(ExecNode *, int) override {
      return arrow::Status::OK();
    }
    arrow::Status StartProducing() override {
      return arrow::Status::OK();
    }
    void PauseProducing(ExecNode *, int32_t counter) override {
      inputs_[0]->PauseProducing(this, counter);
    }
    void ResumeProducing(ExecNode *, int32_t counter) override {
      inputs_[0]->ResumeProducing(this, counter);
    }

   protected:
    arrow::Status StopProducingImpl() override {
      return arrow::Status::OK();
    }
  };
  static const auto registered =
      arrow::acero::default_exec_factory_registry()->AddFactory(
          "zvec_test_reject_delivery",
          [](arrow::acero::ExecPlan *plan,
             std::vector<arrow::acero::ExecNode *> inputs,
             const arrow::acero::ExecNodeOptions &)
              -> arrow::Result<arrow::acero::ExecNode *> {
            return plan->EmplaceNode<RejectDelivery>(plan, inputs.at(0));
          });
  ASSERT_TRUE(registered.ok());

  auto schema = arrow::schema({arrow::field("row", arrow::int32())});
  arrow::Int32Builder builder;
  ASSERT_TRUE(builder.AppendValues(std::vector<int32_t>(1024, 42)).ok());
  auto batch = arrow::RecordBatch::Make(schema, 1024,
                                        {builder.Finish().ValueOrDie()});
  auto started = arrow::Future<>::Make();
  auto gate = arrow::Future<Batch>::Make();
  auto release = ailego::ScopeGuard::Make([&] {
    if (!gate.is_finished()) gate.MarkFinished(Batch{});
  });
  arrow::AsyncGenerator<Batch> source =
      [batch, started, gate, count = 0]() mutable -> arrow::Future<Batch> {
    // Fill the source's 64-delivery window, then gate the next prefetch.
    if (++count == 65) {
      started.MarkFinished();
      return gate;
    }
    return Batch{arrow::compute::ExecBatch(*batch)};
  };
  PlanInfo plan(
      arrow::acero::Declaration::Sequence(
          {{"source", arrow::acero::SourceNodeOptions{
                           schema, source,
                           arrow::compute::Ordering::Implicit()}},
           {"zvec_test_reject_delivery", arrow::acero::ExecNodeOptions{}}}),
      schema, true);
  auto stream = plan.execute_to_generator(8192);
  ASSERT_TRUE(stream);
  ASSERT_TRUE(started.Wait(3));
  auto finished = stream->close();
  EXPECT_FALSE(finished.Wait(0.1));
  gate.MarkFinished(Batch{});
  ASSERT_TRUE(finished.Wait(3));
  EXPECT_TRUE(finished.status().IsIOError()) << finished.status().ToString();
  EXPECT_NE(std::string::npos,
            finished.status().ToString().find("injected delivery failure"));
}

TEST(SegmentStreamTest, ReadySourceHonorsPressureBeforePullingMore) {
  const auto capacity = arrow::GetCpuThreadPoolCapacity();
  ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(arrow::SetCpuThreadPoolCapacity(capacity).ok()); });
  for (const auto &[limit, max_calls] :
       {std::pair<size_t, int>{1, 1}, std::pair<size_t, int>{8192, 2}}) {
    SCOPED_TRACE(limit);
    auto calls = std::make_shared<std::atomic<int>>(0);
    auto plan = make_source(calls, std::make_shared<int>(1), false);
    auto stream = plan->execute_to_generator(limit);
    ASSERT_TRUE(stream);
    auto barrier = arrow::internal::GetCpuThreadPool()->Submit([] {});
    ASSERT_TRUE(barrier.ok());
    barrier->Wait();
    // Each batch is 4 KiB: pause at exactly two batches for an 8 KiB limit.
    // A single oversized batch is allowed, and even a one-byte limit must
    // resume after consumption so the entire scan can finish.
    EXPECT_LE(calls->load(), max_calls);
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
    // Up to two queued batches, one consumer-held batch, one completed Future
    // still forwarding a batch, and one in-progress builder. The builder also
    // owns a validity bitmap until Finish(), even though these rows are all
    // valid. Keep the same peak-byte bound while accounting for that lifetime.
    constexpr int64_t kValidityBytes = (kRows + 7) / 8;
    EXPECT_LE(memory.max_memory(), 5 * kBytes + kValidityBytes);
  }
  EXPECT_TRUE(expires(weak));
  // Releasing the source lease can precede destruction of the worker's last
  // completed Future (and its batch). Wait for that asynchronous cleanup too;
  // neither the peak-byte bound nor the required zero live bytes changes.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (memory.bytes_allocated() != 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
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
TEST(SegmentStreamTest, ManagedPlansDoNotDependOnHostExecutors) {
  // These executor tests configure a standalone payload pool; process-budget
  // startup and stack reservations are covered in global_resource_test.
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(128UL << 20));
  auto &resources = GlobalResource::Instance();
  auto *cpu = resources.arrow_query_thread_pool();
  auto *io = resources.arrow_io_thread_pool();
  ASSERT_NE(nullptr, cpu);
  ASSERT_NE(nullptr, io);
  const auto cpu_count = cpu->GetCapacity();
  const auto io_count = io->GetCapacity();
  ASSERT_TRUE(cpu->SetCapacity(1).ok());
  ASSERT_TRUE(io->SetCapacity(1).ok());
  auto restore_managed = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(cpu->SetCapacity(cpu_count).ok());
    EXPECT_TRUE(io->SetCapacity(io_count).ok());
  });
  auto *host = arrow::internal::GetCpuThreadPool();
  const auto capacity = host->GetCapacity();
  ASSERT_TRUE(host->SetCapacity(1).ok());
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(host->SetCapacity(capacity).ok()); });
  host->WaitForIdle();
  const auto io_capacity = arrow::io::GetIOThreadPoolCapacity();
  ASSERT_TRUE(arrow::io::SetIOThreadPoolCapacity(1).ok());
  auto restore_io = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(arrow::io::SetIOThreadPoolCapacity(io_capacity).ok());
  });
  std::promise<void> io_started, started, release;
  auto gate = release.get_future().share();
  auto blocker = host->Submit([&started, gate] {
    started.set_value();
    gate.wait();
  });
  ASSERT_TRUE(blocker.ok());
  auto io_blocker =
      arrow::io::default_io_context().executor()->Submit([&io_started, gate] {
        io_started.set_value();
        gate.wait();
      });
  bool released = false;
  auto unblock = ailego::ScopeGuard::Make([&] {
    if (!released) release.set_value();
    blocker->Wait();
    if (io_blocker.ok()) io_blocker->Wait();
  });
  ASSERT_TRUE(io_blocker.ok());
  started.get_future().wait();
  io_started.get_future().wait();
  auto run = std::async(std::launch::async, [] {
    auto schema = arrow::schema({arrow::field("row", arrow::int32())});
    auto source = arrow::RecordBatchReader::Make(
        std::vector<std::shared_ptr<arrow::RecordBatch>>{}, schema);
    if (!source.ok()) return false;
    PlanInfo plan({"record_batch_reader_source",
                   arrow::acero::RecordBatchReaderSourceNodeOptions{*source}},
                  schema, /*budgeted=*/true);
    auto reader = plan.execute_to_reader();
    if (!reader) return false;
    std::shared_ptr<arrow::RecordBatch> batch;
    if (!(*reader)->ReadNext(&batch).ok() || batch) return false;
    auto stream = plan.execute_to_generator(8192);
    if (!stream) return false;
    auto end = (*stream)().result();
    return end.ok() && !end->has_value();
  });
  const auto ready = run.wait_for(std::chrono::seconds(3));
  // Release before joining even on failure, so a regression fails rather
  // than hanging behind the intentionally occupied process-global worker.
  release.set_value();
  blocker->Wait();
  io_blocker->Wait();
  released = true;
  EXPECT_EQ(std::future_status::ready, ready);
  EXPECT_TRUE(run.get());
  EXPECT_EQ(1, host->GetCapacity());
}

TEST(SegmentStreamTest,
     BufferPlanAccountsKernelOutputAndDeniesBeforeAllocation) {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const auto capacity = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] { pool.init(capacity); });
  pool.init(1024 * 1024);
  // PlanInfo charges its output to the query pool, separate from column cache.
  auto *arrow_pool = detail::GetParquetQueryMemoryPool().get();
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

TEST(SegmentStreamTest, IteratorFilterDeniesUnfundedArrowOutput) {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0, pool.init(1024 * 1024));
  auto restore = ailego::ScopeGuard::Make([&] { pool.init(previous); });
  auto *arrow_pool = detail::GetParquetMemoryPool().get();
  const auto baseline = arrow_pool->charged_bytes();
  auto schema = arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64())});
  arrow::UInt64Builder builder;
  ASSERT_TRUE(builder.AppendValues(std::vector<uint64_t>(16383, 7)).ok());
  ASSERT_TRUE(builder.Append(8).ok());
  auto input =
      arrow::RecordBatch::Make(schema, 16384, {builder.Finish().ValueOrDie()});
  auto filter = EasyIndexFilter::Create([](uint64_t id) { return id == 8; });
  auto make_reader = [&] {
    auto source = arrow::RecordBatchReader::Make({input}, schema).ValueOrDie();
    return FilteringReader::Make(std::move(source), filter, arrow_pool);
  };
  std::shared_ptr<arrow::RecordBatch> output;
  {
    auto reader = make_reader();
    // The bit mask fits in one small Arrow slab. The result needs 128 KiB.
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used() - 96 * 1024);
    EXPECT_TRUE(reader->ReadNext(&output).IsOutOfMemory());
    EXPECT_EQ(nullptr, output);
    EXPECT_EQ(baseline, arrow_pool->charged_bytes());
  }
  auto reader = make_reader();
  ASSERT_TRUE(reader->ReadNext(&output).ok());
  ASSERT_NE(nullptr, output);
  EXPECT_EQ(16383, output->num_rows());
  EXPECT_GE(arrow_pool->charged_bytes(), baseline + 16383 * sizeof(uint64_t));
  output.reset();
  EXPECT_EQ(baseline, arrow_pool->charged_bytes());
}

TEST(SegmentStreamTest, NestedScanDeliveryIsBoundedAndCancels) {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(2048UL << 20));
  auto calls = std::make_shared<std::atomic<int>>(0);
  std::weak_ptr<arrow::RecordBatchReader> weak;
  {
    auto *pool = detail::GetParquetQueryMemoryPool().get();
    auto child = std::make_shared<arrow::DoubleBuilder>(pool);
    arrow::ListBuilder values(pool, child);
    for (int i = 0; i < 2048; ++i) {
      ASSERT_TRUE(values.Append().ok());
      ASSERT_TRUE(child->AppendValues(std::vector<double>{1., 2.}).ok());
    }
    auto schema =
        arrow::schema({arrow::field("payload", arrow::list(arrow::float64()))});
    auto batch =
        arrow::RecordBatch::Make(schema, 2048, {values.Finish().ValueOrDie()});
    class Reader : public arrow::RecordBatchReader {
     public:
      Reader(std::shared_ptr<arrow::RecordBatch> batch,
             std::shared_ptr<std::atomic<int>> calls)
          : batch_(std::move(batch)), calls_(std::move(calls)) {}
      std::shared_ptr<arrow::Schema> schema() const override {
        return batch_->schema();
      }
      arrow::Status ReadNext(
          std::shared_ptr<arrow::RecordBatch> *out) override {
        *out = calls_->fetch_add(1) < 200 ? batch_ : nullptr;
        return arrow::Status::OK();
      }

     private:
      std::shared_ptr<arrow::RecordBatch> batch_;
      std::shared_ptr<std::atomic<int>> calls_;
    };
    auto reader = std::make_shared<Reader>(batch, calls);
    weak = reader;
    auto predicate = arrow::compute::call(
        "is_valid", {arrow::compute::field_ref("payload")});
    PlanInfo plan(
        arrow::acero::Declaration::Sequence(
            {{"record_batch_reader_source",
              arrow::acero::RecordBatchReaderSourceNodeOptions{reader}},
             {"filter", arrow::acero::FilterNodeOptions{predicate}}}),
        schema, true);
    auto output = plan.execute_to_reader(4096);
    ASSERT_TRUE(output);
    std::shared_ptr<arrow::RecordBatch> first;
    ASSERT_TRUE((*output)->ReadNext(&first).ok());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->num_rows(), 2048);
    EXPECT_TRUE((*output)->Close().ok());
    // Delivery window, buffered group and at most one in-flight reader group.
    EXPECT_LE(calls->load(), 72);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!weak.expired() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_TRUE(weak.expired());
}

TEST(SegmentStreamTest, WideLikeReadAheadIsBoundedAndCancels) {
  ASSERT_TRUE(arrow::compute::Initialize().ok());
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(2048UL << 20));
  for (const auto [batch_rows, value_bytes] :
       {std::pair{64, 4096}, std::pair{2048, 4096}, std::pair{2048, 16384}}) {
    auto calls = std::make_shared<std::atomic<int>>(0);
    std::weak_ptr<arrow::RecordBatchReader> weak;
    {
      arrow::StringBuilder values(detail::GetParquetQueryMemoryPool().get());
      for (int i = 0; i < batch_rows; ++i) {
        ASSERT_TRUE(
            values.Append((i == 0 ? "hit" : "xxx") +
                          std::string(value_bytes - 3, 'x'))
                .ok());
      }
      auto schema = arrow::schema({arrow::field("payload", arrow::utf8())});
      auto batch =
          arrow::RecordBatch::Make(schema, batch_rows, {values.Finish().ValueOrDie()});
      class Reader : public ResidentRecordBatchReader {
       public:
        Reader(std::shared_ptr<arrow::RecordBatch> batch,
               std::shared_ptr<std::atomic<int>> calls)
            : batch_(std::move(batch)), calls_(std::move(calls)) {}
        std::shared_ptr<arrow::Schema> schema() const override {
          return batch_->schema();
        }
        arrow::Status ReadNext(
            std::shared_ptr<arrow::RecordBatch> *out) override {
          *out = calls_->fetch_add(1) < 200 ? batch_ : nullptr;
          return arrow::Status::OK();
        }
        arrow::Result<bool> ReadNextResident(
            std::shared_ptr<arrow::RecordBatch> *out) override {
          ARROW_RETURN_NOT_OK(ReadNext(out));
          return true;
        }

       private:
        std::shared_ptr<arrow::RecordBatch> batch_;
        std::shared_ptr<std::atomic<int>> calls_;
      };
      auto reader = std::make_shared<Reader>(batch, calls);
      weak = reader;
      auto predicate = arrow::compute::call(
          "starts_with", {arrow::compute::field_ref("payload")},
          arrow::compute::MatchSubstringOptions("hit"));
      PlanInfo plan(
          arrow::acero::Declaration::Sequence(
              {{"record_batch_reader_source",
                arrow::acero::RecordBatchReaderSourceNodeOptions{reader}},
               {"filter", arrow::acero::FilterNodeOptions{predicate}}}),
          schema, true);
      auto output = plan.execute_to_reader(4096);
      ASSERT_TRUE(output);
      std::shared_ptr<arrow::RecordBatch> first;
      ASSERT_TRUE((*output)->ReadNext(&first).ok());
      ASSERT_NE(first, nullptr);
      EXPECT_EQ(first->num_rows(), 1);
      EXPECT_TRUE((*output)->Close().ok());
      // Delivery window, buffered group and at most one in-flight reader group.
      EXPECT_LE(calls->load(), 96);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!weak.expired() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(weak.expired());
  }
}
}  // namespace
}  // namespace zvec::sqlengine

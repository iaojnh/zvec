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

#include "plan_info.h"
#include <algorithm>
#include <mutex>
#include <arrow/acero/query_context.h>
#include <arrow/api.h>
#include <arrow/io/interfaces.h>
#include <arrow/util/align_util.h>
#include <arrow/util/async_generator.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include "db/common/error_code.h"
#include "db/common/global_resource.h"
#include "db/index/storage/parquet_memory_pool.h"

namespace zvec::sqlengine {

namespace {
arrow::acero::BackpressureOptions SourceBackpressure(size_t max_queued_bytes) {
  if (!max_queued_bytes) return {};
  // Arrow pauses strictly above its watermark. Pause when the byte limit is
  // reached, before another whole batch can overlap consumer-held buffers and
  // completed Futures still forwarding earlier batches.
  const auto resume = std::max<size_t>(1, max_queued_bytes / 2);
  const auto pause = std::max<size_t>(resume, max_queued_bytes - 1);
  return {resume, pause};
}

arrow::Result<arrow::acero::QueryOptions> QueryOptions(arrow::MemoryPool *pool,
                                                       bool budgeted) {
  arrow::acero::QueryOptions options;
  options.memory_pool = pool;
  if (budgeted) {
    auto &resources = GlobalResource::Instance();
    options.custom_cpu_executor = resources.arrow_query_thread_pool();
    options.custom_io_executor = resources.arrow_io_thread_pool();
    if (!options.custom_cpu_executor || !options.custom_io_executor)
      return arrow::Status::OutOfMemory("Managed Arrow executors unavailable");
  } else {
    options.custom_cpu_executor = arrow::internal::GetCpuThreadPool();
  }
  return options;
}

// Arrow's generic source queues delivery tasks before the sink can pause it.
// A ready generator can therefore enqueue its entire scan. Deliver one batch
// before pulling another and await backpressure without blocking an executor.
// This bounds source read-ahead, not the size of a batch or a blocking
// operator.
class BoundedSource : public arrow::acero::ExecNode {
 public:
  using Batch = std::optional<arrow::compute::ExecBatch>;
  BoundedSource(arrow::acero::ExecPlan *plan,
                const arrow::acero::SourceNodeOptions &options)
      : ExecNode(plan, {}, {}, options.output_schema),
        generator_(options.generator),
        ordering_(options.ordering) {
    if (auto *async = dynamic_cast<const AsyncSourceNodeOptions *>(&options))
      finish_ = async->finish;
  }
  const char *kind_name() const override {
    return "ZvecBoundedSource";
  }
  const arrow::compute::Ordering &ordering() const override {
    return ordering_;
  }
  arrow::Status InputReceived(ExecNode *, arrow::compute::ExecBatch) override {
    return arrow::Status::Invalid("Source has no inputs");
  }
  arrow::Status InputFinished(ExecNode *, int) override {
    return arrow::Status::Invalid("Source has no inputs");
  }
  arrow::Status StartProducing() override {
    ARROW_ASSIGN_OR_RAISE(
        auto done, plan_->query_context()->BeginExternalTask("zvec source"));
    if (!done.is_valid()) return arrow::Status::OK();
    arrow::CallbackOptions options;
    options.executor = plan_->query_context()->executor();
    options.should_schedule = arrow::ShouldSchedule::IfDifferentExecutor;
    arrow::Loop([this, options]() -> arrow::Future<arrow::ControlFlow<int>> {
      if (stopped()) return arrow::Break(count_);
      return pull().Then(
          [this](const Batch &batch) { return consume(batch); },
          [](const arrow::Status &error)
              -> arrow::Future<arrow::ControlFlow<int>> { return error; },
          options);
    }).AddCallback([this, done](const arrow::Result<int> &result) mutable {
      auto status = result.status();
      auto finished = finish_ ? finish_() : arrow::Future<>::MakeFinished();
      finished.AddCallback(
          [this, done, status, count = result.ok() ? *result : 0](
              const arrow::Status &cleanup) mutable {
            // Child cancellation is part of this source's external task.
            // Releasing a child generator without awaiting it lets
            // Segment::close() race with global Arrow/RocksDB destruction after
            // the caller has returned.
            generator_ = {};
            finish_ = {};
            if (status.ok()) status = cleanup;
            if (status.ok()) status = output_->InputFinished(this, count);
            done.MarkFinished(status);
          });
    });
    return arrow::Status::OK();
  }
  void PauseProducing(ExecNode *, int32_t counter) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (counter <= counter_ || stopped_) return;
    counter_ = counter;
    if (resume_.is_finished()) resume_ = arrow::Future<>::Make();
  }
  void ResumeProducing(ExecNode *, int32_t counter) override {
    arrow::Future<> wake;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (counter <= counter_) return;
      counter_ = counter;
      wake = resume_;
      resume_ = arrow::Future<>::MakeFinished();
    }
    if (!wake.is_finished()) wake.MarkFinished();
  }

 protected:
  arrow::Status StopProducingImpl() override {
    arrow::Future<> wake;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
      wake = resume_;
      resume_ = arrow::Future<>::MakeFinished();
    }
    if (!wake.is_finished()) wake.MarkFinished();
    return arrow::Status::OK();
  }

 private:
  bool stopped() {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopped_;
  }
  arrow::Future<Batch> pull() try {
    return generator_();
  } catch (const std::bad_alloc &) {
    return arrow::Status::OutOfMemory("Source allocation failed");
  } catch (const std::exception &error) {
    return arrow::Status::ExecutionError(error.what());
  }
  arrow::Future<arrow::ControlFlow<int>> consume(const Batch &next) try {
    if (!next || stopped()) return arrow::Break(count_);
    // Keep Arrow's batch-size contract. Slices share a single source buffer;
    // backpressure prevents fetching the next independently allocated batch.
    const bool legacy = plan_->query_context()->options().use_legacy_batching;
    int64_t offset = 0;
    do {
      const int64_t size =
          legacy ? next->length
                 : std::min<int64_t>(next->length - offset,
                                     arrow::acero::ExecPlan::kMaxBatchSize);
      ARROW_RETURN_NOT_OK(deliver(next->Slice(offset, size)));
      offset += size;
    } while (offset < next->length && !stopped());
    arrow::Future<> resume;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      resume = resume_;
    }
    return resume.Then(
        []() -> arrow::ControlFlow<int> { return arrow::Continue(); });
  } catch (const std::bad_alloc &) {
    return arrow::Status::OutOfMemory("Source delivery allocation failed");
  } catch (const std::exception &error) {
    return arrow::Status::ExecutionError(error.what());
  }
  arrow::Status deliver(arrow::compute::ExecBatch batch) {
    const auto handling =
        plan_->query_context()->options().unaligned_buffer_handling.value_or(
            arrow::acero::GetDefaultUnalignedBufferHandling());
    for (auto &value : batch.values) {
      if (!value.is_array() ||
          handling == arrow::acero::UnalignedBufferHandling::kIgnore)
        continue;
      if (arrow::util::CheckAlignment(*value.array(),
                                      arrow::util::kValueAlignment))
        continue;
      if (handling == arrow::acero::UnalignedBufferHandling::kError)
        return arrow::Status::Invalid("Unaligned source buffer");
      if (handling == arrow::acero::UnalignedBufferHandling::kReallocate) {
        ARROW_ASSIGN_OR_RAISE(value,
                              arrow::util::EnsureAlignment(
                                  value.array(), arrow::util::kValueAlignment,
                                  plan_->query_context()->memory_pool()));
      } else
        LOG_WARN("Unaligned source buffer");
    }
    if (!ordering_.is_unordered()) batch.index = count_;
    ++count_;
    return output_->InputReceived(this, std::move(batch));
  }
  arrow::AsyncGenerator<Batch> generator_;
  std::function<arrow::Future<>()> finish_;
  arrow::compute::Ordering ordering_;
  std::mutex mutex_;
  bool stopped_{false};
  int32_t counter_{0};
  int count_{0};
  arrow::Future<> resume_ = arrow::Future<>::MakeFinished();
};

arrow::Result<arrow::acero::Declaration> bound_sources(
    arrow::acero::Declaration plan,
    arrow::internal::Executor *managed_io = nullptr) {
  static const auto registered =
      arrow::acero::default_exec_factory_registry()->AddFactory(
          "zvec_bounded_source",
          [](arrow::acero::ExecPlan *exec,
             std::vector<arrow::acero::ExecNode *> inputs,
             const arrow::acero::ExecNodeOptions &options)
              -> arrow::Result<arrow::acero::ExecNode *> {
            const auto *source =
                dynamic_cast<const arrow::acero::SourceNodeOptions *>(&options);
            if (!inputs.empty() || !source || !source->generator ||
                !source->output_schema)
              return arrow::Status::Invalid("Invalid bounded source options");
            return exec->EmplaceNode<BoundedSource>(exec, *source);
          });
  ARROW_RETURN_NOT_OK(registered);
  if (plan.factory_name == "record_batch_reader_source") {
    auto *options =
        dynamic_cast<arrow::acero::RecordBatchReaderSourceNodeOptions *>(
            plan.options.get());
    if (!options || !options->reader)
      return arrow::Status::Invalid("Missing source reader");
    auto reader = options->reader;
    auto *executor = managed_io ? managed_io : options->io_executor;
    if (!executor) executor = arrow::io::default_io_context().executor();
    arrow::AsyncGenerator<BoundedSource::Batch> generator =
        [reader, executor]() -> arrow::Future<BoundedSource::Batch> {
      auto future =
          executor->Submit([reader]() -> arrow::Result<BoundedSource::Batch> {
            try {
              std::shared_ptr<arrow::RecordBatch> batch;
              ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
              if (!batch) return BoundedSource::Batch{};
              return BoundedSource::Batch{arrow::compute::ExecBatch(*batch)};
            } catch (const std::bad_alloc &) {
              return arrow::Status::OutOfMemory(
                  "Source reader allocation failed");
            } catch (const std::exception &error) {
              return arrow::Status::ExecutionError(error.what());
            }
          });
      if (!future.ok()) return future.status();
      return std::move(*future);
    };
    plan.options = std::make_shared<arrow::acero::SourceNodeOptions>(
        reader->schema(), std::move(generator),
        arrow::compute::Ordering::Implicit());
    plan.factory_name = "source";
  }
  if (plan.factory_name == "source") plan.factory_name = "zvec_bounded_source";
  for (auto &input : plan.inputs) {
    if (auto *child = std::get_if<arrow::acero::Declaration>(&input)) {
      ARROW_ASSIGN_OR_RAISE(*child, bound_sources(*child, managed_io));
    }
  }
  return plan;
}

class AsyncPlanStream {
 public:
  using Batch = std::optional<arrow::compute::ExecBatch>;
  ~AsyncPlanStream() {
    if (plan && started && !plan->finished().is_finished()) {
      plan->StopProducing();
      // Destruction can run on the sole CPU worker: never wait here for tasks
      // queued on that worker. Keep the plan alive until cancellation finishes.
      plan->finished().AddCallback([keep = plan](const arrow::Status &) {});
    }
  }
  std::shared_ptr<arrow::acero::ExecPlan> plan;
  arrow::AsyncGenerator<Batch> generator;
  bool started{false};
};

bool has_async_source(const arrow::acero::Declaration &plan) {
  if (dynamic_cast<const AsyncSourceNodeOptions *>(plan.options.get()))
    return true;
  for (const auto &input : plan.inputs) {
    if (auto *child = std::get_if<arrow::acero::Declaration>(&input)) {
      if (has_async_source(*child)) return true;
    }
  }
  return false;
}

// DeclarationToReader's sink has no backpressure in our Arrow version. Use an
// explicit sink for potentially unbounded scans. This limits queued output, not
// an individual batch, in-flight I/O, or memory held by upstream operators.
class BoundedPlanReader : public arrow::RecordBatchReader {
 public:
  static arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> Make(
      const arrow::acero::Declaration &declaration, size_t max_queued_bytes,
      arrow::MemoryPool *pool, bool budgeted) {
    ARROW_ASSIGN_OR_RAISE(auto options, QueryOptions(pool, budgeted));
    arrow::compute::ExecContext context(pool, options.custom_cpu_executor);
    ARROW_ASSIGN_OR_RAISE(auto plan,
                          arrow::acero::ExecPlan::Make(options, context));
    ARROW_ASSIGN_OR_RAISE(
        auto source, bound_sources(declaration, options.custom_io_executor));
    arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>> generator;
    std::shared_ptr<arrow::Schema> schema;
    auto with_sink = arrow::acero::Declaration::Sequence(
        {source,
         {"sink",
          arrow::acero::SinkNodeOptions{
              &generator, &schema, SourceBackpressure(max_queued_bytes)}}});
    ARROW_RETURN_NOT_OK(with_sink.AddToPlan(plan.get()));
    ARROW_RETURN_NOT_OK(plan->Validate());
    auto reader =
        arrow::acero::MakeGeneratorReader(schema, std::move(generator), pool);
    auto result = std::make_unique<BoundedPlanReader>(plan, std::move(reader));
    plan->StartProducing();
    result->started_ = true;
    return result;
  }

  BoundedPlanReader(std::shared_ptr<arrow::acero::ExecPlan> plan,
                    std::shared_ptr<arrow::RecordBatchReader> reader)
      : plan_(std::move(plan)),
        reader_(std::move(reader)),
        schema_(reader_->schema()) {}

  ~BoundedPlanReader() override {
    if (started_) static_cast<void>(Close());
  }

  std::shared_ptr<arrow::Schema> schema() const override {
    return schema_;
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    if (!reader_) return arrow::Status::Invalid("Document ID reader is closed");
    auto status = reader_->ReadNext(batch);
    if (!status.ok()) {
      static_cast<void>(Close());
      return status;
    }
    if (!*batch) return plan_->finished().status();
    return status;
  }

  arrow::Status Close() override {
    if (!reader_) return arrow::Status::OK();
    plan_->StopProducing();
    auto status = plan_->finished().status();
    // Release queued batches only after upstream tasks have stopped.
    reader_.reset();
    // An explicit early close is successful cancellation, not a query error.
    return status.IsCancelled() ? arrow::Status::OK() : status;
  }

 private:
  std::shared_ptr<arrow::acero::ExecPlan> plan_;
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  bool started_{false};
};
}  // namespace

Result<PlanStream> PlanInfo::execute_to_generator(size_t max_queued_bytes) {
  auto make = [&]() -> arrow::Result<PlanStream> {
    auto stream = std::make_shared<AsyncPlanStream>();
    auto *pool = budgeted_ ? detail::GetParquetMemoryPool().get()
                           : arrow::default_memory_pool();
    ARROW_ASSIGN_OR_RAISE(auto options, QueryOptions(pool, budgeted_));
    ARROW_ASSIGN_OR_RAISE(auto source,
                          bound_sources(plan_, options.custom_io_executor));
    arrow::compute::ExecContext context(pool, options.custom_cpu_executor);
    ARROW_ASSIGN_OR_RAISE(stream->plan,
                          arrow::acero::ExecPlan::Make(options, context));
    auto with_sink = arrow::acero::Declaration::Sequence(
        {source,
         {"sink", arrow::acero::SinkNodeOptions{
                      &stream->generator, nullptr,
                      SourceBackpressure(max_queued_bytes)}}});
    ARROW_RETURN_NOT_OK(with_sink.AddToPlan(stream->plan.get()));
    ARROW_RETURN_NOT_OK(stream->plan->Validate());
    stream->started = true;
    stream->plan->StartProducing();
    PlanStream result;
    result.generator = [stream] {
      return stream->generator().Then(
          [stream](const AsyncPlanStream::Batch &batch)
              -> arrow::Future<AsyncPlanStream::Batch> {
            if (batch)
              return arrow::Future<AsyncPlanStream::Batch>::MakeFinished(batch);
            return stream->plan->finished().Then(
                [] { return AsyncPlanStream::Batch{}; });
          });
    };
    result.close = [stream] {
      stream->plan->StopProducing();
      return stream->plan->finished();
    };
    return result;
  };
  auto result = make();
  if (!result.ok()) {
    if (result.status().IsOutOfMemory())
      return tl::make_unexpected(
          Status(StatusCode::RESOURCE_EXHAUSTED, result.status().ToString()));
    return tl::make_unexpected(
        Status::InternalError(result.status().ToString()));
  }
  return std::move(*result);
}

Result<std::unique_ptr<arrow::RecordBatchReader>> PlanInfo::execute_to_reader(
    size_t max_queued_bytes) {
  auto *pool = budgeted_ ? detail::GetParquetMemoryPool().get()
                         : arrow::default_memory_pool();
  if (budgeted_ && !max_queued_bytes) max_queued_bytes = 64 * 1024;
  // An unbudgeted parent reaching its limit must also await cancellation of
  // nested segment streams. Arrow's generic source cannot run our finalizer.
  // A zero queue limit retains the existing unbounded sink behavior.
  auto res =
      max_queued_bytes || has_async_source(plan_)
          ? BoundedPlanReader::Make(plan_, max_queued_bytes, pool, budgeted_)
          : arrow::acero::DeclarationToReader(plan_, true, pool);
  if (!res.ok()) {
    if (res.status().IsOutOfMemory())
      return tl::make_unexpected(
          Status(StatusCode::RESOURCE_EXHAUSTED, res.status().ToString()));
    return tl::make_unexpected(Status::InternalError(
        "execute plan_info failed: ", res.status().ToString()));
  }
  return res.MoveValueUnsafe();
}

}  // namespace zvec::sqlengine

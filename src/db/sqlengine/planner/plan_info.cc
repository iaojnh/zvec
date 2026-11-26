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
#include <arrow/api.h>
#include <arrow/util/async_generator.h>
#include <zvec/ailego/pattern/expected.hpp>
#include "db/common/managed_arrow.h"
#include "db/index/storage/parquet_memory_pool.h"

namespace zvec::sqlengine {
namespace {

class ManagedPlanReader : public arrow::RecordBatchReader {
 public:
  static arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> Make(
      arrow::acero::Declaration declaration) {
    auto *pool = detail::GetParquetQueryMemoryPool().get();
    ARROW_ASSIGN_OR_RAISE(auto options, detail::ManagedArrowOptions(pool));
    detail::BindManagedArrowIo(declaration, options.custom_io_executor);
    arrow::compute::ExecContext context(pool, options.custom_cpu_executor);
    ARROW_ASSIGN_OR_RAISE(auto plan,
                          arrow::acero::ExecPlan::Make(options, context));
    arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>> batches;
    std::shared_ptr<arrow::Schema> schema;
    auto with_sink = arrow::acero::Declaration::Sequence(
        {std::move(declaration),
         {"sink", arrow::acero::SinkNodeOptions{
                      &batches, &schema, {32 * 1024, 64 * 1024 - 1}}}});
    ARROW_RETURN_NOT_OK(with_sink.AddToPlan(plan.get()));
    ARROW_RETURN_NOT_OK(plan->Validate());
    auto reader = arrow::acero::MakeGeneratorReader(
        schema, std::move(batches), pool);
    auto result = std::make_unique<ManagedPlanReader>(plan, std::move(reader));
    // Arrow 21 retains the completion state during callback dispatch when
    // a callback is present. Register before producers can reach completion.
    plan->finished().AddCallback([](const arrow::Status &) {});
    plan->StartProducing();
    result->started_ = true;
    return result;
  }

  ManagedPlanReader(std::shared_ptr<arrow::acero::ExecPlan> plan,
                    std::shared_ptr<arrow::RecordBatchReader> reader)
      : plan_(std::move(plan)),
        reader_(std::move(reader)),
        schema_(reader_->schema()) {}

  ~ManagedPlanReader() override {
    if (started_) static_cast<void>(Close());
  }

  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    if (!reader_) return arrow::Status::Invalid("Query reader is closed");
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
    reader_.reset();
    return status.IsCancelled() ? arrow::Status::OK() : status;
  }

 private:
  std::shared_ptr<arrow::acero::ExecPlan> plan_;
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  bool started_{false};
};

}  // namespace

Result<std::unique_ptr<arrow::RecordBatchReader>> PlanInfo::execute_to_reader()
    try {
  auto make_reader = [&]() -> arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> {
    if (!managed_) return arrow::acero::DeclarationToReader(plan_);
    if (!reads_segment_plans_) return ManagedPlanReader::Make(plan_);
    // The merge source reads child plans synchronously. Run the merge on the
    // consumer so child work can always run on the managed CPU executor.
    ARROW_ASSIGN_OR_RAISE(auto options, detail::ManagedArrowOptions(
        detail::GetParquetQueryMemoryPool().get(), /*use_threads=*/false));
    return arrow::acero::DeclarationToReader(plan_, options);
  };
  auto reader = make_reader();
  if (!reader.ok()) {
    return tl::make_unexpected(detail::ArrowQueryError(reader.status()));
  }
  return std::move(*reader);
} catch (const std::bad_alloc &) {
  return tl::make_unexpected(
      Status(StatusCode::RESOURCE_EXHAUSTED, "Query reader allocation refused"));
}

}  // namespace zvec::sqlengine

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
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include "db/common/error_code.h"

namespace zvec::sqlengine {

namespace {
// DeclarationToReader's sink has no backpressure in our Arrow version. Use an
// explicit sink for potentially unbounded scans. This limits queued output, not
// an individual batch, in-flight I/O, or memory held by upstream operators.
class BoundedPlanReader : public arrow::RecordBatchReader {
 public:
  static arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> Make(
      const arrow::acero::Declaration &declaration, size_t max_queued_bytes) {
    ARROW_ASSIGN_OR_RAISE(auto plan, arrow::acero::ExecPlan::Make());
    arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>> generator;
    std::shared_ptr<arrow::Schema> schema;
    auto with_sink = arrow::acero::Declaration::Sequence(
        {declaration,
         {"sink", arrow::acero::SinkNodeOptions{
                      &generator, &schema,
                      arrow::acero::BackpressureOptions{max_queued_bytes / 2,
                                                        max_queued_bytes}}}});
    ARROW_RETURN_NOT_OK(with_sink.AddToPlan(plan.get()));
    ARROW_RETURN_NOT_OK(plan->Validate());
    auto reader = arrow::acero::MakeGeneratorReader(
        schema, std::move(generator), arrow::default_memory_pool());
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

Result<std::unique_ptr<arrow::RecordBatchReader>> PlanInfo::execute_to_reader(
    size_t max_queued_bytes) {
  auto res = max_queued_bytes ? BoundedPlanReader::Make(plan_, max_queued_bytes)
                              : arrow::acero::DeclarationToReader(plan_);
  if (!res.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "execute plan_info failed: ", res.status().ToString()));
  }
  return res.MoveValueUnsafe();
}

}  // namespace zvec::sqlengine

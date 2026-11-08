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

#pragma once

#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <arrow/acero/api.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/util/async_generator.h>
#include <zvec/db/status.h>

namespace zvec::sqlengine {

// A composed source must finish cancelling its child plans before the parent
// publishes completion. Destroying an asynchronous generator alone cannot wait
// for cancellation when it runs on the same executor as the child.
struct AsyncSourceNodeOptions : arrow::acero::SourceNodeOptions {
  AsyncSourceNodeOptions(
      std::shared_ptr<arrow::Schema> schema,
      arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>> generator,
      std::function<arrow::Future<>()> finish)
      : SourceNodeOptions(std::move(schema), std::move(generator),
                          arrow::compute::Ordering::Implicit()),
        finish(std::move(finish)) {}
  std::function<arrow::Future<>()> finish;
};

struct PlanStream {
  arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>> generator;
  std::function<arrow::Future<>()> close;

  arrow::Future<std::optional<arrow::compute::ExecBatch>> operator()() const {
    return generator();
  }
};

class PlanInfo {
 public:
  using Ptr = std::shared_ptr<PlanInfo>;

  PlanInfo(arrow::acero::Declaration plan,
           std::shared_ptr<arrow::Schema> schema, bool budgeted = false)
      : plan_(std::move(plan)),
        schema_(std::move(schema)),
        budgeted_(budgeted) {}

  Result<std::unique_ptr<arrow::RecordBatchReader>> execute_to_reader(
      size_t max_queued_bytes = 0);

  std::shared_ptr<arrow::Schema> schema() const {
    return schema_;
  }

  // Non-blocking pull interface for composing plans on Arrow's own executor.
  Result<PlanStream> execute_to_generator(size_t max_queued_bytes);

 private:
  arrow::acero::Declaration plan_;
  std::shared_ptr<arrow::Schema> schema_;
  bool budgeted_;
};

}  // namespace zvec::sqlengine

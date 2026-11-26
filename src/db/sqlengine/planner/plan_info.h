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

#include <list>
#include <memory>
#include <string>
#include <arrow/acero/api.h>
#include <arrow/acero/exec_plan.h>
#include <zvec/db/status.h>

namespace zvec::sqlengine {

class PlanInfo {
 public:
  using Ptr = std::shared_ptr<PlanInfo>;

  PlanInfo(arrow::acero::Declaration plan,
           std::shared_ptr<arrow::Schema> schema, bool managed = false,
           bool reads_segment_plans = false)
      : plan_(std::move(plan)),
        schema_(std::move(schema)),
        managed_(managed),
        reads_segment_plans_(reads_segment_plans) {}

  Result<std::unique_ptr<arrow::RecordBatchReader>> execute_to_reader();

  std::shared_ptr<arrow::Schema> schema() const {
    return schema_;
  }

 private:
  arrow::acero::Declaration plan_;
  std::shared_ptr<arrow::Schema> schema_;
  bool managed_;
  bool reads_segment_plans_;
};

}  // namespace zvec::sqlengine

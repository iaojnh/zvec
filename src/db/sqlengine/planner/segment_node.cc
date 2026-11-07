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

#include "db/sqlengine/planner/segment_node.h"
#include <memory>
#include <optional>
#include <arrow/record_batch.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>

namespace zvec::sqlengine {

namespace cp = arrow::compute;

arrow::AsyncGenerator<std::optional<cp::ExecBatch>> SegmentNode::gen() {
  return [self = shared_from_this()] {
    using Batch = std::optional<cp::ExecBatch>;
    // Scalar queries do not need every segment's candidates simultaneously.
    // Preserve reverse traversal, start one bounded source at a time, and never
    // block the CPU executor waiting for a child plan on that same executor.
    return arrow::Loop([self]() -> arrow::Future<arrow::ControlFlow<Batch>> {
      try {
        if (!self->current_) {
          if (self->segment_plans_.empty()) return arrow::Break(Batch{});
          auto result =
              self->segment_plans_.back()->execute_to_generator(64 * 1024);
          if (!result)
            return arrow::Status::ExecutionError(result.error().message());
          self->current_ = std::move(*result);
          self->segment_plans_.pop_back();
        }
        return self->current_().Then(
            [self](const Batch &batch) -> arrow::ControlFlow<Batch> {
              if (batch) return arrow::Break(batch);
              self->current_ = {};
              return arrow::ControlFlow<Batch>{};
            });
      } catch (const std::bad_alloc &) {
        return arrow::Status::OutOfMemory("Segment stream allocation failed");
      } catch (const std::exception &error) {
        return arrow::Status::ExecutionError(error.what());
      }
    });
  };
}

}  // namespace zvec::sqlengine

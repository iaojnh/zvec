// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <arrow/acero/exec_plan.h>
#include <zvec/db/status.h>

namespace zvec::detail {

// Apply the DB RSS limit when BP first uses managed resources. An explicitly
// configured process limit continues to govern all callers.
arrow::Status ConfigureManagedDbRss();

arrow::Result<arrow::acero::QueryOptions> ManagedArrowOptions(
    arrow::MemoryPool *pool, bool use_threads = true);

// Record-batch source nodes do not inherit the plan's I/O executor.
void BindManagedArrowIo(arrow::acero::Declaration &declaration,
                        arrow::internal::Executor *executor);

Status ArrowQueryError(const arrow::Status &status);

}  // namespace zvec::detail

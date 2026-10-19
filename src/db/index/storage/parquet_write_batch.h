// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <arrow/api.h>
#include <arrow/util/byte_size.h>
#include <parquet/arrow/writer.h>

namespace zvec::detail {

// WriteTable writes columns directly rather than retaining buffered row groups
// across WriteRecordBatch calls. Slicing shares input buffers; no concatenated
// copy is needed. The row count is a sizing hint, while the writer's memory
// pool enforces actual allocations (including large individual values).
inline arrow::Status WriteParquetTable(parquet::arrow::FileWriter *writer,
                                       const arrow::Table &table) {
  if (!table.num_rows()) return arrow::Status::OK();
  const auto bytes = arrow::util::TotalBufferSize(table);
  const auto per_row = std::max<int64_t>(1, bytes / table.num_rows());
  const auto rows =
      std::max<int64_t>(1, std::min<int64_t>(8192, (1024 * 1024) / per_row));
  return writer->WriteTable(table, rows);
}

inline arrow::Status WriteParquetBatch(parquet::arrow::FileWriter *writer,
                                       const arrow::RecordBatch &batch) {
  arrow::ChunkedArrayVector columns;
  columns.reserve(batch.num_columns());
  for (const auto &column : batch.columns())
    columns.push_back(std::make_shared<arrow::ChunkedArray>(column));
  return WriteParquetTable(
      writer, *arrow::Table::Make(batch.schema(), columns, batch.num_rows()));
}
}  // namespace zvec::detail

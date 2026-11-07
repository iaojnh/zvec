// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <arrow/api.h>
#include <arrow/util/byte_size.h>
#include <parquet/arrow/writer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::detail {

// A decoder batch is not a row group: creating one group per small batch
// accumulates file metadata (and another copy at Close) outside Arrow's pool.
// Share a bounded encoded row group across calls instead. Input arrays are not
// retained; all encoding buffers still use the writer's budgeted memory pool.
class ParquetBatchWriter {
 public:
  explicit ParquetBatchWriter(int64_t max_rows = 0)
      : max_rows_(max_rows > 0 ? std::min<int64_t>(8192, max_rows) : 8192) {
    const auto capacity = ailego::MemoryLimitPool::get_instance().capacity();
    if (capacity != 0) {
      // Leave room for dictionaries, encoding scratch, readers and other
      // writers. This is a batching hint, not a replacement for admission.
      target_bytes_ = std::max<size_t>(
          64 * 1024, std::min<size_t>(target_bytes_, capacity / 16));
    }
  }

  arrow::Status write(parquet::arrow::FileWriter *writer,
                      const arrow::RecordBatch &batch) {
    if (!status_.ok()) return status_;
    status_ = write_batch(writer, batch);
    return status_;
  }

  arrow::Status write(parquet::arrow::FileWriter *writer,
                      const arrow::Table &table) {
    if (!status_.ok()) return status_;
    auto validation = table.Validate();
    if (!validation.ok()) return status_ = validation;
    if (!table.schema()->Equals(*writer->schema(), false)) {
      return status_ = arrow::Status::Invalid("Parquet writer schema mismatch");
    }
    arrow::TableBatchReader reader(table);
    reader.set_chunksize(max_rows_);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      status_ = reader.ReadNext(&batch);
      if (!status_.ok() || !batch) return status_;
      ARROW_RETURN_NOT_OK(write(writer, *batch));
    }
  }

  const arrow::Status &status() const {
    return status_;
  }

 private:
  arrow::Status write_batch(parquet::arrow::FileWriter *writer,
                            const arrow::RecordBatch &batch) {
    ARROW_RETURN_NOT_OK(batch.Validate());
    if (!batch.schema()->Equals(*writer->schema(), false)) {
      return arrow::Status::Invalid("Parquet writer schema mismatch");
    }
    int64_t offset = 0;
    while (offset < batch.num_rows()) {
      int64_t rows = std::min(max_rows_, batch.num_rows() - offset);
      auto slice = batch.Slice(offset, rows);
      ARROW_ASSIGN_OR_RAISE(auto bytes,
                            arrow::util::ReferencedBufferSize(*slice));
      // Count the referenced slice, not its entire parent buffer. Halving also
      // handles highly variable field widths without trusting a row average.
      while (rows > 1 && bytes > target_bytes_) {
        rows /= 2;
        slice = batch.Slice(offset, rows);
        ARROW_ASSIGN_OR_RAISE(bytes, arrow::util::ReferencedBufferSize(*slice));
      }
      if (rows_ != 0 && (rows > max_rows_ - rows_ || bytes_ >= target_bytes_ ||
                         bytes > target_bytes_ - bytes_)) {
        ARROW_RETURN_NOT_OK(writer->NewBufferedRowGroup());
        rows_ = 0;
        bytes_ = 0;
      }
      // A single oversized value is still subject to pool admission. It is
      // isolated from the next batch rather than silently exceeding a group.
      ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*slice));
      rows_ += rows;
      bytes_ += bytes;
      offset += rows;
    }
    return arrow::Status::OK();
  }

  int64_t max_rows_;
  int64_t target_bytes_ = 4 * 1024 * 1024;
  int64_t rows_ = 0;
  int64_t bytes_ = 0;
  arrow::Status status_;
};
}  // namespace zvec::detail

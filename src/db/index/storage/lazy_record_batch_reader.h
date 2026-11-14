
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

#include <arrow/ipc/reader.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include "db/common/constants.h"
#include "parquet_buffer_pool.h"
#include "parquet_memory_pool.h"


namespace zvec {

class IPCRecordBatchReader : public arrow::RecordBatchReader {
 public:
  IPCRecordBatchReader(
      std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader,
      const std::vector<std::string> &columns,
      std::shared_ptr<arrow::Schema> schema)
      : reader_(std::move(reader)),
        schema_(std::move(schema)),
        columns_(columns) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto &col : columns) {
      int index = schema_->GetFieldIndex(col);
      if (index != -1) {
        fields.push_back(schema_->field(index));
        col_indices_.push_back(index);
      }
    }
    projected_schema_ = arrow::schema(fields);
    num_record_batches_ = reader_->num_record_batches();
  }

  std::shared_ptr<arrow::Schema> schema() const override {
    return projected_schema_;
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    if (current_batch_ >= num_record_batches_) {
      *batch = nullptr;
      return arrow::Status::OK();
    }

    ARROW_ASSIGN_OR_RAISE(auto full_batch,
                          reader_->ReadRecordBatch(current_batch_));
    current_batch_++;

    std::vector<std::shared_ptr<arrow::Array>> projected_arrays;
    for (int index : col_indices_) {
      projected_arrays.push_back(full_batch->column(index));
    }

    *batch = arrow::RecordBatch::Make(projected_schema_, full_batch->num_rows(),
                                      projected_arrays);
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> projected_schema_;
  std::vector<std::string> columns_;
  std::vector<int> col_indices_;
  int current_batch_ = 0;
  int num_record_batches_ = 0;
};


// Keep the BP reader implementation out of mmap's shared reader header.
class ParquetRecordBatchReader : public arrow::RecordBatchReader {
 public:
  ParquetRecordBatchReader(std::unique_ptr<parquet::arrow::FileReader> &reader,
                           const std::vector<std::string> &columns,
                           std::shared_ptr<arrow::Schema> schema,
                           const std::string &file_path, bool with_cache = true,
                           bool stream_uncached = false,
                           std::shared_ptr<arrow::MemoryPool> memory_pool = {},
                           ParquetBufferID metadata_id = {});

  ~ParquetRecordBatchReader() override;

  ParquetRecordBatchReader(const std::vector<std::string> &columns,
                           std::shared_ptr<arrow::Schema> schema,
                           const std::string &file_path,
                           std::shared_ptr<arrow::MemoryPool> memory_pool,
                           std::shared_ptr<arrow::io::RandomAccessFile> input,
                           std::shared_ptr<parquet::FileMetaData> metadata,
                           ParquetBufferID metadata_id,
                           const std::vector<int64_t> &row_group_rows);

  std::shared_ptr<arrow::Schema> schema() const override {
    return projected_schema_;
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  // Preserve the existing uncached mmap reader's row-group batch boundaries.
  arrow::Status read_row_group(std::shared_ptr<arrow::RecordBatch> *batch);

  arrow::Status read_next_uncached(std::shared_ptr<arrow::RecordBatch> *batch);

  void project_columns();
  arrow::Status ensure_reader();

  // FileReader must outlive the batch reader (members are destroyed in reverse
  // order). The reader and its file remain alive even if the store is closed.
  std::shared_ptr<arrow::MemoryPool> memory_pool_;
  std::shared_ptr<arrow::io::RandomAccessFile> input_;
  std::shared_ptr<parquet::FileMetaData> metadata_;
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::unique_ptr<arrow::RecordBatchReader> row_group_reader_;
  std::unique_ptr<arrow::TableBatchReader> cached_batches_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> projected_schema_;
  std::vector<std::string> columns_;
  std::vector<int> col_indices_;
  std::string file_path_;
  ParquetBufferID metadata_id_;

  int64_t current_row_group_ = 0;
  int64_t num_row_groups_ = 0;
  int64_t total_rows_ = 0;
  std::vector<int64_t> row_group_offsets_;
  std::vector<int64_t> row_group_row_nums_;
  bool with_cache_;
  bool stream_uncached_;
};


}  // namespace zvec

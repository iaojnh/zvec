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

#include "bufferpool_forward_store.h"
#include <arrow/acero/exec_plan.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/logger/logger.h>
#include "db/index/storage/store_helper.h"
#include "fetch_memory.h"
#include "lazy_record_batch_reader.h"
#include "parquet_buffer_pool.h"
#include "parquet_memory_pool.h"
#include "parquet_metadata_budget.h"


namespace zvec {

BufferPoolForwardStore::BufferPoolForwardStore(const std::string &uri)
    : file_path_(uri) {}

BufferPoolForwardStore::~BufferPoolForwardStore() {
  if (parquet_reader_) {
    // Compaction retires immutable files. Their decoded columns must not
    // accumulate until global pressure happens to evict them. Escaped Arrow
    // views retain their pins and stay valid until their last consumer exits.
    ParquetBufferPool::get_instance().retire_file(metadata_id_);
  }
}

Status BufferPoolForwardStore::open() {
  std::string uri = file_path_;
  // The footer reader lives as long as this store. A mapped file here would
  // keep faulted file pages resident outside the pool's eviction mechanism.
  // ReadAt's temporary buffers must use the same managed allocator as decode.
  auto status = CreateRandomAccessFileByUri(
      uri, &file_, &file_path_, /*enable_mmap=*/false,
      detail::GetParquetMemoryPool().get());
  if (!status.ok()) {
    return Status::InternalError("Failed to create random access uri: ", uri,
                                 " : ", status.ToString());
  }
  auto format = InferFileFormat(file_path_);
  if (format == FileFormat::PARQUET) {
    status = open_parquet(file_);
    if (!status.ok()) {
      if (status.IsOutOfMemory()) {
        return Status(StatusCode::RESOURCE_EXHAUSTED, status.ToString());
      }
      return Status::InternalError("Failed to open parquet file: ", file_path_,
                                   " : ", status.ToString());
    }
  } else {
    return Status::InternalError("Unsupported format, file: ", file_path_);
  }
  return Status::OK();
}
arrow::Status BufferPoolForwardStore::open_parquet(
    const std::shared_ptr<arrow::io::RandomAccessFile> &file) {
  metadata_id_ = ParquetBufferID(file_path_, 0, 0);
  auto memory_pool = detail::GetParquetMemoryPool();
  ARROW_ASSIGN_OR_RAISE(
      auto parquet_file_reader,
      detail::OpenBudgetedParquetReader(
          file, parquet::ReaderProperties(memory_pool.get())));
  ARROW_RETURN_NOT_OK(parquet::arrow::FileReader::Make(
      memory_pool.get(), std::move(parquet_file_reader), &parquet_reader_));

  auto parquet_metadata = parquet_reader_->parquet_reader()->metadata();
  metadata_id_.metadata = parquet_metadata;
  num_rows_ = parquet_metadata->num_rows();
  num_row_groups_ = parquet_metadata->num_row_groups();

  // Initialize row group offsets and row counts
  int64_t offset = 0;
  for (int64_t rg = 0; rg < num_row_groups_; ++rg) {
    auto row_group_metadata = parquet_metadata->RowGroup(rg);
    int64_t num_rows_in_group = row_group_metadata->num_rows();
    row_group_row_nums_.push_back(num_rows_in_group);
    row_group_offsets_.push_back(offset);
    offset += num_rows_in_group;
  }

  ARROW_RETURN_NOT_OK(parquet_reader_->GetSchema(&physic_schema_));

  LOG_INFO("Opened Parquet with %lld rows, %d cols, %d row groups",
           static_cast<long long>(num_rows_), physic_schema_->num_fields(),
           parquet_metadata->num_row_groups());

  return arrow::Status::OK();
}


bool BufferPoolForwardStore::validate(
    const std::vector<std::string> &columns) const {
  if (columns.empty()) {
    LOG_ERROR("Empty columns");
    return false;
  }
  // TODO : for persist segment, after add new column, this check is not
  // correct.
  for (auto &column : columns) {
    if (column == LOCAL_ROW_ID) {
      continue;
    }
    if (physic_schema_->GetFieldIndex(column) == -1) {
      LOG_ERROR("Validate failed. unknown column: %s", column.c_str());
      return false;
    }
  }
  return true;
}

int BufferPoolForwardStore::find_row_group_for_row(int64_t row) {
  auto it = std::upper_bound(row_group_offsets_.begin(),
                             row_group_offsets_.end(), row);
  if (it == row_group_offsets_.begin()) {
    return 0;
  }
  return static_cast<int>(std::distance(row_group_offsets_.begin(), it) - 1);
}

int64_t BufferPoolForwardStore::get_row_group_offset(int rg_id) {
  if (rg_id < 0 || rg_id >= static_cast<int>(row_group_offsets_.size())) {
    LOG_ERROR("Invalid row group id: %d, max: %zu", rg_id,
              row_group_offsets_.size());
    return -1;
  }
  return row_group_offsets_[rg_id];
}


TablePtr BufferPoolForwardStore::fetch(const std::vector<std::string> &columns,
                                       const std::vector<int> &indices) {
  return fetch(columns, indices, detail::GetParquetMemoryPool().get());
}

TablePtr BufferPoolForwardStore::fetch(const std::vector<std::string> &columns,
                                       const std::vector<int> &indices,
                                       arrow::MemoryPool *pool) {
  auto result = fetch_checked(columns, indices, pool);
  if (!result.ok()) {
    LOG_ERROR("Forward fetch failed: %s", result.status().ToString().c_str());
    return nullptr;
  }
  return std::move(result).ValueOrDie();
}

arrow::Result<TablePtr> BufferPoolForwardStore::fetch_checked(
    const std::vector<std::string> &columns, const std::vector<int> &indices,
    arrow::MemoryPool *pool) try {
  if (!pool) pool = detail::GetParquetMemoryPool().get();
  if (!validate(columns)) {
    return arrow::Status::Invalid("Invalid forward fetch columns");
  }

  if (indices.empty()) {
    arrow::ArrayVector empty_arrays;
    auto fields = SelectFields(physic_schema_, columns);
    for (const auto &field : fields) {
      ARROW_ASSIGN_OR_RAISE(auto array,
                            arrow::MakeEmptyArray(field->type(), pool));
      empty_arrays.push_back(std::move(array));
    }
    return arrow::Table::Make(std::make_shared<arrow::Schema>(fields),
                              empty_arrays, 0);
  }

  bool need_local_doc_id = false;
  std::vector<int> col_indices;
  std::vector<int> data_column_positions;
  std::vector<std::shared_ptr<arrow::Field>> all_fields(columns.size());

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == LOCAL_ROW_ID) {
      need_local_doc_id = true;
      all_fields[i] = arrow::field(LOCAL_ROW_ID, arrow::uint64());
    } else {
      int idx = physic_schema_->GetFieldIndex(columns[i]);
      if (idx == -1) {
        LOG_ERROR("Unknown column: %s", columns[i].c_str());
        return arrow::Status::Invalid("Unknown forward column: ", columns[i]);
      }
      col_indices.push_back(idx);
      data_column_positions.push_back(static_cast<int>(i));
      all_fields[i] = physic_schema_->GetFieldByName(columns[i]);
    }
  }

  std::unordered_map<int, std::vector<std::pair<int, uint64_t>>> rg_to_local;
  std::vector<std::pair<int, int64_t>>
      local_doc_id_pairs;  // (output_row, global_row)

  int output_row = 0;
  for (int global_row : indices) {
    if (global_row < 0 || global_row >= num_rows_) {
      LOG_ERROR("Invalid row index: %d, max: %lld", global_row,
                static_cast<long long>(num_rows_));
      return arrow::Status::IndexError("Invalid forward row: ", global_row);
    }
    int rg_id = find_row_group_for_row(global_row);
    int64_t offset = get_row_group_offset(rg_id);
    if (offset == -1) {
      LOG_ERROR("Failed to get row group offset for row: %d", global_row);
      return arrow::Status::Invalid("Missing parquet row group for row: ",
                                    global_row);
    }
    uint64_t local_in_rg = global_row - offset;
    rg_to_local[rg_id].emplace_back(output_row, local_in_rg);

    if (need_local_doc_id) {
      local_doc_id_pairs.emplace_back(output_row,
                                      static_cast<int64_t>(global_row));
    }
    ++output_row;
  }

  // Copy selected rows directly into budgeted builders. Keeping GetScalar
  // results either copies strings outside the pool or pins whole list columns.
  std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
  for (int column : col_indices) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    ARROW_RETURN_NOT_OK(arrow::MakeBuilder(
        pool, physic_schema_->field(column)->type(), &builder));
    ARROW_RETURN_NOT_OK(builder->Reserve(indices.size()));
    builders.push_back(std::move(builder));
  }
  auto order_result =
      arrow::AllocateBuffer(indices.size() * sizeof(int64_t), pool);
  ARROW_RETURN_NOT_OK(order_result.status());
  std::shared_ptr<arrow::Buffer> order_buffer =
      std::move(order_result).ValueOrDie();
  auto *order = reinterpret_cast<int64_t *>(order_buffer->mutable_data());
  bool needs_reordering = false;

  for (const auto &[rg_id, pairs] : rg_to_local) {
    for (size_t i = 0; i < col_indices.size(); ++i) {
      int col_idx = col_indices[i];
      auto buffer_id =
          ParquetBufferID(file_path_, col_idx, rg_id, metadata_id_);
      ARROW_ASSIGN_OR_RAISE(
          auto buffer_handle,
          ParquetBufferPool::get_instance().acquire_buffer_checked(buffer_id));
      std::shared_ptr<arrow::ChunkedArray> col_chunked_array =
          buffer_handle.data();
      if (!col_chunked_array) {
        LOG_ERROR(
            "Failed to pin parquet data for file: %s, column: %d, row_group: "
            "%d",
            file_path_.c_str(), col_idx, rg_id);
        return arrow::Status::Invalid("Parquet cache returned an empty column");
      }

      if (col_chunked_array->num_chunks() == 0) {
        LOG_WARN(
            "No chunks in chunked array for file: %s, column: %d, row_group: "
            "%d",
            file_path_.c_str(), col_idx, rg_id);
        return arrow::Status::Invalid("Parquet column has no chunks");
      }

      for (const auto &[tmp_output_row, local_idx] : pairs) {
        if ((size_t)local_idx >= (size_t)col_chunked_array->length()) {
          LOG_ERROR("Local index %ld out of bounds for array length %zu",
                    static_cast<long>(local_idx),
                    (size_t)col_chunked_array->length());
          return arrow::Status::IndexError(
              "Parquet row outside decoded column: ", local_idx);
        }
        if (i == 0) {
          order[tmp_output_row] = builders[i]->length();
          needs_reordering |= order[tmp_output_row] != tmp_output_row;
        }
        ARROW_RETURN_NOT_OK(detail::AppendFetchRow(
            builders[i].get(), *col_chunked_array, local_idx));
      }
    }
  }

  std::vector<std::shared_ptr<arrow::Array>> result_arrays(columns.size());
  for (size_t i = 0; i < builders.size(); ++i) {
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builders[i]->Finish(&arr));
    const int position = data_column_positions[i];
    if (!needs_reordering) {
      result_arrays[position] = std::move(arr);
      continue;
    }
    ARROW_RETURN_NOT_OK(builders[i]->Reserve(indices.size()));
    for (size_t row = 0; row < indices.size(); ++row) {
      ARROW_RETURN_NOT_OK(
          builders[i]->AppendArraySlice(*arr->data(), order[row], 1));
    }
    ARROW_RETURN_NOT_OK(builders[i]->Finish(&result_arrays[position]));
  }

  if (need_local_doc_id) {
    std::sort(local_doc_id_pairs.begin(), local_doc_id_pairs.end());
    std::vector<uint64_t> values;
    values.reserve(local_doc_id_pairs.size());
    for (const auto &p : local_doc_id_pairs) {
      values.push_back(p.second);
    }

    // Create UInt64Array
    auto buffer_result =
        arrow::AllocateBuffer(values.size() * sizeof(uint64_t), pool);
    ARROW_RETURN_NOT_OK(buffer_result.status());
    auto buffer = std::move(buffer_result.ValueOrDie());
    std::memcpy(buffer->mutable_data(), values.data(),
                values.size() * sizeof(uint64_t));

    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    buffers.push_back(nullptr);  // no null bitmap
    buffers.push_back(std::shared_ptr<arrow::Buffer>(buffer.release()));

    auto data = arrow::ArrayData::Make(arrow::uint64(),
                                       static_cast<uint64_t>(values.size()),
                                       std::move(buffers), /*null_count=*/0);

    for (size_t i = 0; i < columns.size(); ++i) {
      if (columns[i] == LOCAL_ROW_ID) {
        result_arrays[i] = std::make_shared<arrow::UInt64Array>(data);
      }
    }
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> result_columns;
  result_columns.reserve(result_arrays.size());
  for (auto &arr : result_arrays) {
    result_columns.emplace_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  auto out_schema = std::make_shared<arrow::Schema>(all_fields);
  return arrow::Table::Make(out_schema, result_columns,
                            static_cast<int64_t>(indices.size()));
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("Forward fetch allocation failed");
}

ExecBatchPtr BufferPoolForwardStore::fetch(
    const std::vector<std::string> &columns, int index) {
  if (!validate(columns) || index < 0 || index >= num_rows_) {
    return nullptr;
  }

  std::vector<int> col_indices;
  for (const auto &col : columns) {
    int idx = physic_schema_->GetFieldIndex(col);
    if (idx == -1) {
      LOG_ERROR("Unknown column: %s", col.c_str());
      return nullptr;
    }
    col_indices.push_back(idx);
  }

  int rg_id = find_row_group_for_row(index);
  int64_t offset = get_row_group_offset(rg_id);

  std::vector<arrow::Datum> scalars;
  for (size_t i = 0; i < col_indices.size(); ++i) {
    int col_idx = col_indices[i];
    auto buffer_id = ParquetBufferID(file_path_, col_idx, rg_id, metadata_id_);
    auto buffer_handle =
        ParquetBufferPool::get_instance().acquire_buffer(buffer_id);
    std::shared_ptr<arrow::ChunkedArray> col_chunked_array =
        buffer_handle.data();

    if (!col_chunked_array) {
      LOG_ERROR(
          "Failed to pin parquet data for file: %s, column: %d, row_group: "
          "%d",
          file_path_.c_str(), col_idx, rg_id);
      return nullptr;
    }

    // Resolve the row within its chunk instead of copying the entire decoded
    // row-group column for every scalar fetch.
    const int64_t local_index = index - offset;
    auto scalar_result = col_chunked_array->GetScalar(local_index);
    if (!scalar_result.ok()) {
      LOG_ERROR("Failed to get scalar for row %zu status: %s",
                static_cast<size_t>(local_index),
                scalar_result.status().ToString().c_str());
      return nullptr;
    }

    scalars.emplace_back(std::move(scalar_result.ValueOrDie()));
  }

  return std::make_shared<arrow::compute::ExecBatch>(std::move(scalars), 1);
}

RecordBatchReaderPtr BufferPoolForwardStore::scan(
    const std::vector<std::string> &columns) {
  if (!validate(columns)) {
    return nullptr;
  }

  // Create a new parquet reader for scanning
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  auto memory_pool = detail::GetParquetMemoryPool();
  // A cache miss may fall back to streaming. Do not prefetch an entire column
  // chunk before returning its first batch.
  parquet::ReaderProperties read_properties(memory_pool.get());
  read_properties.enable_buffered_stream();
  read_properties.set_buffer_size(64 * 1024);
  auto parquet_file_reader = parquet::ParquetFileReader::Open(
      file_, read_properties, parquet_reader_->parquet_reader()->metadata());
  parquet::ArrowReaderProperties arrow_properties;
  arrow_properties.set_pre_buffer(false);
  arrow_properties.set_batch_size(8192);
  auto status = parquet::arrow::FileReader::Make(
      memory_pool.get(), std::move(parquet_file_reader), arrow_properties,
      &parquet_reader);
  if (!status.ok()) {
    LOG_ERROR("Failed to create parquet reader: %s", status.message().c_str());
    return nullptr;
  }

  // Segment initialization copies docids into its own paged map. Avoid caching
  // another full column; the streaming decoder still reserves its workspace.
  const bool with_cache =
      columns.size() != 1 || columns.front() != GLOBAL_DOC_ID;
  return std::make_shared<ParquetRecordBatchReader>(
      parquet_reader, columns, physic_schema_, file_path_, with_cache,
      /*stream_uncached=*/true, std::move(memory_pool), metadata_id_);
}

}  // namespace zvec

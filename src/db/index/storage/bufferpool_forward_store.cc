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
#include <limits>
#include <type_traits>
#include <arrow/acero/exec_plan.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/doc.h>
#include "db/index/storage/store_helper.h"
#include "lazy_record_batch_reader.h"
#include "parquet_buffer_pool.h"
#include "selected_rows_plan.h"


namespace zvec {

namespace {

template <typename ArrayType, typename ScalarType>
std::shared_ptr<arrow::Scalar> ReadOwnedScalar(const arrow::Array &array,
                                               int64_t row) {
  const auto &values = static_cast<const ArrayType &>(array);
  if (values.IsNull(row)) return arrow::MakeNullScalar(values.type());
  if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                std::is_same_v<ArrayType, arrow::BinaryArray>) {
    // Own the selected bytes, so returning a scalar cannot keep an entire
    // cached column alive after the buffer handle is released.
    return std::make_shared<ScalarType>(std::string(values.GetView(row)));
  } else {
    return std::make_shared<ScalarType>(values.Value(row));
  }
}

std::shared_ptr<arrow::Scalar> ReadOwnedScalar(const arrow::Array &array,
                                               int64_t row) {
  switch (array.type_id()) {
    case arrow::Type::BOOL:
      return ReadOwnedScalar<arrow::BooleanArray, arrow::BooleanScalar>(array,
                                                                        row);
    case arrow::Type::INT32:
      return ReadOwnedScalar<arrow::Int32Array, arrow::Int32Scalar>(array, row);
    case arrow::Type::INT64:
      return ReadOwnedScalar<arrow::Int64Array, arrow::Int64Scalar>(array, row);
    case arrow::Type::UINT32:
      return ReadOwnedScalar<arrow::UInt32Array, arrow::UInt32Scalar>(array,
                                                                      row);
    case arrow::Type::UINT64:
      return ReadOwnedScalar<arrow::UInt64Array, arrow::UInt64Scalar>(array,
                                                                      row);
    case arrow::Type::FLOAT:
      return ReadOwnedScalar<arrow::FloatArray, arrow::FloatScalar>(array, row);
    case arrow::Type::DOUBLE:
      return ReadOwnedScalar<arrow::DoubleArray, arrow::DoubleScalar>(array,
                                                                      row);
    case arrow::Type::STRING:
      return ReadOwnedScalar<arrow::StringArray, arrow::StringScalar>(array,
                                                                      row);
    case arrow::Type::BINARY:
      return ReadOwnedScalar<arrow::BinaryArray, arrow::BinaryScalar>(array,
                                                                      row);
    default:
      return nullptr;
  }
}

using SelectedRows = std::vector<std::pair<int, uint64_t>>;

template <typename ArrayType>
void ReadDocField(const arrow::Array &array, int64_t row,
                  const std::string &name, Doc &doc) {
  const auto &values = static_cast<const ArrayType &>(array);
  if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                std::is_same_v<ArrayType, arrow::BinaryArray>) {
    doc.set(name, std::string(values.GetView(row)));
  } else {
    doc.set(name, values.Value(row));
  }
}

template <typename ArrayType, typename BuilderType>
arrow::Status AppendTypedRows(const arrow::ChunkedArray &column,
                              const detail::SelectedRowsPlan &plan,
                              arrow::ArrayBuilder *builder) {
  auto &typed_builder = static_cast<BuilderType &>(*builder);
  const bool no_null = column.null_count() == 0;
  return plan.Visit<ArrayType>(column, [&](const ArrayType &array, int64_t row,
                                          int) {
    if (!no_null && array.IsNull(row)) return typed_builder.AppendNull();
    if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                  std::is_same_v<ArrayType, arrow::BinaryArray>) {
      return typed_builder.Append(array.GetView(row));
    } else {
      return typed_builder.Append(array.Value(row));
    }
  });
}

template <typename ArrayType, typename BuilderType>
arrow::Status ReserveSelectedBytes(const arrow::ChunkedArray &column,
                                   const detail::SelectedRowsPlan &plan,
                                   arrow::ArrayBuilder *builder) {
  int64_t bytes = 0;
  const bool no_null = column.null_count() == 0;
  ARROW_RETURN_NOT_OK(plan.Visit<ArrayType>(
      column, [&](const ArrayType &array, int64_t row, int) {
        if (no_null || !array.IsNull(row)) {
          const int64_t length = array.value_length(row);
          if (length > std::numeric_limits<int64_t>::max() - bytes) {
            return arrow::Status::CapacityError("Selected strings are too large");
          }
          bytes += length;
        }
        return arrow::Status::OK();
      }));
  return static_cast<BuilderType &>(*builder).ReserveData(bytes);
}

arrow::Status AppendSelectedRows(const arrow::ChunkedArray &column,
                                 detail::SelectedRowsPlan &plan,
                                 arrow::ArrayBuilder *builder) {
  ARROW_RETURN_NOT_OK(plan.Prepare(column));
  // Reserve selected bytes while the source column is pinned.
  // Reserve(row_count) alone only reserves the string offsets and null bitmap.
  if (column.type()->id() == arrow::Type::STRING) {
    ARROW_RETURN_NOT_OK(
        (ReserveSelectedBytes<arrow::StringArray, arrow::StringBuilder>(
            column, plan, builder)));
  } else if (column.type()->id() == arrow::Type::BINARY) {
    ARROW_RETURN_NOT_OK(
        (ReserveSelectedBytes<arrow::BinaryArray, arrow::BinaryBuilder>(
            column, plan, builder)));
  }
  switch (column.type()->id()) {
    case arrow::Type::BOOL:
      return AppendTypedRows<arrow::BooleanArray, arrow::BooleanBuilder>(
          column, plan, builder);
    case arrow::Type::INT32:
      return AppendTypedRows<arrow::Int32Array, arrow::Int32Builder>(
          column, plan, builder);
    case arrow::Type::INT64:
      return AppendTypedRows<arrow::Int64Array, arrow::Int64Builder>(
          column, plan, builder);
    case arrow::Type::UINT32:
      return AppendTypedRows<arrow::UInt32Array, arrow::UInt32Builder>(
          column, plan, builder);
    case arrow::Type::UINT64:
      return AppendTypedRows<arrow::UInt64Array, arrow::UInt64Builder>(
          column, plan, builder);
    case arrow::Type::FLOAT:
      return AppendTypedRows<arrow::FloatArray, arrow::FloatBuilder>(
          column, plan, builder);
    case arrow::Type::DOUBLE:
      return AppendTypedRows<arrow::DoubleArray, arrow::DoubleBuilder>(
          column, plan, builder);
    case arrow::Type::STRING:
      return AppendTypedRows<arrow::StringArray, arrow::StringBuilder>(
          column, plan, builder);
    case arrow::Type::BINARY:
      return AppendTypedRows<arrow::BinaryArray, arrow::BinaryBuilder>(
          column, plan, builder);
    default:
      break;
  }
  // Nested and other uncommon types keep Arrow's general append semantics.
  std::vector<arrow::ArraySpan> spans;
  spans.reserve(column.num_chunks());
  for (const auto &chunk : column.chunks()) spans.emplace_back(*chunk->data());
  return plan.Visit<arrow::Array>(
      column, [&](const arrow::Array &, int64_t row, int chunk) {
        return builder->AppendArraySlice(spans[chunk], row, 1);
      });
}

}  // namespace

BufferPoolForwardStore::BufferPoolForwardStore(const std::string &uri)
    : file_path_(uri) {}

Status BufferPoolForwardStore::open() {
  std::string uri = file_path_;
  auto status = CreateRandomAccessFileByUri(uri, &file_, &file_path_);
  if (!status.ok()) {
    return Status::InternalError("Failed to create random access uri: ", uri,
                                 " : ", status.ToString());
  }
  auto format = InferFileFormat(file_path_);
  if (format == FileFormat::PARQUET) {
    auto context = ParquetFileContext::Make(file_path_, file_);
    if (!context.ok()) {
      return Status::InternalError("Failed to bind parquet file: ", file_path_,
                                   " : ", context.status().ToString());
    }
    file_context_ = std::move(context).ValueOrDie();
    file_ = file_context_->file;
    status = open_parquet(file_);
    if (!status.ok()) {
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
  auto parquet_file_reader = parquet::ParquetFileReader::Open(file);
  ARROW_RETURN_NOT_OK(parquet::arrow::FileReader::Make(
      arrow::default_memory_pool(), std::move(parquet_file_reader),
      &parquet_reader_));

  auto parquet_metadata = parquet_reader_->parquet_reader()->metadata();
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
  if (!validate(columns)) {
    return nullptr;
  }

  if (indices.empty()) {
    arrow::ArrayVector empty_arrays;
    auto fields = SelectFields(physic_schema_, columns);
    for (const auto &field : fields) {
      empty_arrays.push_back(arrow::MakeEmptyArray(field->type()).ValueOrDie());
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
        return nullptr;
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
      return nullptr;
    }
    int rg_id = find_row_group_for_row(global_row);
    int64_t offset = get_row_group_offset(rg_id);
    if (offset == -1) {
      LOG_ERROR("Failed to get row group offset for row: %d", global_row);
      return nullptr;
    }
    uint64_t local_in_rg = global_row - offset;
    rg_to_local[rg_id].emplace_back(output_row, local_in_rg);

    if (need_local_doc_id) {
      local_doc_id_pairs.emplace_back(output_row,
                                      static_cast<int64_t>(global_row));
    }
    ++output_row;
  }

  std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
  for (int column : col_indices) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    auto status =
        arrow::MakeBuilder(arrow::default_memory_pool(),
                           physic_schema_->field(column)->type(), &builder);
    if (status.ok()) status = builder->Reserve(indices.size());
    if (!status.ok()) {
      LOG_ERROR("Create forward builder failed: %s", status.ToString().c_str());
      return nullptr;
    }
    builders.push_back(std::move(builder));
  }
  std::vector<int64_t> order(indices.size());
  bool needs_reordering = false;

  const auto &file_identity = file_context_->buffer_id;
  for (const auto &[rg_id, pairs] : rg_to_local) {
    detail::SelectedRowsPlan row_plan(pairs);
    for (size_t i = 0; i < col_indices.size(); ++i) {
      int col_idx = col_indices[i];
      auto buffer_id = file_identity;
      buffer_id.column = col_idx;
      buffer_id.row_group = rg_id;
      auto buffer_handle = ParquetBufferPool::get_instance().acquire_buffer(
          std::move(buffer_id));
      std::shared_ptr<arrow::ChunkedArray> col_chunked_array =
          buffer_handle.data();
      if (!col_chunked_array) {
        LOG_ERROR(
            "Failed to pin parquet data for file: %s, column: %d, row_group: "
            "%d",
            file_path_.c_str(), col_idx, rg_id);
        return nullptr;
      }

      if (col_chunked_array->num_chunks() == 0) {
        LOG_WARN(
            "No chunks in chunked array for file: %s, column: %d, row_group: "
            "%d",
            file_path_.c_str(), col_idx, rg_id);
        continue;
      }

      if (i == 0) {
        int64_t appended_row = builders[i]->length();
        for (const auto &row : pairs) {
          order[row.first] = appended_row++;
          needs_reordering |= order[row.first] != row.first;
        }
      }
      auto status =
          AppendSelectedRows(*col_chunked_array, row_plan, builders[i].get());
      if (!status.ok()) {
        LOG_ERROR("Append forward row failed: %s", status.ToString().c_str());
        return nullptr;
      }
    }
  }

  SelectedRows reordered_rows;
  if (needs_reordering) {
    reordered_rows.reserve(order.size());
    for (size_t row = 0; row < order.size(); ++row) {
      reordered_rows.emplace_back(static_cast<int>(row), order[row]);
    }
  }
  std::vector<std::shared_ptr<arrow::Array>> result_arrays(columns.size());
  detail::SelectedRowsPlan reorder_plan(reordered_rows);
  for (size_t i = 0; i < builders.size(); ++i) {
    std::shared_ptr<arrow::Array> arr;
    auto status = builders[i]->Finish(&arr);
    if (status.ok() && needs_reordering) {
      status = builders[i]->Reserve(indices.size());
      if (status.ok()) {
        const arrow::ChunkedArray gathered(arr);
        status =
            AppendSelectedRows(gathered, reorder_plan, builders[i].get());
      }
      if (status.ok()) status = builders[i]->Finish(&arr);
    }
    if (!status.ok()) {
      LOG_ERROR("Build forward array failed: %s", status.ToString().c_str());
      return nullptr;
    }
    result_arrays[data_column_positions[i]] = std::move(arr);
  }

  if (need_local_doc_id) {
    std::sort(local_doc_id_pairs.begin(), local_doc_id_pairs.end());
    std::vector<uint64_t> values;
    values.reserve(local_doc_id_pairs.size());
    for (const auto &p : local_doc_id_pairs) {
      values.push_back(p.second);
    }

    // Create UInt64Array
    auto buffer_result = arrow::AllocateBuffer(values.size() * sizeof(uint64_t),
                                               arrow::default_memory_pool());
    if (!buffer_result.ok()) return nullptr;
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
}

ExecBatchPtr BufferPoolForwardStore::fetch(
    const std::vector<std::string> &columns, int index) {
  if (columns.empty()) {
    LOG_ERROR("Empty columns");
    return nullptr;
  }
  if (index < 0 || index >= num_rows_) {
    return nullptr;
  }

  // Resolving each physical column also validates the projection.
  std::vector<int> col_indices;
  col_indices.reserve(columns.size());
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

  const auto &file_identity = file_context_->buffer_id;
  std::vector<arrow::Datum> scalars;
  scalars.reserve(col_indices.size());
  for (size_t i = 0; i < col_indices.size(); ++i) {
    int col_idx = col_indices[i];
    auto buffer_id = file_identity;
    buffer_id.column = col_idx;
    buffer_id.row_group = rg_id;
    auto buffer_handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(buffer_id));
    std::shared_ptr<arrow::ChunkedArray> col_chunked_array =
        buffer_handle.data();

    if (!col_chunked_array) {
      LOG_ERROR(
          "Failed to pin parquet data for file: %s, column: %d, row_group: "
          "%d",
          file_path_.c_str(), col_idx, rg_id);
      return nullptr;
    }

    if (col_chunked_array->num_chunks() == 0) {
      LOG_WARN(
          "No chunks in chunked array for file: %s, column: %d, row_group: "
          "%d",
          file_path_.c_str(), col_idx, rg_id);
      continue;
    }
    int64_t row_in_chunk = index - offset;
    std::shared_ptr<arrow::Scalar> owned_scalar;
    for (const auto &chunk : col_chunked_array->chunks()) {
      if (row_in_chunk < chunk->length()) {
        owned_scalar = ReadOwnedScalar(*chunk, row_in_chunk);
        break;
      }
      row_in_chunk -= chunk->length();
    }
    if (owned_scalar) {
      scalars.emplace_back(std::move(owned_scalar));
      continue;
    }

    auto row = col_chunked_array->Slice(index - offset, 1);
    auto concat_result =
        arrow::Concatenate(row->chunks(), arrow::default_memory_pool());
    if (!concat_result.ok()) {
      LOG_ERROR("Concatenate failed for file: %s, column: %d, row_group: %d",
                file_path_.c_str(), col_idx, rg_id);
      return nullptr;
    }
    auto concat = concat_result.ValueOrDie();
    auto scalar_result = concat->GetScalar(0);
    if (!scalar_result.ok()) {
      LOG_ERROR("Failed to get scalar for row %zu status: %s", (size_t)offset,
                scalar_result.status().ToString().c_str());
    }

    scalars.emplace_back(std::move(scalar_result.ValueOrDie()));
  }

  return std::make_shared<arrow::compute::ExecBatch>(std::move(scalars), 1);
}

std::shared_ptr<Doc> BufferPoolForwardStore::fetch_doc(
    const std::vector<std::string> &columns,
    const std::shared_ptr<arrow::Schema> &schema, int index) {
  if (index < 0 || index >= num_rows_ || columns.size() < 2 || !schema ||
      schema->num_fields() != static_cast<int>(columns.size()) ||
      columns[0] != GLOBAL_DOC_ID || columns[1] != USER_ID) {
    return nullptr;
  }
  std::vector<int> selected;
  selected.reserve(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    const int column = physic_schema_->GetFieldIndex(columns[i]);
    if (column < 0 || schema->field(i)->name() != columns[i] ||
        !schema->field(i)->type()->Equals(
            physic_schema_->field(column)->type())) {
      return nullptr;
    }
    const auto type = schema->field(i)->type()->id();
    if ((i == 0 && type != arrow::Type::UINT64) ||
        (i == 1 && type != arrow::Type::STRING)) {
      return nullptr;
    }
    switch (type) {
      case arrow::Type::BOOL:
      case arrow::Type::INT32:
      case arrow::Type::INT64:
      case arrow::Type::UINT32:
      case arrow::Type::UINT64:
      case arrow::Type::FLOAT:
      case arrow::Type::DOUBLE:
      case arrow::Type::STRING:
      case arrow::Type::BINARY:
        break;
      default:
        return nullptr;
    }
    selected.push_back(column);
  }
  const int group = find_row_group_for_row(index);
  const int64_t local_row = index - get_row_group_offset(group);
  auto doc = std::make_shared<Doc>();
  for (size_t i = 0; i < selected.size(); ++i) {
    auto key = file_context_->buffer_id;
    key.column = selected[i];
    key.row_group = group;
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    auto column = handle.data();
    if (!column) return nullptr;
    int64_t row = local_row;
    const arrow::Array *array = nullptr;
    for (const auto &chunk : column->chunks()) {
      if (row < chunk->length()) {
        array = chunk.get();
        break;
      }
      row -= chunk->length();
    }
    if (!array) return nullptr;
    if (array->IsNull(row)) {
      if (i < 2) return nullptr;
      continue;
    }
    if (i == 0) {
      doc->set_doc_id(
          static_cast<const arrow::UInt64Array *>(array)->Value(row));
    } else if (i == 1) {
      doc->set_pk(std::string(
          static_cast<const arrow::StringArray *>(array)->GetView(row)));
    } else {
      // Copies become owned Doc values before this column's pin is released.
      switch (array->type_id()) {
        case arrow::Type::BOOL:
          ReadDocField<arrow::BooleanArray>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::INT32:
          ReadDocField<arrow::Int32Array>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::INT64:
          ReadDocField<arrow::Int64Array>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::UINT32:
          ReadDocField<arrow::UInt32Array>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::UINT64:
          ReadDocField<arrow::UInt64Array>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::FLOAT:
          ReadDocField<arrow::FloatArray>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::DOUBLE:
          ReadDocField<arrow::DoubleArray>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::STRING:
          ReadDocField<arrow::StringArray>(*array, row, columns[i], *doc);
          break;
        case arrow::Type::BINARY:
          ReadDocField<arrow::BinaryArray>(*array, row, columns[i], *doc);
          break;
        default:
          return nullptr;
      }
    }
  }
  return doc;
}

RecordBatchReaderPtr BufferPoolForwardStore::scan(
    const std::vector<std::string> &columns) {
  if (!validate(columns)) {
    return nullptr;
  }

  // Create a new parquet reader for scanning
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  auto parquet_file_reader = parquet::ParquetFileReader::Open(file_);
  auto status = parquet::arrow::FileReader::Make(arrow::default_memory_pool(),
                                                 std::move(parquet_file_reader),
                                                 &parquet_reader);
  if (!status.ok()) {
    LOG_ERROR("Failed to create parquet reader: %s", status.message().c_str());
    return nullptr;
  }

  return std::make_shared<ParquetRecordBatchReader>(
      parquet_reader, columns, physic_schema_, file_path_, true, file_context_);
}

}  // namespace zvec

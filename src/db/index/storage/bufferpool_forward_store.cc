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
#include <cstring>
#include <limits>
#include <map>
#include <type_traits>
#include <arrow/acero/exec_plan.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/doc.h>
#include "db/common/query_result_memory.h"
#include "db/index/storage/store_helper.h"
#include "fetch_memory.h"
#include "lazy_record_batch_reader.h"
#include "parquet_buffer_pool.h"
#include "parquet_memory_pool.h"
#include "parquet_metadata_budget.h"
#include "selected_rows_plan.h"


namespace zvec {

ParquetRecordBatchReader::ParquetRecordBatchReader(
    std::unique_ptr<parquet::arrow::FileReader> &reader,
    const std::vector<std::string> &columns,
    std::shared_ptr<arrow::Schema> schema, const std::string &file_path,
    bool with_cache, bool stream_uncached,
    std::shared_ptr<arrow::MemoryPool> memory_pool, ParquetBufferID metadata_id)
    : memory_pool_(std::move(memory_pool)),
      reader_(std::move(reader)),
      schema_(std::move(schema)),
      columns_(columns),
      file_path_(file_path),
      with_cache_(with_cache),
      stream_uncached_(stream_uncached) {
  project_columns();

  auto parquet_metadata = reader_->parquet_reader()->metadata();
  metadata_id_ = metadata_id.filename().empty()
                     ? ParquetBufferID(file_path_, 0, 0)
                     : std::move(metadata_id);
  metadata_id_.metadata = parquet_metadata;
  total_rows_ = parquet_metadata->num_rows();
  num_row_groups_ = parquet_metadata->num_row_groups();
  int64_t offset = 0;
  for (int64_t rg = 0; rg < num_row_groups_; ++rg) {
    auto row_group_metadata = parquet_metadata->RowGroup(rg);
    int64_t num_rows_in_group = row_group_metadata->num_rows();
    row_group_row_nums_.push_back(num_rows_in_group);
    row_group_offsets_.push_back(offset);
    offset += num_rows_in_group;
  }
}

ParquetRecordBatchReader::ParquetRecordBatchReader(
    const std::vector<std::string> &columns,
    std::shared_ptr<arrow::Schema> schema, const std::string &file_path,
    std::shared_ptr<arrow::MemoryPool> memory_pool,
    std::shared_ptr<arrow::io::RandomAccessFile> input,
    std::shared_ptr<parquet::FileMetaData> metadata,
    ParquetBufferID metadata_id, const std::vector<int64_t> &row_group_rows)
    : memory_pool_(std::move(memory_pool)),
      input_(std::move(input)),
      metadata_(std::move(metadata)),
      schema_(std::move(schema)),
      columns_(columns),
      file_path_(file_path),
      metadata_id_(std::move(metadata_id)),
      num_row_groups_(row_group_rows.size()),
      row_group_row_nums_(row_group_rows),
      with_cache_(true),
      stream_uncached_(true) {
  project_columns();
}

void ParquetRecordBatchReader::project_columns() {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto &col : columns_) {
    int index = schema_->GetFieldIndex(col);
    if (index != -1) {
      fields.push_back(schema_->field(index));
      col_indices_.push_back(index);
    }
  }
  projected_schema_ = arrow::schema(fields);
}

arrow::Status ParquetRecordBatchReader::ensure_reader() {
  if (reader_) return arrow::Status::OK();
  parquet::ReaderProperties properties(memory_pool_.get());
  properties.enable_buffered_stream();
  properties.set_buffer_size(64 * 1024);
  auto parquet_reader =
      parquet::ParquetFileReader::Open(input_, properties, metadata_);
  auto arrow_properties = detail::BudgetedParquetArrowProperties();
  arrow_properties.set_batch_size(8192);
  return parquet::arrow::FileReader::Make(memory_pool_.get(),
                                          std::move(parquet_reader),
                                          arrow_properties, &reader_);
}

ParquetRecordBatchReader::~ParquetRecordBatchReader() = default;

arrow::Status ParquetRecordBatchReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *batch) try {
  *batch = nullptr;
  while (true) {
    if (cached_batches_) {
      ARROW_RETURN_NOT_OK(cached_batches_->ReadNext(batch));
      if (*batch) {
        if ((*batch)->num_rows() != 0) return arrow::Status::OK();
        continue;
      }
      cached_batches_.reset();
    }
    if (current_row_group_ >= num_row_groups_) {
      return arrow::Status::OK();
    }

    if (!with_cache_) {
      return stream_uncached_ ? read_next_uncached(batch)
                              : read_row_group(batch);
    }

    int64_t rg_id = current_row_group_;
    int64_t num_rows_in_rg = row_group_row_nums_[rg_id];

    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns(
        col_indices_.size());
    for (size_t col_idx = 0; col_idx < col_indices_.size(); ++col_idx) {
      // A one-pass scan must not fill the point-read cache with entire cold
      // columns. Reuse resident columns without copying; on a miss, stream
      // bounded batches through the same budgeted decoder instead. Preserve
      // the legacy reader's cache behavior when no budgeted pool is supplied.
      std::shared_ptr<arrow::ChunkedArray> col_chunked_array;
      try {
        auto buffer_id = metadata_id_;
        buffer_id.column = col_indices_[col_idx];
        buffer_id.row_group = rg_id;
        auto &cache = ParquetBufferPool::get_instance();
        auto buffer_handle = memory_pool_ ? cache.try_acquire_buffer(buffer_id)
                                          : cache.acquire_buffer(buffer_id);
        col_chunked_array = buffer_handle.data();
      } catch (const std::bad_alloc &) {
        // Cache keys and escaped-view metadata also require admission.
        // A refusal follows the same bounded streaming path as a cold miss.
      }
      if (!col_chunked_array) {
        // Cache residency is optional. Drop partial columns before reading
        // the same row group in bounded batches, without publishing any rows
        // twice or passing null columns to Arrow. Stay uncached for this scan
        // to avoid repeatedly decoding columns that cannot fit in the pool.
        columns.clear();
        with_cache_ = false;
        stream_uncached_ = true;
        return read_next_uncached(batch);
      }
      columns[col_idx] = std::move(col_chunked_array);
    }

    // Slice existing chunks and retain their cache pins; concatenation made
    // a second, unbudgeted copy of every decoded row-group column.
    cached_batches_ =
        std::make_unique<arrow::TableBatchReader>(arrow::Table::Make(
            projected_schema_, std::move(columns), num_rows_in_rg));
    cached_batches_->set_chunksize(8192);
    current_row_group_++;
  }
} catch (const std::bad_alloc &) {
  *batch = nullptr;
  return arrow::Status::OutOfMemory("Parquet scan metadata budget exhausted");
}

arrow::Status ParquetRecordBatchReader::read_row_group(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader_->RowGroup(static_cast<int>(current_row_group_))
                          ->ReadTable(col_indices_, &table));
  std::vector<std::shared_ptr<arrow::Array>> chunks(col_indices_.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(chunks[i],
                          arrow::Concatenate(table->column(i)->chunks(),
                                             arrow::default_memory_pool()));
  }
  *batch = arrow::RecordBatch::Make(
      projected_schema_, row_group_row_nums_[current_row_group_], chunks);
  ++current_row_group_;
  return arrow::Status::OK();
}

arrow::Status ParquetRecordBatchReader::read_next_uncached(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  ARROW_RETURN_NOT_OK(ensure_reader());
  while (current_row_group_ < num_row_groups_) {
    if (!row_group_reader_) {
      const auto metadata = reader_->parquet_reader()->metadata();
      const auto leaves = detail::ParquetLeafColumns(*metadata, col_indices_);
      if (memory_pool_) {
        reader_->set_batch_size(detail::ParquetDecodeBatchRows(
            *metadata, static_cast<int>(current_row_group_), leaves));
      }
      ARROW_ASSIGN_OR_RAISE(
          row_group_reader_,
          reader_->GetRecordBatchReader({static_cast<int>(current_row_group_)},
                                        leaves));
    }
    std::shared_ptr<arrow::RecordBatch> next;
    ARROW_RETURN_NOT_OK(row_group_reader_->ReadNext(&next));
    if (next) {
      // Arrow deduplicates projected Parquet columns. Preserve the caller's
      // order and duplicates, matching cached reads, without copying data.
      std::vector<std::shared_ptr<arrow::Array>> columns;
      columns.reserve(projected_schema_->num_fields());
      for (const auto &field : projected_schema_->fields()) {
        auto column = next->GetColumnByName(field->name());
        if (!column) {
          return arrow::Status::Invalid("Missing scanned column: ",
                                        field->name());
        }
        columns.push_back(std::move(column));
      }
      *batch = arrow::RecordBatch::Make(projected_schema_, next->num_rows(),
                                        std::move(columns));
      return arrow::Status::OK();
    }
    row_group_reader_.reset();
    ++current_row_group_;
  }
  return arrow::Status::OK();
}

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
    auto allocated = arrow::AllocateBuffer(
        values.value_length(row), detail::GetParquetMemoryPool().get());
    if (!allocated.ok()) {
      if (allocated.status().IsOutOfMemory()) throw std::bad_alloc();
      return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = std::move(*allocated);
    const auto bytes = values.GetView(row);
    if (!bytes.empty()) {
      std::memcpy(buffer->mutable_data(), bytes.data(), bytes.size());
    }
    return std::make_shared<ScalarType>(std::move(buffer));
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

using SelectedRows = detail::SelectedRowsPlan::Rows;

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

bool SupportsDirectDocValues(arrow::Type::type type) {
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
      return true;
    default:
      return false;
  }
}

template <typename ArrayType, typename ValueType>
void ReadListDocField(const arrow::Array &child, int64_t offset, int64_t length,
                      const std::string &name, Doc &doc) {
  const auto &values = static_cast<const ArrayType &>(child);
  std::vector<ValueType> owned;
  owned.reserve(length);
  for (int64_t i = offset; i < offset + length; ++i) {
    if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                  std::is_same_v<ArrayType, arrow::BinaryArray>) {
      owned.emplace_back(values.GetView(i));
    } else {
      owned.emplace_back(values.Value(i));
    }
  }
  doc.set(name, std::move(owned));
}

bool ReadListDocField(const arrow::ListArray &array, int64_t row,
                      const std::string &name, Doc &doc) {
  const auto &child = *array.values();
  const int64_t offset = array.value_offset(row);
  const int64_t length = array.value_length(row);
  // zvec array fields do not support null elements. Keep Arrow's existing
  // scalar fallback for externally supplied lists that contain them.
  if (child.null_count() != 0) {
    for (int64_t i = offset; i < offset + length; ++i) {
      if (child.IsNull(i)) return false;
    }
  }
  // Copy directly from the selected child range while the column is pinned.
  // In particular, do not allocate a sliced Arrow array or ListScalar.
  switch (child.type_id()) {
    case arrow::Type::BOOL:
      ReadListDocField<arrow::BooleanArray, bool>(child, offset, length, name,
                                                  doc);
      break;
    case arrow::Type::INT32:
      ReadListDocField<arrow::Int32Array, int32_t>(child, offset, length, name,
                                                   doc);
      break;
    case arrow::Type::INT64:
      ReadListDocField<arrow::Int64Array, int64_t>(child, offset, length, name,
                                                   doc);
      break;
    case arrow::Type::UINT32:
      ReadListDocField<arrow::UInt32Array, uint32_t>(child, offset, length,
                                                     name, doc);
      break;
    case arrow::Type::UINT64:
      ReadListDocField<arrow::UInt64Array, uint64_t>(child, offset, length,
                                                     name, doc);
      break;
    case arrow::Type::FLOAT:
      ReadListDocField<arrow::FloatArray, float>(child, offset, length, name,
                                                 doc);
      break;
    case arrow::Type::DOUBLE:
      ReadListDocField<arrow::DoubleArray, double>(child, offset, length, name,
                                                   doc);
      break;
    case arrow::Type::STRING:
      ReadListDocField<arrow::StringArray, std::string>(child, offset, length,
                                                        name, doc);
      break;
    case arrow::Type::BINARY:
      ReadListDocField<arrow::BinaryArray, std::string>(child, offset, length,
                                                        name, doc);
      break;
    default:
      return false;
  }
  return true;
}

template <typename ArrayType, typename BuilderType>
arrow::Status AppendTypedRows(const arrow::ChunkedArray &column,
                              const detail::SelectedRowsPlan &plan,
                              arrow::ArrayBuilder *builder) {
  auto &typed_builder = static_cast<BuilderType &>(*builder);
  const bool no_null = column.null_count() == 0;
  return plan.visit<ArrayType>(
      column, [&](const ArrayType &array, int64_t row, int) {
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
  ARROW_RETURN_NOT_OK(plan.visit<ArrayType>(column, [&](const ArrayType &array,
                                                        int64_t row, int) {
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
  ARROW_RETURN_NOT_OK(plan.prepare(column));
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
  ailego::BudgetedVector<arrow::ArraySpan> spans;
  spans.reserve(column.num_chunks());
  for (const auto &chunk : column.chunks()) spans.emplace_back(*chunk->data());
  return plan.visit<arrow::Array>(
      column, [&](const arrow::Array &, int64_t row, int chunk) {
        return builder->AppendArraySlice(spans[chunk], row, 1);
      });
}

}  // namespace

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

Status BufferPoolForwardStore::open() try {
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
} catch (const std::bad_alloc &) {
  return Status(StatusCode::RESOURCE_EXHAUSTED,
                "Forward store metadata budget exhausted");
}
arrow::Status BufferPoolForwardStore::open_parquet(
    const std::shared_ptr<arrow::io::RandomAccessFile> &file) {
  metadata_id_ = ParquetBufferID(file_path_, file, 0, 0);
  auto memory_pool = detail::GetParquetMemoryPool();
  ARROW_ASSIGN_OR_RAISE(
      auto parquet_file_reader,
      detail::OpenBudgetedParquetReader(
          file, parquet::ReaderProperties(memory_pool.get())));
  ARROW_RETURN_NOT_OK(parquet::arrow::FileReader::Make(
      memory_pool.get(), std::move(parquet_file_reader),
      detail::BudgetedParquetArrowProperties(), &parquet_reader_));

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


TablePtr BufferPoolForwardStore::fetch_blocks(
    const std::vector<std::string> &columns,
    const std::vector<FetchBlock> &blocks, int row_count,
    arrow::MemoryPool *pool) try {
  if (!pool) pool = detail::GetParquetMemoryPool().get();
  if (columns.empty() || blocks.empty() || row_count <= 0) return nullptr;
  using GroupKey = std::pair<size_t, int>;
  using Entry = std::pair<const GroupKey, detail::SelectedRowsPlan::Rows>;
  std::map<GroupKey, detail::SelectedRowsPlan::Rows, std::less<GroupKey>,
           ailego::BudgetedAllocator<Entry>>
      groups;
  ailego::BudgetedVector<uint8_t> seen(row_count, 0);
  size_t selected_rows = 0;
  for (size_t block = 0; block < blocks.size(); ++block) {
    const auto &request = blocks[block];
    if (!request.store || !request.store->validate(columns)) return nullptr;
    for (const auto &[output, row] : request.rows) {
      if (output < 0 || output >= row_count || seen[output] || row < 0 ||
          row >= request.store->num_rows_)
        return nullptr;
      seen[output] = true;
      ++selected_rows;
      const int rg = request.store->find_row_group_for_row(row);
      groups[{block, rg}].emplace_back(
          output, row - request.store->row_group_offsets_[rg]);
    }
  }
  if (selected_rows != static_cast<size_t>(row_count)) return nullptr;
  detail::SelectedRowsPlan::Rows selected(row_count);
  int64_t offset = 0;
  for (const auto &[key, rows] : groups) {
    for (const auto &[output, local] : rows)
      selected[output] = {output, offset + local};
    offset += blocks[key.first].store->row_group_row_nums_[key.second];
  }
  detail::SelectedRowsPlan plan(selected);
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  for (const auto &name : columns) {
    const auto field =
        blocks.front().store->physic_schema_->GetFieldByName(name);
    if (!field) return nullptr;
    arrow::ArrayVector chunks;
    for (const auto &[key, rows] : groups) {
      auto *store = blocks[key.first].store;
      const int column = store->physic_schema_->GetFieldIndex(name);
      if (column < 0 ||
          !store->physic_schema_->field(column)->type()->Equals(field->type()))
        return nullptr;
      auto id = store->metadata_id_;
      id.column = column;
      id.row_group = key.second;
      auto handle = ParquetBufferPool::get_instance().try_acquire_buffer(id);
      auto borrowed = handle.data();
      if (!borrowed ||
          borrowed->length() != store->row_group_row_nums_[key.second])
        return nullptr;
      chunks.insert(chunks.end(), borrowed->chunks().begin(),
                    borrowed->chunks().end());
    }
    // Only resident sources are pinned while this column is copied. A cache
    // miss or allocation refusal selects the bounded per-block fallback.
    const arrow::ChunkedArray source(std::move(chunks), field->type());
    auto builder = arrow::MakeBuilder(field->type(), pool);
    if (!builder.ok() || !(*builder)->Reserve(row_count).ok() ||
        !AppendSelectedRows(source, plan, builder->get()).ok())
      return nullptr;
    auto result = (*builder)->Finish();
    if (!result.ok()) return nullptr;
    arrays.push_back(std::move(*result));
    fields.push_back(field);
  }
  return arrow::Table::Make(arrow::schema(std::move(fields)), std::move(arrays),
                            row_count);
} catch (const std::bad_alloc &) {
  return nullptr;
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

  using RowGroupRows = ailego::BudgetedVector<std::pair<size_t, uint64_t>>;
  using RowGroupEntry = std::pair<const int, RowGroupRows>;
  // Preserve increasing query row order and avoid an extra gather copy.
  std::map<int, RowGroupRows, std::less<int>,
           ailego::BudgetedAllocator<RowGroupEntry>>
      rg_to_local;
  size_t output_row = 0;
  for (int global_row : indices) {
    if (global_row < 0 || global_row >= num_rows_) {
      LOG_ERROR("Invalid row index: %d, max: %lld", global_row,
                static_cast<long long>(num_rows_));
      return arrow::Status::IndexError("Invalid forward row: ", global_row);
    }
    if (col_indices.empty()) continue;
    int rg_id = find_row_group_for_row(global_row);
    int64_t offset = get_row_group_offset(rg_id);
    if (offset == -1) {
      LOG_ERROR("Failed to get row group offset for row: %d", global_row);
      return arrow::Status::Invalid("Missing parquet row group for row: ",
                                    global_row);
    }
    uint64_t local_in_rg = global_row - offset;
    rg_to_local[rg_id].emplace_back(output_row++, local_in_rg);
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
  std::unique_ptr<arrow::Buffer> order_buffer;
  int64_t *order = nullptr;
  if (!col_indices.empty()) {
    ARROW_ASSIGN_OR_RAISE(
        order_buffer,
        arrow::AllocateBuffer(indices.size() * sizeof(int64_t), pool));
    order = reinterpret_cast<int64_t *>(order_buffer->mutable_data());
  }
  bool needs_reordering = false;

  for (const auto &[rg_id, pairs] : rg_to_local) {
    detail::SelectedRowsPlan row_plan(pairs);
    for (size_t i = 0; i < col_indices.size(); ++i) {
      int col_idx = col_indices[i];
      auto buffer_id = metadata_id_;
      buffer_id.column = col_idx;
      buffer_id.row_group = rg_id;
      ARROW_ASSIGN_OR_RAISE(
          auto buffer_handle,
          ParquetBufferPool::get_instance().acquire_buffer_checked(buffer_id));
      const auto *col_chunked_array = buffer_handle.borrowed_data();
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

      if (i == 0) {
        int64_t appended = builders[i]->length();
        for (const auto &row : pairs) {
          order[row.first] = appended++;
          needs_reordering |=
              order[row.first] != static_cast<int64_t>(row.first);
        }
      }
      ARROW_RETURN_NOT_OK(
          AppendSelectedRows(*col_chunked_array, row_plan, builders[i].get()));
    }
  }

  SelectedRows reordered;
  if (needs_reordering) {
    reordered.reserve(indices.size());
    for (size_t row = 0; row < indices.size(); ++row)
      reordered.emplace_back(row, static_cast<uint64_t>(order[row]));
  }
  detail::SelectedRowsPlan reorder_plan(reordered);
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
    const arrow::ChunkedArray gathered(arr);
    ARROW_RETURN_NOT_OK(
        AppendSelectedRows(gathered, reorder_plan, builders[i].get()));
    ARROW_RETURN_NOT_OK(builders[i]->Finish(&result_arrays[position]));
  }

  if (need_local_doc_id) {
    arrow::UInt64Builder rows(pool);
    ARROW_RETURN_NOT_OK(rows.Reserve(indices.size()));
    for (int row : indices) rows.UnsafeAppend(static_cast<uint64_t>(row));
    ARROW_ASSIGN_OR_RAISE(auto row_ids, rows.Finish());
    for (size_t i = 0; i < columns.size(); ++i) {
      if (columns[i] == LOCAL_ROW_ID) result_arrays[i] = row_ids;
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
    auto buffer_id = metadata_id_;
    buffer_id.column = col_idx;
    buffer_id.row_group = rg_id;
    auto acquired =
        ParquetBufferPool::get_instance().acquire_buffer_checked(buffer_id);
    if (!acquired.ok()) {
      if (acquired.status().IsOutOfMemory()) throw std::bad_alloc();
      LOG_ERROR("Failed to acquire parquet scalar: %s",
                acquired.status().ToString().c_str());
      return nullptr;
    }
    auto buffer_handle = std::move(acquired).ValueOrDie();
    const auto *col_chunked_array = buffer_handle.borrowed_data();

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
    std::shared_ptr<arrow::Scalar> owned;
    int64_t row = local_index;
    for (const auto &chunk : col_chunked_array->chunks()) {
      if (row < chunk->length()) {
        owned = ReadOwnedScalar(*chunk, row);
        break;
      }
      row -= chunk->length();
    }
    auto scalar_result =
        owned ? arrow::Result<std::shared_ptr<arrow::Scalar>>(std::move(owned))
              : buffer_handle.data()->GetScalar(local_index);
    if (!scalar_result.ok()) {
      if (scalar_result.status().IsOutOfMemory()) throw std::bad_alloc();
      LOG_ERROR("Failed to get scalar for row %zu status: %s",
                static_cast<size_t>(local_index),
                scalar_result.status().ToString().c_str());
      return nullptr;
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
  ailego::BudgetedVector<int> selected;
  selected.reserve(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    const int column = physic_schema_->GetFieldIndex(columns[i]);
    if (column < 0 || schema->field(i)->name() != columns[i] ||
        !schema->field(i)->type()->Equals(
            physic_schema_->field(column)->type())) {
      return nullptr;
    }
    const auto &field_type = schema->field(i)->type();
    const auto type = field_type->id();
    if ((i == 0 && type != arrow::Type::UINT64) ||
        (i == 1 && type != arrow::Type::STRING)) {
      return nullptr;
    }
    if (type == arrow::Type::LIST) {
      const auto &list_type = static_cast<const arrow::ListType &>(*field_type);
      if (!SupportsDirectDocValues(list_type.value_type()->id()))
        return nullptr;
    } else if (!SupportsDirectDocValues(type)) {
      return nullptr;
    }
    selected.push_back(column);
  }
  const int group = find_row_group_for_row(index);
  const int64_t local_row = index - get_row_group_offset(group);
  ailego::BudgetedVector<ParquetBufferContextHandle> pins;
  ailego::BudgetedVector<std::pair<const arrow::Array *, int64_t>> source_rows;
  pins.reserve(selected.size());
  source_rows.reserve(selected.size());
  detail::QueryResultEstimate estimate;
  estimate.add(sizeof(Doc) + 256);
  for (size_t i = 0; i < selected.size(); ++i) {
    auto key = metadata_id_;
    key.column = selected[i];
    key.row_group = group;
    auto acquired =
        ParquetBufferPool::get_instance().acquire_buffer_checked(key);
    if (!acquired.ok()) {
      if (acquired.status().IsOutOfMemory()) throw std::bad_alloc();
      return nullptr;
    }
    pins.push_back(std::move(acquired).ValueOrDie());
    const auto *column = pins.back().borrowed_data();
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
    if (!array || (i < 2 && array->IsNull(row))) return nullptr;
    estimate.add(192 + columns[i].size() * 2);
    estimate.value(*array, row);
    source_rows.emplace_back(array, row);
  }
  // Match Segment's result-copy admission before allocating any Doc fields.
  auto memory = std::make_shared<detail::ResultMemoryAllowance::Lease>(
      &detail::SharedResultMemoryAllowance());
  detail::ResultMemoryAllowance::PendingCopy pending;
  memory->add_pending(estimate.bytes(), pending);
  pending.start();
  auto doc = detail::MakeQueryDocument(memory);
  for (size_t i = 0; i < source_rows.size(); ++i) {
    const auto [array, row] = source_rows[i];
    if (array->IsNull(row)) continue;
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
        case arrow::Type::LIST:
          if (!ReadListDocField(static_cast<const arrow::ListArray &>(*array),
                                row, columns[i], *doc)) {
            return nullptr;
          }
          break;
        default:
          return nullptr;
      }
    }
  }
  return doc;
}

RecordBatchReaderPtr BufferPoolForwardStore::scan(
    const std::vector<std::string> &columns) try {
  if (!validate(columns)) {
    return nullptr;
  }

  // Create a new parquet reader for scanning
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  auto memory_pool = detail::GetParquetMemoryPool();
  if (columns.size() != 1 || columns.front() != GLOBAL_DOC_ID) {
    return std::make_shared<ParquetRecordBatchReader>(
        columns, physic_schema_, file_path_, std::move(memory_pool), file_,
        parquet_reader_->parquet_reader()->metadata(), metadata_id_,
        row_group_row_nums_);
  }
  // A cache miss may fall back to streaming. Do not prefetch an entire column
  // chunk before returning its first batch.
  parquet::ReaderProperties read_properties(memory_pool.get());
  read_properties.enable_buffered_stream();
  read_properties.set_buffer_size(64 * 1024);
  auto parquet_file_reader = parquet::ParquetFileReader::Open(
      file_, read_properties, parquet_reader_->parquet_reader()->metadata());
  auto arrow_properties = detail::BudgetedParquetArrowProperties();
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
} catch (const std::bad_alloc &) {
  LOG_ERROR("Forward scan metadata budget exhausted");
  return nullptr;
}

}  // namespace zvec

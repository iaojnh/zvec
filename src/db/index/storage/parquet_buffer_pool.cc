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

#include "parquet_buffer_pool.h"
#include <arrow/array/array_binary.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/util/byte_size.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include "parquet_memory_pool.h"

namespace zvec {

ParquetBufferID::ParquetBufferID(const std::string &filename, int column,
                                 int row_group)
    : filename(filename), column(column), row_group(row_group) {
  const auto path = ailego::FileHelper::PathFromUtf8(filename);
#if defined(_WIN32) || defined(_WIN64)
  struct _stat64 file_stat;
  if (!path.empty() && _wstat64(path.c_str(), &file_stat) == 0) {
#else
  struct stat file_stat;
  if (stat(filename.c_str(), &file_stat) == 0) {
#endif
    file_id = file_stat.st_ino;
  }

  std::error_code ec;
  const auto ftime = std::filesystem::last_write_time(path, ec);
  if (!ec) {
    mtime = static_cast<int64_t>(ftime.time_since_epoch().count());
  }
}

const std::string ParquetBufferID::to_string() const {
  std::string msg{"Buffer["};
  msg += "parquet: " + filename + "[" + std::to_string(file_id) + "]" +
         ", column: " + std::to_string(column) +
         ", row_group: " + std::to_string(row_group);
  msg += ", mtime: " + std::to_string(mtime);
  msg += "]";
  return msg;
}

ParquetBufferContextHandle::ParquetBufferContextHandle(
    const ParquetBufferContextHandle &handle)
    : buffer_id_(handle.buffer_id_) {
  if (handle.arrow_) {
    arrow_ = ParquetBufferPool::get_instance().retain(buffer_id_);
  }
}

ParquetBufferContextHandle::~ParquetBufferContextHandle() {
  if (arrow_) {
    ParquetBufferPool::get_instance().release(buffer_id_);
  }
}

namespace {
struct ParquetLoadError {
  arrow::Status status;
};

arrow::Status LoadColumn(const ParquetBufferID &id,
                         detail::ParquetBufferPayload &payload) {
  payload.memory_pool = detail::GetParquetMemoryPool();
  auto *pool = payload.memory_pool.get();
  ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(id.filename, pool));
  ARROW_ASSIGN_OR_RAISE(auto reader, detail::OpenManagedParquetReader(input, pool));
  if (id.row_group < 0 || id.row_group >= reader->num_row_groups() ||
      id.column < 0 ||
      id.column >= reader->parquet_reader()->metadata()->schema()->group_node()->field_count()) {
    return arrow::Status::Invalid("Invalid Parquet row group or column");
  }
  return reader->RowGroup(id.row_group)->Column(id.column)->Read(&payload.arrow);
}
}  // namespace

bool detail::ParquetBufferLoader::load(const ParquetBufferID &buffer_id,
                                       ParquetBufferPayload &payload,
                                       size_t &size) {
  auto status = LoadColumn(buffer_id, payload);
  if (!status.ok()) throw ParquetLoadError{std::move(status)};
  if (!payload.arrow) {
    throw ParquetLoadError{arrow::Status::Invalid("Missing Parquet column")};
  }
  // The precharged cache accounts for Arrow buffers at allocation. Keep a
  // separate allowance for the object wrappers outside Arrow's MemoryPool.
  payload.metadata.add_array(payload.arrow->num_chunks(),
                             sizeof(arrow::ArrayData) + sizeof(arrow::Array));
  size = arrow::util::TotalBufferSize(*payload.arrow);
  return true;
}

void detail::ParquetBufferLoader::clear(ParquetBufferPayload &payload) const {
  payload.arrow.reset();
  payload.memory_pool.reset();
  payload.metadata.retain_at_most(0);
}

ParquetBufferContextHandle ParquetBufferPool::acquire_buffer(
    ParquetBufferID buffer_id) {
  auto result = acquire_buffer_checked(buffer_id);
  if (!result.ok()) {
    LOG_ERROR("Failed to acquire parquet buffer: %s: %s",
              buffer_id.to_string().c_str(), result.status().ToString().c_str());
    return {};
  }
  return std::move(*result);
}

arrow::Result<ParquetBufferContextHandle> ParquetBufferPool::acquire_buffer_checked(
    const ParquetBufferID &buffer_id) try {
  auto arrow = cache_.acquire(buffer_id);
  if (!arrow) return arrow::Status::OutOfMemory("Parquet cache admission refused");
  try {
    return ParquetBufferContextHandle(buffer_id, std::move(arrow));
  } catch (...) {
    cache_.release(buffer_id);
    throw;
  }
} catch (const ParquetLoadError &error) {
  return error.status;
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("Parquet cache allocation refused");
}

void ParquetBufferPool::release(ParquetBufferID buffer_id) {
  cache_.release(buffer_id);
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::retain(
    ParquetBufferID buffer_id) {
  return cache_.retain(buffer_id);
}

}  // namespace zvec

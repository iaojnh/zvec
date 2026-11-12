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
#include <atomic>
#include <arrow/array/array_binary.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace zvec {

arrow::Result<std::shared_ptr<const ParquetFileContext>>
ParquetFileContext::Make(
    const std::string &filename,
    std::shared_ptr<arrow::io::RandomAccessFile> opened_file) {
  if (auto *mapped =
          dynamic_cast<arrow::io::MemoryMappedFile *>(opened_file.get())) {
    // Keep the existing pread-based cache loader without reopening a path
    // that may now refer to another file.
#if defined(_WIN32) || defined(_WIN64)
    int fd = _dup(mapped->file_descriptor());
#else
    int fd = fcntl(mapped->file_descriptor(), F_DUPFD_CLOEXEC, 0);
#endif
    if (fd < 0) {
      return arrow::Status::IOError("Failed to duplicate opened parquet file: ",
                                    filename);
    }
    auto readable = arrow::io::ReadableFile::Open(fd);
    if (!readable.ok()) {
#if defined(_WIN32) || defined(_WIN64)
      _close(fd);
#else
      close(fd);
#endif
      return readable.status();
    }
    opened_file = std::move(readable).ValueOrDie();
  }
  return std::shared_ptr<const ParquetFileContext>(
      std::make_shared<ParquetFileContext>(filename, std::move(opened_file)));
}

ParquetBufferID::ParquetBufferID(
    const std::string &filename,
    const std::shared_ptr<arrow::io::RandomAccessFile> &file, int column,
    int row_group)
    : filename(filename), column(column), row_group(row_group), input(file) {
  int fd = -1;
  if (auto *readable = dynamic_cast<arrow::io::ReadableFile *>(file.get())) {
    fd = readable->file_descriptor();
  } else if (auto *mapped =
                 dynamic_cast<arrow::io::MemoryMappedFile *>(file.get())) {
    fd = mapped->file_descriptor();
  }
  // Identify the opened file, so a pathname replacement cannot mix the
  // cache identity with bytes from a different file.
#if defined(_WIN32) || defined(_WIN64)
  BY_HANDLE_FILE_INFORMATION info;
  if (fd >= 0 && GetFileInformationByHandle(
                     reinterpret_cast<HANDLE>(_get_osfhandle(fd)), &info)) {
    file_id =
        (static_cast<uint64_t>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    device_id = info.dwVolumeSerialNumber;
    mtime = static_cast<int64_t>(
        (static_cast<uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
        info.ftLastWriteTime.dwLowDateTime);
    return;
  }
#else
  struct stat file_stat;
  if (fd >= 0 && fstat(fd, &file_stat) == 0) {
    file_id = file_stat.st_ino;
    device_id = file_stat.st_dev;
#if defined(__APPLE__)
    mtime = static_cast<int64_t>(file_stat.st_mtimespec.tv_sec) * 1000000000 +
            file_stat.st_mtimespec.tv_nsec;
#else
    mtime = static_cast<int64_t>(file_stat.st_mtim.tv_sec) * 1000000000 +
            file_stat.st_mtim.tv_nsec;
#endif
    return;
  }
#endif
  // Backends without a local descriptor cannot safely share a pathname
  // identity across opens. Isolate their caches for this open lifetime.
  static std::atomic<uint64_t> next_generation{1};
  generation = next_generation.fetch_add(1, std::memory_order_relaxed);
}

const std::string ParquetBufferID::to_string() const {
  std::string msg{"Buffer["};
  msg += "parquet: " + filename + "[" + std::to_string(file_id) + "]" +
         ", column: " + std::to_string(column) +
         ", row_group: " + std::to_string(row_group);
  msg += ", mtime: " + std::to_string(mtime);
  msg += ", device: " + std::to_string(device_id) +
         ", generation: " + std::to_string(generation);
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

bool detail::ParquetBufferLoader::load(const ParquetBufferID &buffer_id,
                                       ParquetBufferPayload &payload,
                                       size_t &size) {
  arrow::MemoryPool *mem_pool = arrow::default_memory_pool();

  auto input = buffer_id.input.lock();
  const auto &file_name = buffer_id.filename;
  if (!input) {
    LOG_ERROR("Opened parquet file is no longer available: %s",
              file_name.c_str());
    return false;
  }

  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto reader_result = parquet::arrow::OpenFile(input, mem_pool);
  if (!reader_result.ok()) {
    LOG_ERROR("Failed to create parquet reader[%s]: %s", file_name.c_str(),
              reader_result.status().ToString().c_str());
    return false;
  }
  reader = std::move(*reader_result);

  int row_group = buffer_id.row_group;
  int column = buffer_id.column;
  auto s = reader->RowGroup(row_group)->Column(column)->Read(&payload.arrow);
  if (!s.ok()) {
    LOG_ERROR("Failed to read parquet file[%s]: %s", file_name.c_str(),
              s.ToString().c_str());
    payload.arrow = nullptr;
    return false;
  }

  size = 0;
  payload.arrow_refs.clear();
  for (auto &array : payload.arrow->chunks()) {
    auto &buffers = array->data()->buffers;
    for (size_t buf_idx = 0; buf_idx < buffers.size(); ++buf_idx) {
      if (buffers[buf_idx] == nullptr) {
        continue;
      }
      payload.arrow_refs.emplace_back(buffers[buf_idx]);
      size += buffers[buf_idx]->capacity();
      std::shared_ptr<arrow::Buffer> hijacked_buffer(buffers[buf_idx].get(),
                                                     [](arrow::Buffer *) {});
      buffers[buf_idx] = hijacked_buffer;
    }
  }

  return true;
}

void detail::ParquetBufferLoader::clear(ParquetBufferPayload &payload) const {
  payload.arrow = nullptr;
  payload.arrow_refs.clear();
}

ParquetBufferContextHandle ParquetBufferPool::acquire_buffer(
    ParquetBufferID buffer_id) {
  auto arrow = cache_.acquire(buffer_id);
  if (!arrow) {
    LOG_ERROR("Failed to acquire parquet buffer: %s",
              buffer_id.to_string().c_str());
    return ParquetBufferContextHandle();
  }
  return ParquetBufferContextHandle(std::move(buffer_id), std::move(arrow));
}

void ParquetBufferPool::release(const ParquetBufferID &buffer_id) {
  cache_.release(buffer_id);
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::retain(
    const ParquetBufferID &buffer_id) {
  return cache_.retain(buffer_id);
}

}  // namespace zvec

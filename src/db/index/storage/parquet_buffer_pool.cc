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
#include <limits>
#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <unordered_set>
#include <arrow/array/array_binary.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include "parquet_memory_pool.h"
#include "parquet_metadata_budget.h"

namespace zvec {

namespace {

#if defined(_WIN32) || defined(_WIN64)
bool ReadFileIdentity(HANDLE handle, ParquetBufferID &id) {
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(handle, &info)) return false;
  id.file_id =
      (static_cast<uint64_t>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
  id.device_id = info.dwVolumeSerialNumber;
  id.mtime = static_cast<int64_t>(
      (static_cast<uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
      info.ftLastWriteTime.dwLowDateTime);
  return true;
}
#else
void ReadFileIdentity(const struct stat &info, ParquetBufferID &id) {
  id.file_id = info.st_ino;
  id.device_id = info.st_dev;
#if defined(__APPLE__)
  id.mtime = static_cast<int64_t>(info.st_mtimespec.tv_sec) * 1000000000 +
             info.st_mtimespec.tv_nsec;
#else
  id.mtime = static_cast<int64_t>(info.st_mtim.tv_sec) * 1000000000 +
             info.st_mtim.tv_nsec;
#endif
}
#endif

bool ReadOpenedFileIdentity(
    const std::shared_ptr<arrow::io::RandomAccessFile> &file,
    ParquetBufferID &id) {
  int fd = -1;
  if (auto *readable = dynamic_cast<arrow::io::ReadableFile *>(file.get())) {
    fd = readable->file_descriptor();
  } else if (auto *mapped =
                 dynamic_cast<arrow::io::MemoryMappedFile *>(file.get())) {
    fd = mapped->file_descriptor();
  }
#if defined(_WIN32) || defined(_WIN64)
  return fd >= 0 &&
         ReadFileIdentity(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), id);
#else
  struct stat info;
  if (fd < 0 || fstat(fd, &info) != 0) return false;
  ReadFileIdentity(info, id);
  return true;
#endif
}

using BufferSet =
    std::unordered_set<const arrow::Buffer *, std::hash<const arrow::Buffer *>,
                       std::equal_to<const arrow::Buffer *>,
                       ailego::BudgetedAllocator<const arrow::Buffer *>>;

// Arrow allocates these object shells with new, independently of MemoryPool.
// This is a conservative planning allowance for the supported Arrow tree,
// not a claim of exact heap/RSS accounting. Raw buffer capacity is excluded.
size_t ArrayMetadataBytes(const arrow::DataType &type) {
  constexpr size_t kArrayShellBytes = 512;
  constexpr size_t kBufferShellBytes = 128;
  const auto buffers = type.layout().buffers.size();
  if (buffers > (SIZE_MAX - kArrayShellBytes) / kBufferShellBytes)
    throw std::bad_alloc();
  size_t bytes = kArrayShellBytes + buffers * kBufferShellBytes;
  const auto add = [&bytes](const arrow::DataType &child) {
    const size_t extra = ArrayMetadataBytes(child);
    if (extra > SIZE_MAX - bytes) throw std::bad_alloc();
    bytes += extra;
  };
  for (const auto &field : type.fields()) add(*field->type());
  if (type.id() == arrow::Type::DICTIONARY)
    add(*static_cast<const arrow::DictionaryType &>(type).value_type());
  return bytes;
}

struct PinnedView {
  ailego::MemoryReservation metadata;
  ParquetBufferContextHandle pin;

  explicit PinnedView(const ParquetBufferContextHandle &handle) : pin(handle) {}
};

// Even a null/empty array without buffers must keep the metadata allowance.
template <typename T>
std::shared_ptr<T> OwnView(std::shared_ptr<T> value,
                           const std::shared_ptr<PinnedView> &pin) {
  struct Owner {
    std::shared_ptr<PinnedView> pin;
    std::shared_ptr<T> value;
  };
  auto *raw = value.get();
  auto owner = std::make_shared<Owner>(Owner{pin, std::move(value)});
  return std::shared_ptr<T>(std::move(owner), raw);
}

class ParquetLoadError : public std::exception {
 public:
  explicit ParquetLoadError(arrow::Status status)
      : status_(std::move(status)) {}
  const char *what() const noexcept override {
    return "Parquet cache load failed";
  }
  const arrow::Status &status() const {
    return status_;
  }

 private:
  arrow::Status status_;
};

void RetainAndDetachArrowBuffers(const std::shared_ptr<arrow::ArrayData> &data,
                                 detail::ParquetBufferPayload *payload,
                                 BufferSet *seen, size_t *size) {
  if (!data) {
    return;
  }
  for (auto &buffer : data->buffers) {
    if (!buffer) {
      continue;
    }
    std::shared_ptr<arrow::Buffer> retained = buffer;
    if (seen->insert(retained.get()).second) {
      payload->arrow_refs.emplace_back(retained);
      const int64_t arrow_capacity = retained->capacity();
      if (arrow_capacity < 0 || static_cast<uint64_t>(arrow_capacity) >
                                    std::numeric_limits<size_t>::max()) {
        *size = std::numeric_limits<size_t>::max();
      } else {
        const size_t capacity = static_cast<size_t>(arrow_capacity);
        *size = capacity > std::numeric_limits<size_t>::max() - *size
                    ? std::numeric_limits<size_t>::max()
                    : *size + capacity;
      }
    }
    buffer = std::shared_ptr<arrow::Buffer>(
        retained.get(), [](arrow::Buffer *) {},
        ailego::BudgetedAllocator<arrow::Buffer>{});
  }
  for (const auto &child : data->child_data) {
    RetainAndDetachArrowBuffers(child, payload, seen, size);
  }
  RetainAndDetachArrowBuffers(data->dictionary, payload, seen, size);
}

std::shared_ptr<arrow::ArrayData> CloneWithPinnedBuffers(
    const std::shared_ptr<arrow::ArrayData> &data,
    const std::shared_ptr<PinnedView> &pin) {
  if (!data) {
    return nullptr;
  }
  auto clone = data->Copy();
  for (auto &buffer : clone->buffers) {
    if (buffer) {
      buffer = std::shared_ptr<arrow::Buffer>(pin, buffer.get());
    }
  }
  for (auto &child : clone->child_data) {
    child = CloneWithPinnedBuffers(child, pin);
  }
  clone->dictionary = CloneWithPinnedBuffers(clone->dictionary, pin);
  return OwnView(std::move(clone), pin);
}

}  // namespace

ParquetBufferID::ParquetBufferID(const std::string &filename, int column,
                                 int row_group)
    : ParquetBufferID(filename, column, row_group, nullptr) {}

ParquetBufferID::ParquetBufferID(
    const std::string &filename, int column, int row_group,
    std::shared_ptr<const Filename> shared_filename)
    : column(column),
      row_group(row_group),
      filename_(shared_filename
                    ? std::move(shared_filename)
                    : std::allocate_shared<Filename>(
                          ailego::CacheMetadataAllocator<Filename, true>{},
                          filename.data(), filename.size())) {
  const auto path = ailego::FileHelper::PathFromUtf8(filename);
#if defined(_WIN32) || defined(_WIN64)
  const HANDLE handle =
      CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle != INVALID_HANDLE_VALUE) {
    const bool identified = ReadFileIdentity(handle, *this);
    CloseHandle(handle);
    if (identified) return;
  }
#else
  struct stat info;
  if (stat(path.c_str(), &info) == 0) {
    ReadFileIdentity(info, *this);
    return;
  }
#endif
  bool has_stat_mtime = false;
  int64_t stat_mtime = 0;
#if defined(_WIN32) || defined(_WIN64)
  struct _stat64 file_stat;
  if (!path.empty() && _wstat64(path.c_str(), &file_stat) == 0) {
#else
  struct stat file_stat;
  if (stat(path.c_str(), &file_stat) == 0) {
#endif
    file_id = file_stat.st_ino;
    has_stat_mtime = true;
    stat_mtime = static_cast<int64_t>(file_stat.st_mtime);
  }

  std::error_code ec;
  const auto ftime = std::filesystem::last_write_time(path, ec);
  if (!ec) {
    mtime = static_cast<int64_t>(ftime.time_since_epoch().count());
  } else if (has_stat_mtime) {
    mtime = stat_mtime;
  }
}

ParquetBufferID::ParquetBufferID(const std::string &filename, int column,
                                 int row_group,
                                 const ParquetBufferID &metadata_source)
    : ParquetBufferID(filename, column, row_group,
                      filename == metadata_source.filename()
                          ? metadata_source.filename_
                          : nullptr) {
  if (this->filename() == metadata_source.filename() &&
      file_id == metadata_source.file_id && mtime == metadata_source.mtime &&
      device_id == metadata_source.device_id &&
      generation == metadata_source.generation) {
    metadata = metadata_source.metadata;
  }
}

ParquetBufferID::ParquetBufferID(
    const std::string &filename,
    const std::shared_ptr<arrow::io::RandomAccessFile> &file, int column,
    int row_group)
    : ParquetBufferID(filename, column, row_group) {
  input = file;
  opened_file = true;
  // Identify the opened file, so a pathname replacement cannot mix the
  // cache identity with bytes from a different file.
  if (ReadOpenedFileIdentity(file, *this)) return;
  // Backends without a local descriptor cannot safely share a pathname
  // identity across opens. Isolate their caches for this open lifetime.
  static std::atomic<uint64_t> next_generation{1};
  generation = next_generation.fetch_add(1, std::memory_order_relaxed);
}

ParquetBufferID ParquetBufferID::for_column(int new_column,
                                            int new_row_group) const {
  auto result = *this;
  result.column = new_column;
  result.row_group = new_row_group;
  if (opened_file) {
    auto file = input.lock();
    if (file && ReadOpenedFileIdentity(file, result) &&
        (result.file_id != file_id || result.device_id != device_id ||
         result.mtime != mtime)) {
      result.metadata.reset();
    }
  }
  return result;
}

const std::string ParquetBufferID::to_string() const {
  std::string msg{"Buffer["};
  msg += "parquet: " + std::string(filename()) + "[" + std::to_string(file_id) +
         "]" + ", column: " + std::to_string(column) +
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
    // Drop this shell owner before the final pin can retire the payload and
    // refund its metadata allowance. The cache still owns it until release.
    arrow_.reset();
    ParquetBufferPool::get_instance().release(buffer_id_);
  }
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferContextHandle::data() const {
  if (!arrow_) {
    return nullptr;
  }
  auto pin = std::make_shared<PinnedView>(*this);
  pin->metadata.add(512);
  pin->metadata.add_array(static_cast<size_t>(arrow_->num_chunks()),
                          ArrayMetadataBytes(*arrow_->type()) + 128);
  // Alias every Arrow buffer to the cache pin while sharing payload bytes.
  arrow::ArrayVector chunks;
  chunks.reserve(arrow_->num_chunks());
  for (const auto &chunk : arrow_->chunks()) {
    chunks.emplace_back(
        arrow::MakeArray(CloneWithPinnedBuffers(chunk->data(), pin)));
  }
  return OwnView(
      std::make_shared<arrow::ChunkedArray>(std::move(chunks), arrow_->type()),
      pin);
}

bool detail::ParquetBufferLoader::load(const ParquetBufferID &buffer_id,
                                       ParquetBufferPayload &payload,
                                       size_t &size) {
  // ExternalCache transports exceptions to every waiter of the same flight.
  // Keep Arrow's typed cause intact until the checked storage boundary.
  auto status = load_checked(buffer_id, payload, size);
  if (!status.ok()) throw ParquetLoadError(std::move(status));
  return true;
}

arrow::Status detail::ParquetBufferLoader::load_checked(
    const ParquetBufferID &buffer_id, ParquetBufferPayload &payload,
    size_t &size) try {
  payload.memory_pool = GetParquetMemoryPool();
  arrow::MemoryPool *mem_pool = payload.memory_pool.get();

  auto input = buffer_id.input.lock();
  const auto file_name = buffer_id.filename();
  if (!input) {
    if (buffer_id.opened_file) {
      return arrow::Status::Invalid(
          "Opened parquet file is no longer available");
    }
    ARROW_ASSIGN_OR_RAISE(
        input, arrow::io::ReadableFile::Open(
                   std::string(file_name.data(), file_name.size()), mem_pool));
  }

  std::unique_ptr<parquet::arrow::FileReader> reader;
  // Configure both the low-level Parquet reader and Arrow output builders;
  // OpenFile(input, pool) alone leaves ReaderProperties on the default pool.
  parquet::ReaderProperties read_properties(mem_pool);
  read_properties.enable_buffered_stream();
  read_properties.set_buffer_size(64 * 1024);
  auto arrow_properties = BudgetedParquetArrowProperties();
  // Bound output-builder growth even while retaining the resulting chunks
  // for cache publication. Individual dictionary/data pages still need a
  // contiguous decode workspace and are subject to allocation admission.
  // Each load owns a separate reader/decoder; only the immutable file footer
  // is shared. An expired hint must also reserve metadata before parsing.
  auto file_reader = OpenBudgetedParquetReader(input, read_properties,
                                               buffer_id.metadata.lock());
  if (!file_reader.ok()) {
    LOG_ERROR("Failed to open parquet reader[%s]: %s", file_name.data(),
              file_reader.status().ToString().c_str());
    return file_reader.status();
  }
  auto reader_status = parquet::arrow::FileReader::Make(
      mem_pool, std::move(file_reader).ValueOrDie(), arrow_properties, &reader);
  if (!reader_status.ok()) {
    LOG_ERROR("Failed to create parquet reader[%s]: %s", file_name.data(),
              reader_status.ToString().c_str());
    return reader_status;
  }

  int row_group = buffer_id.row_group;
  int column = buffer_id.column;
  const auto metadata = reader->parquet_reader()->metadata();
  if (row_group < 0 || row_group >= metadata->num_row_groups()) {
    return arrow::Status::Invalid("Invalid parquet row group: ", row_group);
  }
  const auto leaves = ParquetLeafColumns(*metadata, {column});
  if (leaves.empty())
    return arrow::Status::Invalid("Invalid parquet column: ", column);
  const auto batch_rows = ParquetDecodeBatchRows(*metadata, row_group, leaves);
  reader->set_batch_size(batch_rows);
  auto batches = reader->GetRecordBatchReader({row_group}, leaves);
  if (!batches.ok()) {
    LOG_ERROR("Failed to create parquet column reader[%s]: %s",
              file_name.data(), batches.status().ToString().c_str());
    return batches.status();
  }
  const auto schema = (*batches)->schema();
  if (schema->num_fields() != 1)
    return arrow::Status::Invalid("Expected one parquet output column");
  const auto rows = metadata->RowGroup(row_group)->num_rows();
  if (rows < 0 || batch_rows <= 0)
    return arrow::Status::Invalid("Invalid parquet batch size");
  const size_t chunks =
      static_cast<size_t>(rows / batch_rows) + (rows % batch_rows != 0);
  payload.metadata_memory.add(512);
  payload.metadata_memory.add_array(
      chunks, ArrayMetadataBytes(*schema->field(0)->type()) + 128);
  auto result = (*batches)->ToTable();
  auto s = result.status();
  if (!s.ok()) {
    LOG_ERROR("Failed to read parquet file[%s]: %s", file_name.data(),
              s.ToString().c_str());
    payload.arrow = nullptr;
    return s;
  }
  auto table = *result;
  if (table->num_columns() != 1) {
    return arrow::Status::Invalid("Expected one parquet output column");
  }
  payload.arrow = table->column(0);

  retain_buffers(payload, size);
  return arrow::Status::OK();
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("Parquet cache metadata budget exhausted");
}

void detail::ParquetBufferLoader::retain_buffers(ParquetBufferPayload &payload,
                                                 size_t &size) const {
  // ExternalCache runs the loader outside its mutex. These nodes and control
  // blocks may reclaim cached pages before admission; the no-reclaim allocator
  // is only needed for cache bookkeeping performed while holding that mutex.
  size = 0;
  payload.arrow_refs.clear();
  BufferSet seen;
  for (auto &array : payload.arrow->chunks()) {
    RetainAndDetachArrowBuffers(array->data(), &payload, &seen, &size);
  }
}

void detail::ParquetBufferLoader::clear(ParquetBufferPayload &payload) const {
  payload.arrow = nullptr;
  decltype(payload.arrow_refs){}.swap(payload.arrow_refs);
  payload.memory_pool.reset();
  payload.metadata_memory.retain_at_most(0);
}

ParquetBufferContextHandle ParquetBufferPool::acquire_buffer(
    const ParquetBufferID &buffer_id) {
  auto result = acquire_buffer_checked(buffer_id);
  if (!result.ok()) {
    LOG_ERROR("Failed to acquire parquet buffer: %s: %s",
              buffer_id.to_string().c_str(),
              result.status().ToString().c_str());
    return ParquetBufferContextHandle();
  }
  return std::move(result).ValueOrDie();
}

ParquetBufferContextHandle ParquetBufferPool::try_acquire_buffer(
    const ParquetBufferID &buffer_id) {
  auto &cache = cache_for(buffer_id);
  auto arrow = cache.retain(buffer_id);
  if (!arrow) return {};
  try {
    return ParquetBufferContextHandle(buffer_id, std::move(arrow));
  } catch (...) {
    cache.release(buffer_id);
    throw;
  }
}

arrow::Result<ParquetBufferContextHandle>
ParquetBufferPool::acquire_buffer_checked(
    const ParquetBufferID &buffer_id) try {
  auto &cache = cache_for(buffer_id);
  auto arrow = cache.retain(buffer_id);
  if (!arrow) {
    // Hold the shared load slot through decoding, charging and publication.
    // acquire() rechecks the entry and coalesces concurrent loads of one key.
    LoadSlot slot(*this);
    arrow = cache.acquire(buffer_id);
  }
  if (!arrow) {
    // This loader throws on read/format errors. An empty result can only be
    // the cache's admission refusal, before any payload is published.
    return arrow::Status::OutOfMemory(
        "Parquet cache admission exceeds shared budget: ",
        buffer_id.to_string());
  }
  try {
    return ParquetBufferContextHandle(buffer_id, arrow);
  } catch (...) {
    cache.release(buffer_id);
    throw;
  }
} catch (const ParquetLoadError &error) {
  return error.status();
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("Parquet cache allocation failed");
}

void ParquetBufferPool::release(const ParquetBufferID &buffer_id) {
  cache_for(buffer_id).release(buffer_id);
}

std::shared_ptr<arrow::ChunkedArray> ParquetBufferPool::retain(
    const ParquetBufferID &buffer_id) {
  return cache_for(buffer_id).retain(buffer_id);
}

ParquetBufferPool::Cache &ParquetBufferPool::cache_for(
    const ParquetBufferID &buffer_id) {
  // Stable identity components select the lock shard; the cache still checks
  // the full key, including the pathname, for equality.
  const uint64_t hash = buffer_id.file_id ^ buffer_id.generation ^
                        (static_cast<uint64_t>(buffer_id.column) * 31) ^
                        static_cast<uint64_t>(buffer_id.row_group);
  return caches_[hash % kCacheShards].cache;
}

ParquetBufferPool::LoadSlot::LoadSlot(ParquetBufferPool &pool) : pool_(pool) {
  std::unique_lock<std::mutex> lock(pool_.load_mutex_);
  pool_.load_cv_.wait(
      lock, [&] { return pool_.active_loads_ < kMaxConcurrentLoads; });
  ++pool_.active_loads_;
}

ParquetBufferPool::LoadSlot::~LoadSlot() {
  {
    std::lock_guard<std::mutex> lock(pool_.load_mutex_);
    --pool_.active_loads_;
  }
  pool_.load_cv_.notify_one();
}

}  // namespace zvec

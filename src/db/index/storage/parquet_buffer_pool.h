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

#include <sys/stat.h>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <arrow/result.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/external_cache.h>

namespace arrow {
class Buffer;
class ChunkedArray;
namespace io {
class RandomAccessFile;
}
}  // namespace arrow

namespace parquet {
class FileMetaData;
}  // namespace parquet

namespace zvec {

struct ParquetBufferID {
  int column{0};
  int row_group{0};
  uint64_t file_id{0};
  int64_t mtime{0};
  uint64_t device_id{0};
  uint64_t generation{0};
  bool opened_file{false};
  std::weak_ptr<arrow::io::RandomAccessFile> input;

  // A reader may lend its already parsed footer to a cache miss. Cache keys
  // must not keep that footer alive after the owning reader is closed.
  std::weak_ptr<parquet::FileMetaData> metadata;

  ParquetBufferID() = default;
  ParquetBufferID(const std::string &filename,
                  const std::shared_ptr<arrow::io::RandomAccessFile> &file,
                  int column, int row_group);
  ParquetBufferID(const std::string &filename, int column, int row_group);
  ParquetBufferID(const std::string &filename, int column, int row_group,
                  const ParquetBufferID &metadata_source);

  // Reuse the admitted filename owner, but refresh the opened file's identity.
  // Path replacements must keep reading the original descriptor, while writes
  // to that descriptor must not reuse a decoded column from an older version.
  ParquetBufferID for_column(int column, int row_group) const;

  std::string_view filename() const noexcept {
    return filename_ ? std::string_view(filename_->data(), filename_->size())
                     : std::string_view("");
  }

  const std::string to_string() const;

 private:
  using Filename =
      std::basic_string<char, std::char_traits<char>,
                        ailego::CacheMetadataAllocator<char, true>>;
  ParquetBufferID(const std::string &filename, int column, int row_group,
                  std::shared_ptr<const Filename> shared_filename);

  // Created with reclaim-capable admission outside the cache lock. Copying a
  // key (including cache nodes and escaped views) only shares this immutable
  // owner, so it neither allocates nor enters reclamation under that lock.
  // Both string capacity and the shared control block remain budgeted until
  // the final key owner is destroyed.
  std::shared_ptr<const Filename> filename_;
};

struct ParquetBufferIDHash {
  size_t operator()(const ParquetBufferID &buffer_id) const {
    size_t hash = std::hash<std::string_view>{}(buffer_id.filename());
    const auto combine = [&hash](size_t value) {
      hash ^= value + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    };
    combine(std::hash<uint64_t>{}(buffer_id.file_id));
    combine(std::hash<int64_t>{}(buffer_id.mtime));
    combine(std::hash<uint64_t>{}(buffer_id.device_id));
    combine(std::hash<uint64_t>{}(buffer_id.generation));
    combine(std::hash<int>{}(buffer_id.column));
    combine(std::hash<int>{}(buffer_id.row_group));
    return hash;
  }
};

struct ParquetBufferIDEqual {
  bool operator()(const ParquetBufferID &a, const ParquetBufferID &b) const {
    if (a.filename() != b.filename()) {
      return false;
    }
    if (a.file_id != b.file_id) {
      return false;
    }
    if (a.device_id != b.device_id || a.generation != b.generation)
      return false;
    if (a.mtime != b.mtime) {
      return false;
    }
    return a.column == b.column && a.row_group == b.row_group;
  }
};

namespace detail {

class ParquetMemoryPool;

struct ParquetBufferPayload {
  // Arrow's MemoryPool covers buffer bytes, not Array/ArrayData/Buffer
  // wrappers. Admit a schema/chunk-based allowance before decoding and keep
  // it until all cached wrappers have been destroyed.
  ailego::MemoryReservation metadata_memory;
  // Arrow buffers store a raw pool pointer. Destroy their owners before this
  // shared pool owner, including when a cached payload outlives its reader.
  std::shared_ptr<ParquetMemoryPool> memory_pool;
  std::shared_ptr<arrow::ChunkedArray> arrow{nullptr};
  ailego::BudgetedVector<std::shared_ptr<arrow::Buffer>> arrow_refs{};
};

class ParquetBufferLoader {
 public:
  using Value = std::shared_ptr<arrow::ChunkedArray>;

  bool load(const ParquetBufferID &buffer_id, ParquetBufferPayload &payload,
            size_t &size);

  arrow::Status load_checked(const ParquetBufferID &buffer_id,
                             ParquetBufferPayload &payload, size_t &size);

  Value value(const ParquetBufferPayload &payload) const {
    return payload.arrow;
  }

  void clear(ParquetBufferPayload &payload) const;

  // Also used to retain buffers materialized outside the Parquet reader.
  void retain_buffers(ParquetBufferPayload &payload, size_t &size) const;
};

}  // namespace detail

class ParquetBufferContextHandle {
 public:
  ParquetBufferContextHandle() = default;
  ParquetBufferContextHandle(const ParquetBufferID &buffer_id,
                             std::shared_ptr<arrow::ChunkedArray> arrow)
      : buffer_id_(buffer_id), arrow_(std::move(arrow)) {}
  ParquetBufferContextHandle(const ParquetBufferContextHandle &handle);
  ParquetBufferContextHandle(ParquetBufferContextHandle &&handle)
      : buffer_id_(std::move(handle.buffer_id_)),
        arrow_(std::move(handle.arrow_)) {}

  ~ParquetBufferContextHandle();

  //! Return an Arrow view whose buffers retain the cache pin.
  std::shared_ptr<arrow::ChunkedArray> data() const;

  // Only for synchronous reads completed before this handle is destroyed.
  // Escaping arrays/scalars must continue using data() to retain their pin.
  const arrow::ChunkedArray *borrowed_data() const & noexcept {
    return arrow_.get();
  }
  const arrow::ChunkedArray *borrowed_data() const && = delete;

 private:
  ParquetBufferID buffer_id_;
  std::shared_ptr<arrow::ChunkedArray> arrow_{nullptr};
};

class ParquetBufferPool {
 public:
  typedef std::shared_ptr<ParquetBufferPool> Pointer;

  ParquetBufferContextHandle acquire_buffer(const ParquetBufferID &buffer_id);

  //! Pin an already loaded column without decoding or admitting a cold miss.
  ParquetBufferContextHandle try_acquire_buffer(
      const ParquetBufferID &buffer_id);

  arrow::Result<ParquetBufferContextHandle> acquire_buffer_checked(
      const ParquetBufferID &buffer_id);

  void retire_file(const ParquetBufferID &file) {
    for (auto &shard : caches_) {
      shard.cache.retire_if([&file](const ParquetBufferID &key) {
        return key.filename() == file.filename() &&
               key.file_id == file.file_id && key.mtime == file.mtime &&
               key.device_id == file.device_id &&
               key.generation == file.generation;
      });
    }
  }

  static ParquetBufferPool &get_instance() {
    static ParquetBufferPool instance;
    return instance;
  }

  ParquetBufferPool(const ParquetBufferPool &) = delete;
  ParquetBufferPool &operator=(const ParquetBufferPool &) = delete;
  ParquetBufferPool(ParquetBufferPool &&) = delete;
  ParquetBufferPool &operator=(ParquetBufferPool &&) = delete;

 private:
  static constexpr size_t kMaxConcurrentLoads = 4;

  friend class ParquetBufferContextHandle;
  friend struct ParquetBufferPoolTestPeer;

  using Cache =
      ailego::ExternalCache<ParquetBufferID, detail::ParquetBufferPayload,
                            detail::ParquetBufferLoader, ParquetBufferIDHash,
                            ParquetBufferIDEqual, /*PayloadOwnsBudget=*/true>;

  ParquetBufferPool() = default;

  static constexpr size_t kCacheShards = 16;
  struct alignas(AILEGO_CACHE_LINE_SIZE) CacheShard {
    Cache cache;
  };

  class LoadSlot {
   public:
    explicit LoadSlot(ParquetBufferPool &pool);
    ~LoadSlot();
    LoadSlot(const LoadSlot &) = delete;
    LoadSlot &operator=(const LoadSlot &) = delete;

   private:
    ParquetBufferPool &pool_;
  };

  Cache &cache_for(const ParquetBufferID &buffer_id);

  std::shared_ptr<arrow::ChunkedArray> retain(const ParquetBufferID &buffer_id);

  void release(const ParquetBufferID &buffer_id);

 private:
  std::array<CacheShard, kCacheShards> caches_;
  // Misses alone share the loader ceiling. Hot pins touch only their shard.
  std::mutex load_mutex_;
  std::condition_variable load_cv_;
  size_t active_loads_{0};
};

}  // namespace zvec

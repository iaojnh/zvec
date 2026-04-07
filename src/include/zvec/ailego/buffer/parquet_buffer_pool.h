#pragma once

#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <arrow/api.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/pattern/singleton.h>
#include "lru_cache.h"

namespace arrow {
class ChunkedArray;
class Array;
class DataType;
class Scalar;
template <typename T>
class Result;
class Status;
class Buffer;
}  // namespace arrow

namespace zvec {
namespace ailego {

using block_id_t = size_t;
using version_t = size_t;

class LRUCache;

struct ParquetBufferID {
  std::string filename;
  int column;
  int row_group;
  uint64_t file_id;
  long mtime;
  ParquetBufferID(std::string &filename, int column, int row_group)
      : filename(filename), column(column), row_group(row_group) {
    struct stat file_stat;
    if (stat(filename.c_str(), &file_stat) == 0) {
      // file_stat.st_ino contains the inode number
      // file_stat.st_dev contains the device ID
      // Together they uniquely identify a file
      file_id = file_stat.st_ino;
      std::filesystem::path p(filename);
      auto ftime = std::filesystem::last_write_time(p);
      mtime = static_cast<std::uint64_t>(ftime.time_since_epoch().count());
    }
  }
};

struct IDHash {
  size_t operator()(const ParquetBufferID &buffer_id) const {
    size_t hash = std::hash<int>{}(1);
    hash = hash ^ (std::hash<uint64_t>{}(buffer_id.file_id));
    hash = hash * 31 + std::hash<int>{}(buffer_id.column);
    hash = hash * 31 + std::hash<int>{}(buffer_id.row_group);
    return hash;
  }
};

struct IDEqual {
  bool operator()(const ParquetBufferID &a, const ParquetBufferID &b) const {
    if (a.filename != b.filename) {
      return false;
    }
    if (a.file_id != b.file_id) {
      return false;
    }
    if (a.mtime != b.mtime) {
      return false;
    }
    return a.column == b.column && a.row_group == b.row_group;
  }
};


class ParquetBufferPool {
 public:
  typedef std::shared_ptr<ParquetBufferPool> Pointer;

  struct ParquetBufferContext {
    // A shared pointer to the buffers allocated for arrow parquet data
    std::shared_ptr<arrow::ChunkedArray> arrow{nullptr};

    // Guard original arrow buffers to prevent premature deletion
    std::vector<std::shared_ptr<arrow::Buffer>> arrow_refs{};

    size_t size;
    alignas(64) std::atomic<int> ref_count{std::numeric_limits<int>::min()};
    alignas(64) std::atomic<version_t> load_count{0};
  };

  struct ArrowBufferDeleter {
    explicit ArrowBufferDeleter(ParquetBufferPool *c, ParquetBufferID i)
        : pool(c), id(i) {}
    ParquetBufferPool *pool;
    ParquetBufferID id;
    // Only reduces the reference count but does not actually release the
    // buffer, since the buffer memory is managed by the BufferManager.
    void operator()(arrow::Buffer *) {
      return;
    }
  };

  using Table = std::unordered_map<ParquetBufferID, ParquetBufferContext,
                                   IDHash, IDEqual>;

  arrow::Status acquire(ParquetBufferID buffer_id,
                        ParquetBufferContext &context) {
    // TODO: file handler and memory pool can be optimized
    arrow::MemoryPool *mem_pool = arrow::default_memory_pool();

    // Open file
    std::shared_ptr<arrow::io::RandomAccessFile> input;
    const auto &file_name = buffer_id.filename;
    ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(file_name));

    // Open reader
    std::unique_ptr<parquet::arrow::FileReader> reader;
    ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(input, mem_pool));

    // Perform read
    int row_group = buffer_id.row_group;
    int column = buffer_id.column;
    auto s = reader->RowGroup(row_group)->Column(column)->Read(&context.arrow);
    if (!s.ok()) {
      LOG_ERROR("Failed to read parquet file[%s]", file_name.c_str());
      context.arrow = nullptr;
      return s;
    }

    context.size = 0;
    context.arrow_refs.clear();
    // Compute the memory usage and hijack Arrow's buffers with our
    // implementation
    for (auto &array : context.arrow->chunks()) {
      auto &buffers = array->data()->buffers;
      for (size_t buf_idx = 0; buf_idx < buffers.size(); ++buf_idx) {
        if (buffers[buf_idx] == nullptr) {
          continue;
        }
        // Keep references to original buffers to prevent premature deletion
        context.arrow_refs.emplace_back(buffers[buf_idx]);
        context.size += buffers[buf_idx]->capacity();
        // Create hijacked buffer with custom deleter that notifies us when
        // Arrow is finished with the buffer
        std::shared_ptr<arrow::Buffer> hijacked_buffer(
            buffers[buf_idx].get(), ArrowBufferDeleter(this, buffer_id));
        buffers[buf_idx] = hijacked_buffer;
      }
    }

    return arrow::Status::OK();
  }

  bool acquire_buffer(ParquetBufferID buffer_id,
                      std::shared_ptr<arrow::ChunkedArray> &arrow) {
    {
      std::shared_lock<std::shared_mutex> lock(table_mutex_);
      auto iter = table_.find(buffer_id);
      if (iter != table_.end()) {
        arrow = acquire(buffer_id);
        if (arrow != nullptr) {
          return true;
        }
      }
    }
    {
      std::unique_lock<std::shared_mutex> lock(table_mutex_);
      {
        bool found = MemoryLimitPool::get_instance().try_acquire_parquet(0);
        if (!found) {
          for (int i = 0; i < 5; i++) {
            LRUCache::get_instance().recycle();
            found = MemoryLimitPool::get_instance().try_acquire_parquet(0);
            if (found) {
              break;
            }
          }
        }
        if (!found) {
          LOG_ERROR("Failed to acquire parquet buffer");
          return false;
        }
      }
      if (acquire(buffer_id, table_[buffer_id]).ok()) {
        arrow = set_block_acquired(buffer_id);
        return true;
      } else {
        LOG_ERROR("Failed to acquire parquet buffer");
        return false;
      }
    }
  }

  std::shared_ptr<arrow::ChunkedArray> set_block_acquired(
      ParquetBufferID buffer_id) {
    ParquetBufferContext &context = table_[buffer_id];
    while (true) {
      int current_count = context.ref_count.load(std::memory_order_relaxed);
      if (current_count >= 0) {
        if (context.ref_count.compare_exchange_weak(
                current_count, current_count + 1, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          return context.arrow;
        }
      } else {
        if (context.ref_count.compare_exchange_weak(
                current_count, 1, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          context.load_count.fetch_add(1, std::memory_order_relaxed);
          return context.arrow;
        }
      }
    }
  }
  std::shared_ptr<arrow::ChunkedArray> acquire(ParquetBufferID buffer_id) {
    auto iter = table_.find(buffer_id);
    if (iter == table_.end()) {
      return nullptr;
    }
    ParquetBufferContext &context = table_[buffer_id];
    while (true) {
      int current_count = context.ref_count.load(std::memory_order_acquire);
      if (current_count < 0) {
        return nullptr;
      }
      if (context.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        if (current_count == 0) {
          context.load_count.fetch_add(1, std::memory_order_relaxed);
        }
        return context.arrow;
      }
    }
  }

  void release(ParquetBufferID buffer_id) {
    std::shared_lock<std::shared_mutex> lock(table_mutex_);
    auto iter = table_.find(buffer_id);
    if (iter == table_.end()) {
      return;
    }
    ParquetBufferContext &context = table_[buffer_id];
    if (context.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      LRUCache::BlockType block;
      // TODO: set block
      LRUCache::get_instance().add_single_block(block, 0);
    }
  }

  void evict(ParquetBufferID buffer_id) {
    std::unique_lock<std::shared_mutex> lock(table_mutex_);
    auto iter = table_.find(buffer_id);
    if (iter == table_.end()) {
      return;
    }
    ParquetBufferContext &context = table_[buffer_id];
    int expected = 0;
    if (context.ref_count.compare_exchange_strong(
            expected, std::numeric_limits<int>::min())) {
      MemoryLimitPool::get_instance().release_parquet(context.size);
      table_.erase(buffer_id);
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
  ParquetBufferPool() = default;

 private:
  Table table_;
  std::shared_mutex table_mutex_;
};

}  // namespace ailego
}  // namespace zvec
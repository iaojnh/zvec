#pragma once

#include <sys/stat.h>
#include <fcntl.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <zvec/ailego/internal/platform.h>
#include "concurrentqueue.h"
#include "lru_cache.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

namespace zvec {
namespace ailego {

using block_id_t = size_t;
using version_t = size_t;

class VectorPageTable {
  struct Entry {
    alignas(64) std::atomic<int> ref_count;
    alignas(64) std::atomic<version_t> load_count;
    alignas(64) std::atomic<version_t> in_lru_version;
    char *buffer;
    size_t size;
  };

 public:
  VectorPageTable() : entry_num_(0), entries_(nullptr) {
    LRUCache::get_instance().set_valid(this);
  }
  ~VectorPageTable() {
    delete[] entries_;
    LRUCache::get_instance().set_invalid(this);
  }

  void init(size_t entry_num);

  char *acquire_block(block_id_t block_id);

  void release_block(block_id_t block_id);

  char *evict_block(block_id_t block_id);

  char *set_block_acquired(block_id_t block_id, char *buffer, size_t size);

  size_t entry_num() const {
    return entry_num_;
  }

  inline bool isDeadBlock(LRUCache::BlockType block) const {
    Entry &entry = entries_[block.block.first];
    return block.block.second != entry.load_count.load();
  }

 private:
  size_t entry_num_{0};
  Entry *entries_{nullptr};
};

class VecBufferPoolHandle;

class VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  VecBufferPool(const std::string &filename);
  ~VecBufferPool() {
    // Free any buffers still pinned in the map
    for (size_t i = 0; i < lp_map_.entry_num(); ++i) {
      lp_map_.evict_block(i);
    }
#if defined(_MSC_VER)
    _close(fd_);
#else
    close(fd_);
#endif
  }

  int init(size_t pool_capacity, size_t block_size, size_t segment_count);

  VecBufferPoolHandle get_handle();

  char *acquire_buffer(block_id_t block_id, size_t offset, size_t size,
                       int retry = 0);

  int get_meta(size_t offset, size_t length, char *buffer);

  size_t file_size() const {
    return file_size_;
  }

 private:
  int fd_;
  size_t file_size_;
  size_t pool_capacity_;

 public:
  VectorPageTable lp_map_;

 private:
  std::vector<std::unique_ptr<std::mutex>> mutex_vec_;
};

class VecBufferPoolHandle {
 public:
  VecBufferPoolHandle(VecBufferPool &pool) : pool_(pool) {}
  VecBufferPoolHandle(VecBufferPoolHandle &&other) : pool_(other.pool_) {}

  ~VecBufferPoolHandle() = default;

  typedef std::shared_ptr<VecBufferPoolHandle> Pointer;

  char *get_block(size_t offset, size_t size, size_t block_id);

  int get_meta(size_t offset, size_t length, char *buffer);

  void release_one(block_id_t block_id);

  void acquire_one(block_id_t block_id);

 private:
  VecBufferPool &pool_;
};

}  // namespace ailego
}  // namespace zvec
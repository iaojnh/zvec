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
    alignas(64) std::atomic<version_t> lru_version;
    char *buffer;
    size_t size;
  };

 public:
  VectorPageTable() : entry_num_(0), entries_(nullptr) {
    LRUCache::get_instance().set_valid(this);
  }
  ~VectorPageTable() {
    LRUCache::get_instance().set_invalid(this);
    delete[] entries_;
  }

  VectorPageTable(const VectorPageTable &) = delete;
  VectorPageTable &operator=(const VectorPageTable &) = delete;
  VectorPageTable(VectorPageTable &&) = delete;
  VectorPageTable &operator=(VectorPageTable &&) = delete;

  void init(size_t entry_num);

  char *acquire_block(block_id_t block_id);

  void release_block(block_id_t block_id);

  void evict_block(block_id_t block_id);

  char *set_block_acquired(block_id_t block_id, char *buffer, size_t size);

  size_t entry_num() const {
    return entry_num_;
  }

  // Returns true if the block has no active references (ref_count <= 0).
  // Used by VecBufferPool destructor to assert all handles are released.
  bool is_released(block_id_t block_id) const {
    assert(block_id < entry_num_);
    return entries_[block_id].ref_count.load(std::memory_order_relaxed) <= 0;
  }

  inline bool is_dead_block(LRUCache::BlockType block) const {
    Entry &entry = entries_[block.vector_block.first];
    return block.vector_block.second != entry.load_count.load();
  }

 private:
  size_t entry_num_{0};
  Entry *entries_{nullptr};
  moodycamel::ConcurrentQueue<block_id_t> evict_cache_;
};

class VecBufferPoolHandle;

class VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  VecBufferPool(const std::string &filename);
  ~VecBufferPool() {
    for (size_t i = 0; i < page_table_.entry_num(); ++i) {
      // A positive ref_count means a VecBufferPoolHandle is still alive,
      // which is a contract violation: all handles must be destroyed before
      // the pool itself is destroyed.
      assert(page_table_.is_released(i));
      page_table_.evict_block(i);
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

 public:
  VectorPageTable page_table_;

 private:
  std::vector<std::unique_ptr<std::mutex>> block_mutexes_;
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
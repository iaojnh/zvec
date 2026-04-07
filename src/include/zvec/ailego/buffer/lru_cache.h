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
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_logger.h>
#include "concurrentqueue.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

namespace zvec {
namespace ailego {

class LPMap;

using block_id_t = size_t;
using version_t = size_t;

struct ParquetBufferID {
  std::string filename;
  int column;
  int row_group;
  uint64_t file_id;
  long mtime;
  ParquetBufferID() {}
  ParquetBufferID(std::string &filename, int column, int row_group);
};

class LRUCache {
 public:
  struct BlockType {
    std::pair<block_id_t, version_t> block;
    std::pair<ParquetBufferID, version_t> parquet_buffer_block;
    LPMap *lp_map{nullptr};
  };
  typedef moodycamel::ConcurrentQueue<BlockType> ConcurrentQueue;

  static LRUCache &get_instance() {
    static LRUCache instance;
    return instance;
  }
  LRUCache(const LRUCache &) = delete;
  LRUCache &operator=(const LRUCache &) = delete;
  LRUCache(LRUCache &&) = delete;
  LRUCache &operator=(LRUCache &&) = delete;

  int init();

  bool evict_single_block(BlockType &item);

  bool evict_block(BlockType &item);

  bool add_single_block(const BlockType &block, int block_type);

  void clear_dead_node();

  bool is_valid(LPMap *lp_map) {
    std::shared_lock<std::shared_mutex> lock(valid_lp_maps_mutex_);
    return valid_lp_maps_.find(lp_map) != valid_lp_maps_.end();
  }

  void set_valid(LPMap *lp_map) {
    std::unique_lock<std::shared_mutex> lock(valid_lp_maps_mutex_);
    valid_lp_maps_.insert(lp_map);
  }

  void set_invalid(LPMap *lp_map) {
    std::unique_lock<std::shared_mutex> lock(valid_lp_maps_mutex_);
    valid_lp_maps_.erase(lp_map);
  }

  bool recycle();

 private:
  LRUCache() {
    init();
  }

 private:
  constexpr static size_t CATCH_QUEUE_NUM = 3;
  size_t block_size_{0};
  std::vector<ConcurrentQueue> queues_;
  alignas(64) std::atomic<size_t> evict_queue_insertions_{0};
  std::unordered_set<LPMap *> valid_lp_maps_;
  std::shared_mutex valid_lp_maps_mutex_;
};

class MemoryLimitPool {
 public:
  static MemoryLimitPool &get_instance() {
    static MemoryLimitPool instance;
    return instance;
  }
  MemoryLimitPool(const MemoryLimitPool &) = delete;
  MemoryLimitPool &operator=(const MemoryLimitPool &) = delete;
  MemoryLimitPool(MemoryLimitPool &&) = delete;
  MemoryLimitPool &operator=(MemoryLimitPool &&) = delete;

  int init(size_t pool_size);

  bool try_acquire_buffer(const size_t buffer_size, char *&buffer);

  void acquire_parquet(const size_t buffer_size);

  void release_buffer(char *buffer, const size_t buffer_size);

  void release_parquet(const size_t buffer_size);

  bool is_full();

 private:
  MemoryLimitPool() = default;

 private:
  size_t pool_size_{0};
  std::atomic<size_t> used_size_{0};
};

}  // namespace ailego
}  // namespace zvec
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
#include "buffer_pool.h"
#include "concurrentqueue.h"
#include <zvec/core/framework/index_logger.h>

#if defined(_MSC_VER)
#include <io.h>
#endif

namespace zvec {
namespace ailego {

class LPMap;

using block_id_t = size_t;
using version_t = size_t;

class LRUCache {
 public:
  struct BlockType {
    std::pair<block_id_t, version_t> block;
    LPMap *lp_map;
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

  void clear_dead_node(const LPMap *lp_map);

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

  void release_buffer(char *buffer, const size_t buffer_size);

  bool is_full();

 private:
  MemoryLimitPool() = default;

 private:
  size_t pool_size_{0};
  std::atomic<size_t> used_size_{0};
};

}  // namespace ailego
}  // namespace zvec
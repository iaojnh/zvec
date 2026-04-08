#include <zvec/ailego/buffer/vector_buffer_pool.h>
#include <zvec/ailego/buffer/parquet_buffer_pool.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace ailego {

int LRUCache::init() {
  block_size_ = 512;
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    queues_.push_back(ConcurrentQueue(block_size_ * 200));
  }
  return 0;
}

bool LRUCache::evict_single_block(BlockType &item) {
  bool found = false;
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    found = queues_[i].try_dequeue(item);
    if (found) {
      break;
    }
  }
  return found;
}

bool LRUCache::evict_block(BlockType &item) {
  bool ok = false;
  do {
    ok = LRUCache::get_instance().evict_single_block(item);
    if (!ok) {
      return false;
    }
    if (item.lp_map == nullptr) {
      if (!ParquetBufferPool::get_instance().is_dead_node(item)) {
        break;
      } else {
        continue;
      }
    }
  } while (!is_valid(item.lp_map) || item.lp_map->isDeadBlock(item));
  return ok;
}

bool LRUCache::recycle() {
  BlockType item;
  while (MemoryLimitPool::get_instance().is_full() && evict_block(item)) {
    if (item.lp_map) {
      item.lp_map->evict_block(item.block.first);
    } else {
      ParquetBufferPool::get_instance().evict(item.parquet_buffer_block.first);
    }
  }
  return MemoryLimitPool::get_instance().is_full();
}

bool LRUCache::add_single_block(const BlockType &block, int block_type) {
  bool ok = queues_[block_type].enqueue(block);
  if (!ok) {
    LOG_ERROR("enqueue failed.");
    return false;
  }
  static thread_local int evict_queue_insertions = 0;
  if (evict_queue_insertions++ > block_size_) {
    this->clear_dead_node();
    evict_queue_insertions = 0;
  }
  return true;
}

void LRUCache::clear_dead_node() {
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    size_t clear_size = block_size_;
    if (queues_[i].size_approx() < block_size_) {
      continue;
    }
    if (queues_[i].size_approx() > block_size_ * 8) {
      clear_size *= 2;
    }
    size_t clear_count = 0;
    BlockType item;
    ConcurrentQueue tmp_queue(block_size_ * 200);
    while (queues_[i].try_dequeue(item) && (clear_count++ < clear_size)) {
      if (item.lp_map == nullptr) {
        if (!ParquetBufferPool::get_instance().is_dead_node(item)) {
          tmp_queue.enqueue(item);
        }
      } else if (is_valid(item.lp_map) && !item.lp_map->isDeadBlock(item)) {
        tmp_queue.enqueue(item);
      }
    }
    while (tmp_queue.try_dequeue(item)) {
      queues_[i].enqueue(item);
    }
  }
}

int MemoryLimitPool::init(size_t pool_size) {
  pool_size_ = 0;
  LRUCache::get_instance().recycle();
  pool_size_ = pool_size;
  return 0;
}

bool MemoryLimitPool::try_acquire_buffer(const size_t buffer_size,
                                         char *&buffer) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    if (expected >= pool_size_) {
      // LOG_ERROR("expected: %lu, pool_size: %lu", expected, pool_size_);
      return false;
    }
    desired = expected + buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));
  buffer = (char *)ailego_malloc(buffer_size);
  return true;
}

void MemoryLimitPool::acquire_parquet(const size_t buffer_size) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    desired = expected + buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));
}

void MemoryLimitPool::release_buffer(char *buffer, const size_t buffer_size) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    desired = expected - buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));
  ailego_free(buffer);
}

void MemoryLimitPool::release_parquet(const size_t buffer_size) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    desired = expected - buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));
}

bool MemoryLimitPool::is_full() {
  return used_size_.load() >= pool_size_;
}

bool MemoryLimitPool::is_hot_level1() {
  return used_size_.load() >= pool_size_ * 3 / 5;
}

bool MemoryLimitPool::is_hot_level2() {
  return used_size_.load() >= pool_size_ * 4 / 5;
}

}  // namespace ailego
}  // namespace zvec
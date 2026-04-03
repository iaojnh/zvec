#include <zvec/ailego/buffer/buffer_pool.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace ailego {

int LRUCache::init() {
  block_size_ = 512;
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    queues_.push_back(ConcurrentQueue());
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
  } while (item.lp_map->isDeadBlock(item));
  return ok;
}

bool LRUCache::recycle() {
  BlockType item;
  while (MemoryLimitPool::get_instance().is_full() && evict_block(item)) {
    item.lp_map->evict_block(item.block.first);
  }
  return MemoryLimitPool::get_instance().is_full();
}

bool LRUCache::add_single_block(const BlockType &block, int block_type) {
  bool ok = queues_[block_type].enqueue(block);
  if (!ok) {
    LOG_ERROR("enqueue failed.");
    return false;
  }
  evict_queue_insertions_.fetch_add(1, std::memory_order_relaxed);
  if (evict_queue_insertions_ % block_size_ == 0) {
    this->clear_dead_node(block.lp_map);
  }
  return true;
}

void LRUCache::clear_dead_node(const LPMap *lp_map) {
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    size_t clear_size = block_size_ * 2;
    if (queues_[i].size_approx() < clear_size * 4) {
      continue;
    }
    size_t clear_count = 0;
    ConcurrentQueue tmp(block_size_);
    BlockType item;
    while (queues_[i].try_dequeue(item) && (clear_count++ < clear_size)) {
      if (!lp_map->isDeadBlock(item)) {
        if (!tmp.enqueue(item)) {
          LOG_ERROR("enqueue failed.");
        }
      }
    }
    while (tmp.try_dequeue(item)) {
      if (!lp_map->isDeadBlock(item)) {
        if (!queues_[i].enqueue(item)) {
          LOG_ERROR("enqueue failed.");
        }
      }
    }
  }
}
}  // namespace ailego
}  // namespace zvec
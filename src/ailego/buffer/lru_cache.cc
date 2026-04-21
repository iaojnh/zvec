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

#include <zvec/ailego/buffer/parquet_hash_table.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace ailego {

int LRUCache::init() {
  evict_batch_size_ = 512;
  for (size_t i = 0; i < CACHE_QUEUE_NUM; i++) {
    evict_queues_.push_back(ConcurrentQueue(evict_batch_size_ * 200));
  }
  return 0;
}

bool LRUCache::evict_single_block(BlockType &item) {
  bool found = false;
  for (size_t i = 0; i < CACHE_QUEUE_NUM; i++) {
    found = evict_queues_[i].try_dequeue(item);
    if (found) {
      break;
    }
  }
  return found;
}

bool LRUCache::is_valid_and_alive(const BlockType &item) {
  std::shared_lock<std::shared_mutex> lock(valid_page_tables_mutex_);
  if (valid_page_tables_.find(item.page_table) == valid_page_tables_.end()) {
    return false;
  }
  // is_dead_block accesses entries_ under the same shared lock, so the
  // VectorPageTable destructor (which holds the unique lock via set_invalid)
  // cannot free entries_ while this check is in progress.
  return !item.page_table->is_dead_block(item);
}

bool LRUCache::evict_block(BlockType &item) {
  bool ok = false;
  do {
    ok = evict_single_block(item);
    if (!ok) {
      return false;
    }
    if (item.page_table == nullptr) {
      if (!ParquetBufferPool::get_instance().is_dead_node(item)) {
        break;
      } else {
        continue;
      }
    }
  } while (!is_valid_and_alive(item));
  return ok;
}

void LRUCache::recycle() {
  BlockType item;
  while (MemoryLimitPool::get_instance().is_full() && evict_block(item)) {
    if (item.page_table) {
      // Hold the shared lock across the eviction call to prevent
      // use-after-free if the VectorPageTable is concurrently destroyed.
      std::shared_lock<std::shared_mutex> lock(valid_page_tables_mutex_);
      if (valid_page_tables_.find(item.page_table) !=
          valid_page_tables_.end()) {
        item.page_table->evict_block(item.vector_block.first);
      }
    } else {
      ParquetBufferPool::get_instance().evict(item.parquet_buffer_block.first);
    }
  }
}

bool LRUCache::add_single_block(const BlockType &block, int queue_index) {
  bool ok = evict_queues_[queue_index].enqueue(block);
  if (!ok) {
    LOG_ERROR("enqueue failed.");
    return false;
  }
  static thread_local int evict_queue_insertions = 0;
  if (evict_queue_insertions++ > evict_batch_size_) {
    this->clear_dead_node();
    evict_queue_insertions = 0;
  }
  return true;
}

void LRUCache::clear_dead_node() {
  for (size_t i = 0; i < CACHE_QUEUE_NUM; i++) {
    size_t clear_size = evict_batch_size_;
    if (evict_queues_[i].size_approx() < evict_batch_size_) {
      continue;
    }
    if (evict_queues_[i].size_approx() > evict_batch_size_ * 8) {
      clear_size *= 2;
    }
    size_t clear_count = 0;
    BlockType item;
    ConcurrentQueue live_blocks_queue(evict_batch_size_ * 200);
    while (evict_queues_[i].try_dequeue(item) && (clear_count++ < clear_size)) {
      if (item.page_table == nullptr) {
        if (!ParquetBufferPool::get_instance().is_dead_node(item)) {
          live_blocks_queue.enqueue(item);
        }
      } else if (is_valid_and_alive(item)) {
        live_blocks_queue.enqueue(item);
      }
    }
    while (live_blocks_queue.try_dequeue(item)) {
      evict_queues_[i].enqueue(item);
    }
  }
}

int MemoryLimitPool::init(size_t pool_size) {
  pool_size_ = 0;
  LRUCache::get_instance().recycle();
  pool_size_ = pool_size;
  LOG_INFO("MemoryLimitPool initialized with pool size: %lu", pool_size_);
  return 0;
}

bool MemoryLimitPool::try_acquire_buffer(const size_t buffer_size,
                                         char *&buffer) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    if (expected >= pool_size_) {
      return false;
    }
    desired = expected + buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));
  // buffer = (char *)ailego_aligned_malloc(buffer_size, 64);
  // if (!buffer) {
  //   used_size_.fetch_sub(buffer_size);
  //   return false;
  // }
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
    assert(expected >= buffer_size);
  } while (!used_size_.compare_exchange_weak(expected, desired));
  // ailego_free(buffer);
}

void MemoryLimitPool::release_parquet(const size_t buffer_size) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    desired = expected - buffer_size;
    assert(expected >= buffer_size);
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
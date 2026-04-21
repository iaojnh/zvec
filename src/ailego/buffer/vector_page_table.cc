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

#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_logger.h>

#if !defined(_MSC_VER)
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace zvec {
namespace ailego {

void VectorPageTable::init(size_t entry_num) {
  if (entries_) {
    delete[] entries_;
  }
  entry_num_ = entry_num;
  entries_ = new Entry[entry_num_];
  for (size_t i = 0; i < entry_num_; i++) {
    entries_[i].ref_count.store(std::numeric_limits<int>::min());
    entries_[i].load_count.store(0);
    entries_[i].lru_version.store(0);
    entries_[i].buffer = nullptr;
  }
}

char *VectorPageTable::acquire_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_acquire);
    if (current_count < 0) {
      return nullptr;
    }
    if (entry.ref_count.compare_exchange_weak(current_count, current_count + 1,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      if (current_count == 0) {
        entry.load_count.fetch_add(1, std::memory_order_relaxed);
      }
      return entry.buffer;
    }
  }
}

void VectorPageTable::release_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];

  if (entry.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    if (MemoryLimitPool::get_instance().is_hot_level1()) {
      LRUCache::BlockType block;
      block.page_table = this;
      block.vector_block.first = block_id;
      version_t v = entry.load_count.load(std::memory_order_relaxed);
      block.vector_block.second = v;
      entry.lru_version.store(v, std::memory_order_relaxed);
      LRUCache::get_instance().add_single_block(block, 0);
    } else {
      // Two separate relaxed loads: a concurrent acquire_block may increment
      // load_count between the two reads, making the condition transiently
      // false (missed enqueue). This is benign: the block will satisfy the
      // condition again on the next release cycle, and hot_level1 pressure
      // will add it to LRU directly regardless.
      if (entry.lru_version.load(std::memory_order_relaxed) + 1 ==
          entry.load_count.load(std::memory_order_relaxed)) {
        evict_cache_.enqueue(block_id);
      }
    }
  }
}

void VectorPageTable::evict_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  char *buffer = entry.buffer;
  size_t size = entry.size;
  int expected = 0;
  if (entry.ref_count.compare_exchange_strong(
          expected, std::numeric_limits<int>::min())) {
    if (buffer) {
      madvise(buffer, size, MADV_DONTNEED);
      MemoryLimitPool::get_instance().release_buffer(buffer, size);
    }
  }
}

char *VectorPageTable::set_block_acquired(block_id_t block_id, char *buffer,
                                          size_t size) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  if (MemoryLimitPool::get_instance().is_hot_level2()) {
    size_t evict_block_id = 0;
    while (evict_cache_.try_dequeue(evict_block_id)) {
      Entry &hot_entry = entries_[evict_block_id];
      if (hot_entry.ref_count.load() != 0) {
        continue;
      }
      // Snapshot load_count once. We only need to advance lru_version to this
      // snapshot version; chasing subsequent increments is unnecessary and can
      // cause unbounded spinning under high concurrency.
      // If the CAS fails, another thread has already advanced lru_version (to
      // at least this version), so the block is already queued in LRU.
      version_t desired = hot_entry.load_count.load(std::memory_order_relaxed);
      version_t current = hot_entry.lru_version.load(std::memory_order_relaxed);
      if (current != desired) {
        if (hot_entry.lru_version.compare_exchange_strong(
                current, desired, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          LRUCache::BlockType block;
          block.page_table = this;
          block.vector_block.first = evict_block_id;
          block.vector_block.second = desired;
          LRUCache::get_instance().add_single_block(block, 0);
        }
      }
    }
  }
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_relaxed);
    if (current_count >= 0) {
      // Defensive branch: in practice this path should never be reached.
      // set_block_acquired() is always called under block_mutexes_[block_id],
      // and the caller (acquire_buffer) re-checks acquire_block() inside the
      // same lock before invoking this function. Therefore, if we get here,
      // ref_count must still be negative (unloaded). This branch is retained
      // as a safety net in case the locking contract is violated in the future,
      // e.g. if set_block_acquired is called from an unlocked context.
      if (entry.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        madvise(buffer, size, MADV_DONTNEED);
        MemoryLimitPool::get_instance().release_buffer(buffer, size);
        return entry.buffer;
      }
    } else {
      entry.buffer = buffer;
      entry.size = size;
      entry.load_count.fetch_add(1, std::memory_order_relaxed);
      entry.ref_count.store(1, std::memory_order_release);
      return entry.buffer;
    }
  }
}

VecBufferPool::VecBufferPool(const std::string &filename) {
#if defined(_MSC_VER)
  fd_ = _open(filename.c_str(), O_RDONLY | _O_BINARY);
#else
  fd_ = open(filename.c_str(), O_RDONLY);
#endif
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
#if defined(_MSC_VER)
  struct _stat64 st;
  if (_fstat64(fd_, &st) < 0) {
    _close(fd_);
    throw std::runtime_error("Failed to stat file: " + filename);
  }
  file_size_ = st.st_size;
  if (file_size_ > 0) {
    HANDLE file_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
    HANDLE mapping_handle =
        CreateFileMapping(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mapping_handle == NULL) {
      _close(fd_);
      throw std::runtime_error("Failed to create file mapping: " + filename);
    }
    mmap_addr_ =
        reinterpret_cast<char *>(MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0));
    CloseHandle(mapping_handle);
    if (!mmap_addr_) {
      _close(fd_);
      throw std::runtime_error("Failed to map view of file: " + filename);
    }
  }
#else
  struct stat st;
  if (fstat(fd_, &st) < 0) {
    ::close(fd_);
    throw std::runtime_error("Failed to stat file: " + filename);
  }
  file_size_ = st.st_size;
  if (file_size_ > 0) {
    mmap_addr_ = reinterpret_cast<char *>(
        mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0));
    if (mmap_addr_ == MAP_FAILED) {
      mmap_addr_ = nullptr;
      ::close(fd_);
      throw std::runtime_error("Failed to mmap file: " + filename);
    }
  }
#endif
}

int VecBufferPool::init(size_t /*pool_capacity*/, size_t block_size,
                        size_t segment_count) {
  if (block_size == 0) {
    LOG_ERROR("block_size must not be 0");
    return -1;
  }
  size_t block_num = segment_count + 10;
  page_table_.init(block_num);
  block_mutexes_.clear();
  block_mutexes_.reserve(block_num);
  for (size_t i = 0; i < block_num; i++) {
    block_mutexes_.emplace_back(std::make_unique<std::mutex>());
  }
  LOG_DEBUG("entry num: %zu", page_table_.entry_num());
  return 0;
}

VecBufferPoolHandle VecBufferPool::get_handle() {
  return VecBufferPoolHandle(*this);
}

char *VecBufferPool::acquire_buffer(block_id_t block_id, size_t offset,
                                    size_t size, int retry) {
  assert(block_id < block_mutexes_.size());
  char *buffer = page_table_.acquire_block(block_id);
  if (buffer) {
    return buffer;
  }
  std::lock_guard<std::mutex> lock(*block_mutexes_[block_id]);
  buffer = page_table_.acquire_block(block_id);
  if (buffer) {
    return buffer;
  }
  {
    bool found =
        MemoryLimitPool::get_instance().try_acquire_buffer(size, buffer);
    if (!found) {
      for (int i = 0; i < retry; i++) {
        LRUCache::get_instance().recycle();
        found =
            MemoryLimitPool::get_instance().try_acquire_buffer(size, buffer);
        if (found) {
          break;
        }
      }
    }
    if (!found) {
      LOG_ERROR("Buffer pool failed to get free buffer");
      return nullptr;
    }
  }

  if (!mmap_addr_) {
    LOG_ERROR("Buffer pool mmap region is not available");
    madvise(buffer, size, MADV_DONTNEED);
    MemoryLimitPool::get_instance().release_buffer(buffer, size);
    return nullptr;
  }
  buffer = mmap_addr_ + offset;
  // memcpy(buffer, mmap_addr_ + offset, size);
  return page_table_.set_block_acquired(block_id, buffer, size);
}

int VecBufferPool::get_meta(size_t offset, size_t length, char *buffer) {
  if (!mmap_addr_) {
    LOG_ERROR("Buffer pool mmap region is not available");
    return -1;
  }
  memcpy(buffer, mmap_addr_ + offset, length);
  return 0;
}

char *VecBufferPoolHandle::get_block(size_t offset, size_t size,
                                     size_t block_id) {
  char *buffer = pool_.acquire_buffer(block_id, offset, size, 5);
  return buffer;
}

int VecBufferPoolHandle::get_meta(size_t offset, size_t length, char *buffer) {
  return pool_.get_meta(offset, length, buffer);
}

void VecBufferPoolHandle::release_one(block_id_t block_id) {
  pool_.page_table_.release_block(block_id);
}

void VecBufferPoolHandle::acquire_one(block_id_t block_id) {
  // The caller must guarantee the block is already loaded before calling
  // acquire_one(). The return value of acquire_block() is intentionally
  // ignored here, as a null return would indicate a contract violation.
  pool_.page_table_.acquire_block(block_id);
}

}  // namespace ailego
}  // namespace zvec
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

#include <zvec/ailego/buffer/vector_buffer_pool.h>
#include <zvec/core/framework/index_logger.h>

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
static ssize_t zvec_pread(int fd, void *buf, size_t count, size_t offset) {
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) return -1;
  OVERLAPPED ov = {};
  ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytes_read = 0;
  if (!ReadFile(handle, buf, static_cast<DWORD>(count), &bytes_read, &ov)) {
    return -1;
  }
  return static_cast<ssize_t>(bytes_read);
}
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
  if (MemoryLimitPool::get_instance().is_hot_level2()) {
    for (int i = 0; i < entry_num_; i++) {
      Entry &hot_entry = entries_[i];
      if (hot_entry.ref_count.load() != 0) {
        continue;
      }
      while (true) {
        int current = hot_entry.lru_version.load(std::memory_order_relaxed);
        int expected = hot_entry.load_count.load(std::memory_order_relaxed);
        if (current == expected) {
          break;
        }
        if (hot_entry.ref_count.compare_exchange_weak(
                current, expected, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          LRUCache::BlockType block;
          block.page_table = this;
          block.vector_block.first = i;
          block.vector_block.second = expected;
          LRUCache::get_instance().add_single_block(block, 0);
        }
      }
    }
  }
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
      block.vector_block.second = entry.load_count.load();
      entry.lru_version = entry.load_count.load();
      LRUCache::get_instance().add_single_block(block, 0);
    }
  }
}

char *VectorPageTable::evict_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  int expected = 0;
  if (entry.ref_count.compare_exchange_strong(
          expected, std::numeric_limits<int>::min())) {
    char *buffer = entry.buffer;
    if (buffer) {
      MemoryLimitPool::get_instance().release_buffer(buffer, entry.size);
      entry.buffer = nullptr;
    }
    return buffer;
  } else {
    return nullptr;
  }
}

char *VectorPageTable::set_block_acquired(block_id_t block_id, char *buffer,
                                          size_t size) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  entry.size = size;
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_relaxed);
    if (current_count >= 0) {
      // Another thread has already loaded this block. Release the buffer we
      // allocated since it won't be used, then pin the existing entry.
      if (entry.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        MemoryLimitPool::get_instance().release_buffer(buffer, size);
        return entry.buffer;
      }
    } else {
      // Block is unloaded (ref_count < 0). Take ownership of buffer.
      if (entry.ref_count.compare_exchange_weak(current_count, 1,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
        entry.buffer = buffer;
        entry.load_count.fetch_add(1, std::memory_order_relaxed);
        return entry.buffer;
      }
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
#else
  struct stat st;
  if (fstat(fd_, &st) < 0) {
    ::close(fd_);
#endif
    throw std::runtime_error("Failed to stat file: " + filename);
  }
  file_size_ = st.st_size;
}

int VecBufferPool::init(size_t /*pool_capacity*/, size_t block_size,
                        size_t segment_count) {
  if (block_size == 0) {
    LOG_ERROR("block_size must not be 0");
    return -1;
  }
  size_t block_num = segment_count + 10;
  page_table_.init(block_num);
  block_mutexes_.reserve(block_num);
  for (int i = 0; i < block_num; i++) {
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

#if defined(_MSC_VER)
  ssize_t read_bytes = zvec_pread(fd_, buffer, size, offset);
#else
  ssize_t read_bytes = pread(fd_, buffer, size, offset);
#endif
  if (read_bytes != static_cast<ssize_t>(size)) {
    LOG_ERROR("Buffer pool failed to read file at offset: %zu", offset);
    MemoryLimitPool::get_instance().release_buffer(buffer, size);
    return nullptr;
  }
  return page_table_.set_block_acquired(block_id, buffer, size);
}

int VecBufferPool::get_meta(size_t offset, size_t length, char *buffer) {
#if defined(_MSC_VER)
  ssize_t read_bytes = zvec_pread(fd_, buffer, length, offset);
#else
  ssize_t read_bytes = pread(fd_, buffer, length, offset);
#endif
  if (read_bytes != static_cast<ssize_t>(length)) {
    LOG_ERROR("Buffer pool failed to read file at offset: %zu", offset);
    return -1;
  }
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
  pool_.page_table_.acquire_block(block_id);
}

}  // namespace ailego
}  // namespace zvec
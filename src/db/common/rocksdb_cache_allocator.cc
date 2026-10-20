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
#include "rocksdb_cache_allocator.h"
#include <limits>
#include <new>
#include <zvec/ailego/io/file.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace zvec {
namespace {

size_t CachePageSize() {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  ::GetSystemInfo(&info);
  return info.dwPageSize;
#else
  const auto size = ::sysconf(_SC_PAGESIZE);
  return size > 0 ? static_cast<size_t>(size) : 0;
#endif
}

}  // namespace

RocksdbCacheAllocator::RocksdbCacheAllocator() : page_size_(CachePageSize()) {
  if (page_size_ == 0) {
    throw std::bad_alloc();
  }
}

void *RocksdbCacheAllocator::Allocate(size_t size) {
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  if (size > kMax - sizeof(Header) ||
      size + sizeof(Header) > kMax - (page_size_ - 1)) {
    throw std::bad_alloc();
  }
  const size_t bytes =
      (size + sizeof(Header) + page_size_ - 1) / page_size_ * page_size_;
  void *mapping = ailego::File::MemoryMap(bytes, 0);
  if (mapping == nullptr) {
    throw std::bad_alloc();
  }
  auto *header = new (mapping) Header{bytes};
  return header + 1;
}

void RocksdbCacheAllocator::Deallocate(void *pointer) {
  if (pointer != nullptr) {
    auto *header = static_cast<Header *>(pointer) - 1;
    ailego::File::MemoryUnmap(header, header->mapped_bytes);
  }
}

size_t RocksdbCacheAllocator::UsableSize(void *pointer,
                                         size_t /*allocation_size*/) const {
  return (static_cast<Header *>(pointer) - 1)->mapped_bytes;
}

}  // namespace zvec

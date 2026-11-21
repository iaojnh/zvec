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

#include <algorithm>
#include <cstring>
#include <utility>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

// Immutable range backed by an index segment. mmap retains its direct view;
// Buffer Storage retains no resident pages between reads. Each caller owns
// the guard for its bounded read, including cross-page fallback allocations.
class ImmutableSegmentView {
 public:
  int load(const IndexStorage::Pointer &storage, const std::string &id,
           size_t offset, size_t bytes) {
    if (!storage) return IndexError_InvalidArgument;
    auto segment = storage->get(id);
    if (!segment || offset > segment->data_size() ||
        bytes > segment->data_size() - offset)
      return IndexError_InvalidFormat;
    IndexStorage::MemoryBlock direct;
    if (bytes && !storage->vec_buffer_pool() &&
        segment->read(offset, direct, bytes) != bytes)
      return IndexError_ReadData;
    direct_ = std::move(direct);
    segment_ = std::move(segment);
    storage_ = storage;
    offset_ = offset;
    size_ = bytes;
    return 0;
  }

  int view(size_t offset, size_t bytes, IndexStorage::MemoryBlock *guard,
           const void **out) const {
    if (!guard || !out) return IndexError_InvalidArgument;
    guard->reset();
    *out = nullptr;
    if (!segment_ || offset > size_ || bytes > size_ - offset)
      return IndexError_InvalidArgument;
    if (!bytes) return 0;
    if (direct_.data()) {
      *out = static_cast<const char *>(direct_.data()) + offset;
    } else {
      if (segment_->read(offset_ + offset, *guard, bytes) != bytes)
        return IndexError_ReadData;
      *out = guard->data();
    }
    return *out ? 0 : IndexError_ReadData;
  }

  int read(size_t offset, void *out, size_t bytes) const {
    if (!out && bytes) return IndexError_InvalidArgument;
    IndexStorage::MemoryBlock guard;
    const void *data = nullptr;
    int ret = view(offset, bytes, &guard, &data);
    if (ret == 0 && bytes) std::memcpy(out, data, bytes);
    return ret;
  }

  template <typename Writer>
  int copy_to(Writer &&write) const {
    for (size_t offset = 0; offset < size_;) {
      const size_t bytes = std::min<size_t>(64 * 1024, size_ - offset);
      IndexStorage::MemoryBlock guard;
      const void *data = nullptr;
      int ret = view(offset, bytes, &guard, &data);
      if (ret != 0) return ret;
      ret = write(offset, data, bytes);
      if (ret != 0) return ret;
      offset += bytes;
    }
    return 0;
  }

  size_t size() const {
    return size_;
  }
  bool empty() const {
    return size_ == 0;
  }
  const void *data() const {
    return direct_.data();
  }

 private:
  IndexStorage::Pointer storage_;
  IndexStorage::Segment::Pointer segment_;
  IndexStorage::MemoryBlock direct_;
  size_t offset_{0};
  size_t size_{0};
};

}  // namespace zvec::core

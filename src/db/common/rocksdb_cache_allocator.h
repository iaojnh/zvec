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

#include <cstddef>
#include <rocksdb/memory_allocator.h>

namespace zvec {

// Keep long-lived SST cache payloads out of the heap used by RocksDB's
// temporary flush/compaction buffers. Otherwise a small live cache block can
// retain much larger freed regions of that heap across collection lifecycles.
//
// The cache, not BufferPool, owns this budget. UsableSize reports the full
// mapping, including its header and page rounding. Deallocate unmaps it rather
// than retaining freed blocks in a separate, unaccounted allocator cache.
class RocksdbCacheAllocator final : public rocksdb::MemoryAllocator {
 public:
  RocksdbCacheAllocator();

  const char *Name() const override {
    return "ZvecMappedCacheAllocator";
  }

  void *Allocate(size_t size) override;
  void Deallocate(void *pointer) override;
  size_t UsableSize(void *pointer, size_t allocation_size) const override;

 private:
  struct alignas(std::max_align_t) Header {
    size_t mapped_bytes;
  };

  const size_t page_size_;
};

}  // namespace zvec

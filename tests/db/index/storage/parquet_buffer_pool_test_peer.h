// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "db/index/storage/parquet_buffer_pool.h"

namespace zvec {

// Preserve the class declaration and MSVC member decoration while tests
// inspect or damage cached payloads. Redefining private changes that ABI.
struct ParquetBufferPoolTestPeer {
  static auto &cache_for(const ParquetBufferID &key) {
    return ParquetBufferPool::get_instance().cache_for(key);
  }

  static size_t entry_count() {
    size_t count = 0;
    for (auto &shard : ParquetBufferPool::get_instance().caches_)
      count += shard.cache.entry_count();
    return count;
  }
};

}  // namespace zvec

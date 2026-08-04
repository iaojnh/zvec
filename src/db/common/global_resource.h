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

#include <cstdint>
#include <memory>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/pattern/singleton.h>

namespace rocksdb {
class Cache;
}

namespace zvec {

class GlobalResource : public ailego::Singleton<GlobalResource> {
 public:
  struct MemoryStats {
    bool explicit_memory_budget_enabled{false};
    uint64_t total_capacity{0};
    uint64_t shared_cache_capacity{0};
    uint64_t shared_cache_used{0};
    uint64_t shared_cache_committed{0};
    uint64_t shared_cache_page_used{0};
    uint64_t shared_cache_external_used{0};
    uint64_t shared_cache_rejections{0};
    uint64_t rocksdb_block_cache_capacity{0};
    uint64_t rocksdb_block_cache_used{0};
    uint64_t query_working_capacity{0};
    uint64_t query_working_used{0};
    uint64_t query_working_rejections{0};
    uint64_t resident_metadata_capacity{0};
    uint64_t resident_metadata_used{0};
    uint64_t resident_metadata_rejections{0};
    uint64_t safety_reserve_capacity{0};
  };

  void initialize();

  MemoryStats memory_stats();

  ailego::ThreadPool *query_thread_pool() {
    initialize();
    return query_thread_pool_.get();
  }

  ailego::ThreadPool *optimize_thread_pool() {
    initialize();
    return optimize_thread_pool_.get();
  }

  std::shared_ptr<rocksdb::Cache> rocksdb_block_cache() {
    initialize();
    return rocksdb_block_cache_;
  }

 private:
  std::unique_ptr<ailego::ThreadPool> query_thread_pool_;
  std::unique_ptr<ailego::ThreadPool> optimize_thread_pool_;
  std::shared_ptr<rocksdb::Cache> rocksdb_block_cache_;
};

}  // namespace zvec

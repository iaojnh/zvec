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
#include "db/common/global_resource.h"
#include <mutex>
#include <rocksdb/cache.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_budget.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>

namespace zvec {

void GlobalResource::initialize() {
  static std::once_flag flag;
  std::call_once(flag, [this]() {
    this->query_thread_pool_.reset(new ailego::ThreadPool(
        GlobalConfig::Instance().query_thread_count(),
        GlobalConfig::Instance().query_thread_binding()));
    this->optimize_thread_pool_.reset(new ailego::ThreadPool(
        GlobalConfig::Instance().optimize_thread_count(),
        GlobalConfig::Instance().optimize_thread_binding()));
    const auto &config = GlobalConfig::Instance();
    ailego::MemoryBudgetManager::Config budget;
    budget.total_bytes = config.memory_limit_bytes();
    budget.enforce_accounting = config.explicit_memory_budget();
    budget.buffer_cache_bytes = config.buffer_cache_bytes();
    budget.rocksdb_block_cache_bytes = config.rocksdb_block_cache_bytes();
    budget.query_working_bytes = config.query_working_memory_bytes();
    budget.resident_metadata_bytes = config.resident_metadata_bytes();
    budget.safety_reserve_bytes = config.safety_reserve_bytes();
    if (!ailego::MemoryBudgetManager::get_instance().configure(budget)) {
      LOG_FATAL("Invalid process memory budget partition");
    }
    zvec::ailego::MemoryLimitPool::get_instance().init(
        static_cast<size_t>(budget.buffer_cache_bytes));
    if (budget.rocksdb_block_cache_bytes != 0) {
      rocksdb_block_cache_ = rocksdb::NewLRUCache(
          static_cast<size_t>(budget.rocksdb_block_cache_bytes),
          /*num_shard_bits=*/-1, /*strict_capacity_limit=*/true);
    }
    LOG_INFO(
        "Memory budget initialized: total=%llu buffer_cache=%llu "
        "rocksdb_block_cache=%llu query_working=%llu "
        "resident_metadata=%llu safety_reserve=%llu",
        static_cast<unsigned long long>(budget.total_bytes),
        static_cast<unsigned long long>(budget.buffer_cache_bytes),
        static_cast<unsigned long long>(budget.rocksdb_block_cache_bytes),
        static_cast<unsigned long long>(budget.query_working_bytes),
        static_cast<unsigned long long>(budget.resident_metadata_bytes),
        static_cast<unsigned long long>(budget.safety_reserve_bytes));
  });
}

}  // namespace zvec

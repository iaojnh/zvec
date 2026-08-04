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
#include <algorithm>
#include <thread>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_budget.h>
#include <zvec/db/config.h>

using namespace zvec;

TEST(GlobalResource, UsesEffectiveCountsAndDisablesBindingByDefault) {
  GlobalConfig::ConfigData config;
  config.query_thread_count = 2;
  config.optimize_thread_count = 1;
  const uint64_t partition = config.memory_limit_bytes / 16;
  config.rocksdb_block_cache_bytes = partition;
  config.query_working_memory_bytes = partition;
  config.resident_metadata_bytes = partition;
  config.safety_reserve_bytes = partition;
  EXPECT_FALSE(config.query_thread_binding);
  EXPECT_FALSE(config.optimize_thread_binding);

  const auto status = GlobalConfig::Instance().Initialize(config);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FALSE(GlobalConfig::Instance().query_thread_binding());
  EXPECT_FALSE(GlobalConfig::Instance().optimize_thread_binding());

  const auto max_workers =
      std::max(std::thread::hardware_concurrency(), 1u);
  const auto expected_query_workers = std::min(
      GlobalConfig::Instance().query_thread_count(), max_workers);
  const auto expected_optimize_workers = std::min(
      GlobalConfig::Instance().optimize_thread_count(), max_workers);

  EXPECT_EQ(expected_query_workers,
            GlobalResource::Instance().query_thread_pool()->count());
  EXPECT_EQ(expected_optimize_workers,
            GlobalResource::Instance().optimize_thread_pool()->count());

  const auto budget =
      zvec::ailego::MemoryBudgetManager::get_instance().snapshot();
  EXPECT_TRUE(budget.config.enforce_accounting);
  EXPECT_EQ(GlobalConfig::Instance().shared_cache_bytes(),
            budget.config.buffer_cache_bytes);
  EXPECT_EQ(GlobalConfig::Instance().shared_cache_bytes(),
            zvec::ailego::MemoryLimitPool::get_instance().capacity());

  auto rocksdb_cache = GlobalResource::Instance().rocksdb_block_cache();
  ASSERT_NE(nullptr, rocksdb_cache);
  EXPECT_EQ(partition, rocksdb_cache->GetCapacity());

  const auto stats = GlobalResource::Instance().memory_stats();
  EXPECT_TRUE(stats.explicit_memory_budget_enabled);
  EXPECT_EQ(config.memory_limit_bytes, stats.total_capacity);
  EXPECT_EQ(GlobalConfig::Instance().shared_cache_bytes(),
            stats.shared_cache_capacity);
  EXPECT_EQ(0u, stats.shared_cache_used);
  EXPECT_EQ(0u, stats.shared_cache_external_used);
  EXPECT_EQ(partition, stats.rocksdb_block_cache_capacity);
  EXPECT_EQ(partition, stats.query_working_capacity);
  EXPECT_EQ(partition, stats.resident_metadata_capacity);
  EXPECT_EQ(partition, stats.safety_reserve_capacity);

  auto &manager = zvec::ailego::MemoryBudgetManager::get_instance();
  ASSERT_TRUE(manager.try_charge(
      zvec::ailego::MemoryBudgetManager::Category::QueryWorking, partition));
  EXPECT_FALSE(manager.try_charge(
      zvec::ailego::MemoryBudgetManager::Category::QueryWorking, 1));
  const auto charged_stats = GlobalResource::Instance().memory_stats();
  EXPECT_EQ(partition, charged_stats.query_working_used);
  EXPECT_EQ(1u, charged_stats.query_working_rejections);
  manager.release(zvec::ailego::MemoryBudgetManager::Category::QueryWorking,
                  partition);
  EXPECT_EQ(0u,
            GlobalResource::Instance().memory_stats().query_working_used);
}

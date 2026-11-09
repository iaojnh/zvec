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
#include <cstdlib>
#include <thread>
#include <ailego/utility/memory_helper.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/config.h>
#include "db/common/native_image_memory.h"
#include "db/common/native_worker_memory.h"

using namespace zvec;

#if (defined(__linux__) || defined(__APPLE__) || defined(_WIN32)) && \
    GTEST_HAS_DEATH_TEST
TEST(GlobalResourceStartupDeathTest,
     PartialBudgetFailureCannotBecomeStandalone) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        constexpr size_t limit = 128UL << 20;
        if (pool.init(
                GlobalResource::calculate_buffer_pool_memory_budget(limit)))
          std::_Exit(1);
        // Simulate existing mandatory backing exhausting the pool exactly at
        // native image/worker reservation, after pool setup has succeeded.
        const size_t occupied = pool.available();
        if (!pool.try_charge_external(occupied)) std::_Exit(2);
        GlobalConfig::ConfigData config;
        config.memory_limit_bytes = limit;
        config.query_thread_count = config.optimize_thread_count = 1;
        if (GlobalConfig::Instance().initialize(config).ok()) std::_Exit(3);
        pool.release_external(occupied);
        auto &resources = GlobalResource::Instance();
        if (resources.initialize() == 0 || resources.initialized() ||
            resources.query_thread_pool() || resources.rocksdb_block_cache())
          std::_Exit(4);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}
#endif

TEST(GlobalResource, UsesEffectiveCountsAndDisablesBindingByDefault) {
  GlobalConfig::ConfigData config;
  config.query_thread_count = 2;
  config.optimize_thread_count = 1;
  EXPECT_FALSE(config.query_thread_binding);
  EXPECT_FALSE(config.optimize_thread_binding);

  const auto status = GlobalConfig::Instance().initialize(config);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FALSE(GlobalConfig::Instance().query_thread_binding());
  EXPECT_FALSE(GlobalConfig::Instance().optimize_thread_binding());

  const auto max_workers = std::max(std::thread::hardware_concurrency(), 1u);
  const auto expected_query_workers =
      std::min(GlobalConfig::Instance().query_thread_count(), max_workers);
  const auto expected_optimize_workers =
      std::min(GlobalConfig::Instance().optimize_thread_count(), max_workers);

  EXPECT_EQ(expected_query_workers,
            GlobalResource::Instance().query_thread_pool()->count());
  EXPECT_EQ(expected_optimize_workers,
            GlobalResource::Instance().optimize_thread_pool()->count());
  auto *arrow_pool = GlobalResource::Instance().arrow_query_thread_pool();
  ASSERT_NE(nullptr, arrow_pool);
  EXPECT_EQ(expected_query_workers, arrow_pool->GetCapacity());
  EXPECT_NE(arrow::internal::GetCpuThreadPool(), arrow_pool);
  auto *io_pool = GlobalResource::Instance().arrow_io_thread_pool();
  ASSERT_NE(nullptr, io_pool);
  EXPECT_EQ(expected_query_workers, io_pool->GetCapacity());
  EXPECT_NE(arrow_pool, io_pool);
  auto task =
      arrow_pool->Submit([arrow_pool] { return arrow_pool->OwnsThisThread(); });
  ASSERT_TRUE(task.ok());
  ASSERT_TRUE(task->result().ok());
  EXPECT_TRUE(*task->result());

  const uint64_t total_capacity = config.memory_limit_bytes;
  const uint64_t expected_rocksdb_capacity =
      GlobalResource::calculate_rocksdb_memory_budget(total_capacity);
  const uint64_t expected_buffer_capacity =
      GlobalResource::calculate_buffer_pool_memory_budget(total_capacity);
  EXPECT_EQ(expected_buffer_capacity,
            ailego::MemoryLimitPool::get_instance().capacity());

#if defined(__APPLE__) || defined(__linux__) || defined(_WIN32)
  const auto stack_bytes = detail::DefaultNativeWorkerStackBytes();
  const auto native_workers =
      3 * expected_query_workers + expected_optimize_workers +
      detail::RocksDbBackgroundThreads(config.optimize_thread_count) + 2;
  EXPECT_GT(stack_bytes, 0u);
  EXPECT_EQ(native_workers * stack_bytes,
            detail::NativeWorkerStackReservation().bytes());
  EXPECT_GE(ailego::MemoryLimitPool::get_instance().external_used(),
            native_workers * stack_bytes);
  // Repeated access/initialization cannot charge the same process workers
  // again, and collection close does not refund live worker-stack capacity.
  const auto reserved = detail::NativeWorkerStackReservation().bytes();
  EXPECT_EQ(0, GlobalResource::Instance().initialize());
  EXPECT_EQ(reserved, detail::NativeWorkerStackReservation().bytes());
#endif

  const auto image_reserved = detail::NativeImageReservation().bytes();
  EXPECT_GE(ailego::MemoryLimitPool::get_instance().external_used(),
            image_reserved);
  EXPECT_EQ(0, GlobalResource::Instance().initialize());
  EXPECT_EQ(image_reserved, detail::NativeImageReservation().bytes());
  EXPECT_EQ(total_capacity,
            ailego::MemoryLimitPool::get_instance().process_rss_limit());
  EXPECT_GT(ailego::MemoryLimitPool::get_instance().process_rss(), 0u);

  const auto rocksdb_cache = GlobalResource::Instance().rocksdb_block_cache();
  const auto write_buffer_manager =
      GlobalResource::Instance().rocksdb_write_buffer_manager();
  ASSERT_NE(nullptr, rocksdb_cache);
  ASSERT_NE(nullptr, write_buffer_manager);
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  ASSERT_NE(nullptr, rocksdb_cache->memory_allocator());
  EXPECT_STREQ("ZvecMappedCacheAllocator",
               rocksdb_cache->memory_allocator()->Name());
#else
  EXPECT_EQ(nullptr, rocksdb_cache->memory_allocator());
#endif
  EXPECT_EQ(expected_rocksdb_capacity, rocksdb_cache->GetCapacity());
  EXPECT_EQ(GlobalResource::calculate_rocksdb_write_buffer_budget(
                expected_rocksdb_capacity),
            write_buffer_manager->buffer_size());
  EXPECT_TRUE(write_buffer_manager->cost_to_cache());
}

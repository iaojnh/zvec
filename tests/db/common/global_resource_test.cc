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
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <thread>
#include <ailego/utility/memory_helper.h>
#include <arrow/io/interfaces.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/io/file.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include "db/common/native_image_memory.h"
#include "db/common/native_worker_memory.h"

using namespace zvec;

#if (defined(__linux__) || defined(__APPLE__) || defined(_WIN32)) && \
    GTEST_HAS_DEATH_TEST
namespace {

class DeferredConfigValidation : public GlobalConfig::ConsoleLogConfig {
 public:
  DeferredConfigValidation(std::promise<void> &entered,
                           std::future<void> resume)
      : entered_(entered), resume_(std::move(resume)) {}

  std::string get_logger_type() const override {
    entered_.set_value();
    resume_.wait();
    return GlobalConfig::ConsoleLogConfig::get_logger_type();
  }

 private:
  std::promise<void> &entered_;
  std::future<void> resume_;
};

}  // namespace

TEST(GlobalResourceStartupDeathTest, ExplicitStartupOwnsLazyPublication) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        GlobalConfig::ConfigData config;
        config.query_thread_binding = true;
        config.query_thread_count = config.optimize_thread_count = 1;
        std::promise<void> entered;
        std::promise<void> resume;
        auto validating = entered.get_future();
        config.log_config = std::make_shared<DeferredConfigValidation>(
            entered, resume.get_future());
        auto explicit_startup = std::async(std::launch::async, [&config] {
          return GlobalConfig::Instance().initialize(config);
        });
        validating.wait();
        auto lazy_startup = std::async(std::launch::async, [] {
          auto &resources = GlobalResource::Instance();
          return resources.initialize() == 0 &&
                 GlobalConfig::Instance().query_thread_binding() &&
                 resources.query_thread_pool()->count() == 1;
        });
        const bool waited =
            lazy_startup.wait_for(std::chrono::milliseconds(250)) ==
            std::future_status::timeout;
        resume.set_value();
        if (!explicit_startup.get().ok()) std::_Exit(1);
        if (!lazy_startup.get() || !waited) std::_Exit(2);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}

TEST(GlobalResourceStartupDeathTest, UnusedArrowCapacityDoesNotBlockStartup) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        // Raising idle pool capacities does not start Arrow workers. Their
        // configured upper bounds must not consume Zvec's startup budget.
        if (!arrow::SetCpuThreadPoolCapacity(4096).ok() ||
            !arrow::io::SetIOThreadPoolCapacity(4096).ok())
          std::_Exit(1);
        GlobalConfig::ConfigData config;
        config.memory_limit_bytes = 512ULL << 20;
        config.query_thread_count = config.optimize_thread_count = 1;
        if (!GlobalConfig::Instance().initialize(config).ok()) std::_Exit(2);
        auto &resources = GlobalResource::Instance();
        auto &pool = ailego::MemoryLimitPool::get_instance();
        if (!resources.initialized() || !resources.query_thread_pool() ||
            !resources.optimize_thread_pool() ||
            pool.process_rss_limit() != config.memory_limit_bytes)
          std::_Exit(3);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}

TEST(GlobalResourceStartupDeathTest,
     PartialBudgetFailureCannotBecomeStandalone) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        constexpr size_t limit = 128UL << 20;
        if (pool.init(limit)) std::_Exit(1);
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
            resources.query_thread_pool())
          std::_Exit(4);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}

TEST(GlobalResourceStartupDeathTest, ExplicitStartupCompletesLazyMemoryPolicy) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  for (const bool mmap : {false, true}) {
    EXPECT_EXIT(
        {
          auto &pool = ailego::MemoryLimitPool::get_instance();
          GlobalConfig::ConfigData config;
          config.memory_limit_bytes = 512ULL << 20;
          if (pool.init(config.memory_limit_bytes)) std::_Exit(1);
          const auto path = std::filesystem::temp_directory_path() /
                            ("zvec_late_rss_" +
                             std::to_string(std::chrono::steady_clock::now()
                                                .time_since_epoch()
                                                .count()));
          CollectionSchema schema("late_rss");
          if (!schema
                   .add_field(std::make_shared<FieldSchema>(
                       "payload", DataType::STRING, false))
                   .ok())
            std::_Exit(2);
          auto created = Collection::CreateAndOpen(path.u8string(), schema,
                                                   {false, mmap, 1024 * 1024});
          if (!created) std::_Exit(3);
          auto &resources = GlobalResource::Instance();
          auto *query = resources.query_thread_pool();
          auto *optimize = resources.optimize_thread_pool();
          if (!query || !optimize || pool.process_rss_limit() != 0)
            std::_Exit(4);
          if (!GlobalConfig::Instance().initialize(config).ok()) std::_Exit(5);
          if (pool.process_rss_limit() != config.memory_limit_bytes ||
              detail::NativeWorkerStackReservation().bytes() == 0 ||
              resources.query_thread_pool() != query ||
              resources.optimize_thread_pool() != optimize)
            std::_Exit(6);
          const size_t reserved =
              detail::NativeWorkerStackReservation().bytes();
          if (!GlobalConfig::Instance().initialize(config).ok() ||
              resources.initialize() ||
              detail::NativeWorkerStackReservation().bytes() != reserved)
            std::_Exit(7);
          if (!GlobalConfig::Instance()
                   .set_memory_pressure(GlobalConfig::MemoryPressure::kCritical)
                   .ok())
            std::_Exit(8);
          bool refused = false;
          try {
            ailego::ThreadPool temporary(1, false);
          } catch (const std::bad_alloc &) {
            refused = true;
          }
          if (!refused) std::_Exit(9);
          if (!GlobalConfig::Instance()
                   .set_memory_pressure(GlobalConfig::MemoryPressure::kNormal)
                   .ok())
            std::_Exit(10);
          if (!created.value()->close().ok()) std::_Exit(11);
          std::filesystem::remove_all(path);
          std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), "");
  }
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}

TEST(GlobalResourceStartupDeathTest, RealResidentPressureRefusesWorkerUntilReleased) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        constexpr size_t kMiB = 1024 * 1024;
        constexpr size_t held_bytes = 64 * kMiB;
        if (pool.init(128 * kMiB) != 0) std::_Exit(1);
        const size_t baseline = ailego::MemoryHelper::SelfRSS();
        auto *memory = static_cast<char *>(ailego::File::MemoryMap(held_bytes, 0));
        if (!memory) std::_Exit(2);
        // Fault in real pages outside the cache budget, as a caller could.
        auto *resident = static_cast<volatile char *>(memory);
        const size_t page = ailego::MemoryHelper::PageSize();
        for (size_t offset = 0; offset < held_bytes; offset += page)
          resident[offset] = 1;
        const size_t loaded = ailego::MemoryHelper::SelfRSS();
        if (!baseline || loaded < baseline + held_bytes / 2) std::_Exit(3);
        if (!pool.configure_process_rss_limit(baseline + held_bytes / 2))
          std::_Exit(4);

        bool refused = false;
        try {
          ailego::ThreadPool worker(1, false);
        } catch (const std::bad_alloc &) {
          refused = true;
        }
        if (!refused) std::_Exit(5);
        if (!ailego::File::TryMemoryUnmap(memory, held_bytes)) std::_Exit(6);
        for (size_t attempt = 0; attempt < 60; ++attempt) {
          try {
            ailego::ThreadPool worker(1, false);
            bool executed = false;
            worker.execute_and_wait([&] { executed = true; });
            std::_Exit(executed ? 0 : 7);
          } catch (const std::bad_alloc &) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
          }
        }
        std::_Exit(8);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}

TEST(GlobalResourceStartupDeathTest, FailedLatePolicyKeepsPublishedPoolsAlive) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto &pool = ailego::MemoryLimitPool::get_instance();
        GlobalConfig::ConfigData config;
        config.memory_limit_bytes = 512ULL << 20;
        if (pool.init(config.memory_limit_bytes)) std::_Exit(1);
        auto &resources = GlobalResource::Instance();
        auto *query = resources.query_thread_pool();
        auto *optimize = resources.optimize_thread_pool();
        if (!query || !optimize) std::_Exit(2);
        const auto occupied = pool.available();
        if (!pool.try_charge_external(occupied)) std::_Exit(3);
        if (GlobalConfig::Instance().initialize(config).ok()) std::_Exit(4);
        pool.release_external(occupied);
        if (resources.query_thread_pool() != query ||
            resources.optimize_thread_pool() != optimize)
          std::_Exit(5);
        bool completed = false;
        query->execute_and_wait([&] { completed = true; });
        if (!completed) std::_Exit(6);
        if (GlobalConfig::Instance().initialize(config).ok()) std::_Exit(7);
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
  auto &pool = ailego::MemoryLimitPool::get_instance();
  EXPECT_EQ(config.memory_limit_bytes, pool.capacity());

#if defined(__APPLE__) || defined(__linux__) || defined(_WIN32)
  const auto stack_bytes = detail::DefaultNativeWorkerStackBytes();
  const auto native_workers = static_cast<size_t>(expected_query_workers) +
                              expected_optimize_workers +
                              detail::kRocksDbBackgroundThreads + 2;
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
  EXPECT_EQ(config.memory_limit_bytes,
            ailego::MemoryLimitPool::get_instance().process_rss_limit());
  EXPECT_GT(ailego::MemoryLimitPool::get_instance().process_rss(), 0u);

#if defined(__APPLE__) || defined(__linux__) || defined(_WIN32)
  auto &global_config = GlobalConfig::Instance();
  const auto &pressure = GlobalConfig::MemoryPressure::kCritical;
  ASSERT_TRUE(global_config.set_memory_pressure(pressure).ok());
  EXPECT_THROW((ailego::ThreadPool(1, false)), std::bad_alloc);
  ASSERT_TRUE(
      global_config.set_memory_pressure(GlobalConfig::MemoryPressure::kNormal)
          .ok());
  EXPECT_NO_THROW({
    ailego::ThreadPool temporary(1, false);
    EXPECT_EQ(1u, temporary.count());
  });
#endif
}

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/parallel/worker_memory.h>

namespace zvec::ailego {
struct VecBufferPoolTestPeer {
  static bool has_writeback_worker(const VecBufferPool &pool) {
    return pool.writeback_thread_.joinable();
  }
};

TEST(WorkerMemoryTest, PlatformStackAllowanceIsNonzero) {
  EXPECT_GT(DefaultNativeWorkerStackBytes(), 0u);
}

#if GTEST_HAS_DEATH_TEST
class WorkerMemoryDeathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = ::testing::FLAGS_gtest_death_test_style;
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  }
  void TearDown() override {
    ::testing::FLAGS_gtest_death_test_style = previous_;
  }
  static void Initialize() {
    auto &pool = MemoryLimitPool::get_instance();
    if (pool.init(64UL << 20) != 0 ||
        !pool.configure_process_rss_limit(1UL << 30))
      std::_Exit(1);
  }

 private:
  std::string previous_;
};

TEST_F(WorkerMemoryDeathTest, ConcurrentPoolsRetainAndReuseStartedHighWater) {
  EXPECT_EXIT(
      {
        Initialize();
        auto &pool = MemoryLimitPool::get_instance();
        const size_t before = pool.external_used();
        const size_t stack = DefaultNativeWorkerStackBytes();
        {
          ThreadPool first(1, false);
          if (pool.external_used() != before + stack) std::_Exit(2);
          ThreadPool second(1, false);
          if (pool.external_used() != before + 2 * stack) std::_Exit(3);
          bool ran = false;
          second.execute_and_wait([&] { ran = true; });
          if (!ran) std::_Exit(4);
        }
        if (pool.external_used() != before + 2 * stack) std::_Exit(5);
        {
          ThreadPool reused(1, false);
          if (pool.external_used() != before + 2 * stack) std::_Exit(6);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(WorkerMemoryDeathTest, CriticalPressureRejectsRetainedStackReuse) {
  EXPECT_EXIT(
      {
        Initialize();
        auto &pool = MemoryLimitPool::get_instance();
        {
          ThreadPool initial(1, false);
        }
        const size_t before = pool.external_used();
        pool.set_memory_pressure(ProcessRssBudget::PressureLevel::kCritical);
        bool rejected = false;
        try {
          ThreadPool denied(1, false);
        } catch (const std::bad_alloc &) {
          rejected = true;
        }
        if (!rejected || pool.external_used() != before) std::_Exit(2);
        pool.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
        {
          ThreadPool recovered(1, false);
        }
        if (pool.external_used() != before) std::_Exit(3);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(WorkerMemoryDeathTest, FailedCreationRefundsOnlyNeverStartedWorkers) {
  EXPECT_EXIT(
      {
        Initialize();
        auto &pool = MemoryLimitPool::get_instance();
        const size_t before = pool.external_used();
        const size_t stack = DefaultNativeWorkerStackBytes();
        {
          WorkerMemoryLease unstarted(3);
        }
        if (pool.external_used() != before) std::_Exit(2);
        {
          WorkerMemoryLease partial(3);
          partial.worker_started();
        }
        if (pool.external_used() != before + stack) std::_Exit(3);
        bool overflow = false;
        try {
          WorkerMemoryLease huge(std::numeric_limits<size_t>::max());
        } catch (const std::bad_alloc &) {
          overflow = true;
        }
        if (!overflow || pool.external_used() != before + stack) std::_Exit(4);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(WorkerMemoryDeathTest, BudgetExhaustionRejectsBeforeStartingPool) {
  EXPECT_EXIT(
      {
        Initialize();
        auto &pool = MemoryLimitPool::get_instance();
        const size_t before = pool.external_used();
        MemoryReservation occupied;
        occupied.add(pool.available());
        const size_t full = pool.external_used();
        bool rejected = false;
        try {
          ThreadPool denied(1, false);
        } catch (const std::bad_alloc &) {
          rejected = true;
        }
        if (!rejected || pool.external_used() != full) std::_Exit(2);
        occupied.retain_at_most(0);
        {
          ThreadPool recovered(1, false);
          if (pool.external_used() != before + DefaultNativeWorkerStackBytes())
            std::_Exit(3);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(WorkerMemoryDeathTest, NestedPoolRefusalLeavesParentUsable) {
  EXPECT_EXIT(
      {
        Initialize();
        auto &pool = MemoryLimitPool::get_instance();
        ThreadPool parent(1, false);
        MemoryReservation occupied;
        occupied.add(pool.available());
        bool rejected = false;
        parent.execute_and_wait([&] {
          try {
            ThreadPool nested(1, false);
          } catch (const std::bad_alloc &) {
            rejected = true;
          }
        });
        if (!rejected) std::_Exit(2);
        occupied.retain_at_most(0);
        bool ran = false;
        parent.execute_and_wait([&] {
          ThreadPool nested(1, false);
          nested.execute_and_wait([&] { ran = true; });
        });
        if (!ran) std::_Exit(3);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(WorkerMemoryDeathTest, StandalonePayloadBudgetKeepsExistingSemantics) {
  EXPECT_EXIT(
      {
        auto &pool = MemoryLimitPool::get_instance();
        if (pool.init(64UL << 20) != 0) std::_Exit(1);
        const size_t before = pool.external_used();
        {
          ThreadPool standalone(1, false);
        }
        if (pool.external_used() != before) std::_Exit(2);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(WorkerMemoryDeathTest, WritebackFallsBackToSynchronousPersistence) {
  EXPECT_EXIT(
      {
        Initialize();
        auto &memory = MemoryLimitPool::get_instance();
        const char *path = "worker_memory_writeback.bin";
        FILE *file = std::fopen(path, "wb");
        if (!file) std::_Exit(2);
        std::fclose(file);
        {
          MemoryReservation occupied;
          occupied.add(memory.available() -
                       VecBufferPool::metadata_bytes_for_page_count(0, true));
          VecBufferPool pool(path, true);
          if (pool.init() != 0 ||
              VecBufferPoolTestPeer::has_writeback_worker(pool))
            std::_Exit(3);
          occupied.retain_at_most(0);
          if (!pool.extend_file(kVectorPageSize)) std::_Exit(4);
          const char payload[] = "synchronous writeback survives stack refusal";
          if (pool.write_range(0, sizeof(payload), payload) != 0 ||
              pool.flush_all() != 0)
            std::_Exit(5);
        }
        file = std::fopen(path, "rb");
        if (!file) std::_Exit(6);
        char output[64]{};
        const size_t bytes = std::fread(output, 1, sizeof(output), file);
        std::fclose(file);
        std::remove(path);
        if (bytes != sizeof(output) ||
            std::string(output) !=
                "synchronous writeback survives stack refusal")
          std::_Exit(7);
        // Once budget is available, a new writable pool may start its worker.
        file = std::fopen(path, "wb");
        if (!file) std::_Exit(8);
        std::fclose(file);
        {
          VecBufferPool pool(path, true);
          if (pool.init() != 0 ||
              !VecBufferPoolTestPeer::has_writeback_worker(pool))
            std::_Exit(9);
        }
        std::remove(path);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}
#endif
}  // namespace zvec::ailego

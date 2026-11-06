// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {
namespace {
struct FailingIndexQueueTraits : moodycamel::ConcurrentQueueDefaultTraits {
  inline static size_t allocations_left = std::numeric_limits<size_t>::max();
  static void *malloc(size_t bytes) {
    if (allocations_left == 0) return nullptr;
    --allocations_left;
    return std::malloc(bytes);
  }
  static void free(void *ptr) {
    std::free(ptr);
  }
};

TEST(EvictionQueueRecoveryTest, InitialIndexFailureCanRetryWithoutNewProducer) {
  moodycamel::ConcurrentQueue<int, FailingIndexQueueTraits> queue(0);
  // Allow the producer itself, but fail its constructor's block index.
  FailingIndexQueueTraits::allocations_left = 1;
  EXPECT_FALSE(queue.enqueue(1));
  FailingIndexQueueTraits::allocations_left =
      std::numeric_limits<size_t>::max();
  const auto available = FailingIndexQueueTraits::allocations_left;
  EXPECT_FALSE(queue.try_enqueue(2));
  EXPECT_EQ(available, FailingIndexQueueTraits::allocations_left);
  ASSERT_TRUE(queue.enqueue(3));
  int value = 0;
  ASSERT_TRUE(queue.try_dequeue(value));
  EXPECT_EQ(3, value);
  EXPECT_FALSE(queue.try_dequeue(value));
}

TEST(EvictionQueueRecoveryTest, RecycledProducerRecoversItsMissingIndex) {
  moodycamel::ConcurrentQueue<int, FailingIndexQueueTraits> queue(0);
  std::thread first([&] {
    FailingIndexQueueTraits::allocations_left = 1;
    EXPECT_FALSE(queue.enqueue(1));
  });
  first.join();
  FailingIndexQueueTraits::allocations_left =
      std::numeric_limits<size_t>::max();
  std::thread second([&] {
    const int values[] = {2, 3};
    EXPECT_TRUE(queue.enqueue_bulk(values, 2));
  });
  second.join();
  int value = 0;
  ASSERT_TRUE(queue.try_dequeue(value));
  EXPECT_EQ(2, value);
  ASSERT_TRUE(queue.try_dequeue(value));
  EXPECT_EQ(3, value);
  EXPECT_FALSE(queue.try_dequeue(value));
}

class EvictionQueueBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(64UL << 20));
    ASSERT_EQ(pool_.queue_metadata_used(), pool_.used());
  }
  void TearDown() override {
    EXPECT_EQ(pool_.queue_allocated(), pool_.queue_metadata_used());
    EXPECT_EQ(pool_.queue_metadata_used(), pool_.metadata_used());
    EXPECT_EQ(pool_.queue_metadata_used(), pool_.used());
    EXPECT_LE(pool_.committed(), pool_.capacity());
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
};

TEST_F(EvictionQueueBudgetTest, InitialQueuesConsumeTheConfiguredTotalBudget) {
  const size_t queue = pool_.queue_metadata_used();
  EXPECT_GT(queue, 9UL << 20);
  EXPECT_LT(queue, 12UL << 20);
  EXPECT_EQ(64UL << 20, pool_.capacity());
  EXPECT_EQ(pool_.capacity() - queue, pool_.available());
  const auto stats = pool_.stats();
  EXPECT_EQ(queue, stats.queue_metadata_used);
  EXPECT_EQ(queue, stats.queue_allocated);
  EXPECT_EQ(queue, stats.metadata_used);
  EXPECT_EQ(0u, stats.page_used);
  EXPECT_EQ(0u, stats.external_used);
  EXPECT_EQ(queue, stats.committed);
}

TEST_F(EvictionQueueBudgetTest, InsufficientInitializationPreservesOldBudget) {
  const size_t queue = pool_.queue_allocated();
  EXPECT_NE(0, pool_.init(queue - 1));
  EXPECT_EQ(64UL << 20, pool_.capacity());
  EXPECT_EQ(queue, pool_.used());
  EXPECT_EQ(0, pool_.init(queue));
  EXPECT_EQ(0u, pool_.available());
  EXPECT_FALSE(pool_.try_charge_external(1));
  EXPECT_EQ(0, pool_.init(64UL << 20));
}

TEST_F(EvictionQueueBudgetTest,
       DisabledAndReinitializedPoolAdoptsRetainedQueues) {
  const size_t queue = pool_.queue_allocated();
  ASSERT_EQ(0, pool_.init(0));
  EXPECT_EQ(queue, pool_.queue_allocated());
  EXPECT_EQ(0u, pool_.queue_metadata_used());
  {
    BlockEvictionQueue::ConcurrentQueue extra(128);
    const size_t grown = pool_.queue_allocated();
    EXPECT_GT(grown, queue);
    ASSERT_EQ(0, pool_.init(64UL << 20));
    EXPECT_EQ(grown, pool_.used());
    EXPECT_EQ(grown, pool_.queue_metadata_used());
    ASSERT_EQ(0, pool_.init(64UL << 20));
    EXPECT_EQ(grown, pool_.used());
  }
  EXPECT_EQ(queue, pool_.used());
}

TEST_F(EvictionQueueBudgetTest, ActiveConsumerStillPreventsBudgetChanges) {
  const size_t queue = pool_.queue_metadata_used();
  ASSERT_TRUE(pool_.try_charge_external(4096));
  EXPECT_EQ(0, pool_.init(64UL << 20));
  EXPECT_NE(0, pool_.init(32UL << 20));
  EXPECT_NE(0, pool_.init(0));
  EXPECT_EQ(64UL << 20, pool_.capacity());
  EXPECT_EQ(queue + 4096, pool_.used());
  EXPECT_EQ(queue, pool_.queue_metadata_used());
  pool_.release_external(4096);
  EXPECT_EQ(0, pool_.init(32UL << 20));
  EXPECT_EQ(queue, pool_.used());
  EXPECT_EQ(0, pool_.init(64UL << 20));
}

TEST_F(EvictionQueueBudgetTest,
       GrowthFailsWithoutRecursiveEvictionAndCanRetry) {
  const size_t baseline = pool_.used();
  {
    BlockEvictionQueue::ConcurrentQueue queue(32);
    const size_t initial = pool_.used();
    EXPECT_GT(initial, baseline);
    BlockEvictionQueue::BlockType item;
    item.owner_key = 19;
    const size_t pressure = pool_.available();
    ASSERT_TRUE(pool_.try_charge_external(pressure));
    EXPECT_FALSE(queue.enqueue(item));
    EXPECT_EQ(pool_.capacity(), pool_.used());
    pool_.release_external(pressure);
    ASSERT_TRUE(queue.enqueue(item));
    EXPECT_GT(pool_.used(), initial);  // Producer and index allocations.
    BlockEvictionQueue::BlockType result;
    ASSERT_TRUE(queue.try_dequeue(result));
    EXPECT_EQ(19u, result.owner_key);
  }
  EXPECT_EQ(baseline, pool_.used());
  EXPECT_EQ(baseline, pool_.committed());
}

TEST_F(EvictionQueueBudgetTest, DynamicBlocksCannotEscapeTheSharedBudget) {
  const size_t baseline = pool_.used();
  {
    BlockEvictionQueue::ConcurrentQueue queue(32);
    const size_t pressure = pool_.available() - (128UL << 10);
    ASSERT_TRUE(pool_.try_charge_external(pressure));
    size_t accepted = 0;
    for (; accepted < 16384; ++accepted) {
      BlockEvictionQueue::BlockType item;
      item.owner_key = accepted;
      if (!queue.enqueue(item)) break;
    }
    EXPECT_GT(accepted, 32u);
    EXPECT_LT(accepted, 16384u);
    EXPECT_LE(pool_.used(), pool_.capacity());
    EXPECT_LE(pool_.committed(), pool_.capacity());
    for (size_t i = 0; i < accepted; ++i) {
      BlockEvictionQueue::BlockType item;
      ASSERT_TRUE(queue.try_dequeue(item));
      EXPECT_EQ(i, item.owner_key);
    }
    pool_.release_external(pressure);
    ASSERT_TRUE(queue.enqueue({}));
  }
  EXPECT_EQ(baseline, pool_.used());
}

TEST_F(EvictionQueueBudgetTest, OldSlabCannotTouchNewPagesUnderPressure) {
  const size_t page = MemoryLimitPool::page_buffer_size();
  if (page >= (64UL << 10)) GTEST_SKIP() << "Queue slabs fit in one page";
  constexpr size_t bytes = 3001;
  std::vector<void *> slots;
  using Pressure = ProcessRssBudget::PressureLevel;
  const auto cleanup = ScopeGuard::Make([&] {
    pool_.set_memory_pressure(Pressure::kNormal);
    for (void *slot : slots) EvictionQueueTraits::free(slot);
  });

  void *first = EvictionQueueTraits::malloc(bytes);
  ASSERT_NE(nullptr, first);
  slots.push_back(first);
  const size_t committed = pool_.committed();
  pool_.set_memory_pressure(Pressure::kCritical);
  bool denied = false;
  for (size_t i = 0; i < page / bytes + 1; ++i) {
    void *slot = EvictionQueueTraits::malloc(bytes);
    if (!slot) {
      denied = true;
      break;
    }
    slots.push_back(slot);
  }
  EXPECT_TRUE(denied);
  EXPECT_EQ(committed, pool_.committed());

  pool_.set_memory_pressure(Pressure::kNormal);
  void *slot = EvictionQueueTraits::malloc(bytes);
  ASSERT_NE(nullptr, slot);
  slots.push_back(slot);
  EXPECT_EQ(committed, pool_.committed());
  EvictionQueueTraits::free(slot);
  slots.pop_back();
  pool_.set_memory_pressure(Pressure::kCritical);
  void *reused = EvictionQueueTraits::malloc(bytes);
  EXPECT_EQ(slot, reused);
  if (reused) slots.push_back(reused);
}

TEST_F(EvictionQueueBudgetTest, ConcurrentAllocationAndCrossThreadFreeBalance) {
  const size_t baseline = pool_.used();
  std::atomic<size_t> ready{0};
  void *buffers[8]{};
  std::thread threads[8];
  for (size_t i = 0; i < 8; ++i) {
    threads[i] = std::thread([&, i] {
      buffers[i] = EvictionQueueTraits::malloc(1UL << 20);
      ready.fetch_add(1);
      while (ready.load() != 8) std::this_thread::yield();
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_GT(pool_.used(), baseline + (8UL << 20));
  for (void *buffer : buffers) {
    EXPECT_NE(nullptr, buffer);
    EvictionQueueTraits::free(buffer);
  }
  EXPECT_EQ(baseline, pool_.used());
  EXPECT_EQ(nullptr,
            EvictionQueueTraits::malloc(std::numeric_limits<size_t>::max()));
  EXPECT_EQ(baseline, pool_.used());
}
}  // namespace
}  // namespace zvec::ailego

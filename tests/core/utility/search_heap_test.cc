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
#include "utility/search_heap.h"
#include <atomic>
#include <future>
#include <thread>
#include <type_traits>
#include <vector>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>

namespace zvec::core {
namespace {

using TopkHeap = SearchHeap::TopkHeap;

TEST(SearchHeap, MandatoryWorkspaceWaitsForPendingPageRelease) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  // Non-page-sized storage is released directly, without any free-list slack.
  constexpr size_t capacity = 64 * 1024 + 1;
  ASSERT_EQ(0, pool.init(capacity));
  char *buffer = nullptr;
  ASSERT_TRUE(pool.try_acquire_buffer(capacity, buffer));
  const auto waits = pool.stats().capacity_waits;
  EXPECT_FALSE(pool.try_charge_external(1024));
  EXPECT_EQ(waits, pool.stats().capacity_waits);

  std::atomic<bool> done{false};
  std::promise<void> ready;
  auto ready_future = ready.get_future();
  std::thread release([&] {
    ready.set_value();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!done.load() && pool.stats().capacity_waits == waits &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    pool.release_buffer(buffer, capacity);
  });
  AILEGO_DEFER([&] {
    done.store(true);
    release.join();
  });
  ready_future.wait();
  ailego::BudgetedAllocator<char> allocator;
  char *workspace = nullptr;
  EXPECT_NO_THROW(workspace = allocator.allocate(1024));
  EXPECT_NE(nullptr, workspace);
  EXPECT_EQ(waits + 1, pool.stats().capacity_waits);
  if (workspace) allocator.deallocate(workspace, 1024);
}

TEST(SearchHeap, MandatoryWorkspaceRejectsUnreclaimableBudget) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  constexpr size_t capacity = 64 * 1024 + 1;
  ASSERT_EQ(0, pool.init(capacity));
  ASSERT_TRUE(pool.try_charge_external(capacity));
  auto waits = pool.stats().capacity_waits;
  EXPECT_FALSE(pool.try_charge_workspace(1024));
  EXPECT_EQ(waits, pool.stats().capacity_waits);
  pool.release_external(capacity);

  char *buffer = nullptr;
  ASSERT_TRUE(pool.try_acquire_buffer(capacity, buffer));
  const auto timeouts = pool.stats().capacity_wait_timeouts;
  EXPECT_FALSE(pool.try_charge_workspace(1024));
  EXPECT_EQ(waits + 1, pool.stats().capacity_waits);
  EXPECT_EQ(timeouts + 1, pool.stats().capacity_wait_timeouts);
  pool.release_buffer(buffer, capacity);
  EXPECT_TRUE(pool.try_charge_workspace(1024));
  pool.release_external(1024);
}

TEST(SearchHeap, ReusesSelectedTopkAndExportsInDistanceOrder) {
  SearchHeap heap;
  auto &topk = heap.reset<TopkHeap>(3);
  topk.emplace(30, 3.0f);
  topk.emplace(10, 1.0f);
  topk.emplace(20, 2.0f);
  const void *storage = topk.container().data();

  std::vector<uint32_t> ids;
  heap.for_each_sorted(2, [&](uint32_t id, float) {
    ids.push_back(id);
    return true;
  });
  EXPECT_EQ((std::vector<uint32_t>{10, 20}), ids);

  auto &reset = heap.reset<TopkHeap>(2);
  EXPECT_EQ(&topk, &reset);
  EXPECT_EQ(storage, reset.container().data());
  EXPECT_TRUE(reset.empty());
  EXPECT_EQ(2U, reset.limit());
}

TEST(SearchHeap, ExportsLinearPoolWithoutMaterializingTopk) {
  SearchHeap heap;
  auto &pool = heap.reset<LinearPool<float>>(3, 4);
  const uint32_t input_ids[] = {30, 10, 20, 40};
  const float distances[] = {3.0f, 1.0f, 2.0f, 4.0f};
  pool.push_block(distances, input_ids, 4);

  std::vector<uint32_t> ids;
  heap.for_each_sorted(3, [&](uint32_t id, float) {
    ids.push_back(id);
    return true;
  });
  EXPECT_EQ((std::vector<uint32_t>{10, 20, 30}), ids);

  heap.dispatch([&](auto &active) {
    EXPECT_TRUE(
        (std::is_same_v<std::decay_t<decltype(active)>, LinearPool<float>>));
    EXPECT_EQ(static_cast<void *>(&pool), static_cast<void *>(&active));
  });
}

TEST(SearchHeap, SelectsPoolBackendFromCpuFeatures) {
  SearchHeap heap;
  heap.reset_pool(4, 8);
  const bool expect_block = ailego::internal::CpuFeatures::static_flags_.AVX2;

  heap.dispatch([&](auto &active) {
    using Heap = std::decay_t<decltype(active)>;
    EXPECT_EQ(expect_block, (std::is_same_v<Heap, BlockHeap>));
    EXPECT_EQ(!expect_block, (std::is_same_v<Heap, LinearPool<float>>));
    EXPECT_EQ(0, active.size());
  });

  heap.reset_pool(2, 8);
  EXPECT_EQ(0U, heap.size());
}

TEST(SearchHeap, LazyCandidateAdmissionRejectsAndReleasesBudget) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(64 * 1024));
  {
    ailego::BudgetedKeyValueHeap<uint32_t, float, std::greater<float>> heap;
    heap.limit(1000000);
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0u, heap.container().capacity());
    EXPECT_THROW(
        {
          for (uint32_t i = 0; i < 1000000; ++i) heap.emplace(i, float(i));
        },
        std::bad_alloc);
    EXPECT_GT(pool.used(), 0u);
    EXPECT_LE(pool.used(), pool.capacity());
    heap.release();
    EXPECT_EQ(0u, pool.used());
    heap.emplace(1, 1.0f);
    EXPECT_EQ(1u, heap.size());
  }
}

TEST(SearchHeap, AllBackendsAreChargedAndRetryAfterRejection) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(64 * 1024));
  {
    SearchHeap heap;
    EXPECT_THROW(heap.reset<LinearPool<float>>(100000, 32), std::bad_alloc);
    heap.reset<LinearPool<float>>(8, 32);
    EXPECT_GT(pool.used(), 0u);
    EXPECT_THROW(heap.reset<BlockHeap>(100000, 32), std::bad_alloc);
    heap.reset<BlockHeap>(8, 32);
    EXPECT_GT(pool.used(), 0u);
    auto &topk = heap.reset<TopkHeap>(1000000);
    EXPECT_EQ(0u, pool.used());
    topk.emplace(1, 1.0f);
    EXPECT_GT(pool.used(), 0u);
  }
}

TEST(SearchHeap, RareLargeHeapDoesNotStayAtHighWaterMark) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(1024 * 1024));
  {
    TopkHeap heap(20000);
    for (uint32_t i = 0; i < 10000; ++i) heap.emplace(i, float(i));
    EXPECT_GT(pool.used(), 64000u);
    heap.clear();
    EXPECT_EQ(0u, pool.used());
  }
}
}  // namespace
}  // namespace zvec::core

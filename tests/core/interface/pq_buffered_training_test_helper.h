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

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <type_traits>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <turbo/distance/common/fast_scan_common.h>
#include <turbo/quantizer/common/pq_quantizer/pq_opq.h>
#include <turbo/quantizer/pq_fast_quantizer/pq_fast_quantizer.h>
#include <turbo/quantizer/pq_int4_quantizer/pq_int4_quantizer.h>
#include <turbo/quantizer/pq_int8_quantizer/pq_int8_quantizer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_holder.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core_interface {
namespace {
class PqBufferedTrainingTest : public ::testing::Test {
 protected:
  size_t capacity_{0};
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    ASSERT_EQ(pool.init(0), 0);
    // PoolPayloadBudget excludes persistent queue allocator metadata from
    // the matrix/worker payload allowance. Materialize both worker producers
    // before taking that baseline; their platform-dependent headers must not
    // consume the training tile's reserved I/O pages later.
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    std::atomic<size_t> ready{0};
    std::atomic<bool> release{false}, admitted{true};
    std::vector<std::thread> workers;
    auto finish = ailego::ScopeGuard::Make([&] {
      release.store(true, std::memory_order_release);
      for (auto &worker : workers)
        if (worker.joinable()) worker.join();
    });
    for (size_t i = 0; i < 2; ++i) {
      workers.emplace_back([&] {
        for (size_t priority = 0;
             priority < ailego::BlockEvictionQueue::kQueueCount; ++priority) {
          if (!queue.add_single_block({}, static_cast<int>(priority)))
            admitted.store(false);
        }
        ready.fetch_add(1, std::memory_order_release);
        while (!release.load(std::memory_order_acquire))
          std::this_thread::yield();
      });
    }
    while (ready.load(std::memory_order_acquire) != workers.size())
      std::this_thread::yield();
    release.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();
    ASSERT_TRUE(admitted.load());
    ailego::BlockEvictionQueue::BlockType stale;
    while (queue.evict_single_block(stale)) {
    }
  }
  void TearDown() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(capacity_), 0);
  }
};
}  // namespace
}  // namespace zvec::core_interface

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

#include <cstdint>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_budget.h>

extern "C" {
void *zvec_test_dso_memory_budget_manager();
void *zvec_test_dso_memory_limit_pool();
void *zvec_test_dso_block_eviction_queue();
bool zvec_test_dso_charge_query_memory(uint64_t bytes);
void zvec_test_dso_release_query_memory(uint64_t bytes);
}

namespace zvec {
namespace ailego {

TEST(CrossDsoSingletonTest, SharesAddressesAndAccountingState) {
  auto &budget = MemoryBudgetManager::get_instance();
  auto &pool = MemoryLimitPool::get_instance();
  auto &eviction_queue = BlockEvictionQueue::get_instance();

  EXPECT_EQ(&budget, zvec_test_dso_memory_budget_manager());
  EXPECT_EQ(&pool, zvec_test_dso_memory_limit_pool());
  EXPECT_EQ(&eviction_queue, zvec_test_dso_block_eviction_queue());

  MemoryBudgetManager::Config config;
  config.total_bytes = 1024;
  config.enforce_accounting = true;
  config.query_working_bytes = 1024;
  ASSERT_TRUE(budget.configure(config));

  ASSERT_TRUE(zvec_test_dso_charge_query_memory(256));
  EXPECT_EQ(256u, budget.used(MemoryBudgetManager::Category::QueryWorking));
  zvec_test_dso_release_query_memory(256);
  EXPECT_EQ(0u, budget.used(MemoryBudgetManager::Category::QueryWorking));
}

}  // namespace ailego
}  // namespace zvec

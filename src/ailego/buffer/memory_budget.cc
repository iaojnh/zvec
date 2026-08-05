// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <zvec/ailego/buffer/memory_budget.h>

namespace zvec {
namespace ailego {

MemoryBudgetManager &MemoryBudgetManager::get_instance() {
  auto &instance = instance_slot();
  MemoryBudgetManager *current = instance.load(std::memory_order_acquire);
  if (current != nullptr) {
    return *current;
  }

  MemoryBudgetManager *created = new MemoryBudgetManager();
  if (instance.compare_exchange_strong(current, created,
                                       std::memory_order_acq_rel,
                                       std::memory_order_acquire)) {
    return *created;
  }
  delete created;
  return *current;
}

}  // namespace ailego
}  // namespace zvec

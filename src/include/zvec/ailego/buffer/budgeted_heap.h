// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/container/heap.h>

namespace zvec::ailego {
// Search scratch is not evictable while in use. Grow only as candidates arrive
// and fail allocation before exceeding the shared budget.
template <typename T, typename Compare = std::less<T>>
class BudgetedHeap
    : public Heap<T, Compare, std::vector<T, BudgetedAllocator<T>>> {
 public:
  using Container = std::vector<T, BudgetedAllocator<T>>;
  using Base = Heap<T, Compare, Container>;
  using Base::Base;
  size_t limit() const {
    return Base::limit();
  }
  BudgetedHeap() = default;
  explicit BudgetedHeap(size_t max) {
    limit(max);
  }
  void limit(size_t max, bool reserve = false) {
    Base::limit(max, reserve);
  }
  void clear() {
    auto &items = this->mutable_container();
    items.clear();
    if (items.capacity() > 4096) Container().swap(items);
  }
  void release() {
    Container().swap(this->mutable_container());
  }
};

template <typename Key, typename Value, typename Compare = std::less<Value>>
using BudgetedKeyValueHeap =
    BudgetedHeap<std::pair<Key, Value>,
                 KeyValueHeapComparer<Key, Value, Compare>>;
}  // namespace zvec::ailego

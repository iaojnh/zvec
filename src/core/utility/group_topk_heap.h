// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <map>
#include <string>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/container/heap.h>
#include "group_key.h"

namespace zvec::core {
template <typename Key, typename Score>
class GroupTopkHeap
    : public ailego::Heap<
          std::pair<Key, Score>, ailego::KeyValueHeapComparer<Key, Score>,
          std::vector<std::pair<Key, Score>,
                      ailego::BudgetedAllocator<std::pair<Key, Score>>>> {
  using Base = ailego::Heap<
      std::pair<Key, Score>, ailego::KeyValueHeapComparer<Key, Score>,
      std::vector<std::pair<Key, Score>,
                  ailego::BudgetedAllocator<std::pair<Key, Score>>>>;

 public:
  void limit(size_t count) {
    Base::limit(count, false);
  }
};

template <typename Key, typename Score>
using GroupTopkMap =
    std::map<GroupKey, GroupTopkHeap<Key, Score>, std::less<>,
             ailego::BudgetedAllocator<
                 std::pair<const GroupKey, GroupTopkHeap<Key, Score>>>>;

// Same map iteration and score-only tie ordering as the original path, but
// point at selected heaps rather than copying candidate buffers and strings.
template <typename Map>
auto SelectGroupHeaps(Map &groups, size_t count) {
  ailego::BudgetedVector<typename Map::value_type *> result;
  for (auto &group : groups) {
    group.second.sort();
    if (!group.second.empty()) result.push_back(&group);
  }
  std::sort(result.begin(), result.end(), [](const auto *a, const auto *b) {
    return a->second[0].second < b->second[0].second;
  });
  if (result.size() > count) result.resize(count);
  return result;
}
}  // namespace zvec::core

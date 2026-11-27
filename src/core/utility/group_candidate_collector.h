// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <zvec/core/framework/index_document.h>
#include "appendable_build_buffer.h"
#include "group_key.h"

namespace zvec::core {

// Keep only the requested groups' best scores in RAM. Replay a pageable log
// after selection, so a group that leaves and later re-enters the winners does
// not lose its earlier runner-up documents. No second vector scan is needed.
class GroupCandidateCollector {
 public:
  GroupCandidateCollector(size_t groups, size_t topk)
      : groups_(groups),
        topk_(topk),
        heap_only_(ailego::MemoryLimitPool::get_instance().capacity() == 0) {
    status_ = data_.init(0);
  }

  int add(const std::string &group, uint64_t key, float score,
          uint32_t index = 0) try {
    if (status_ || !groups_ || std::isnan(score)) return status_;
    if (group.size() > std::numeric_limits<uint32_t>::max())
      return status_ = IndexError_InvalidArgument;
    auto found = best_.find(group);
    if (found != best_.end()) {
      if (score < found->second) {
        ranked_.erase(&*found);
        found->second = score;
        ranked_.emplace(&*found);
      }
    } else if (best_.size() < groups_ ||
               (score < (*ranked_.rbegin())->second ||
                (score == (*ranked_.rbegin())->second &&
                 group < (*ranked_.rbegin())->first))) {
      if (best_.size() == groups_) {
        auto last = std::prev(ranked_.end());
        auto best = best_.find((*last)->first);
        ranked_.erase(last);
        best_.erase(best);
      }
      auto inserted = best_.emplace(group, score);
      ranked_.emplace(&*inserted.first);
    }
    if (!spilling_) {
      auto heap_entry = heap_groups_.find(group);
      if (heap_entry == heap_groups_.end()) {
        heap_entry = heap_groups_.try_emplace(group).first;
        string_bytes_ += group.size();
      }
      auto &heap = heap_entry->second;
      if (heap.empty()) heap.limit(topk_);
      size_t before = heap.size();
      heap.emplace(key, score, index);
      entries_ += heap.size() - before;
      // Small group queries do not need a temporary file. The working set is
      // bounded independently of cardinality and the caller's per-group topk.
      if (heap_only_ || (heap_groups_.size() <= 64 && entries_ <= 4096 &&
                         string_bytes_ <= kBatch))
        return status_;
      spilling_ = true;
      staging_.resize(kBatch);
      for (const auto &entry : heap_groups_) {
        for (const auto &item : entry.second.container()) {
          Record record{item.key(), item.score(), item.index(),
                        static_cast<uint32_t>(entry.first.size())};
          append(&record, sizeof(record));
          append(entry.first.data(), entry.first.size());
        }
      }
      heap_groups_.clear();
      return status_;
    }
    Record record{key, score, index, static_cast<uint32_t>(group.size())};
    append(&record, sizeof(record));
    append(group.data(), group.size());
    return status_;
  } catch (const std::bad_alloc &) {
    return status_ = IndexError_NoMemory;
  } catch (const std::exception &) {
    return status_ = IndexError_Runtime;
  }

  template <typename Consumer>
  int finish(const Consumer &consume) try {
    if (status_) return status_;
    if (!spilling_) {
      for (const auto &group : best_)
        for (const auto &item : heap_groups_.at(group.first).container())
          consume(group.first, item.key(), item.score(), item.index());
      return status_;
    }
    flush();
    if (status_) return status_;
    size_t offset = 0, cursor = 0, loaded = 0;
    auto read = [&](void *out, size_t bytes) {
      auto *dest = static_cast<char *>(out);
      while (bytes && !status_) {
        if (cursor == loaded) {
          loaded = std::min(kBatch, data_.size() - offset);
          if (!loaded) {
            status_ = IndexError_ReadData;
            break;
          }
          status_ = data_.read(offset, staging_.data(), loaded);
          offset += loaded;
          cursor = 0;
        }
        size_t n = std::min(bytes, loaded - cursor);
        std::memcpy(dest, staging_.data() + cursor, n);
        cursor += n;
        dest += n;
        bytes -= n;
      }
    };
    ailego::MemoryReservation replay_memory;
    std::string group;
    while (!status_ && (offset < data_.size() || cursor < loaded)) {
      Record record{};
      read(&record, sizeof(record));
      if (status_) break;
      if (record.length > group.capacity()) {
        ailego::MemoryReservation replacement_memory;
        replacement_memory.add_array(static_cast<size_t>(record.length) + 1, 2);
        std::string replacement(record.length, '\0');
        replacement_memory.retain_at_most(replacement.capacity() + 1);
        group.swap(replacement);
        replay_memory.swap(replacement_memory);
      } else {
        group.resize(record.length);
      }
      read(group.data(), group.size());
      if (!status_ && best_.find(group) != best_.end())
        consume(group, record.key, record.score, record.index);
    }
    return status_;
  } catch (const std::bad_alloc &) {
    return status_ = IndexError_NoMemory;
  } catch (const std::exception &) {
    return status_ = IndexError_Runtime;
  }

  size_t retained_groups() const {
    return best_.size();
  }
  bool paged() const {
    return spilling_ && data_.paged();
  }
  int status() const {
    return status_;
  }
  // Preserve per-group distance pruning while the bounded fast path is live.
  float threshold_for(const std::string &group, float threshold) const {
    auto it = heap_groups_.find(group);
    if (it != heap_groups_.end() && it->second.full())
      return std::min(threshold, it->second.front().score());
    return threshold;
  }
  template <typename Map>
  int finish_heaps(Map &heaps, size_t topk) {
    heaps.clear();
    return finish(
        [&](const std::string &group, uint64_t key, float score, uint32_t) {
          auto entry = heaps.find(group);
          if (entry == heaps.end()) entry = heaps.try_emplace(group).first;
          auto &heap = entry->second;
          if (heap.empty()) heap.limit(topk);
          heap.emplace(key, score);
        });
  }

 private:
  struct Record {
    uint64_t key;
    float score;
    uint32_t index;
    uint32_t length;
  };
  void append(const void *data, size_t bytes) {
    auto *source = static_cast<const char *>(data);
    while (bytes && !status_) {
      const size_t n = std::min(bytes, kBatch - used_);
      std::memcpy(staging_.data() + used_, source, n);
      used_ += n;
      source += n;
      bytes -= n;
      if (used_ == kBatch) flush();
    }
  }
  void flush() {
    if (!used_ || status_) return;
    const size_t old = data_.size();
    if (used_ > std::numeric_limits<size_t>::max() - old) {
      status_ = IndexError_Overflow;
      return;
    }
    status_ = data_.resize(old + used_);
    if (!status_) status_ = data_.write(old, staging_.data(), used_);
    used_ = 0;
  }
  static constexpr size_t kBatch = 64 * 1024;
  size_t groups_;
  size_t topk_;
  bool heap_only_;
  bool spilling_{false};
  size_t entries_{0}, string_bytes_{0};
  template <typename Key, typename Value>
  using Map = std::map<Key, Value, std::less<>,
                       ailego::BudgetedAllocator<std::pair<const Key, Value>>>;
  Map<GroupKey, IndexCandidateHeap> heap_groups_;
  using BestMap = Map<GroupKey, float>;
  using RankedEntry = const BestMap::value_type *;
  struct RankLess {
    bool operator()(RankedEntry a, RankedEntry b) const {
      return a->second < b->second ||
             (a->second == b->second && a->first < b->first);
    }
  };
  BestMap best_;
  std::set<RankedEntry, RankLess, ailego::BudgetedAllocator<RankedEntry>>
      ranked_;
  AppendableBuildBuffer data_;
  std::vector<char, ailego::BudgetedAllocator<char>> staging_;
  size_t used_{0};
  int status_{0};
};
}  // namespace zvec::core

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <map>
#include <set>
#include <string_view>
#include "core/utility/appendable_build_buffer.h"
#include "topk_candidates.h"

namespace zvec::sqlengine {

// Group ranking depends on its best document. A group may leave and re-enter
// the winners, so dropping its earlier documents would lose runner-up results.
// Replay compact references from pageable storage after selecting the groups;
// neither all groups' heaps nor their wide output fields need to stay in RAM.
class GroupTopKCandidates {
 public:
  template <typename T>
  using Vector = std::vector<T, ailego::BudgetedAllocator<T>>;
  using String = std::basic_string<char, std::char_traits<char>,
                                   ailego::BudgetedAllocator<char>>;
  GroupTopKCandidates(size_t groups, size_t topk, bool descending)
      : limit_(groups),
        topk_(topk),
        better_{descending},
        ranked_(RankLess{better_}) {}

  void offer(std::string_view group, RankedDoc row) {
    if (!limit_ || !topk_) return;
    if (group.size() > std::numeric_limits<uint32_t>::max())
      throw std::length_error("Group key is too long");
    auto found = best_.find(group);
    if (found != best_.end()) {
      if (better_(row, found->second)) {
        ranked_.erase(Rank{found->second, &found->first});
        found->second = row;
        ranked_.insert(Rank{row, &found->first});
      }
    } else if (best_.size() < limit_ || better_(row, ranked_.rbegin()->row)) {
      if (best_.size() == limit_) {
        auto last = std::prev(ranked_.end());
        auto remove = best_.find(*last->name);
        ranked_.erase(last);
        best_.erase(remove);
      }
      auto added = best_.emplace(String(group.data(), group.size()), row).first;
      ranked_.insert(Rank{row, &added->first});
    }
    if (staging_.empty()) staging_.resize(kBatch);
    Record record{row, static_cast<uint32_t>(group.size())};
    append(&record, sizeof(record));
    append(group.data(), group.size());
  }

  void finish() {
    if (best_.empty()) return;
    // Small candidate sets fit in the charged staging area and need no file.
    // Only larger merges spill/replay through the pool-backed scratch store.
    size_t offset = 0, cursor = 0, loaded = 0;
    if (data_.size())
      flush();
    else {
      loaded = used_;
      used_ = 0;
    }
    Map<TopKCandidates> heaps;
    for (const auto &entry : best_)
      heaps.emplace(entry.first, TopKCandidates(topk_, better_.descending));
    auto read = [&](void *out, size_t bytes) {
      auto *dest = static_cast<char *>(out);
      while (bytes) {
        if (cursor == loaded) {
          loaded = std::min(kBatch, data_.size() - offset);
          if (!loaded)
            throw std::runtime_error("Truncated group candidate log");
          check(data_.read(offset, staging_.data(), loaded));
          offset += loaded;
          cursor = 0;
        }
        const size_t count = std::min(bytes, loaded - cursor);
        std::memcpy(dest, staging_.data() + cursor, count);
        dest += count;
        bytes -= count;
        cursor += count;
      }
    };
    String group;
    while (offset < data_.size() || cursor < loaded) {
      Record record{};
      read(&record, sizeof(record));
      group.resize(record.length);
      read(group.data(), group.size());
      auto heap = heaps.find(group);
      if (heap != heaps.end()) heap->second.offer(record.row);
    }
    ranked_.clear();
    best_.clear();
    for (auto it = heaps.begin(); it != heaps.end();) {
      const size_t id = groups_.size();
      groups_.push_back(it->first);
      for (auto row : it->second.rows()) {
        row.group = id;
        rows_.push_back(row);
      }
      it = heaps.erase(it);
    }
    std::sort(rows_.begin(), rows_.end(), better_);
    data_ = core::AppendableBuildBuffer{};
    Vector<char>().swap(staging_);
  }

  const Vector<RankedDoc> &rows() const {
    return rows_;
  }
  const String &group(size_t id) const {
    return groups_.at(id);
  }
  size_t retained_groups() const {
    return best_.size();
  }
  bool paged() const {
    return data_.paged();
  }

 private:
  struct KeyLess {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const {
      return a < b;
    }
  };
  template <typename T>
  using Map = std::map<String, T, KeyLess,
                       ailego::BudgetedAllocator<std::pair<const String, T>>>;
  struct Rank {
    RankedDoc row;
    const String *name;
  };
  struct RankLess {
    TopKCandidates::Better better;
    bool operator()(const Rank &a, const Rank &b) const {
      if (better(a.row, b.row)) return true;
      if (better(b.row, a.row)) return false;
      return *a.name < *b.name;
    }
  };
  struct Record {
    RankedDoc row;
    uint32_t length;
  };
  static void check(int status) {
    if (status == core::IndexError_NoMemory) throw std::bad_alloc();
    if (status)
      throw std::runtime_error("Group candidate storage failed: " +
                               std::to_string(status));
  }
  void append(const void *input, size_t bytes) {
    const auto *src = static_cast<const char *>(input);
    while (bytes) {
      const size_t count = std::min(bytes, kBatch - used_);
      std::memcpy(staging_.data() + used_, src, count);
      used_ += count;
      src += count;
      bytes -= count;
      if (used_ == kBatch) flush();
    }
  }
  void flush() {
    if (!used_) return;
    const size_t offset = data_.size();
    if (used_ > std::numeric_limits<size_t>::max() - offset)
      throw std::bad_alloc();
    check(data_.resize(offset + used_));
    check(data_.write(offset, staging_.data(), used_));
    used_ = 0;
  }
  static constexpr size_t kBatch = 64 * 1024;
  size_t limit_, topk_;
  TopKCandidates::Better better_;
  Map<RankedDoc> best_;
  std::set<Rank, RankLess, ailego::BudgetedAllocator<Rank>> ranked_;
  core::AppendableBuildBuffer data_;
  Vector<char> staging_;
  size_t used_{0};
  Vector<RankedDoc> rows_;
  Vector<String> groups_;
};
}  // namespace zvec::sqlengine

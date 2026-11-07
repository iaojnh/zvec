// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <zvec/core/framework/index_build_partitions.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>
#include "paged_record_sort.h"

namespace zvec::core {

struct IndexBuildPartitions::State {
  size_t partitions{0};
  size_t records{0};
  bool paged{false};
  bool finished{false};
  std::atomic<size_t> assigned{0};
  std::mutex mutex;
  ailego::BudgetedVector<ailego::BudgetedVector<uint32_t>> buckets;
  ailego::BudgetedVector<size_t> offsets;
  PagedBuildBuffer data;
};

IndexBuildPartitions::IndexBuildPartitions() = default;
IndexBuildPartitions::~IndexBuildPartitions() = default;
IndexBuildPartitions::IndexBuildPartitions(IndexBuildPartitions &&) noexcept =
    default;
IndexBuildPartitions &IndexBuildPartitions::operator=(
    IndexBuildPartitions &&) noexcept = default;

int IndexBuildPartitions::init(size_t partitions, size_t records) try {
  if (!partitions || partitions == SIZE_MAX ||
      records > std::numeric_limits<uint32_t>::max() ||
      records > SIZE_MAX / sizeof(KeyOrderRecord))
    return IndexError_InvalidArgument;
  auto candidate = std::make_unique<State>();
  candidate->partitions = partitions;
  candidate->records = records;
  candidate->paged = ailego::MemoryLimitPool::get_instance().capacity() != 0;
  if (candidate->paged) {
    int ret = candidate->data.init(records * sizeof(KeyOrderRecord), 64 * 1024);
    if (ret != 0) return ret;
  } else {
    candidate->buckets.resize(partitions);
  }
  state_ = std::move(candidate);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_InvalidArgument;
}

int IndexBuildPartitions::assign(size_t ordinal, size_t partition) try {
  if (!state_ || state_->finished) return IndexError_NoReady;
  auto &state = *state_;
  if (ordinal >= state.records || partition >= state.partitions)
    return IndexError_OutOfRange;
  if (state.paged) {
    const KeyOrderRecord record{partition, static_cast<uint32_t>(ordinal), 0};
    int ret =
        state.data.write(ordinal * sizeof(record), &record, sizeof(record));
    if (ret != 0) return ret;
  } else {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.buckets[partition].push_back(static_cast<uint32_t>(ordinal));
  }
  state.assigned.fetch_add(1, std::memory_order_relaxed);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int IndexBuildPartitions::finish() try {
  if (!state_) return IndexError_NoReady;
  auto &state = *state_;
  if (state.finished) return 0;
  if (state.assigned.load(std::memory_order_relaxed) != state.records)
    return IndexError_Mismatch;
  if (state.paged) {
    int ret = SortPagedRecords<KeyOrderRecord>(
        &state.data, [](const auto &a, const auto &b) {
          return a.key != b.key ? a.key < b.key : a.id < b.id;
        });
    if (ret != 0) return ret;
    state.offsets.assign(state.partitions + 1, 0);
    for (size_t i = 0; i < state.records; ++i) {
      KeyOrderRecord record;
      ret = state.data.read(i * sizeof(record), &record, sizeof(record));
      if (ret != 0) return ret;
      if (record.key >= state.partitions || record.id >= state.records)
        return IndexError_InvalidFormat;
      ++state.offsets[record.key + 1];
    }
    for (size_t i = 1; i < state.offsets.size(); ++i)
      state.offsets[i] += state.offsets[i - 1];
  } else {
    for (auto &bucket : state.buckets) std::sort(bucket.begin(), bucket.end());
  }
  state.finished = true;
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

void IndexBuildPartitions::clear() {
  state_.reset();
}
size_t IndexBuildPartitions::size() const {
  return state_ ? state_->partitions : 0;
}
size_t IndexBuildPartitions::count(size_t partition) const {
  if (!state_ || !state_->finished || partition >= size()) return 0;
  return state_->paged
             ? state_->offsets[partition + 1] - state_->offsets[partition]
             : state_->buckets[partition].size();
}
int IndexBuildPartitions::read(size_t partition, size_t offset,
                               uint32_t *ordinal) const {
  if (!ordinal) return IndexError_InvalidArgument;
  if (!state_ || !state_->finished) return IndexError_NoReady;
  if (partition >= size() || offset >= count(partition))
    return IndexError_OutOfRange;
  if (!state_->paged) {
    *ordinal = state_->buckets[partition][offset];
    return 0;
  }
  KeyOrderRecord record;
  int ret = state_->data.read(
      (state_->offsets[partition] + offset) * sizeof(record), &record,
      sizeof(record));
  if (ret != 0) return ret;
  if (record.key != partition) return IndexError_InvalidFormat;
  *ordinal = record.id;
  return 0;
}
}  // namespace zvec::core

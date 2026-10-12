// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <stdexcept>
#include <zvec/db/status.h>
#include "core/utility/appendable_build_buffer.h"

namespace zvec {

// Monotonic global IDs, addressed by dense segment-local position. Consecutive
// IDs need only one inline range. Gaps add pageable range records, not a second
// resident copy of every persisted ID. The owner provides synchronization.
class DocIdMap {
 public:
  class ReadError : public std::runtime_error {
   public:
    explicit ReadError(const Status &status)
        : std::runtime_error(status.message()), status_(status) {}
    const Status &status() const {
      return status_;
    }

   private:
    Status status_;
  };
  // Prepare all fallible storage work before publishing a document. A failed
  // insert can abandon this preparation; the next one overwrites the same slot.
  Status prepare_append(uint64_t id) {
    if (size_ && id <= last_) {
      return Status::InvalidArgument(
          "Document IDs must be strictly increasing");
    }
    if (size_ == std::numeric_limits<size_t>::max()) {
      return Status(StatusCode::RESOURCE_EXHAUSTED,
                    "Document ID mapping is full");
    }
    if (size_ && id - last_ != 1) {
      if (runs_ > std::numeric_limits<size_t>::max() / sizeof(Run)) {
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "Document ID ranges overflow");
      }
      Run run{size_, id};
      int ret = ranges_.resize(runs_ * sizeof(Run));
      if (ret == 0)
        ret = ranges_.write((runs_ - 1) * sizeof(Run), &run, sizeof(run));
      if (ret != 0)
        return Status(ret == core::IndexError_NoMemory
                          ? StatusCode::RESOURCE_EXHAUSTED
                          : StatusCode::INTERNAL_ERROR,
                      "Cannot store document ID range: " + std::to_string(ret));
    }
    return Status::OK();
  }

  // Must follow a successful prepare_append(id), under the same owner lock.
  void commit_append(uint64_t id) noexcept {
    if (!size_) {
      first_ = id;
      runs_ = 1;
    } else if (id - last_ != 1) {
      ++runs_;
    }
    last_ = id;
    ++size_;
  }

  Status append(uint64_t id) {
    auto status = prepare_append(id);
    if (status.ok()) commit_append(id);
    return status;
  }

  Result<uint64_t> at(size_t index) const {
    if (index >= size_)
      return tl::make_unexpected(
          Status::InvalidArgument("Document position out of range"));
    auto run = preceding(index, false);
    if (!run) return tl::make_unexpected(run.error());
    return run->global + (index - run->local);
  }

  Result<size_t> find(uint64_t id) const {
    if (!size_ || id < first_ || id > last_)
      return tl::make_unexpected(Status::NotFound("Document ID not found"));
    auto run = preceding(id, true);
    if (!run) return tl::make_unexpected(run.error());
    const uint64_t delta = id - run->global;
    if (delta >= size_ - run->local)
      return tl::make_unexpected(Status::NotFound("Document ID not found"));
    const size_t index = run->local + static_cast<size_t>(delta);
    auto actual = at(index);
    if (!actual) return tl::make_unexpected(actual.error());
    if (*actual != id)
      return tl::make_unexpected(Status::NotFound("Document ID lies in a gap"));
    return index;
  }

  // Counting cannot report an I/O status through its legacy uint64_t API.
  // Fail explicitly rather than returning a false empty segment on read error.
  template <typename Filter>
  size_t count_filtered(const Filter &filter) const {
    size_t count = 0;
    for (size_t r = 0; r < runs_; ++r) {
      auto current = read_run(r);
      if (!current) throw ReadError(current.error());
      size_t end = size_;
      if (r + 1 < runs_) {
        auto next = read_run(r + 1);
        if (!next) throw ReadError(next.error());
        end = next->local;
      }
      for (size_t i = current->local; i < end; ++i) {
        count += filter(current->global + (i - current->local));
      }
    }
    return count;
  }

  size_t size() const {
    return size_;
  }
  size_t range_count() const {
    return runs_;
  }
  uint64_t front() const {
    return first_;
  }
  uint64_t back() const {
    return last_;
  }
  bool paged() const {
    return ranges_.paged();
  }

 private:
  struct Run {
    size_t local;
    uint64_t global;
  };
  Result<Run> read_run(size_t index) const {
    if (!index) return Run{0, first_};
    Run run{};
    int ret = ranges_.read((index - 1) * sizeof(Run), &run, sizeof(run));
    if (ret != 0)
      return tl::make_unexpected(
          Status::InternalError("Cannot read document ID range: ", ret));
    return run;
  }
  Result<Run> preceding(uint64_t value, bool global) const {
    Run found{0, first_};
    size_t lo = 1, hi = runs_;
    while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      auto run = read_run(mid);
      if (!run) return tl::make_unexpected(run.error());
      if ((global ? run->global : run->local) <= value) {
        found = *run;
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return found;
  }
  core::AppendableBuildBuffer ranges_;
  size_t size_{0};
  size_t runs_{0};
  uint64_t first_{0};
  uint64_t last_{0};
};
}  // namespace zvec

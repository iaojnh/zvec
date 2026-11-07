// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <zvec/core/framework/index_error.h>

namespace zvec::core {

// One partition assignment per input ordinal. The owner chooses storage and
// manages sorting scratch, capacity admission and cleanup. Readers see input
// ordinals in ascending order within each partition after finish().
class IndexBuildPartitions {
 public:
  IndexBuildPartitions();
  ~IndexBuildPartitions();
  IndexBuildPartitions(const IndexBuildPartitions &) = delete;
  IndexBuildPartitions &operator=(const IndexBuildPartitions &) = delete;
  IndexBuildPartitions(IndexBuildPartitions &&) noexcept;
  IndexBuildPartitions &operator=(IndexBuildPartitions &&) noexcept;

  // A rejected replacement leaves the previous index unchanged.
  int init(size_t partitions, size_t records);
  // Assign distinct ordinals concurrently, exactly once per ordinal. Join all
  // writers before finish(), clear(), init() or destruction.
  int assign(size_t ordinal, size_t partition);
  int finish();
  void clear();
  size_t size() const;
  size_t count(size_t partition) const;
  int read(size_t partition, size_t offset, uint32_t *ordinal) const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace zvec::core

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace zvec::core {
// Read-only, call-scoped adaptation of the legacy batched key API.
class IndexKeyBatchView {
 public:
  IndexKeyBatchView() = default;
  IndexKeyBatchView(const std::vector<uint64_t> &keys) : single_(&keys) {}
  IndexKeyBatchView(const std::vector<std::vector<uint64_t>> &keys)
      : batch_(&keys) {}
  size_t size() const {
    return single_ ? 1 : batch_ ? batch_->size() : 0;
  }
  const std::vector<uint64_t> &operator[](size_t i) const {
    return single_ ? *single_ : (*batch_)[i];
  }

 private:
  const std::vector<uint64_t> *single_{nullptr};
  const std::vector<std::vector<uint64_t>> *batch_{nullptr};
};
}  // namespace zvec::core

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <limits>
#include <zvec/db/status.h>
#include "core/utility/appendable_build_buffer.h"

namespace zvec {

// Temporary matched IDs, not Doc objects or primary-key strings. When a pool is
// configured, the contents are pageable; replay uses fixed-size scratch. The
// caller finishes the complete scan before applying any mutations.
class DocIdBuffer {
 public:
  Status append(const uint64_t *ids, size_t count) {
    if (!ids && count) return Status::InvalidArgument("Null document IDs");
    if (count > std::numeric_limits<size_t>::max() / sizeof(uint64_t) - size_) {
      return Status(StatusCode::RESOURCE_EXHAUSTED,
                    "Matched document IDs overflow");
    }
    if (!count) return Status::OK();
    int ret = buffer_.resize((size_ + count) * sizeof(uint64_t));
    if (ret == 0) {
      ret = buffer_.write(size_ * sizeof(uint64_t), ids,
                          count * sizeof(uint64_t));
    }
    if (ret != 0) return storage_error(ret);
    size_ += count;
    return Status::OK();
  }

  template <typename Consumer>
  Status for_each(const Consumer &consume) const {
    std::array<uint64_t, 512> ids;
    for (size_t offset = 0; offset < size_;) {
      const size_t count = std::min(ids.size(), size_ - offset);
      int ret = buffer_.read(offset * sizeof(uint64_t), ids.data(),
                             count * sizeof(uint64_t));
      if (ret != 0) return storage_error(ret);
      for (size_t i = 0; i < count; ++i) {
        auto status = consume(ids[i]);
        if (!status.ok()) return status;
      }
      offset += count;
    }
    return Status::OK();
  }

  size_t size() const {
    return size_;
  }

 private:
  static Status storage_error(int ret) {
    return Status(
        ret == core::IndexError_NoMemory ? StatusCode::RESOURCE_EXHAUSTED
                                         : StatusCode::INTERNAL_ERROR,
        "Cannot store/read matched document IDs: " + std::to_string(ret));
  }

  core::AppendableBuildBuffer buffer_;
  size_t size_{0};
};

}  // namespace zvec

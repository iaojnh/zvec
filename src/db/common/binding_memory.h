// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <limits>
#include <new>

namespace zvec::detail {

inline size_t BindingMemoryBytes(size_t count, size_t width, size_t extra = 0) {
  if (width && count > (std::numeric_limits<size_t>::max() - extra) / width)
    throw std::bad_alloc();
  return count * width + extra;
}

}  // namespace zvec::detail

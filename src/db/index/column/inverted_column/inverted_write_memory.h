// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include "db/common/binding_memory.h"

namespace zvec::detail {

inline size_t InvertedArrayWriteBytes(size_t terms, size_t payload) {
  // Account for the input string vector used by compaction, its encoded
  // counterpart, and overlapping RocksDB WriteBatch backing during growth.
  return BindingMemoryBytes(
      payload, 8, BindingMemoryBytes(terms, 3 * sizeof(std::string) + 96));
}

}  // namespace zvec::detail

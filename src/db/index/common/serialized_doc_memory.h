// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace zvec::detail {

// Validate without allocating and estimate decoded Doc payload, hash nodes,
// buckets and string-array elements. The caller admits this estimate before
// deserializing and retains the allowance through the decoded Doc's lifetime.
// This is an ordinary-heap planning estimate, not an allocator RSS guarantee.
// Invalid/truncated input leaves *bytes unchanged.
bool EstimateSerializedDocMemory(const uint8_t *data, size_t size,
                                 size_t *bytes);

}  // namespace zvec::detail

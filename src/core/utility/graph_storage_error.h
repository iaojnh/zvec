// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <zvec/core/framework/index_error.h>
namespace zvec::core {
// Propagate storage failures through graph helpers that return no status.
// Public index operations translate this into their existing error codes.
constexpr uint32_t kGraphBuildFailed = 1U << 31;
struct GraphStorageError {
  int code;
};
inline void CheckGraphStorage(int code) {
  if (code != 0) throw GraphStorageError{code};
}
}  // namespace zvec::core

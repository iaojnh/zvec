// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace zvec::detail {

#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
constexpr bool kMobileMemoryPolicy = true;
#else
constexpr bool kMobileMemoryPolicy = false;
#endif

constexpr uint64_t kMobileMinimumMemoryLimit = uint64_t{16} << 20;

// This is a conservative startup policy, not an OS per-app entitlement. A host
// may override it explicitly; native pressure can still restrict admission.
inline uint64_t MobileDefaultMemoryLimit(uint64_t detected_bytes) {
  return std::min(detected_bytes / 8, uint64_t{128} << 20);
}

}  // namespace zvec::detail

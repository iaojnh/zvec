// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace zvec {

// Four buffer-equivalents plus encoding headroom are reserved before writes.
// Keep the mobile default compatible with its 64/128 MiB process budgets.
const uint32_t MOBILE_DEFAULT_MAX_BUFFER_SIZE = 4 * 1024 * 1024;
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
const uint32_t DEFAULT_MAX_BUFFER_SIZE = MOBILE_DEFAULT_MAX_BUFFER_SIZE;
#else
const uint32_t DEFAULT_MAX_BUFFER_SIZE = 64 * 1024 * 1024;
#endif

struct CollectionOptions {
  // Pending WAL records require a writable open and close before read-only.
  bool read_only_{false};
  bool enable_mmap_{true};  // ignored when load collection
  uint32_t max_buffer_size_{
      DEFAULT_MAX_BUFFER_SIZE};  // ignored when read_only=true

  bool operator==(const CollectionOptions &other) const {
    return read_only_ == other.read_only_ &&
           enable_mmap_ == other.enable_mmap_ &&
           max_buffer_size_ == other.max_buffer_size_;
  }

  bool operator!=(const CollectionOptions &other) const {
    return !(*this == other);
  }

  CollectionOptions() = default;

  CollectionOptions(bool read_only, bool enable_mmap,
                    uint32_t max_buffer_size = DEFAULT_MAX_BUFFER_SIZE)
      : read_only_(read_only),
        enable_mmap_(enable_mmap),
        max_buffer_size_(max_buffer_size) {}
};

struct SegmentOptions {
  bool read_only_;
  bool enable_mmap_;
  uint32_t max_buffer_size_{DEFAULT_MAX_BUFFER_SIZE};
};

struct CreateIndexOptions {
  int concurrency_{0};  // default use config.optimize_thread_pool
};

struct OptimizeOptions {
  int concurrency_{0};
};

struct AddColumnOptions {
  int concurrency_{0};
};

struct AlterColumnOptions {
  int concurrency_{0};
};

struct IteratorOptions {
  std::optional<std::vector<std::string>> output_fields_;
  bool include_vector_{true};
};

}  // namespace zvec

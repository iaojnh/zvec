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
#include <memory>
#include <string>
#include <vector>
#include <roaring/roaring.h>
#include "db/common/rocksdb_context.h"

namespace zvec {

enum class RangeQueryPath { ScanLeft, ScanRight, PrefixAdd, PrefixSubtract };

struct RangeQueryPlan {
  RangeQueryPath path{RangeQueryPath::ScanLeft};
  size_t prefix_index{SIZE_MAX};
  size_t range_begin{0};
  size_t range_end{0};
  size_t boundary_bucket{SIZE_MAX};
  bool left_inclusive{false};
  bool flip{false};
  uint64_t estimated_work{0};
};

// Only small planning metadata is resident. Portable posting bitmaps live in
// the field's $PREFIXES CF and use RocksDB's ordinary read/cache path.
class InvertedRangePrefix {
 public:
  struct BitmapDeleter {
    void operator()(roaring_bitmap_t *bitmap) const {
      roaring_bitmap_free(bitmap);
    }
  };
  using BitmapPtr = std::unique_ptr<roaring_bitmap_t, BitmapDeleter>;
  struct Bucket {
    std::string key;
    std::string begin;
    std::string end;
    uint64_t cardinality{0};
    uint64_t bytes{0};
    uint64_t cumulative_cardinality{0};
    uint64_t cumulative_bytes{0};
  };
  struct Prefix {
    size_t end_bucket{0};
    uint64_t serialized_bytes{0};
  };

  // Called before the sealed marker is published. Without a manifest,
  // the original RANGES/TERMS search remains in use.
  static Status Build(RocksdbContext &context,
                      rocksdb::ColumnFamilyHandle *ranges,
                      rocksdb::ColumnFamilyHandle *prefixes, uint32_t max_id,
                      const roaring_bitmap_t *null_bitmap);
  static std::unique_ptr<InvertedRangePrefix> Load(
      RocksdbContext &context, rocksdb::ColumnFamilyHandle *prefixes,
      uint32_t max_id, const roaring_bitmap_t *null_bitmap);

  std::vector<Bucket> buckets;
  std::vector<Prefix> prefixes;
  uint64_t total_cardinality{0};
};

}  // namespace zvec

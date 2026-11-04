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

#include <limits>
#include <memory>
#include <utility>
#include <rocksdb/advanced_cache.h>

namespace zvec {

// RocksDB 8.1 charges full-filter contents but omits its FilterBitsReader
// and part of the ParsedFullFilterBlock object. Reserve a conservative fixed
// allowance for those small built-in Bloom/Ribbon reader objects, including
// allocator rounding. The underlying strict LRU owns the charge until the
// entry's final reference is released, including after Erase or DB close.
// Do not apply this to WBM/filter-construction reservations or data/index
// blocks, whose existing charge must remain unchanged.
class RocksdbBlockCache final : public rocksdb::CacheWrapper {
 public:
  static constexpr size_t kFilterMetadataCharge = 128;

  explicit RocksdbBlockCache(std::shared_ptr<rocksdb::Cache> target)
      : rocksdb::CacheWrapper(std::move(target)) {
    // CacheWrapper does not forward this nonvirtual accessor. Preserve the
    // mapped payload allocator; otherwise wrapping would revert SST reads
    // to the default heap allocator.
    memory_allocator_ = std::shared_ptr<rocksdb::MemoryAllocator>(
        target_, target_->memory_allocator());
  }

  const char *Name() const override {
    return "ZvecBudgetedBlockCache";
  }

  rocksdb::Status Insert(const rocksdb::Slice &key, ObjectPtr value,
                         const CacheItemHelper *helper, size_t charge,
                         Handle **handle = nullptr,
                         Priority priority = Priority::LOW) override {
    if (is_filter(helper)) {
      if (charge > std::numeric_limits<size_t>::max() - kFilterMetadataCharge) {
        if (handle != nullptr) {
          *handle = nullptr;
        }
        return rocksdb::Status::MemoryLimit("Filter metadata charge overflow");
      }
      charge += kFilterMetadataCharge;
    }
    return target_->Insert(key, value, helper, charge, handle, priority);
  }

  Handle *CreateStandalone(const rocksdb::Slice &key, ObjectPtr value,
                           const CacheItemHelper *helper, size_t charge,
                           bool allow_uncharged) override {
    if (is_filter(helper)) {
      if (charge > std::numeric_limits<size_t>::max() - kFilterMetadataCharge) {
        return nullptr;
      }
      charge += kFilterMetadataCharge;
      // A live filter must not escape accounting when the cache is full.
      allow_uncharged = false;
    }
    return target_->CreateStandalone(key, value, helper, charge,
                                     allow_uncharged);
  }

 private:
  static bool is_filter(const CacheItemHelper *helper) {
    return helper != nullptr &&
           helper->role == rocksdb::CacheEntryRole::kFilterBlock;
  }
};

}  // namespace zvec

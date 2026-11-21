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

#include <array>
#include <numeric>
#include "utility/appendable_build_buffer.h"
#include "utility/paged_record_sort.h"
#include "flat_utility.h"

namespace zvec::core {

// Collect once while writing features; dump keys in input order, then sort
// (key, ordinal) records in the same paged file and emit the legacy uint32 map.
class FlatDumpKeys {
 public:
  int append(uint64_t key) try {
    if (count_ >= std::numeric_limits<uint32_t>::max() ||
        count_ >=
            std::numeric_limits<size_t>::max() / (2 * sizeof(KeyOrderRecord)))
      return IndexError_IndexFull;
    if (!paged_) {
      keys_.push_back(key);
      ++count_;
      return 0;
    }
    const KeyOrderRecord record{key, static_cast<uint32_t>(count_), 0};
    int ret = records_.resize((count_ + 1) * sizeof(record));
    if (ret != 0) return ret;
    ret = records_.write(count_ * sizeof(record), &record, sizeof(record));
    if (ret == 0) ++count_;
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  size_t size() const {
    return count_;
  }

  int write_keys(IndexDumper *dumper,
                 const std::string &name = FLAT_SEGMENT_KEYS_SEG_ID) const {
    return write_values<uint64_t>(
        dumper, name, [](const auto &record) { return record.key; });
  }

  int write_mapping(IndexDumper *dumper,
                    const std::string &name = FLAT_SEGMENT_MAPPING_SEG_ID) try {
    if (!paged_) {
      mapping_.resize(count_);
      std::iota(mapping_.begin(), mapping_.end(), 0);
      std::sort(mapping_.begin(), mapping_.end(), [&](uint32_t a, uint32_t b) {
        return keys_[a] != keys_[b] ? keys_[a] < keys_[b] : a < b;
      });
      return write_values<uint32_t>(
          dumper, name, [](const auto &record) { return record.id; });
    }
    int ret = records_.resize(2 * count_ * sizeof(KeyOrderRecord));
    if (ret != 0) return ret;
    ret = SortPagedRecords<KeyOrderRecord>(
        &records_,
        [](const auto &a, const auto &b) {
          return a.key != b.key ? a.key < b.key : a.id < b.id;
        },
        count_);
    if (ret != 0) return ret;
    return write_values<uint32_t>(dumper, name,
                                  [](const auto &record) { return record.id; });
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

 private:
  template <typename T, typename Extract>
  int write_values(IndexDumper *dumper, const std::string &name,
                   Extract extract) const {
    if (!dumper) return IndexError_InvalidArgument;
    std::array<KeyOrderRecord, 1024> records{};
    std::array<T, 1024> values{};
    for (size_t first = 0; first < count_; first += records.size()) {
      size_t n = std::min(records.size(), count_ - first);
      if (paged_) {
        int ret = records_.read(first * sizeof(KeyOrderRecord), records.data(),
                                n * sizeof(KeyOrderRecord));
        if (ret != 0) return ret;
        for (size_t i = 0; i < n; ++i) values[i] = extract(records[i]);
      } else {
        for (size_t i = 0; i < n; ++i) {
          if constexpr (std::is_same_v<T, uint64_t>)
            values[i] = keys_[first + i];
          else
            values[i] = mapping_[first + i];
        }
      }
      if (dumper->write(values.data(), n * sizeof(T)) != n * sizeof(T))
        return IndexError_WriteData;
    }
    const size_t bytes = count_ * sizeof(T);
    const size_t padding_size = (32 - bytes % 32) % 32;
    const char padding[32]{};
    if (padding_size && dumper->write(padding, padding_size) != padding_size)
      return IndexError_WriteData;
    return dumper->append(name, bytes, padding_size, 0);
  }

  AppendableBuildBuffer records_;
  size_t count_{0};
  const bool paged_{ailego::MemoryLimitPool::get_instance().capacity() != 0};
  // Preserve the compact 8-byte keys + 4-byte mapping when no pool exists.
  std::vector<uint64_t> keys_;
  std::vector<uint32_t> mapping_;
};

}  // namespace zvec::core

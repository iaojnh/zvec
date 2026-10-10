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
#include <type_traits>
#include "paged_build_buffer.h"

namespace zvec::core {

// External merge sort for fixed-width scratch records. Only three bounded
// tiles are on the heap; the runs and merge output share the process pool.
// Equal records retain input order across merge passes.
template <typename T, typename Compare>
int SortPagedRecords(PagedBuildBuffer *records, Compare less) try {
  static_assert(std::is_trivially_copyable<T>::value,
                "Scratch records must be POD");
  if (!records || records->size() % sizeof(T))
    return IndexError_InvalidArgument;
  const size_t count = records->size() / sizeof(T);
  if (count < 2) return 0;
  const size_t tile =
      records->paged() ? std::max<size_t>(1, 64 * 1024 / sizeof(T)) : count;
  std::vector<T> left(std::min(tile, count));
  for (size_t first = 0; first < count; first += tile) {
    const size_t n = std::min(tile, count - first);
    int ret = records->read(first * sizeof(T), left.data(), n * sizeof(T));
    if (ret != 0) return ret;
    std::sort(left.begin(), left.begin() + n, less);
    ret = records->write(first * sizeof(T), left.data(), n * sizeof(T));
    if (ret != 0) return ret;
  }
  if (count <= tile) return 0;
  PagedBuildBuffer output;
  int ret = output.init(records->size());
  if (ret != 0) return ret;
  std::vector<T> right(left.size()), merged(left.size());
  for (size_t width = tile; width < count;) {
    for (size_t first = 0; first < count;) {
      const size_t middle = first + std::min(width, count - first);
      const size_t end = middle + std::min(width, count - middle);
      size_t l = first, r = middle, ln = 0, rn = 0, li = 0, ri = 0;
      size_t written = first, pending = 0;
      while (l < middle || r < end || li < ln || ri < rn) {
        if (li == ln && l < middle) {
          ln = std::min(tile, middle - l);
          ret = records->read(l * sizeof(T), left.data(), ln * sizeof(T));
          if (ret != 0) return ret;
          l += ln;
          li = 0;
        }
        if (ri == rn && r < end) {
          rn = std::min(tile, end - r);
          ret = records->read(r * sizeof(T), right.data(), rn * sizeof(T));
          if (ret != 0) return ret;
          r += rn;
          ri = 0;
        }
        if (ri == rn || (li < ln && !less(right[ri], left[li]))) {
          merged[pending++] = left[li++];
        } else {
          merged[pending++] = right[ri++];
        }
        if (pending == merged.size()) {
          ret = output.write(written * sizeof(T), merged.data(),
                             pending * sizeof(T));
          if (ret != 0) return ret;
          written += pending;
          pending = 0;
        }
      }
      if (pending) {
        ret = output.write(written * sizeof(T), merged.data(),
                           pending * sizeof(T));
        if (ret != 0) return ret;
      }
      first = end;
    }
    std::swap(*records, output);
    width = width > count / 2 ? count : width * 2;
  }
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

struct KeyOrderRecord {
  uint64_t key{0};
  uint32_t id{0};
  uint32_t padding{0};
};

template <typename ReadKey>
int BuildPagedKeyMapping(size_t count, ReadKey &&read_key,
                         PagedBuildBuffer *mapping) try {
  if (!mapping || count > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<size_t>::max() / sizeof(KeyOrderRecord))
    return IndexError_InvalidArgument;
  PagedBuildBuffer records;
  int ret = records.init(count * sizeof(KeyOrderRecord), 64 * 1024);
  if (ret != 0) return ret;
  const size_t tile = 4096;
  std::vector<KeyOrderRecord> batch(std::min(tile, count));
  for (size_t first = 0; first < count; first += tile) {
    const size_t n = std::min(tile, count - first);
    for (size_t i = 0; i < n; ++i) {
      batch[i].id = static_cast<uint32_t>(first + i);
      ret = read_key(first + i, &batch[i].key);
      if (ret != 0) return ret;
    }
    ret = records.write(first * sizeof(KeyOrderRecord), batch.data(),
                        n * sizeof(KeyOrderRecord));
    if (ret != 0) return ret;
  }
  ret = SortPagedRecords<KeyOrderRecord>(
      &records, [](const auto &a, const auto &b) {
        return a.key != b.key ? a.key < b.key : a.id < b.id;
      });
  if (ret != 0) return ret;
  // Compact ids into the front of the same file. Every input tile is consumed
  // before its prefix is overwritten, so no second result file is needed.
  std::vector<uint32_t> ids(batch.size());
  for (size_t first = 0; first < count; first += tile) {
    const size_t n = std::min(tile, count - first);
    ret = records.read(first * sizeof(KeyOrderRecord), batch.data(),
                       n * sizeof(KeyOrderRecord));
    if (ret != 0) return ret;
    for (size_t i = 0; i < n; ++i) ids[i] = batch[i].id;
    ret = records.write(first * sizeof(uint32_t), ids.data(),
                        n * sizeof(uint32_t));
    if (ret != 0) return ret;
  }
  ret = records.truncate(count * sizeof(uint32_t));
  if (ret != 0) return ret;
  *mapping = std::move(records);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

}  // namespace zvec::core

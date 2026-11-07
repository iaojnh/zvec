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
// With an explicit count, records must reserve space for twice that many
// records. Merge passes use the second half of the same file, avoiding a
// second page table. The sorted result always ends at offset zero.
template <typename T, typename Compare, typename Buffer>
int SortPagedRecords(Buffer *records, Compare less,
                     size_t count = std::numeric_limits<size_t>::max()) try {
  static_assert(std::is_trivially_copyable<T>::value,
                "Scratch records must be POD");
  if (!records || records->size() % sizeof(T))
    return IndexError_InvalidArgument;
  const bool shared_scratch = count != std::numeric_limits<size_t>::max();
  if (shared_scratch) {
    if (count > records->size() / sizeof(T) / 2)
      return IndexError_InvalidArgument;
  } else {
    count = records->size() / sizeof(T);
  }
  if (count < 2) return 0;
  const size_t tile =
      records->paged() ? std::max<size_t>(1, 64 * 1024 / sizeof(T)) : count;
  ailego::BudgetedVector<T> left(std::min(tile, count));
  for (size_t first = 0; first < count; first += tile) {
    const size_t n = std::min(tile, count - first);
    int ret = records->read(first * sizeof(T), left.data(), n * sizeof(T));
    if (ret != 0) return ret;
    std::sort(left.begin(), left.begin() + n, less);
    ret = records->write(first * sizeof(T), left.data(), n * sizeof(T));
    if (ret != 0) return ret;
  }
  if (count <= tile) return 0;
  Buffer separate_output;
  auto *output = shared_scratch ? records : &separate_output;
  int ret = 0;
  if (!shared_scratch) {
    ret = output->init(records->size());
    if (ret != 0) return ret;
  }
  size_t input_base = 0;
  size_t output_base = shared_scratch ? count * sizeof(T) : 0;
  ailego::BudgetedVector<T> right(left.size()), merged(left.size());
  for (size_t width = tile; width < count;) {
    for (size_t first = 0; first < count;) {
      const size_t middle = first + std::min(width, count - first);
      const size_t end = middle + std::min(width, count - middle);
      size_t l = first, r = middle, ln = 0, rn = 0, li = 0, ri = 0;
      size_t written = first, pending = 0;
      while (l < middle || r < end || li < ln || ri < rn) {
        if (li == ln && l < middle) {
          ln = std::min(tile, middle - l);
          ret = records->read(input_base + l * sizeof(T), left.data(),
                              ln * sizeof(T));
          if (ret != 0) return ret;
          l += ln;
          li = 0;
        }
        if (ri == rn && r < end) {
          rn = std::min(tile, end - r);
          ret = records->read(input_base + r * sizeof(T), right.data(),
                              rn * sizeof(T));
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
          ret = output->write(output_base + written * sizeof(T), merged.data(),
                              pending * sizeof(T));
          if (ret != 0) return ret;
          written += pending;
          pending = 0;
        }
      }
      if (pending) {
        ret = output->write(output_base + written * sizeof(T), merged.data(),
                            pending * sizeof(T));
        if (ret != 0) return ret;
      }
      first = end;
    }
    if (shared_scratch)
      std::swap(input_base, output_base);
    else
      std::swap(*records, *output);
    width = width > count / 2 ? count : width * 2;
  }
  if (input_base != 0) {
    for (size_t first = 0; first < count; first += tile) {
      const size_t bytes = std::min(tile, count - first) * sizeof(T);
      ret = records->read(input_base + first * sizeof(T), left.data(), bytes);
      if (ret != 0) return ret;
      ret = records->write(first * sizeof(T), left.data(), bytes);
      if (ret != 0) return ret;
    }
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
  ailego::BudgetedVector<KeyOrderRecord> batch(std::min(tile, count));
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
  ailego::BudgetedVector<uint32_t> ids(batch.size());
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

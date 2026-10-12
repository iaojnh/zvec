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

#include "utility/paged_record_sort.h"
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>

namespace zvec::core {
TEST(PagedRecordSortTest, KeyMappingHandlesMergeBoundariesAndDuplicateKeys) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(pool.used(), 0u);
  auto restore = ailego::ScopeGuard::Make([&]() {
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(old), 0);
  });
  const size_t metadata =
      ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
  for (size_t budget : {size_t{0}, 2 * metadata + 128 * 1024}) {
    ASSERT_EQ(pool.init(budget), 0);
    for (size_t count :
         {size_t{0}, size_t{1}, size_t{4096}, size_t{4097}, size_t{24577}}) {
      std::vector<KeyOrderRecord> reference(count);
      for (size_t i = 0; i < count; ++i) {
        reference[i].key = (i * 7879) % 6553;
        reference[i].id = static_cast<uint32_t>(i);
      }
      auto key = [](size_t i, uint64_t *out) {
        *out = (i * 7879) % 6553;
        return 0;
      };
      PagedBuildBuffer mapping;
      ASSERT_EQ(BuildPagedKeyMapping(count, key, &mapping), 0);
      std::sort(reference.begin(), reference.end(),
                [](const auto &a, const auto &b) {
                  return a.key != b.key ? a.key < b.key : a.id < b.id;
                });
      std::vector<uint32_t> actual(count);
      ASSERT_EQ(
          mapping.read(0, actual.data(), actual.size() * sizeof(uint32_t)), 0);
      for (size_t i = 0; i < count; ++i) ASSERT_EQ(actual[i], reference[i].id);
      if (budget) EXPECT_LE(pool.used(), pool.capacity());
    }
  }
}

TEST(PagedRecordSortTest, SourceErrorDoesNotPublishPartialMapping) {
  PagedBuildBuffer mapping;
  ASSERT_EQ(mapping.init(sizeof(uint32_t)), 0);
  const uint32_t marker = 31;
  ASSERT_EQ(mapping.write(0, &marker, sizeof(marker)), 0);
  EXPECT_EQ(BuildPagedKeyMapping(
                5000,
                [](size_t id, uint64_t *key) -> int {
                  if (id == 4173) return IndexError_ReadData;
                  *key = id;
                  return 0;
                },
                &mapping),
            IndexError_ReadData);
  uint32_t value = 0;
  ASSERT_EQ(mapping.read(0, &value, sizeof(value)), 0);
  EXPECT_EQ(value, marker);
}

TEST(PagedRecordSortTest, MergeScratchSharesOnePageTable) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(pool.used(), 0u);
  auto restore = ailego::ScopeGuard::Make([&]() {
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(old), 0);
  });
  const size_t metadata =
      ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
  ASSERT_EQ(pool.init(metadata + 64 * 1024), 0);
  for (size_t count : {size_t{0}, size_t{1}, size_t{4096}, size_t{4097},
                       size_t{8193}, size_t{24577}}) {
    SCOPED_TRACE(count);
    PagedBuildBuffer records;
    ASSERT_EQ(records.init(count * sizeof(KeyOrderRecord) * 2), 0);
    std::vector<KeyOrderRecord> reference(count);
    for (size_t i = 0; i < count; ++i) {
      reference[i].key = (i * 7879) % 6553;
      reference[i].id = static_cast<uint32_t>(i);
    }
    ASSERT_EQ(
        records.write(0, reference.data(), count * sizeof(KeyOrderRecord)), 0);
    auto less = [](const auto &a, const auto &b) {
      return a.key != b.key ? a.key < b.key : a.id < b.id;
    };
    EXPECT_EQ(SortPagedRecords<KeyOrderRecord>(&records, less, count + 1),
              IndexError_InvalidArgument);
    ASSERT_EQ(SortPagedRecords<KeyOrderRecord>(&records, less, count), 0);
    std::sort(reference.begin(), reference.end(), less);
    for (size_t i = 0; i < count; ++i) {
      KeyOrderRecord actual;
      ASSERT_EQ(records.read(i * sizeof(actual), &actual, sizeof(actual)), 0);
      ASSERT_EQ(actual.key, reference[i].key);
      ASSERT_EQ(actual.id, reference[i].id);
    }
    EXPECT_LE(pool.used(), pool.capacity());
  }
}
}  // namespace zvec::core

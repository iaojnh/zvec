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

#include <gtest/gtest.h>
#include "interface/indexes/holder_builder.h"

namespace zvec::core_interface {
TEST(DocCacheHolderTest, BorrowsRowsSkipsHolesAndRetainsInput) {
  auto cache = std::make_shared<DocCacheHolder::Cache>();
  cache->emplace_back(9, std::string(12, 'a'));
  cache->emplace_back(kInvalidKey, std::string{});
  cache->emplace_back(4, std::string(12, 'b'));
  core::IndexQueryMeta meta(core::IndexMeta::DT_FP32, 3);
  auto holder = std::make_shared<DocCacheHolder>(meta, cache);
  EXPECT_EQ(holder->count(), 2u);
  core::OrdinalAccessHolder::Reader::Pointer reader;
  ASSERT_EQ(holder->create_ordinal_reader(&reader), 0);
  auto iter = holder->create_iterator();
  EXPECT_EQ(iter->data(), (*cache)[0].second.data());
  iter->next();
  EXPECT_EQ(iter->data(), (*cache)[2].second.data());
  uint64_t key = 0;
  const void *data = nullptr;
  ASSERT_EQ(reader->read(1, &key, &data), 0);
  EXPECT_EQ(data, iter->data());
  EXPECT_EQ(key, 4u);
  std::weak_ptr<DocCacheHolder::Cache> weak = cache;
  cache.reset();
  holder.reset();
  iter.reset();
  EXPECT_FALSE(weak.expired());
  ASSERT_EQ(reader->read(0, &key, &data), 0);
  EXPECT_EQ(key, 9u);
  EXPECT_EQ(std::string(static_cast<const char *>(data), 12),
            std::string(12, 'a'));
  EXPECT_EQ(reader->read(2, &key, &data), core::IndexError_OutOfRange);
  reader.reset();
  EXPECT_TRUE(weak.expired());
}
TEST(DocCacheHolderTest, DenseInputAndMalformedRows) {
  auto cache = std::make_shared<DocCacheHolder::Cache>();
  cache->emplace_back(1, std::string(12, 'a'));
  cache->emplace_back(2, std::string(11, 'b'));
  auto holder = std::make_shared<DocCacheHolder>(
      core::IndexQueryMeta(core::IndexMeta::DT_FP32, 3), cache);
  ASSERT_EQ(holder->count(), 2u);
  auto iter = holder->create_iterator();
  ASSERT_TRUE(iter->is_valid());
  EXPECT_EQ(iter->data(), (*cache)[0].second.data());
  iter->next();
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(iter->status(), core::IndexError_InvalidLength);
  EXPECT_EQ(iter->data(), nullptr);
  core::OrdinalAccessHolder::Reader::Pointer reader;
  ASSERT_EQ(holder->create_ordinal_reader(&reader), 0);
  uint64_t key;
  const void *data;
  EXPECT_EQ(reader->read(1, &key, &data), core::IndexError_InvalidLength);
}
}  // namespace zvec::core_interface

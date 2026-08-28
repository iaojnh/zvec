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

#include "mixed_reducer/merged_provider_index_holder.h"
#include <atomic>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace core {
namespace {

constexpr size_t kDimension = 2;

IndexProvider::Pointer MakeProvider(
    const std::vector<std::pair<uint64_t, float>> &docs) {
  auto provider =
      std::make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  for (const auto &doc : docs) {
    ailego::NumericalVector<float> vector(kDimension);
    vector[0] = doc.second;
    vector[1] = doc.second + 0.5f;
    EXPECT_TRUE(provider->emplace(doc.first, std::move(vector)));
  }
  return provider;
}

MergedProviderIndexHolder::Source MakeSource(
    const IndexProvider::Pointer &provider) {
  MergedProviderIndexHolder::Source source;
  source.provider = provider;
  source.provider_meta =
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension);
  return source;
}

std::vector<std::pair<uint64_t, float>> ReadAll(
    MergedProviderIndexHolder *holder) {
  std::vector<std::pair<uint64_t, float>> docs;
  auto iter = holder->create_iterator();
  EXPECT_TRUE(iter != nullptr);
  while (iter && iter->is_valid()) {
    const float *data = static_cast<const float *>(iter->data());
    EXPECT_TRUE(data != nullptr);
    if (!data) {
      break;
    }
    docs.emplace_back(iter->key(), data[0]);
    iter->next();
  }
  return docs;
}

TEST(MergedProviderIndexHolderTest, FiltersRewritesIdsAndSupportsMultiPass) {
  auto first = MakeProvider({{0, 10.0f}, {1, 11.0f}});
  auto second = MakeProvider({{0, 20.0f}, {1, 21.0f}, {2, 22.0f}});

  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(first));
  sources.emplace_back(MakeSource(second));

  size_t filter_calls = 0;
  IndexFilter filter;
  filter.set([&filter_calls](uint64_t logical_id) {
    ++filter_calls;
    return logical_id == 1 || logical_id == 3;
  });

  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      std::move(sources));
  ASSERT_EQ(0, holder.init(filter));
  EXPECT_EQ(3u, holder.count());
  EXPECT_EQ(2u, holder.filtered_count());
  EXPECT_EQ(5u, filter_calls);
  EXPECT_TRUE(holder.multipass());

  const std::vector<std::pair<uint64_t, float>> expected = {
      {0, 10.0f}, {1, 20.0f}, {2, 22.0f}};
  EXPECT_EQ(expected, ReadAll(&holder));
  EXPECT_EQ(expected, ReadAll(&holder));

  // Filter decisions are planned once and do not depend on filter lifetime or
  // side effects during later builder passes.
  EXPECT_EQ(5u, filter_calls);
  EXPECT_EQ(0, holder.status());
}

TEST(MergedProviderIndexHolderTest, RejectsUnreformedMetaMismatch) {
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(MakeProvider({{0, 10.0f}})));

  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension + 1),
      std::move(sources));
  EXPECT_EQ(IndexError_Mismatch, holder.init(IndexFilter()));
}

TEST(MergedProviderIndexHolderTest, StopsDuringFilterPlanning) {
  std::vector<MergedProviderIndexHolder::Source> sources;
  sources.emplace_back(MakeSource(MakeProvider({{0, 10.0f}})));

  std::atomic<bool> stop{true};
  MergedProviderIndexHolder holder(
      IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
      std::move(sources));
  EXPECT_EQ(IndexError_Canceled, holder.init(IndexFilter(), &stop));
}

}  // namespace
}  // namespace core
}  // namespace zvec

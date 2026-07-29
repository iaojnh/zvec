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
#include <zvec/core/framework/index_factory.h>
#include "diskann_cache_budget.h"
#include "diskann_params.h"

using namespace zvec::core;

TEST(DiskAnnCacheBudgetTest, AccountsForPayloadAndMapBookkeeping) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 768);
  constexpr uint32_t kMaxDegree = 32;

  const uint64_t payload =
      DiskAnnCacheBudget::PayloadBytesPerNode(meta, kMaxDegree);
  const uint64_t estimated =
      DiskAnnCacheBudget::EstimatedBytesPerNode(meta, kMaxDegree);

  EXPECT_EQ(payload, 768u * sizeof(float) + 33u * sizeof(diskann_id_t));
  EXPECT_GT(estimated, payload);
}

TEST(DiskAnnCacheBudgetTest, ResolvesBudgetAndAppliesDocumentCountCap) {
  constexpr uint64_t kBytesPerNode = 2048;
  constexpr uint64_t kDocCount = 10000;

  EXPECT_EQ(
      DiskAnnCacheBudget::ResolveNodeCount(100 * kBytesPerNode, kDocCount,
                                           kBytesPerNode),
      100u);
  EXPECT_EQ(
      DiskAnnCacheBudget::ResolveNodeCount(2000 * kBytesPerNode, kDocCount,
                                           kBytesPerNode),
      2000u);
  EXPECT_EQ(
      DiskAnnCacheBudget::ResolveNodeCount(20000 * kBytesPerNode, kDocCount,
                                           kBytesPerNode),
      10000u);
  EXPECT_EQ(
      DiskAnnCacheBudget::ResolveNodeCount(kBytesPerNode - 1, kDocCount,
                                           kBytesPerNode),
      0u);
}

TEST(DiskAnnCacheBudgetTest, RejectsConflictingNodeCountAndByteBudget) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, 100);
  params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_BUDGET_BYTES, 1024UL * 1024UL);
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(params));
}

TEST(DiskAnnCacheBudgetTest, RejectsUnknownNodePagePolicy) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_PAGE_POLICY,
             std::string("unknown"));
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(params));
}

TEST(DiskAnnCacheBudgetTest, RejectsUnknownWarmupMode) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_WARMUP_MODE, std::string("unknown"));
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(params));
}

TEST(DiskAnnCacheBudgetTest, RejectsWarmupWithoutNodeCount) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_WARMUP_MODE, DISKANN_WARMUP_MODE_BFS);
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(params));
}

TEST(DiskAnnCacheBudgetTest, RejectsQuerySampleWithoutNodeList) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_WARMUP_MODE,
             DISKANN_WARMUP_MODE_QUERY_SAMPLE);
  params.set(PARAM_DISKANN_SEARCHER_WARMUP_NODE_NUM, 100);
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(params));
}

TEST(DiskAnnCacheBudgetTest, AcceptsBfsAndQuerySampleWarmup) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_WARMUP_MODE, DISKANN_WARMUP_MODE_BFS);
  params.set(PARAM_DISKANN_SEARCHER_WARMUP_NODE_NUM, 100);
  EXPECT_EQ(0, searcher->init(params));

  searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  params.set(PARAM_DISKANN_SEARCHER_WARMUP_MODE,
             DISKANN_WARMUP_MODE_QUERY_SAMPLE);
  params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_LIST_PATH,
             std::string("/tmp/query-sample-nodes.txt"));
  EXPECT_EQ(0, searcher->init(params));
}

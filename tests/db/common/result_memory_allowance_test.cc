// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/common/result_memory_allowance.h"
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "db/common/query_result_memory.h"
#include "db/index/column/fts_column/fts_memory.h"
#include "db/index/column/inverted_column/inverted_search_result.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::detail {
namespace {

constexpr size_t kMiB = 1024 * 1024;

class ResultMemoryAllowanceTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(64 * kMiB));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
};

TEST_F(ResultMemoryAllowanceTest, RetainsAndReusesBackingAfterCallsFinish) {
  ResultMemoryAllowance allowance;
  for (int i = 0; i < 3; ++i) {
    {
      ResultMemoryAllowance::Lease lease(&allowance);
      lease.add(2 * kMiB);
      lease.add(kMiB);
      EXPECT_EQ(3 * kMiB, allowance.active_bytes());
      EXPECT_EQ(6 * kMiB, allowance.reserved_bytes());
    }
    EXPECT_EQ(0u, allowance.active_bytes());
    EXPECT_EQ(6 * kMiB, pool_.external_used());
  }
}

TEST_F(ResultMemoryAllowanceTest, ConcurrentCallsShareOneHighWaterAllowance) {
  ResultMemoryAllowance allowance;
  std::atomic<size_t> ready{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      ResultMemoryAllowance::Lease lease(&allowance);
      lease.add(kMiB);
      ++ready;
      while (!release.load()) std::this_thread::yield();
    });
  }
  while (ready.load() != 4) std::this_thread::yield();
  EXPECT_EQ(4 * kMiB, allowance.active_bytes());
  EXPECT_EQ(8 * kMiB, pool_.external_used());
  release = true;
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(8 * kMiB, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, RejectedGrowthLeavesLeaseAndChargeUnchanged) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease lease(&allowance);
    lease.add(kMiB);
    const size_t pressure = pool_.capacity() - pool_.used();
    ASSERT_TRUE(pool_.try_charge_external(pressure));
    EXPECT_THROW(lease.add(kMiB), std::bad_alloc);
    EXPECT_EQ(kMiB, allowance.active_bytes());
    EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
    pool_.release_external(pressure);
    // Retry succeeds once the competing owner returns real capacity.
    lease.add(kMiB);
    EXPECT_EQ(2 * kMiB, allowance.active_bytes());
  }
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, OverflowDoesNotConsumeBudget) {
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance);
  EXPECT_THROW(lease.add(std::numeric_limits<size_t>::max()), std::bad_alloc);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, AnalyzerExpansionIsAdmittedBeforeAllocation) {
  FtsIndexParams standard;
  FtsIndexParams ngram("ngram", {}, R"({"ngram_min":2048,"ngram_max":2048})");
  const std::string text(32768, 'a');
  const size_t normal = EstimateFtsWorkspace(standard, text);
  const size_t expanded = EstimateFtsWorkspace(ngram, text);
  EXPECT_GT(expanded, 512 * kMiB);
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance);
  lease.add(normal);
  const size_t reserved = allowance.reserved_bytes();
  EXPECT_THROW(lease.add(expanded), std::bad_alloc);
  EXPECT_EQ(reserved, allowance.reserved_bytes());
  EXPECT_EQ(normal, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest, MaterializersWaitBeforeAccumulatingChunks) {
  ResultMemoryAllowance allowance;
  auto first = std::make_unique<ResultMemoryAllowance::Lease>(&allowance, true);
  first->add(kMiB);
  std::promise<void> entered;
  auto waiting = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease second(&allowance, true);
    entered.set_value();
    second.add(kMiB);
    second.add(kMiB);
    EXPECT_EQ(2 * kMiB, allowance.active_bytes());
  });
  entered.get_future().wait();
  EXPECT_EQ(std::future_status::timeout,
            waiting.wait_for(std::chrono::milliseconds(20)));
  // The first call can grow to completion while another call waits with no
  // partial result charge. No sleep or retry is needed for forward progress.
  first->add(2 * kMiB);
  EXPECT_EQ(3 * kMiB, allowance.active_bytes());
  first.reset();
  EXPECT_EQ(std::future_status::ready,
            waiting.wait_for(std::chrono::seconds(5)));
  waiting.get();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(6 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, FailedMaterializerUnblocksNextCall) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease oversized(&allowance, true);
    EXPECT_THROW(oversized.add(pool_.capacity()), std::bad_alloc);
  }
  auto next = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease lease(&allowance, true);
    lease.add(kMiB);
  });
  EXPECT_EQ(std::future_status::ready, next.wait_for(std::chrono::seconds(5)));
  next.get();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, OversizedCallDoesNotExhaustFutureCacheSpace) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease lease(&allowance);
    lease.add(kMiB);
    EXPECT_THROW(lease.add(pool_.capacity()), std::bad_alloc);
    EXPECT_EQ(kMiB, allowance.active_bytes());
    EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
  }
  ResultMemoryAllowance::Lease next(&allowance);
  EXPECT_NO_THROW(next.add(kMiB));
  EXPECT_EQ(2 * kMiB, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, DisabledAndUnconfiguredPathsDoNotReserve) {
  ResultMemoryAllowance::Lease disabled(nullptr);
  EXPECT_FALSE(disabled.enabled());
  disabled.add(kMiB);
  EXPECT_EQ(0u, pool_.external_used());
  ASSERT_EQ(0, pool_.init(0));
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease lease(&allowance);
  lease.add(kMiB);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, BitmapFilterAndIteratorKeepAdmission) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  memory->add(kMiB);
  auto *bitmap = roaring_bitmap_create();
  ASSERT_NE(nullptr, bitmap);
  roaring_bitmap_add(bitmap, 7);
  auto result = std::make_shared<InvertedSearchResult>(bitmap);
  result->keep_memory(memory);
  auto filter = result->make_filter();
  auto iter = result->create_iterator();
  memory.reset();
  result.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_FALSE(filter->is_filtered(7));
  filter.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_EQ(7u, iter->doc_id());
  iter.reset();
  EXPECT_EQ(0u, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest, QueryDocumentKeepsAdmissionUntilLastOwner) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  memory->add(kMiB);
  auto first = MakeQueryDocument(memory);
  first->set_pk("retained");
  auto second = first;
  memory.reset();
  first.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_EQ("retained", second->pk_ref());
  second.reset();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, QueryEstimateCountsOnlySelectedStringSlice) {
  arrow::StringBuilder builder;
  ASSERT_TRUE(builder.Append(std::string(4096, 'x')).ok());
  ASSERT_TRUE(builder.Append("abc").ok());
  std::shared_ptr<arrow::Array> values;
  ASSERT_TRUE(builder.Finish(&values).ok());
  auto schema = arrow::schema({arrow::field("payload", arrow::utf8())});
  auto small = arrow::RecordBatch::Make(schema, 1, {values->Slice(1, 1)});
  auto large = arrow::RecordBatch::Make(schema, 1, {values->Slice(0, 1)});
  EXPECT_EQ(4096u - 3u,
            EstimateQueryResult(*large) - EstimateQueryResult(*small));
}

TEST_F(ResultMemoryAllowanceTest, QueryEstimateCountsNestedStringsAndNulls) {
  auto child = std::make_shared<arrow::StringBuilder>();
  arrow::ListBuilder builder(arrow::default_memory_pool(), child);
  ASSERT_TRUE(builder.Append().ok());
  ASSERT_TRUE(child->Append(std::string(4096, 'a')).ok());
  ASSERT_TRUE(child->Append("b").ok());
  ASSERT_TRUE(builder.AppendNull().ok());
  std::shared_ptr<arrow::Array> values;
  ASSERT_TRUE(builder.Finish(&values).ok());
  QueryResultEstimate estimate;
  estimate.value(*values, 1);
  EXPECT_EQ(0u, estimate.bytes());
  estimate.value(*values, 0);
  EXPECT_GE(estimate.bytes(), 4097u + 2 * 32u);
  EXPECT_THROW(estimate.add(std::numeric_limits<size_t>::max(), 2),
               std::bad_alloc);
}

}  // namespace
}  // namespace zvec::detail

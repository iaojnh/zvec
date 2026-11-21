// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_dumper.h>
#include "diskann_pq_read_batch.h"
#include "diskann_resident_pq_batch.h"
#include "utility/utility_params.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class ResidentPqBatchTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(8UL << 20));
    path_ = "resident-pq-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_));
    // Unaligned segment start also exercises crossings at nonstandard IDs.
    const char prefix[32]{};
    ASSERT_EQ(sizeof(prefix), dumper->write(prefix, sizeof(prefix)));
    ASSERT_EQ(0, dumper->append("prefix", sizeof(prefix), 0, 0));
    codes_.resize(2048 * 384);
    for (size_t i = 0; i < codes_.size(); ++i) codes_[i] = i % 251;
    ASSERT_EQ(codes_.size(), dumper->write(codes_.data(), codes_.size()));
    const std::string padding(4096 - codes_.size() % 4096, '\0');
    ASSERT_EQ(padding.size(), dumper->write(padding.data(), padding.size()));
    ASSERT_EQ(0, dumper->append("codes", codes_.size(), padding.size(), 0));
    ASSERT_EQ(0, dumper->close());
    storage_ = IndexFactory::CreateStorage("BufferReadStorage");
    ASSERT_NE(nullptr, storage_);
    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE, BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_EQ(0, storage_->init(params));
    ASSERT_EQ(0, storage_->open(path_, false));
    segment_ = storage_->get("codes");
    ASSERT_NE(nullptr, segment_);
  }
  void TearDown() override {
    segment_.reset();
    if (storage_) EXPECT_EQ(0, storage_->close());
    storage_.reset();
    std::remove(path_.c_str());
  }
  void warm(const uint32_t *ids, size_t count) {
    ailego::BudgetedVector<IndexStorage::MemoryBlock> blocks;
    ailego::BudgetedVector<const void *> views;
    ASSERT_EQ(0, ReadPagedQuantizedCodes(*segment_, ids, count, 384, blocks, views));
  }
  void evict_page(size_t page) {
    auto pool = storage_->vec_buffer_pool();
    for (size_t i = 0; i < 64 && pool->is_page_resident(page); ++i)
      ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  }
  std::string path_, codes_;
  IndexStorage::Pointer storage_;
  IndexStorage::Segment::Pointer segment_;
};

TEST_F(ResidentPqBatchTest, ResidentCrossingsAndDuplicatesMatchOriginalBytes) {
  uint32_t ids[130];
  for (size_t i = 0; i < 130; ++i) ids[i] = (i * 7) % 301;
  ids[129] = ids[0];
  warm(ids, 130);
  ResidentPqBatch batch;
  ailego::BudgetedVector<const void *> views;
  ASSERT_TRUE(batch.read(*storage_->vec_buffer_pool(), *segment_, ids, 130, 384, views));
  for (size_t i = 0; i < 130; ++i)
    EXPECT_EQ(0, std::memcmp(views[i], codes_.data() + ids[i] * 384, 384));
  batch.release();
  // The same scratch is reused even when the crossing count changes.
  const uint32_t other[]{0, 10, 21, 32};
  warm(other, 4);
  ASSERT_TRUE(batch.read(*storage_->vec_buffer_pool(), *segment_, other, 4, 384, views));
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(0, std::memcmp(views[i], codes_.data() + other[i] * 384, 384));
}

TEST_F(ResidentPqBatchTest, MissRollsBackPinsBeforeOriginalPagedFallback) {
  const uint32_t first[]{0};
  warm(first, 1);
  const uint32_t ids[]{0, 1000};
  auto pool = storage_->vec_buffer_pool();
  const auto before = ailego::MemoryLimitPool::get_instance().stats();
  ResidentPqBatch batch;
  ailego::BudgetedVector<const void *> views;
  EXPECT_FALSE(batch.read(*pool, *segment_, ids, 2, 384, views));
  EXPECT_EQ(before.page_used, ailego::MemoryLimitPool::get_instance().stats().page_used);
  const size_t page = segment_->data_offset() / ailego::kVectorPageSize;
  evict_page(page);
  EXPECT_FALSE(pool->is_page_resident(page));
  ailego::BudgetedVector<IndexStorage::MemoryBlock> blocks;
  ASSERT_EQ(0, ReadPagedQuantizedCodes(*segment_, ids, 2, 384, blocks, views));
  for (size_t i = 0; i < 2; ++i)
    EXPECT_EQ(0, std::memcmp(views[i], codes_.data() + ids[i] * 384, 384));
}

TEST_F(ResidentPqBatchTest, ReleaseMakesDeduplicatedPagesEvictable) {
  const uint32_t ids[]{0, 0, 0, 1};
  warm(ids, 4);
  auto pool = storage_->vec_buffer_pool();
  const size_t page = segment_->data_offset() / ailego::kVectorPageSize;
  ResidentPqBatch batch;
  ailego::BudgetedVector<const void *> views;
  ASSERT_TRUE(batch.read(*pool, *segment_, ids, 4, 384, views));
  evict_page(page);
  EXPECT_TRUE(pool->is_page_resident(page));
  batch.release();
  batch.release();
  evict_page(page);
  EXPECT_FALSE(pool->is_page_resident(page));
}

TEST_F(ResidentPqBatchTest, InvalidIdsAndLargeBatchesUseFallbackWithoutPins) {
  uint32_t ids[257]{};
  ResidentPqBatch batch;
  ailego::BudgetedVector<const void *> views;
  EXPECT_FALSE(batch.read(*storage_->vec_buffer_pool(), *segment_, ids, 257, 384, views));
  ids[0] = UINT32_MAX;
  EXPECT_FALSE(batch.read(*storage_->vec_buffer_pool(), *segment_, ids, 1, 384, views));
  EXPECT_FALSE(batch.read(*storage_->vec_buffer_pool(), *segment_, nullptr, 1, 384, views));
}

TEST_F(ResidentPqBatchTest, MixedColdBatchPreservesCrossingsDuplicatesAndOrder) {
  const uint32_t warm_ids[]{0, 10};
  warm(warm_ids, 2);
  uint32_t ids[130];
  for (size_t i = 0; i < 130; ++i) ids[i] = (i * 13) % 2048;
  ids[129] = ids[0];
  auto pool = storage_->vec_buffer_pool();
  const auto misses = pool->stats().miss;
  ResidentPqBatch batch;
  ailego::BudgetedVector<const void *> views;
  ASSERT_TRUE(batch.read(*pool, *segment_, ids, 130, 384, views,
                         /*load_misses=*/true));
  EXPECT_GT(pool->stats().miss, misses);
  for (size_t i = 0; i < 130; ++i)
    EXPECT_EQ(0, std::memcmp(views[i], codes_.data() + ids[i] * 384, 384));
  batch.release();
  const auto after_cold = pool->stats().miss;
  ASSERT_TRUE(batch.read(*pool, *segment_, ids, 130, 384, views,
                         /*load_misses=*/true));
  EXPECT_EQ(after_cold, pool->stats().miss);
  batch.release();
  const size_t first = segment_->data_offset() / ailego::kVectorPageSize;
  evict_page(first);
  EXPECT_FALSE(pool->is_page_resident(first));
}

TEST_F(ResidentPqBatchTest, InvalidColdBatchDoesNotReadOrLeavePins) {
  const uint32_t ids[]{0, UINT32_MAX};
  auto pool = storage_->vec_buffer_pool();
  const auto before = pool->stats().miss;
  ResidentPqBatch batch;
  ailego::BudgetedVector<const void *> views;
  EXPECT_FALSE(batch.read(*pool, *segment_, ids, 2, 384, views,
                          /*load_misses=*/true));
  EXPECT_EQ(before, pool->stats().miss);
}

TEST_F(ResidentPqBatchTest, ConcurrentColdBatchesSharePagesWithoutChangingCodes) {
  std::atomic<bool> start{false};
  std::atomic<int> errors{0};
  std::vector<std::thread> threads;
  auto pool = storage_->vec_buffer_pool();
  for (size_t worker = 0; worker < 4; ++worker) {
    threads.emplace_back([&, worker] {
      ResidentPqBatch batch;
      ailego::BudgetedVector<const void *> views;
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (size_t pass = 0; pass < 8; ++pass) {
        uint32_t ids[130];
        for (size_t i = 0; i < 130; ++i)
          ids[i] = (i * 13 + pass * 7 + worker) % 2048;
        if (!batch.read(*pool, *segment_, ids, 130, 384, views,
                         /*load_misses=*/true)) {
          ++errors;
          continue;
        }
        for (size_t i = 0; i < 130; ++i)
          if (std::memcmp(views[i], codes_.data() + ids[i] * 384, 384) != 0)
            ++errors;
        batch.release();
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(0, errors.load());
}

TEST_F(ResidentPqBatchTest, CapacityFailureReleasesPinsAndAllowsLaterReload) {
  uint32_t ids[130];
  for (size_t i = 0; i < 130; ++i) ids[i] = (i * 13) % 2048;
  auto pool = storage_->vec_buffer_pool();
  auto &budget = ailego::MemoryLimitPool::get_instance();
  ResidentPqBatch batch;
  ailego::BudgetedVector<const void *> views;
  {
    // Leave room for scratch and a few pages, but not every page in this
    // batch. Optional population must release partial pins/load claims.
    ASSERT_GT(budget.available(), 128UL << 10);
    ailego::MemoryReservation other_workspace;
    other_workspace.add(budget.available() - (128UL << 10));
    EXPECT_FALSE(batch.read(*pool, *segment_, ids, 130, 384, views,
                            /*load_misses=*/true));
    EXPECT_LE(budget.used(), budget.capacity());
  }
  ASSERT_TRUE(batch.read(*pool, *segment_, ids, 130, 384, views,
                         /*load_misses=*/true));
  for (size_t i = 0; i < 130; ++i)
    EXPECT_EQ(0, std::memcmp(views[i], codes_.data() + ids[i] * 384, 384));
}
}  // namespace
}  // namespace zvec::core

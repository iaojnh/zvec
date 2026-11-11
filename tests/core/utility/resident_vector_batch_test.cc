// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_factory.h>
#include "utility/resident_vector_batch.h"
#include "utility/utility_params.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class ResidentVectorBatchTest : public testing::Test {
 protected:
  static constexpr size_t kRows = 128, kBytes = 3072, kStride = 3104;
  void SetUp() override {
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(16UL << 20));
    path_ = "resident-vector-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_));
    const std::string prefix(32, '\0');
    ASSERT_EQ(prefix.size(), dumper->write(prefix.data(), prefix.size()));
    ASSERT_EQ(0, dumper->append("prefix", prefix.size(), 0, 0));
    for (size_t chunk = 0; chunk < 2; ++chunk) {
      payload_[chunk].resize(kRows * kStride);
      for (size_t i = 0; i < payload_[chunk].size(); ++i)
        payload_[chunk][i] = (i + chunk * 41) % 251;
      ASSERT_EQ(payload_[chunk].size(), dumper->write(payload_[chunk].data(), payload_[chunk].size()));
      ASSERT_EQ(0, dumper->append("chunk" + std::to_string(chunk), payload_[chunk].size(), 0, 0));
    }
    ASSERT_EQ(0, dumper->close());
    storage_ = IndexFactory::CreateStorage("BufferReadStorage");
    ASSERT_NE(nullptr, storage_);
    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE, BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_EQ(0, storage_->init(params));
    ASSERT_EQ(0, storage_->open(path_, false));
    for (size_t chunk = 0; chunk < 2; ++chunk) {
      segments_[chunk] = storage_->get("chunk" + std::to_string(chunk));
      ASSERT_NE(nullptr, segments_[chunk]);
    }
  }
  void TearDown() override {
    segments_[0].reset(); segments_[1].reset();
    if (storage_) EXPECT_EQ(0, storage_->close());
    storage_.reset();
    std::remove(path_.c_str());
  }
  void warm(const uint32_t *ids, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      IndexStorage::MemoryBlock block;
      ASSERT_EQ(kBytes, segments_[ids[i] / kRows]->read_borrowed(
          (ids[i] % kRows) * kStride, block, kBytes));
    }
  }
  bool read(ResidentVectorBatch &batch, const uint32_t *ids, size_t count,
            ailego::BudgetedVector<const void *> &views) {
    return batch.read_offsets(*storage_->vec_buffer_pool(), ids, count, kBytes,
        [&](uint32_t id, size_t *offset) {
          if (id >= 2 * kRows) return false;
          *offset = segments_[id / kRows]->data_offset() + (id % kRows) * kStride;
          return true;
        }, views);
  }
  const char *expected(uint32_t id) {
    return payload_[id / kRows].data() + (id % kRows) * kStride;
  }
  void evict(size_t page) {
    auto pool = storage_->vec_buffer_pool();
    for (size_t i = 0; i < 64 && pool->is_page_resident(page); ++i)
      ailego::BlockEvictionQueue::get_instance().batch_recycle(256);
  }
  std::string path_, payload_[2];
  IndexStorage::Pointer storage_;
  IndexStorage::Segment::Pointer segments_[2];
};

TEST_F(ResidentVectorBatchTest, CrossSegmentStridedVectorsMatchAllOriginalBytes) {
  uint32_t ids[80];
  for (size_t i = 0; i < 80; ++i) ids[i] = (i * 17) % 256;
  ids[79] = ids[0];
  warm(ids, 80);
  ResidentVectorBatch batch;
  ailego::BudgetedVector<const void *> views;
  ASSERT_TRUE(read(batch, ids, 80, views));
  for (size_t i = 0; i < 80; ++i)
    EXPECT_EQ(0, std::memcmp(expected(ids[i]), views[i], kBytes));
  batch.release();
  const uint32_t next[]{2, 129, 200};
  warm(next, 3);
  ASSERT_TRUE(read(batch, next, 3, views));
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(0, std::memcmp(expected(next[i]), views[i], kBytes));
}

TEST_F(ResidentVectorBatchTest, MissingPageRollsBackPinsAndNeverReadsDisk) {
  const uint32_t first[]{0};
  warm(first, 1);
  const uint32_t ids[]{0, 250};
  auto pool = storage_->vec_buffer_pool();
  const auto page = segments_[0]->data_offset() / ailego::kVectorPageSize;
  const auto before = ailego::MemoryLimitPool::get_instance().stats().page_used;
  ResidentVectorBatch batch;
  ailego::BudgetedVector<const void *> views;
  EXPECT_FALSE(read(batch, ids, 2, views));
  EXPECT_EQ(before, ailego::MemoryLimitPool::get_instance().stats().page_used);
  evict(page);
  EXPECT_FALSE(pool->is_page_resident(page));
  // The original read still works after the failed resident-only probe.
  warm(ids, 2);
  ASSERT_TRUE(read(batch, ids, 2, views));
  EXPECT_EQ(0, std::memcmp(expected(250), views[1], kBytes));
}

TEST_F(ResidentVectorBatchTest, InvalidLocationRetainsNoPages) {
  const uint32_t ids[]{0, 500};
  warm(ids, 1);
  ResidentVectorBatch batch;
  ailego::BudgetedVector<const void *> views;
  EXPECT_FALSE(read(batch, ids, 2, views));
  const auto page = segments_[0]->data_offset() / ailego::kVectorPageSize;
  evict(page);
  EXPECT_FALSE(storage_->vec_buffer_pool()->is_page_resident(page));
}

TEST_F(ResidentVectorBatchTest, BudgetPressureKeepsReadsBoundedAndReleasesPins) {
  const uint32_t ids[]{0, 129};
  warm(ids, 2);
  ResidentVectorBatch batch;
  ailego::BudgetedVector<const void *> views;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.available());
    // Reclaimed pages or reusable scratch can make either outcome valid.
    // A successful probe must retain correct bytes; refusal must stay bounded.
    if (read(batch, ids, 2, views))
      for (size_t i = 0; i < 2; ++i)
        EXPECT_EQ(0, std::memcmp(expected(ids[i]), views[i], kBytes));
    batch.release();
    EXPECT_LE(pool.used(), pool.capacity());
  }
  const auto page = segments_[0]->data_offset() / ailego::kVectorPageSize;
  evict(page);
  EXPECT_FALSE(storage_->vec_buffer_pool()->is_page_resident(page));
  warm(ids, 2);
  ASSERT_TRUE(read(batch, ids, 2, views));
  for (size_t i = 0; i < 2; ++i)
    EXPECT_EQ(0, std::memcmp(expected(ids[i]), views[i], kBytes));
}

TEST_F(ResidentVectorBatchTest, UnsupportedWidthFallsBackBeforeAcquiringPages) {
  ResidentVectorBatch batch;
  ailego::BudgetedVector<const void *> views;
  const uint32_t ids[]{0};
  bool located = false;
  EXPECT_FALSE(batch.read_offsets(*storage_->vec_buffer_pool(), ids, 1,
      ailego::kVectorPageSize + 1,
      [&](uint32_t, size_t *) { located = true; return true; }, views));
  EXPECT_FALSE(located);
}
}  // namespace
}  // namespace zvec::core

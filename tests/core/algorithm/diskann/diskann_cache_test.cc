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

#include <limits>
#include <cstdio>
#include <cstring>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_factory.h>
#include "diskann_pq_read_batch.h"
#include "diskann_params.h"
#include "diskann_util.h"
#include "utility/utility_params.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

using namespace zvec::core;

namespace {

class CountingPqSegment : public IndexStorage::Segment {
 public:
  explicit CountingPqSegment(Pointer source) : source_(std::move(source)) {}
  size_t data_size() const override { return source_->data_size(); }
  size_t data_offset() const override { return source_->data_offset(); }
  uint32_t data_crc() const override { return source_->data_crc(); }
  size_t padding_size() const override { return source_->padding_size(); }
  size_t capacity() const override { return source_->capacity(); }
  size_t fetch(size_t offset, void *data, size_t length) const override {
    return source_->fetch(offset, data, length);
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &data,
              size_t length) override {
    return source_->read(offset, data, length);
  }
  size_t read_borrowed(size_t offset, IndexStorage::MemoryBlock &data,
                       size_t length) override {
    ++read_calls;
    return source_->read_borrowed(offset, data, length);
  }
  int read_error() const override { return source_->read_error(); }
  size_t write(size_t, const void *, size_t) override { return 0; }
  size_t resize(size_t) override { return 0; }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override { return source_->clone(); }
  size_t read_calls{0};

 private:
  Pointer source_;
};

class DiskAnnPqReadBatchTest : public testing::Test {
 protected:
  void SetUp() override {
    namespace ailego = zvec::ailego;
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(16UL << 20));
    payload_.resize(12 * ailego::kVectorPageSize);
    for (size_t i = 0; i < payload_.size(); ++i)
      payload_[i] = static_cast<char>(i % 251);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_));
    ASSERT_EQ(payload_.size(), dumper->write(payload_.data(), payload_.size()));
    ASSERT_EQ(0, dumper->append("pq", payload_.size(), 0, 0));
    ASSERT_EQ(0, dumper->close());
    storage_ = IndexFactory::CreateStorage("BufferReadStorage");
    ASSERT_NE(nullptr, storage_);
    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE, BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_EQ(0, storage_->init(params));
    ASSERT_EQ(0, storage_->open(path_, false));
    auto segment = storage_->get("pq");
    ASSERT_NE(nullptr, segment);
    segment_ = std::make_shared<CountingPqSegment>(segment);
    ASSERT_EQ(0u, segment_->data_offset() % 16);
  }
  void TearDown() override {
    blocks_ = decltype(blocks_){};
    views_ = decltype(views_){};
    segment_.reset();
    if (storage_) {
      EXPECT_EQ(0, storage_->close());
    }
    storage_.reset();
    EXPECT_EQ(0u, zvec::ailego::PoolPayloadBudget::get_instance().used());
    std::remove(path_);
  }
  void expect_codes(const uint32_t *ids, size_t count, size_t size,
                     const zvec::ailego::BudgetedVector<const void *> &views) {
    ASSERT_EQ(count, views.size());
    for (size_t i = 0; i < count; ++i) {
      ASSERT_NE(nullptr, views[i]);
      EXPECT_EQ(0, std::memcmp(views[i], payload_.data() + ids[i] * size, size));
    }
  }
  uint32_t cross_page_id(size_t code_size) const {
    const size_t page = zvec::ailego::kVectorPageSize;
    for (size_t offset = 0; offset + code_size <= payload_.size();
         offset += code_size) {
      if ((segment_->data_offset() + offset) % page + code_size > page)
        return static_cast<uint32_t>(offset / code_size);
    }
    ADD_FAILURE() << "fixture has no cross-page code";
    return 0;
  }
  zvec::ailego::BudgetedVector<IndexStorage::MemoryBlock> blocks_;
  zvec::ailego::BudgetedVector<const void *> views_;
  IndexStorage::Pointer storage_;
  std::shared_ptr<CountingPqSegment> segment_;
  std::string payload_;
  const char *path_{"diskann-pq-read-batch-test-file"};
};

TEST_F(DiskAnnPqReadBatchTest, SamePageCodesShareOneBatchPinUntilRelease) {
  namespace ailego = zvec::ailego;
  const uint32_t ids[]{1, 2, 15, 2};
  ASSERT_EQ(0, ReadPagedQuantizedCodes(*segment_, ids, 4, 16, blocks_, views_));
  EXPECT_EQ(1u, segment_->read_calls);
  size_t pinned = 0;
  for (const auto &block : blocks_)
    if (block.type_ == IndexStorage::MemoryBlock::MBT_BUFFERPOOL) ++pinned;
  EXPECT_EQ(1u, pinned);
  auto pool = storage_->vec_buffer_pool();
  const auto page = (segment_->data_offset() + 16) / ailego::kVectorPageSize;
  for (size_t i = 0; i < 16; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  EXPECT_TRUE(pool->is_page_resident(page));
  expect_codes(ids, 4, 16, views_);
  blocks_.clear();
  for (size_t i = 0; i < 16; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  EXPECT_FALSE(pool->is_page_resident(page));
}

TEST_F(DiskAnnPqReadBatchTest, CacheCollisionKeepsEarlierPageViewsValid) {
  const uint32_t distant = 8 * zvec::ailego::kVectorPageSize / 16 + 1;
  const uint32_t ids[]{1, distant, 3};
  ASSERT_EQ(0, ReadPagedQuantizedCodes(*segment_, ids, 3, 16, blocks_, views_));
  EXPECT_EQ(3u, segment_->read_calls);
  expect_codes(ids, 3, 16, views_);
}

TEST_F(DiskAnnPqReadBatchTest, CrossPageCodesKeepSeparateOwnedReads) {
  const uint32_t cross = cross_page_id(24);
  const uint32_t ids[]{cross, cross};
  ASSERT_GT((segment_->data_offset() + cross * 24) % zvec::ailego::kVectorPageSize +
                24,
            zvec::ailego::kVectorPageSize);
  ASSERT_EQ(0, ReadPagedQuantizedCodes(*segment_, ids, 2, 24, blocks_, views_));
  EXPECT_EQ(2u, segment_->read_calls);
  EXPECT_NE(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, blocks_[0].type_);
  EXPECT_NE(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, blocks_[1].type_);
  EXPECT_NE(views_[0], views_[1]);
  expect_codes(ids, 2, 24, views_);
}

TEST_F(DiskAnnPqReadBatchTest, ShortReadRollsBackEarlierViewsAndPins) {
  const uint32_t ids[]{1, static_cast<uint32_t>(segment_->capacity() / 16 + 1)};
  EXPECT_EQ(IndexError_ReadData,
            ReadPagedQuantizedCodes(*segment_, ids, 2, 16, blocks_, views_));
  EXPECT_TRUE(blocks_.empty());
  EXPECT_TRUE(views_.empty());
  EXPECT_EQ(0, ReadPagedQuantizedCodes(*segment_, nullptr, 0, 16, blocks_, views_));
}

TEST_F(DiskAnnPqReadBatchTest, NoMemoryRollsBackEarlierViewsAndPins) {
  namespace ailego = zvec::ailego;
  blocks_.resize(2);
  views_.resize(2);
  IndexStorage::MemoryBlock warm;
  ASSERT_EQ(24u, segment_->read_borrowed(0, warm, 24));
  auto &budget = ailego::PoolPayloadBudget::get_instance();
  const size_t pressure = budget.available();
  ASSERT_TRUE(budget.try_charge_external(pressure));
  AILEGO_DEFER([&] { budget.release_external(pressure); });
  ASSERT_EQ(0u, budget.available());
  const uint32_t ids[]{0, cross_page_id(24)};
  EXPECT_EQ(IndexError_NoMemory,
            ReadPagedQuantizedCodes(*segment_, ids, 2, 24, blocks_, views_));
  EXPECT_TRUE(blocks_.empty());
  EXPECT_TRUE(views_.empty());
  warm.reset();
  for (size_t i = 0; i < 16; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  EXPECT_FALSE(storage_->vec_buffer_pool()->is_page_resident(
      segment_->data_offset() / ailego::kVectorPageSize));
}

}  // namespace

TEST(DiskAnnCachePreloadTest, BoundsBatchByScratchBufferCapacity) {
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(0), 1u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(1), 128u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(2), 64u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(128), 1u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(129), 1u);
}

TEST(DiskAnnCacheConfigTest, FailedReinitKeepsPreviousValidConfiguration) {
  zvec::ailego::Params valid;
  valid.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 321);
  valid.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, 7);

  zvec::ailego::Params invalid;
  invalid.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, -1);

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(valid));
  ASSERT_EQ(IndexError_InvalidArgument, searcher->init(invalid));
  uint32_t list_size = 0;
  long long cache_nodes = 0;
  EXPECT_TRUE(
      searcher->params().get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size));
  EXPECT_TRUE(searcher->params().get(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM,
                                     &cache_nodes));
  EXPECT_EQ(list_size, 321u);
  EXPECT_EQ(cache_nodes, 7);

  auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
  IndexMeta invalid_meta(IndexMeta::DataType::DT_FP32, 16);
  ASSERT_EQ(0, streamer->init(meta, valid));
  ASSERT_EQ(IndexError_InvalidArgument, streamer->init(invalid_meta, invalid));
  EXPECT_EQ(streamer->meta().dimension(), 8u);
}

TEST(DiskAnnCacheConfigTest, RejectsOutOfRangeNodeCount) {
  zvec::ailego::Params negative;
  negative.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, -1);

  zvec::ailego::Params too_large;
  const long long too_many_nodes =
      static_cast<long long>((std::numeric_limits<uint32_t>::max)()) + 1;
  too_large.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, too_many_nodes);

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(negative));
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(too_large));

  auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
  EXPECT_EQ(IndexError_InvalidArgument, streamer->init(meta, negative));
  EXPECT_EQ(IndexError_InvalidArgument, streamer->init(meta, too_large));
}

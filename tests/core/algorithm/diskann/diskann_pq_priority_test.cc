// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include <turbo/quantizer/quantizer.h>
#include "diskann_searcher_entity.h"
#include "utility/utility_params.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

class DiskAnnPqPriorityTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(8UL << 20));
    path_ = "diskann-pq-priority-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    meta_.set_meta(IndexMeta::DT_FP32, 64);
    meta_.set_metric("SquaredEuclidean", 0, ailego::Params());
    meta_.set_builder("DiskAnnBuilder", DiskAnnEntity::kRevision, ailego::Params());
  }

  void TearDown() override {
    entity_.clear();
    if (storage_) EXPECT_EQ(0, storage_->close());
    storage_.reset();
    std::remove(path_.c_str());
  }

  void make_index(size_t rows = 2048) {
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_));
    auto append = [&](const std::string &name, const void *data, size_t size) {
      EXPECT_EQ(size, dumper->write(data, size));
      const size_t padding = (4096 - size % 4096) % 4096;
      const std::string zeros(padding, '\0');
      EXPECT_EQ(padding, dumper->write(zeros.data(), padding));
      EXPECT_EQ(0, dumper->append(name, size, padding, 0));
    };
    // Deliberately leave all following segments off the pool-page boundary.
    const std::string prefix(32, '\0');
    ASSERT_EQ(prefix.size(), dumper->write(prefix.data(), prefix.size()));
    ASSERT_EQ(0, dumper->append("prefix", prefix.size(), 0, 0));
    DiskAnnMetaHeader header;
    header.doc_cnt = rows;
    header.ndims = 64;
    header.max_node_size = 64 * sizeof(float) + sizeof(uint32_t);
    header.node_per_sector = 4096 / header.max_node_size;
    append(DiskAnnEntity::kDiskAnnMetaSegmentId, &header, sizeof(header));
    DiskAnnPqMeta pq_meta;
    pq_meta.chunk_num = 64;
    // The entity owns serialized metadata; the indexer decodes it separately.
    // Only the serialized header's extent is needed by the entity itself.
    pq_meta.quantizer_meta_buffer_size = sizeof(turbo::QuantizerSerHeader);
    std::string model(reinterpret_cast<const char *>(&pq_meta), sizeof(pq_meta));
    model.append(pq_meta.quantizer_meta_buffer_size, '\0');
    append(DiskAnnEntity::kDiskAnnPqMetaSegmentId, model.data(), model.size());
    codes_.resize(rows * 64);
    for (size_t i = 0; i < codes_.size(); ++i) codes_[i] = static_cast<char>(i % 251);
    append(DiskAnnEntity::kDiskAnnPqDataSegmentId, codes_.data(), codes_.size());
    const size_t sectors = (rows + header.node_per_sector - 1) / header.node_per_sector;
    const std::string graph(sectors * 4096, '\0');
    append(DiskAnnEntity::kDiskAnnVectorSegmentId, graph.data(), graph.size());
    std::vector<diskann_key_t> keys(rows);
    std::vector<diskann_id_t> ids(rows);
    for (size_t i = 0; i < rows; ++i) keys[i] = ids[i] = i;
    append(DiskAnnEntity::kDiskAnnKeySegmentId, keys.data(), keys.size() * sizeof(keys[0]));
    append(DiskAnnEntity::kDiskAnnKeyMappingSegmentId, ids.data(), ids.size() * sizeof(ids[0]));
    const uint32_t entrypoints[]{1, 0};
    append(DiskAnnEntity::kDiskAnnEntryPointSegmentId, entrypoints, sizeof(entrypoints));
    ASSERT_EQ(0, dumper->close());
  }

  void load(const char *type = "BufferReadStorage",
            const std::string &warmup = BUFFER_READ_STORAGE_WARMUP_NONE) {
    storage_ = IndexFactory::CreateStorage(type);
    ASSERT_NE(nullptr, storage_);
    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE, warmup);
    ASSERT_EQ(0, storage_->init(params));
    ASSERT_EQ(0, storage_->open(path_, false));
    ASSERT_EQ(0, entity_.load(meta_, storage_));
  }

  size_t first_pq_page() const {
    return entity_.paged_pq_codes()->data_offset() / ailego::kVectorPageSize;
  }

  void read_code(size_t id = 0) {
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(64u, entity_.paged_pq_codes()->read_borrowed(id * 64, block, 64));
    ASSERT_EQ(0, std::memcmp(block.data(), codes_.data() + id * 64, 64));
  }

  std::string path_, codes_;
  IndexMeta meta_;
  IndexStorage::Pointer storage_;
  DiskAnnSearcherEntity entity_;
};

TEST_F(DiskAnnPqPriorityTest, LoadingOnlySetsPriorityWithoutReadingPq) {
  make_index();
  load();
  auto pool = storage_->vec_buffer_pool();
  ASSERT_NE(nullptr, pool);
  ASSERT_EQ(nullptr, entity_.pq_codes());
  auto segment = entity_.paged_pq_codes();
  ASSERT_NE(nullptr, segment);
  const size_t last = (segment->data_offset() + codes_.size() - 1) / ailego::kVectorPageSize;
  for (size_t page = first_pq_page(); page <= last; ++page)
    EXPECT_FALSE(pool->is_page_resident(page));
  read_code();
  EXPECT_TRUE(pool->is_page_resident(first_pq_page()));
}

TEST_F(DiskAnnPqPriorityTest, AutoPreparationPopulatesOnlyPqAndRemainsEvictable) {
  make_index();
  load("BufferReadStorage", BUFFER_READ_STORAGE_WARMUP_AUTO);
  auto pool = storage_->vec_buffer_pool();
  EXPECT_FALSE(pool->is_page_resident(first_pq_page()));
  entity_.prepare_pq();
  const size_t last = (entity_.paged_pq_codes()->data_offset() +
                       codes_.size() - 1) / ailego::kVectorPageSize;
  for (size_t pg = first_pq_page(); pg <= last; ++pg)
    EXPECT_TRUE(pool->is_page_resident(pg));
  EXPECT_EQ(nullptr, entity_.pq_codes());
  auto graph = entity_.get_vector_segment();
  EXPECT_FALSE(pool->is_page_resident(
      graph->data_offset() / ailego::kVectorPageSize + 1));
  read_code(0);
  read_code(codes_.size() / 64 - 1);
  entity_.prepare_pq();
  EXPECT_TRUE(pool->is_page_resident(last));
  for (size_t i = 0; i < 64 && pool->is_page_resident(first_pq_page()); ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  EXPECT_FALSE(pool->is_page_resident(first_pq_page()));
  read_code();
}

TEST_F(DiskAnnPqPriorityTest, ExplicitNoneKeepsPreparationLazy) {
  make_index();
  load();
  entity_.prepare_pq();
  EXPECT_FALSE(storage_->vec_buffer_pool()->is_page_resident(first_pq_page()));
  read_code();
}

TEST_F(DiskAnnPqPriorityTest, AutoSkipsPqLargerThanBudget) {
  const size_t rows = (ailego::MemoryLimitPool::get_instance().capacity() +
                       (1UL << 20) + 63) / 64;
  make_index(rows);
  load("BufferReadStorage", BUFFER_READ_STORAGE_WARMUP_AUTO);
  entity_.prepare_pq();
  EXPECT_FALSE(storage_->vec_buffer_pool()->is_page_resident(first_pq_page()));
  read_code(0);
  read_code(rows - 1);
}

TEST_F(DiskAnnPqPriorityTest, GraphPagesAreEvictedBeforePqPages) {
  make_index();
  load();
  read_code();
  auto graph = entity_.get_vector_segment();
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, graph->read_borrowed(2 * ailego::kVectorPageSize, block, 64));
  const auto graph_page = (graph->data_offset() + 2 * ailego::kVectorPageSize) / ailego::kVectorPageSize;
  block.reset();
  auto pool = storage_->vec_buffer_pool();
  for (size_t i = 0; i < 64 && pool->is_page_resident(graph_page); ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1);
  EXPECT_FALSE(pool->is_page_resident(graph_page));
  EXPECT_TRUE(pool->is_page_resident(first_pq_page()));
}

TEST_F(DiskAnnPqPriorityTest, HighPriorityPqRemainsEvictableAndReloadsCorrectly) {
  make_index();
  load();
  read_code();
  auto pool = storage_->vec_buffer_pool();
  for (size_t i = 0; i < 64 && pool->is_page_resident(first_pq_page()); ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  EXPECT_FALSE(pool->is_page_resident(first_pq_page()));
  EXPECT_GT(pool->stats().evictions_by_priority[ailego::VecBufferPool::kHighPriority], 0u);
  read_code();
}

TEST_F(DiskAnnPqPriorityTest, BudgetSmallerThanPqStillSupportsPagedReads) {
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(2UL << 20));
  auto &budget = ailego::MemoryLimitPool::get_instance();
  const size_t rows = (budget.capacity() + (1UL << 20) + 63) / 64;
  make_index(rows);
  ASSERT_GT(codes_.size(), budget.capacity());
  load();
  ASSERT_EQ(nullptr, entity_.pq_codes());
  read_code();
  read_code(rows - 1);
  EXPECT_LE(budget.used(), budget.capacity());
}

TEST_F(DiskAnnPqPriorityTest, FileReadStorageRetainsItsResidentPqCopy) {
  make_index();
  load("FileReadStorage");
  EXPECT_EQ(nullptr, storage_->vec_buffer_pool());
  EXPECT_EQ(nullptr, entity_.paged_pq_codes());
  ASSERT_NE(nullptr, entity_.pq_codes());
  EXPECT_EQ(0, std::memcmp(entity_.pq_codes(), codes_.data(), codes_.size()));
}

}  // namespace
}  // namespace zvec::core

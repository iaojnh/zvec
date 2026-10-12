// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "ailego/pattern/defer.h"
#include "diskann_context.h"
#include "diskann_indexer.h"
#include "diskann_searcher_entity.h"

namespace zvec::core {
namespace {
class ResultKeyEntity : public DiskAnnSearcherEntity {
 public:
  int read_key(diskann_id_t id, diskann_key_t *key) const override {
    *key = id + 100;
    return 0;
  }
};
}  // namespace

TEST(DiskAnnQueryScratch, PublishedPayloadOutlivesCandidateAndContext) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(16 * 1024));
  IndexDocumentList output;
  const std::string source(3072, 'v');
  {
    IndexMeta meta(IndexMeta::DT_FP32, 768);
    auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
    ASSERT_EQ(0, metric->init(meta, {}));
    auto entity = std::make_shared<ResultKeyEntity>();
    DiskAnnContext ctx(meta, metric, entity);
    DiskAnnIndexer indexer(meta);
    ctx.set_fetch_vector(true);
    ctx.set_topk(1);
    ctx.resize_results(1);
    indexer.emplace_candidate(&ctx, ctx.topk_heap(), 7, 1, source.data());
    const char *data = ctx.topk_heap()[0].second.vec_->data.data();
    ctx.topk_to_result(0);
    EXPECT_TRUE(ctx.topk_heap().empty());
    ASSERT_EQ(1u, ctx.result().size());
    EXPECT_EQ(107u, ctx.result()[0].key());
    EXPECT_EQ(data, ctx.result()[0].vector_string().data());
    EXPECT_GT(pool.used(), source.size());
    output = std::move(*ctx.mutable_result(0));
    ctx.clear();
    EXPECT_GT(pool.used(), source.size());
  }
  EXPECT_EQ(source, output[0].vector_string());
  EXPECT_GT(pool.used(), source.size());
  output.clear();
  EXPECT_EQ(0u, pool.used());
}

TEST(DiskAnnQueryScratch, CandidatePayloadIsOptionalSharedAndBudgeted) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(64 * 1024));
  IndexMeta meta(IndexMeta::DT_FP32, 768);
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_EQ(0, metric->init(meta, {}));
  auto entity = std::make_shared<DiskAnnSearcherEntity>();
  DiskAnnContext ctx(meta, metric, entity);
  DiskAnnIndexer indexer(meta);
  const std::string source(meta.element_size(), 'x');
  auto &heap = ctx.topk_heap();
  heap.limit(1);
  indexer.emplace_candidate(&ctx, heap, 7, 1, source.data());
  ASSERT_EQ(1u, heap.size());
  EXPECT_EQ(nullptr, heap[0].second.vec_);
  ctx.set_fetch_vector(true);
  indexer.emplace_candidate(&ctx, heap, 8, 0, source.data());
  ASSERT_NE(nullptr, heap[0].second.vec_);
  EXPECT_EQ(source, heap[0].second.vec_->data);
  const size_t used = pool.used();
  indexer.emplace_candidate(&ctx, heap, 9, 2, source.data());
  EXPECT_EQ(used, pool.used());
  EXPECT_EQ(8u, heap[0].first);
  {
    VectorInfo shared = heap[0].second;
    EXPECT_EQ(used, pool.used());
    EXPECT_EQ(shared.vec_.get(), heap[0].second.vec_.get());
    const char *payload = shared.vec_->data.data();
    IndexDocument result(8, 0, 8, shared.vec_);
    EXPECT_EQ(payload, result.vector_string().data());
    EXPECT_EQ(source, result.vector_string());
  }
  heap.release();
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(128));
  EXPECT_THROW(indexer.emplace_candidate(&ctx, heap, 1, 0, source.data()),
               std::bad_alloc);
  EXPECT_TRUE(heap.empty());
  EXPECT_EQ(0u, pool.used());
  ctx.set_fetch_vector(false);
  EXPECT_NO_THROW(indexer.emplace_candidate(&ctx, heap, 1, 0, source.data()));
  heap.release();
}

TEST(DiskAnnQueryScratch, CandidateQueueRejectsBeforeAllocatingAndCanRetry) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(1024));
  NeighborPriorityQueue queue;
  queue.reserve(8);
  queue.insert(Neighbor(3, 1));
  const size_t used = pool.used();
  EXPECT_THROW(queue.reserve(1000000), std::bad_alloc);
  EXPECT_EQ(8u, queue.capacity());
  EXPECT_EQ(3u, queue[0].id);
  EXPECT_EQ(used, pool.used());
  queue.release();
  EXPECT_EQ(0u, pool.used());
  queue.reserve(4);
  queue.insert(Neighbor(1, 0));
  EXPECT_EQ(1u, queue.closest_unexpanded().id);
}

TEST(DiskAnnQueryScratch, LazyAlignedBudgetedResizeAndFailureRecovery) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(64 * 1024));
  IndexMeta meta(IndexMeta::DT_FP32, 8);
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_NE(nullptr, metric);
  ASSERT_EQ(0, metric->init(meta, {}));
  {
    auto entity = std::make_shared<DiskAnnSearcherEntity>();
    DiskAnnContext context(meta, metric, entity);
    ASSERT_EQ(0, context.init(DiskAnnContext::kSearcherContext, 32, 4,
                              meta.element_size(), false));
    EXPECT_EQ(0u, context.sector_buffer_size());
    const size_t baseline = pool.used();
    ASSERT_EQ(0, context.ensure_sector_buffer(2));
    EXPECT_EQ(8192u, context.sector_buffer_size());
    EXPECT_EQ(baseline + 8192, pool.used());
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(context.sector_buffer()) % 4096);
    auto *old_buffer = context.sector_buffer();
    EXPECT_EQ(IndexError_NoMemory, context.ensure_sector_buffer(128));
    EXPECT_EQ(old_buffer, context.sector_buffer());
    EXPECT_EQ(baseline + 8192, pool.used());
    EXPECT_EQ(IndexError_InvalidArgument, context.ensure_sector_buffer(0));
    EXPECT_EQ(IndexError_InvalidArgument, context.ensure_sector_buffer(129));
    ASSERT_EQ(0, context.ensure_sector_buffer(4));
    EXPECT_EQ(baseline + 16384, pool.used());
    ASSERT_EQ(0, context.ensure_sector_buffer(1));
    EXPECT_EQ(baseline + 4096, pool.used());
  }
}
}  // namespace zvec::core

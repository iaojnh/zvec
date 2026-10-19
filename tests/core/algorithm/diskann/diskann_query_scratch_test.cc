// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <thread>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "ailego/pattern/defer.h"
#include "diskann_context.h"
#include "diskann_index_provider.h"
#include "diskann_indexer.h"
#include "diskann_searcher_entity.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
class DiskAnnProviderTestPeer {
 public:
  using Owner = DiskAnnIndexProvider::ResultBufferOwner;
  static ailego::BudgetedString &buffer(const std::shared_ptr<Owner> &owner) {
    return DiskAnnIndexProvider::thread_result_buffer(owner);
  }
};

class DiskAnnCacheTestPeer {
 public:
  static void set_reader(DiskAnnIndexer &indexer,
                         const std::shared_ptr<AlignedFileReader> &reader) {
    indexer.reader_ = reader;
    indexer.node_per_sector_ = 1;
    indexer.max_node_size_ = DiskAnnUtil::kSectorSize;
    indexer.max_degree_ = 4;
    indexer.disk_bytes_per_point_ = indexer.meta_.element_size();
  }
};

namespace {
class ResultKeyEntity : public DiskAnnSearcherEntity {
 public:
  int read_key(diskann_id_t id, diskann_key_t *key) const override {
    *key = id + 100;
    return 0;
  }
};

class BatchReader : public AlignedFileReader {
 public:
  void open(const std::string &) override {}
  void close() override {}
  int read(std::vector<AlignedRead> &requests, IOContext &,
           bool = false) override {
    ++calls;
    max_batch = std::max(max_batch, requests.size());
    for (const auto &request : requests) {
      std::memset(request.buf, 0, request.len);
      *static_cast<float *>(request.buf) =
          static_cast<float>(request.offset / DiskAnnUtil::kSectorSize);
    }
    return 0;
  }
  int submit(PendingBatch &, std::vector<AlignedRead> &, IOContext &) override {
    return IndexError_NotImplemented;
  }
  int get_completed(PendingBatch &, IOContext &, int,
                    std::vector<uint32_t> &) override {
    return IndexError_NotImplemented;
  }
  void release_io_ctx(IOContext &) override {}
  size_t calls{0};
  size_t max_batch{0};
};
}  // namespace

TEST(DiskAnnQueryScratch, PublishedPayloadOutlivesCandidateAndContext) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
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
  EXPECT_GT(pool.used(), 0u);
  IndexDocumentList().swap(output);
  EXPECT_EQ(0u, pool.used());
}

TEST(DiskAnnQueryScratch, CandidatePayloadIsOptionalSharedAndBudgeted) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
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
  auto &pool = ailego::PoolPayloadBudget::get_instance();
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
  auto &pool = ailego::PoolPayloadBudget::get_instance();
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

TEST(DiskAnnQueryScratch, QueryBuffersRejectAdmissionAndRetry) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(16 * 1024));
  IndexMeta meta(IndexMeta::DT_FP32, 768);
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_EQ(0, metric->init(meta, {}));
  auto entity = std::make_shared<DiskAnnSearcherEntity>();
  DiskAnnContext context(meta, metric, entity);
  {
    ailego::MemoryReservation pressure;
    pressure.add(12 * 1024);
    const size_t baseline = pool.used();
    EXPECT_EQ(IndexError_NoMemory,
              context.init(DiskAnnContext::kSearcherContext, 32, 4,
                           meta.element_size(), false));
    EXPECT_EQ(baseline, pool.used());
  }
  ASSERT_EQ(0, context.init(DiskAnnContext::kSearcherContext, 32, 4,
                            meta.element_size(), false));
  EXPECT_GE(pool.used(), 3u * meta.element_size());
}

TEST(DiskAnnQueryScratch, AlignedReplacementChargesOverlapAndPreservesFailure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(4096));
  ailego::BudgetedAlignedBuffer buffer;
  buffer.resize(2048, 256);
  ASSERT_EQ(0u, reinterpret_cast<uintptr_t>(buffer.data()) % 256);
  *static_cast<char *>(buffer.data()) = 'x';
  auto *original = buffer.data();
  EXPECT_THROW(buffer.resize(3072, 256), std::bad_alloc);
  EXPECT_EQ(original, buffer.data());
  EXPECT_EQ('x', *static_cast<char *>(buffer.data()));
  EXPECT_EQ(2048u, pool.used());
  EXPECT_THROW(buffer.resize_array(SIZE_MAX, 2, 256), std::bad_alloc);
  EXPECT_EQ(2048u, pool.used());
  ailego::BudgetedAlignedBuffer moved(std::move(buffer));
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(original, moved.data());
  moved.resize(0, 256);
  EXPECT_EQ(0u, pool.used());
  EXPECT_NO_THROW(moved.resize(3072, 256));
}

TEST(DiskAnnQueryScratch, ProviderTlsRetainsChargeUntilPruningOrThreadExit) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(16 * 1024));
  std::thread worker([&] {
    auto owner = std::make_shared<DiskAnnProviderTestPeer::Owner>();
    auto &buffer = DiskAnnProviderTestPeer::buffer(owner);
    buffer.assign(4096, 'x');
    const size_t used = pool.used();
    EXPECT_GT(used, 4096u);
    EXPECT_THROW(buffer.resize(1000000), std::bad_alloc);
    EXPECT_EQ(4096u, buffer.size());
    EXPECT_EQ(used, pool.used());
    owner.reset();
    EXPECT_EQ(used, pool.used());  // An expired provider does not free TLS.
    auto next = std::make_shared<DiskAnnProviderTestPeer::Owner>();
    auto &replacement = DiskAnnProviderTestPeer::buffer(next);
    EXPECT_LT(pool.used(), used);  // Lookup prunes the expired allocation.
    replacement.assign(2048, 'y');
    EXPECT_GT(pool.used(), 2048u);
  });
  worker.join();
  EXPECT_EQ(0u, pool.used());
}

TEST(DiskAnnQueryScratch, NodeReadsBatchScratchAndRetryAfterBudgetFailure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(1024 * 1024));
  IndexMeta meta(IndexMeta::DT_FP32, 8);
  DiskAnnIndexer indexer(meta);
  auto reader = std::make_shared<BatchReader>();
  DiskAnnCacheTestPeer::set_reader(indexer, reader);
  constexpr size_t count = 300;
  ailego::BudgetedVector<diskann_id_t> ids(count);
  ailego::BudgetedVector<float> data(count * 8);
  ailego::BudgetedVector<void *> buffers(count);
  ailego::BudgetedVector<std::pair<uint32_t, diskann_id_t *>> neighbors(count);
  for (size_t i = 0; i < count; ++i) {
    ids[i] = static_cast<diskann_id_t>(i);
    buffers[i] = data.data() + i * 8;
  }
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used() - 32768);
    const size_t before = pool.used();
    EXPECT_THROW(indexer.read_nodes(ids, buffers, neighbors), std::bad_alloc);
    EXPECT_EQ(before, pool.used());
    EXPECT_EQ(0u, reader->calls);
  }
  const size_t before = pool.used();
  {
    const auto status = indexer.read_nodes(ids, buffers, neighbors);
    ASSERT_EQ(count, status.size());
    for (size_t i = 0; i < count; ++i) {
      EXPECT_TRUE(status[i]);
      EXPECT_FLOAT_EQ(static_cast<float>(i), data[i * 8]);
    }
    EXPECT_EQ(3u, reader->calls);
    EXPECT_LE(reader->max_batch, 128u);
  }
  EXPECT_EQ(before, pool.used());
}
}  // namespace zvec::core

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

// End-to-end consistency test for the DiskAnn VecBufferPool path. The same
// index and queries are searched twice: once through FileReadStorage (direct
// O_DIRECT + libaio reads) and once through BufferReadStorage (VecBufferPool
// paged cache -> BufferPoolAlignedFileReader). Because the POC does not change
// candidate selection or distance math, both paths must return byte-identical
// top-K keys and scores. It also verifies that repeated queries turn logical
// sector reads into buffer-pool cache hits.

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_budget.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "utility/utility_params.h"
#include "diskann_holder.h"
#include "diskann_params.h"
#include "diskann_searcher.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

namespace {

constexpr size_t kDim = 64;
constexpr size_t kDocCnt = 10000UL;
constexpr size_t kTopk = 50;

class DiskAnnBufferPoolSearchTest : public testing::Test {
 protected:
  void SetUp(void) override {
    LoggerBroker::SetLevel(Logger::LEVEL_WARN);
    // Generous cap so the BufferReadStorage path can cache the whole index and
    // repeated queries hit; individual eviction behavior is covered by the
    // reader unit tests.
    MemoryLimitPool::get_instance().init(512UL * 1024UL * 1024UL);
    MemoryBudgetManager::Config budget;
    budget.total_bytes = MemoryLimitPool::get_instance().capacity();
    budget.buffer_cache_bytes = budget.total_bytes;
    ASSERT_TRUE(MemoryBudgetManager::get_instance().configure(budget));
    meta_.reset(new IndexMeta(IndexMeta::DataType::DT_FP32, kDim));
    meta_->set_metric("SquaredEuclidean", 0, Params());
  }

  void TearDown(void) override {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir_.c_str());
    system(cmd);
  }

  // Build a DiskAnn index and dump it to `path`.
  void BuildIndex(const string &path, size_t doc_count = kDocCnt) {
    IndexBuilder::Pointer builder =
        IndexFactory::CreateBuilder("DiskAnnBuilder");
    ASSERT_NE(builder, nullptr);

    auto holder =
        make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(kDim);
    for (size_t i = 0; i < doc_count; i++) {
      NumericalVector<float> vec(kDim);
      for (size_t j = 0; j < kDim; ++j) vec[j] = static_cast<float>(i);
      ASSERT_TRUE(holder->emplace(i, vec));
    }

    Params params;
    params.set("zvec.diskann.builder.max_degree", 32);
    params.set("zvec.diskann.builder.list_size", 300);
    params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
    params.set("zvec.diskann.builder.threads", 4);

    ASSERT_EQ(0, builder->init(*meta_, params));
    ASSERT_EQ(0, builder->train(holder));
    ASSERT_EQ(0, builder->build(holder));

    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(dumper, nullptr);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, builder->dump(dumper));
    ASSERT_EQ(0, dumper->close());
  }

  IndexSearcher::Pointer MakeSearcher(const IndexStorage::Pointer &storage,
                                      uint64_t node_budget_bytes = 0,
                                      const std::string &node_page_policy =
                                          DISKANN_CACHE_NODE_PAGE_POLICY_KEEP) {
    IndexSearcher::Pointer searcher =
        IndexFactory::CreateSearcher("DiskAnnSearcher");
    EXPECT_NE(searcher, nullptr);
    Params sp;
    if (node_budget_bytes == 0) {
      sp.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, 0);
    } else {
      sp.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_BUDGET_BYTES, node_budget_bytes);
    }
    sp.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_PAGE_POLICY, node_page_policy);
    sp.set("zvec.diskann.searcher.list_size", 200);
    EXPECT_EQ(0, searcher->init(sp));
    EXPECT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
    return searcher;
  }

  IndexStorage::Pointer MakeBufferStorage(const string &path,
                                          bool enable_io_profile = false) {
    auto storage = IndexFactory::CreateStorage("BufferReadStorage");
    EXPECT_NE(storage, nullptr);
    if (!storage) {
      return nullptr;
    }

    Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE,
               BUFFER_READ_STORAGE_WARMUP_NONE);
    params.set(BUFFER_READ_STORAGE_ENABLE_IO_PROFILE, enable_io_profile);
    EXPECT_EQ(0, storage->init(params));
    EXPECT_EQ(0, storage->open(path, false));
    return storage;
  }

  string dir_{"DiskAnnBufferPoolSearchTest/"};
  shared_ptr<IndexMeta> meta_;
};

}  // namespace

TEST_F(DiskAnnBufferPoolSearchTest, DirectAndBufferPoolResultsMatch) {
  const string path = dir_ + "/index";
  BuildIndex(path);

  // Direct path: FileReadStorage.
  auto direct_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(direct_storage, nullptr);
  ASSERT_EQ(0, direct_storage->open(path, false));
  ASSERT_EQ(direct_storage->vec_buffer_pool(), nullptr);
  auto direct_searcher = MakeSearcher(direct_storage);

  // Buffer path: BufferReadStorage over a VecBufferPool.
  auto buffer_storage = MakeBufferStorage(path);
  ASSERT_NE(buffer_storage, nullptr);
  ASSERT_NE(buffer_storage->vec_buffer_pool(), nullptr);
  auto buffer_searcher = MakeSearcher(buffer_storage);

  auto direct_ctx = direct_searcher->create_context();
  auto buffer_ctx = buffer_searcher->create_context();
  ASSERT_TRUE(!!direct_ctx);
  ASSERT_TRUE(!!buffer_ctx);
  direct_ctx->set_topk(kTopk);
  buffer_ctx->set_topk(kTopk);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  NumericalVector<float> vec(kDim);

  const size_t step = 500;
  for (size_t i = 0; i < kDocCnt; i += step) {
    for (size_t j = 0; j < kDim; ++j) vec[j] = i + 0.1f;

    ASSERT_EQ(0, direct_searcher->search_impl(vec.data(), qmeta, direct_ctx));
    ASSERT_EQ(0, buffer_searcher->search_impl(vec.data(), qmeta, buffer_ctx));

    auto &dr = direct_ctx->result();
    auto &br = buffer_ctx->result();
    ASSERT_EQ(dr.size(), br.size()) << "query i=" << i;
    for (size_t k = 0; k < dr.size(); ++k) {
      EXPECT_EQ(dr[k].key(), br[k].key())
          << "key mismatch at query i=" << i << " rank " << k;
      EXPECT_NEAR(dr[k].score(), br[k].score(), 1e-4f)
          << "score mismatch at query i=" << i << " rank " << k;
    }
  }
}

TEST_F(DiskAnnBufferPoolSearchTest, RepeatedQueriesProduceCacheHits) {
  const string path = dir_ + "/index_hits";
  BuildIndex(path);

  auto buffer_storage = MakeBufferStorage(path, true);
  ASSERT_NE(buffer_storage, nullptr);
  auto *pool = buffer_storage->vec_buffer_pool();
  ASSERT_NE(pool, nullptr);
  auto searcher = MakeSearcher(buffer_storage);

  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);
  ctx->set_topk(kTopk);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  NumericalVector<float> vec(kDim);
  for (size_t j = 0; j < kDim; ++j) vec[j] = 4200.1f;

  // Warm the cache.
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  uint64_t hit_before = pool->stats().hit;

  // Repeat the identical query: sectors should now be resident -> pool hits.
  for (int r = 0; r < 5; ++r) {
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  }
  uint64_t hit_after = pool->stats().hit;
  EXPECT_GT(hit_after, hit_before);
  auto io_profile = pool->stats().io_profile;
  EXPECT_GT(io_profile.aio_pages + io_profile.sync_reads, 0u);
}

TEST_F(DiskAnnBufferPoolSearchTest,
       StaticNodeCacheSharesBudgetAndCanEvictDuplicatePages) {
  const string path = dir_ + "/index_node_cache";
  BuildIndex(path);

  auto buffer_storage = MakeBufferStorage(path);
  ASSERT_NE(buffer_storage, nullptr);
  auto *pool = buffer_storage->vec_buffer_pool();
  ASSERT_NE(pool, nullptr);

  constexpr uint64_t kNodeBudget = 4ULL * 1024 * 1024;
  const size_t used_before = MemoryLimitPool::get_instance().used();
  const uint64_t evict_before = pool->stats().evict;
  auto searcher = MakeSearcher(buffer_storage, kNodeBudget,
                               DISKANN_CACHE_NODE_PAGE_POLICY_EVICT);
  ASSERT_NE(searcher, nullptr);

  EXPECT_GT(MemoryLimitPool::get_instance().used(), used_before);
  EXPECT_GT(MemoryLimitPool::get_instance().external_used(), 0u);
  EXPECT_GT(pool->stats().evict, evict_before);

  searcher.reset();
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().external_used());
}

TEST_F(DiskAnnBufferPoolSearchTest,
       DirectModeKeepsExplicitPartitionsAndSharedNodeCache) {
  const string path = dir_ + "/index_direct_budget";
  BuildIndex(path);

  constexpr uint64_t kMiB = 1024ULL * 1024ULL;
  MemoryBudgetManager::Config budget;
  budget.total_bytes = 656 * kMiB;
  budget.enforce_accounting = true;
  budget.buffer_cache_bytes = 512 * kMiB;
  budget.query_working_bytes = 64 * kMiB;
  budget.resident_metadata_bytes = 64 * kMiB;
  budget.safety_reserve_bytes = 16 * kMiB;
  ASSERT_TRUE(MemoryBudgetManager::get_instance().configure(budget));

  auto direct_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(direct_storage, nullptr);
  ASSERT_EQ(0, direct_storage->open(path, false));
  ASSERT_EQ(nullptr, direct_storage->vec_buffer_pool());

  constexpr uint64_t kNodeBudget = 4 * kMiB;
  auto searcher = MakeSearcher(direct_storage, kNodeBudget);
  ASSERT_NE(searcher, nullptr);

  auto snapshot = MemoryBudgetManager::get_instance().snapshot();
  EXPECT_TRUE(snapshot.config.enforce_accounting);
  EXPECT_GT(snapshot.resident_metadata_used, 0u);
  EXPECT_GT(MemoryLimitPool::get_instance().external_used(), 0u);
  EXPECT_LE(MemoryLimitPool::get_instance().external_used(), kNodeBudget);

  auto context = searcher->create_context();
  ASSERT_TRUE(!!context);
  EXPECT_GT(MemoryBudgetManager::get_instance()
                .snapshot()
                .query_working_used,
            0u);

  context.reset();
  EXPECT_EQ(0u, MemoryBudgetManager::get_instance()
                    .snapshot()
                    .query_working_used);

  searcher.reset();
  snapshot = MemoryBudgetManager::get_instance().snapshot();
  EXPECT_EQ(0u, snapshot.resident_metadata_used);
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().external_used());
}

TEST_F(DiskAnnBufferPoolSearchTest,
       ExplicitZeroSharedCacheDisablesStaticNodeCache) {
  const string path = dir_ + "/index_zero_shared_cache";
  BuildIndex(path, 1000);

  constexpr uint64_t kMiB = 1024ULL * 1024ULL;
  MemoryBudgetManager::Config budget;
  budget.total_bytes = 512 * kMiB;
  budget.enforce_accounting = true;
  budget.query_working_bytes = 128 * kMiB;
  budget.resident_metadata_bytes = 384 * kMiB;
  ASSERT_TRUE(MemoryBudgetManager::get_instance().configure(budget));
  MemoryLimitPool::get_instance().init(0);

  auto direct_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(direct_storage, nullptr);
  ASSERT_EQ(0, direct_storage->open(path, false));

  const auto rejections_before =
      MemoryLimitPool::get_instance().stats().high_watermark_hits;
  auto searcher = MakeSearcher(direct_storage, 4 * kMiB);
  ASSERT_NE(searcher, nullptr);
  const auto stats = MemoryLimitPool::get_instance().stats();
  EXPECT_EQ(0u, stats.pool_size);
  EXPECT_EQ(0u, stats.used);
  EXPECT_EQ(0u, stats.external_used);
  EXPECT_GT(stats.high_watermark_hits, rejections_before);

  auto context = searcher->create_context();
  ASSERT_TRUE(!!context);
  context->set_topk(kTopk);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  NumericalVector<float> query(kDim);
  for (size_t i = 0; i < kDim; ++i) query[i] = 100.1f;
  EXPECT_EQ(0, searcher->search_impl(query.data(), qmeta, context));
}

TEST_F(DiskAnnBufferPoolSearchTest,
       InsufficientSharedCacheShrinksStaticNodeCache) {
  const string path = dir_ + "/index_small_shared_cache";
  BuildIndex(path, 1000);

  constexpr uint64_t kMiB = 1024ULL * 1024ULL;
  constexpr uint64_t kSharedCacheBytes = 128ULL * 1024ULL;
  MemoryBudgetManager::Config budget;
  budget.total_bytes = 512 * kMiB;
  budget.enforce_accounting = true;
  budget.buffer_cache_bytes = kSharedCacheBytes;
  budget.query_working_bytes = 128 * kMiB;
  budget.resident_metadata_bytes =
      budget.total_bytes - budget.buffer_cache_bytes -
      budget.query_working_bytes;
  ASSERT_TRUE(MemoryBudgetManager::get_instance().configure(budget));
  MemoryLimitPool::get_instance().init(kSharedCacheBytes);

  auto direct_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(direct_storage, nullptr);
  ASSERT_EQ(0, direct_storage->open(path, false));

  const auto rejections_before =
      MemoryLimitPool::get_instance().stats().high_watermark_hits;
  auto searcher = MakeSearcher(direct_storage, 4 * kMiB);
  ASSERT_NE(searcher, nullptr);

  const auto stats = MemoryLimitPool::get_instance().stats();
  EXPECT_GT(stats.external_used, 0u);
  EXPECT_LE(stats.external_used, kSharedCacheBytes);
  EXPECT_GT(stats.high_watermark_hits, rejections_before);

  searcher.reset();
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().external_used());
}

TEST_F(DiskAnnBufferPoolSearchTest,
       ZeroCapacityDisablesCacheWithoutBreakingCoreOnlyUse) {
  const string path = dir_ + "/index_unmanaged_zero_capacity";
  BuildIndex(path, 1000);

  MemoryBudgetManager::Config budget;
  ASSERT_TRUE(MemoryBudgetManager::get_instance().configure(budget));
  ASSERT_FALSE(budget.enforce_accounting);
  MemoryLimitPool::get_instance().init(0);

  auto direct_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(direct_storage, nullptr);
  ASSERT_EQ(0, direct_storage->open(path, false));

  constexpr uint64_t kNodeBudget = 4ULL * 1024ULL * 1024ULL;
  const auto rejections_before =
      MemoryLimitPool::get_instance().stats().high_watermark_hits;
  auto searcher = MakeSearcher(direct_storage, kNodeBudget);
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().capacity());
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().external_used());
  EXPECT_GT(MemoryLimitPool::get_instance().stats().high_watermark_hits,
            rejections_before);
}

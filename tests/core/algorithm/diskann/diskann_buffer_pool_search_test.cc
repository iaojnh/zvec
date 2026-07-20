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

#include "diskann_searcher.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_holder.h"
#include "diskann_params.h"

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
    meta_.reset(new IndexMeta(IndexMeta::DataType::DT_FP32, kDim));
    meta_->set_metric("SquaredEuclidean", 0, Params());
  }

  void TearDown(void) override {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir_.c_str());
    system(cmd);
  }

  // Build a DiskAnn index and dump it to `path`.
  void BuildIndex(const string &path) {
    IndexBuilder::Pointer builder =
        IndexFactory::CreateBuilder("DiskAnnBuilder");
    ASSERT_NE(builder, nullptr);

    auto holder =
        make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(kDim);
    for (size_t i = 0; i < kDocCnt; i++) {
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

  // Create a searcher backed by `storage` with node cache disabled.
  IndexSearcher::Pointer MakeSearcher(const IndexStorage::Pointer &storage) {
    IndexSearcher::Pointer searcher =
        IndexFactory::CreateSearcher("DiskAnnSearcher");
    EXPECT_NE(searcher, nullptr);
    Params sp;
    sp.set("zvec.diskann.searcher.cache_node_num", 0);  // isolate pool value
    sp.set("zvec.diskann.searcher.list_size", 200);
    EXPECT_EQ(0, searcher->init(sp));
    EXPECT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
    return searcher;
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
  auto buffer_storage = IndexFactory::CreateStorage("BufferReadStorage");
  ASSERT_NE(buffer_storage, nullptr);
  ASSERT_EQ(0, buffer_storage->open(path, false));
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

  auto buffer_storage = IndexFactory::CreateStorage("BufferReadStorage");
  ASSERT_NE(buffer_storage, nullptr);
  ASSERT_EQ(0, buffer_storage->open(path, false));
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
}

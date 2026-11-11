// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "ivf_builder.h"
#include "ivf_searcher.h"
#include "ivf_streamer.h"
#include "utility/chunk_read_storage.h"

namespace zvec::core {
namespace {

TEST(IVFMmapPrefetch, PlainRowTailsPreserveFilteredAndUnfilteredResults) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(32UL << 20));
  ASSERT_TRUE(pool.configure_process_rss_limit(0));
  auto close_pool = ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(0)); });
  for (uint32_t dimension : {17U, 33U}) {
    for (uint32_t count : {1U, 33U, 65U}) {
      for (uint32_t centers : {1U, 9U}) {
        if (centers > count) continue;
        SCOPED_TRACE(dimension);
        SCOPED_TRACE(count);
        SCOPED_TRACE(centers);
        const std::string path = "ivf_mmap_prefetch_tail.index";
        auto remove =
            ailego::ScopeGuard::Make([&] { ailego::File::RemovePath(path); });
        IndexMeta meta;
        meta.set_meta(IndexMeta::DT_FP32, dimension);
        meta.set_major_order(IndexMeta::MO_ROW);
        meta.set_metric("SquaredEuclidean", 0, ailego::Params());
        auto holder =
            std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dimension);
        for (uint32_t i = 0; i < count; ++i) {
          ailego::NumericalVector<float> vector(dimension);
          std::fill(vector.begin(), vector.end(), i / 8.0f);
          holder->emplace(i, vector);
        }
        ailego::Params build_params;
        build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, std::to_string(centers));
        build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
        IVFBuilder builder;
        ASSERT_EQ(0, builder.init(meta, build_params));
        ASSERT_EQ(0, builder.train(IndexThreads::Pointer(), holder));
        ASSERT_EQ(0, builder.build(IndexThreads::Pointer(), holder));
        auto dumper = IndexFactory::CreateDumper("FileDumper");
        ASSERT_NE(nullptr, dumper);
        ASSERT_EQ(0, dumper->create(path));
        ASSERT_EQ(0, builder.dump(dumper));
        ASSERT_EQ(0, dumper->close());
        ailego::Params search_params;
        search_params.set(PARAM_IVF_SEARCHER_NPROBE, centers);
        search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
        search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0U);
        const float value = (count - 1) / 8.0f + 0.03125f;
        const std::vector<float> query(dimension, value);
        const IndexQueryMeta query_meta(IndexMeta::DT_FP32, dimension);
        const auto check = [&](auto *searcher) {
          auto context = searcher->create_context();
          ASSERT_NE(nullptr, context);
          context->set_topk(3);
          for (bool filtered : {false, true}) {
            SCOPED_TRACE(filtered);
            context->set_filter(
                [filtered](uint64_t id) { return filtered && id % 2 == 0; });
            std::vector<std::pair<float, uint64_t>> expected;
            for (uint32_t i = 0; i < count; ++i) {
              if (filtered && i % 2 == 0) continue;
              const float delta = value - i / 8.0f;
              expected.emplace_back(delta * delta * dimension, i);
            }
            std::sort(expected.begin(), expected.end());
            expected.resize(std::min<size_t>(3, expected.size()));
            ASSERT_EQ(0,
                      searcher->search_impl(query.data(), query_meta, context));
            const auto &result = context->result(0);
            ASSERT_EQ(expected.size(), result.size());
            for (size_t i = 0; i < result.size(); ++i) {
              EXPECT_EQ(expected[i].second, result[i].key());
              EXPECT_NEAR(expected[i].first, result[i].score(),
                          1e-3f * std::max(1.0f, expected[i].first));
            }
          }
        };
        for (const char *storage_name :
             {"FileReadStorage", "MMapFileReadStorage", "ChunkReadStorage"}) {
          SCOPED_TRACE(storage_name);
          auto storage = IndexFactory::CreateStorage(storage_name);
          ASSERT_NE(nullptr, storage);
          ailego::Params storage_params;
          storage_params.set(CHUNK_READ_STORAGE_ADAPTIVE, true);
          storage_params.set(CHUNK_READ_STORAGE_PREFETCH_DEPTH, size_t{2});
          storage_params.set(CHUNK_READ_STORAGE_PREFETCH_BYTES,
                             8 * ailego::MemoryLimitPool::page_buffer_size());
          ASSERT_EQ(0, storage->init(storage_params));
          ASSERT_EQ(0, storage->open(path, false));
          IVFSearcher searcher;
          ASSERT_EQ(0, searcher.init(search_params));
          ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
          check(&searcher);
          ASSERT_EQ(0, searcher.unload());
          IVFStreamer streamer;
          ASSERT_EQ(0, streamer.init(meta, search_params));
          ASSERT_EQ(0, streamer.open(storage));
          check(&streamer);
          ASSERT_EQ(0, streamer.close());
        }
      }
    }
  }
}

}  // namespace
}  // namespace zvec::core

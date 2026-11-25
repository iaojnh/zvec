// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include "algorithm/hnsw_sparse/hnsw_sparse_context.h"
#include "query_context_oom_fixture.h"

namespace zvec {
namespace {
TEST_P(QueryContextOomTest, SparseSearcherReleasesFailedContext) {
  core::IndexMeta meta(core::IndexMeta::MetaType::MT_SPARSE,
                       core::IndexMeta::DataType::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  const core::IndexQueryMeta query_meta(core::IndexMeta::MetaType::MT_SPARSE,
                                       core::IndexMeta::DataType::DT_FP32);
  const std::array<uint32_t, 2> indices{1, 3};
  const std::array<float, 2> values{0.5F, 1.0F};
  const auto path = (directory_ / "sparse.index").string();
  {
    auto streamer = core::IndexFactory::CreateStreamer("HnswSparseStreamer");
    auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, streamer);
    ASSERT_NE(nullptr, storage);
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, storage->init(ailego::Params{}));
    ASSERT_EQ(0, storage->open((directory_ / "sparse-working").string(), true));
    ASSERT_EQ(0, streamer->init(meta, ailego::Params{}));
    ASSERT_EQ(0, streamer->open(storage));
    auto context = streamer->create_context();
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(0, streamer->add_impl(7, indices.size(), indices.data(),
                                    values.data(), query_meta, context));
    context.reset();
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, streamer->dump(dumper));
    ASSERT_EQ(0, dumper->close());
    ASSERT_EQ(0, streamer->close());
    ASSERT_EQ(0, storage->close());
  }

  core::IndexStorage::Pointer storage;
  ASSERT_NO_FATAL_FAILURE(OpenReadStorage(path, &storage));
  auto searcher = core::IndexFactory::CreateSearcher("HnswSparseSearcher");
  ASSERT_NE(nullptr, searcher);
  ASSERT_EQ(0, searcher->init(ailego::Params{}));
  ASSERT_EQ(0, searcher->load(storage, nullptr));

  ASSERT_NO_FATAL_FAILURE(
      CheckContextFailure(*searcher, sizeof(core::HnswSparseContext),
                          sizeof(core::VisitByteMap::Context)));
  auto context = searcher->create_context();
  ASSERT_NE(nullptr, context);
  context->set_topk(1);
  ASSERT_EQ(0, searcher->search_impl(indices.size(), indices.data(),
                                     values.data(), query_meta, context));
  ASSERT_EQ(1u, context->result().size());
  EXPECT_EQ(7u, context->result()[0].key());
}

INSTANTIATE_TEST_SUITE_P(StorageBackends, QueryContextOomTest, ::testing::Bool());
}  // namespace
}  // namespace zvec

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"

namespace zvec::core {
namespace {
class SparseFlowContext : public IndexContext {
 public:
  void set_topk(uint32_t) override {}
  const IndexDocumentList &result() const override {
    return results.at(0);
  }
  const IndexDocumentList &result(size_t i) const override {
    return results.at(i);
  }
  IndexDocumentList *mutable_result(size_t i) override {
    return &results.at(i);
  }
  void reset() override {
    results.clear();
  }
  std::vector<IndexDocumentList> results;
};

// Observe the actual framework handoff, without introducing another index's
// scratch allocations or training variability into this conversion test.
class SparseFlowSearcher : public IndexSearcher {
 public:
  const IndexMeta &meta() const override {
    return meta_;
  }
  const ailego::Params &params() const override {
    return params_;
  }
  const Stats &stats() const override {
    return stats_;
  }
  int cleanup() override {
    return 0;
  }
  int unload() override {
    return 0;
  }
  int load(IndexStorage::Pointer storage, IndexMetric::Pointer) override {
    return IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  }
  Context::Pointer create_context() const override {
    return std::make_unique<SparseFlowContext>();
  }
  int search_impl(const uint32_t *counts, const uint32_t *, const void *query,
                  const IndexQueryMeta &meta, uint32_t count,
                  Context::Pointer &context) const override {
    ++batches;
    last_input = query;
    if (meta.data_type() != IndexMeta::DT_FP16 || meta.unit_size() != 2)
      return IndexError_Mismatch;
    auto &results = static_cast<SparseFlowContext *>(context.get())->results;
    results.clear();
    results.resize(count);
    const auto *values = static_cast<const ailego::Float16 *>(query);
    for (uint32_t i = 0; i < count; ++i) {
      float sum = 0;
      for (uint32_t j = 0; j < counts[i]; ++j) sum += float(*values++);
      results[i].emplace_back(counts[i], sum);
    }
    return 0;
  }
  int search_bf_impl(const uint32_t *counts, const uint32_t *indices,
                     const void *query, const IndexQueryMeta &meta,
                     uint32_t count, Context::Pointer &ctx) const override {
    return search_impl(counts, indices, query, meta, count, ctx);
  }
  int search_impl(uint32_t count, const uint32_t *indices, const void *query,
                  const IndexQueryMeta &meta,
                  Context::Pointer &ctx) const override {
    return search_impl(&count, indices, query, meta, 1, ctx);
  }
  int search_bf_impl(uint32_t count, const uint32_t *indices, const void *query,
                     const IndexQueryMeta &meta,
                     Context::Pointer &ctx) const override {
    return search_impl(&count, indices, query, meta, 1, ctx);
  }
  mutable size_t batches{0};
  mutable const void *last_input{nullptr};

 private:
  IndexMeta meta_;
  ailego::Params params_;
  Stats stats_;
};

TEST(SparseFlowWorkspace, OneBatchNoRetainedCopyAndRetryAfterBudgetFailure) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "sparse_flow_workspace.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
    std::filesystem::remove(path);
  });
  ASSERT_EQ(0, pool.init(0));
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP16);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  meta.set_reformer("HalfFloatSparseReformer", 0, ailego::Params{});
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexHelper::SerializeToDumper(meta, dumper.get()));
  ASSERT_EQ(0, dumper->close());
  auto searcher = std::make_shared<SparseFlowSearcher>();
  IndexSparseFlow flow;
  ASSERT_EQ(0, flow.set_searcher(searcher));
  ASSERT_EQ(0, flow.load(path));
  auto ctx = flow.create_context();
  ASSERT_NE(nullptr, ctx);
  std::vector<uint32_t> counts(2049);
  std::vector<float> values;
  std::vector<uint32_t> indices;
  for (size_t i = 0; i < counts.size(); ++i) {
    counts[i] = static_cast<uint32_t>(i % 4) * 16;
    for (uint32_t j = 0; j < counts[i]; ++j) {
      indices.push_back(j);
      values.push_back(float(i % 13) / 4);
    }
  }
  IndexQueryMeta query_meta(IndexMeta::DT_FP32);
  for (bool brute : {false, true}) {
    auto search = [&] {
      return brute ? flow.search_bf_impl(counts.data(), indices.data(),
                                         values.data(), query_meta,
                                         counts.size(), ctx)
                   : flow.search_impl(counts.data(), indices.data(),
                                      values.data(), query_meta, counts.size(),
                                      ctx);
    };
    ASSERT_EQ(0, pool.init(4096));
    const size_t before = searcher->batches;
    EXPECT_EQ(IndexError_NoMemory, search());
    EXPECT_EQ(before, searcher->batches);
    EXPECT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(1024 * 1024));
    ASSERT_EQ(0, search());
    EXPECT_EQ(before + 1, searcher->batches);
    EXPECT_EQ(0u, pool.used());
    EXPECT_TRUE(ctx->features().empty());
    for (size_t i = 0; i < counts.size(); ++i) {
      ASSERT_EQ(1u, ctx->result(i).size());
      EXPECT_EQ(counts[i], ctx->result(i)[0].key());
    }
  }
  std::vector<ailego::Float16> half(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    half[i] = ailego::Float16(values[i]);
  ASSERT_EQ(0, pool.init(1));
  ASSERT_EQ(0, flow.search_impl(counts.data(), indices.data(), half.data(),
                                IndexQueryMeta(IndexMeta::DT_FP16),
                                counts.size(), ctx));
  EXPECT_EQ(half.data(), searcher->last_input);
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1024));
  ASSERT_EQ(
      0, flow.search_impl(16, indices.data(), values.data(), query_meta, ctx));
  ASSERT_EQ(0, flow.search_bf_impl(16, indices.data(), values.data(),
                                   query_meta, ctx));
  EXPECT_EQ(0u, pool.used());
}
}  // namespace
}  // namespace zvec::core

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <filesystem>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "flat/flat_searcher.h"

namespace zvec::core {
namespace {
class BatchCountingFlatSearcher : public IndexSearcher {
 public:
  int init(const ailego::Params &params) override {
    return inner_->init(params);
  }
  int init(const ailego::Params &params,
           const std::shared_ptr<turbo::Quantizer> &quantizer) override {
    return inner_->init(params, quantizer);
  }
  int cleanup() override {
    return inner_->cleanup();
  }
  int unload() override {
    return inner_->unload();
  }
  const IndexMeta &meta() const override {
    return inner_->meta();
  }
  const ailego::Params &params() const override {
    return inner_->params();
  }
  const Stats &stats() const override {
    return inner_->stats();
  }
  int load(IndexStorage::Pointer storage,
           IndexMetric::Pointer metric) override {
    return inner_->load(std::move(storage), std::move(metric));
  }
  Context::Pointer create_context() const override {
    return inner_->create_context();
  }
  int search_impl(const void *query, const IndexQueryMeta &meta,
                  Context::Pointer &context) const override {
    return inner_->search_impl(query, meta, context);
  }
  int search_impl(const void *query, const IndexQueryMeta &meta, uint32_t count,
                  Context::Pointer &context) const override {
    ++batches;
    last_count = count;
    return inner_->search_impl(query, meta, count, context);
  }
  int search_bf_impl(const void *query, const IndexQueryMeta &meta,
                     uint32_t count, Context::Pointer &context) const override {
    ++batches;
    last_count = count;
    return inner_->search_bf_impl(query, meta, count, context);
  }
  mutable size_t batches{0}, last_count{0};

 private:
  IndexSearcher::Pointer inner_{IndexFactory::CreateSearcher("FlatSearcher")};
};

TEST(FlatFlowWorkspace,
     ConvertedBatchKeepsOneScanAndCanRetryAfterBudgetFailure) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "flat_flow_workspace_test.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
    std::filesystem::remove(path);
  });
  for (bool quantized : {false, true}) {
    SCOPED_TRACE(quantized);
    ASSERT_EQ(0, pool.init(0));
    constexpr size_t dimension = 8, rows = 257, queries_count = 2053;
    IndexMeta meta(IndexMeta::DT_FP16, dimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
    std::shared_ptr<turbo::Quantizer> quantizer;
    if (quantized) {
      IndexMeta raw(IndexMeta::DT_FP32, dimension);
      raw.set_metric("SquaredEuclidean", 0, ailego::Params{});
      quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
      ASSERT_EQ(0, quantizer->init(raw, {}));
      meta = quantizer->meta();
      meta.set_quantizer("Fp16Quantizer", 0, ailego::Params{});
    } else {
      meta.set_reformer("HalfFloatReformer", 0, ailego::Params{});
    }
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP16>>(dimension);
    for (size_t i = 0; i < rows; ++i) {
      ailego::NumericalVector<ailego::Float16> row(dimension);
      for (size_t d = 0; d < dimension; ++d) row[d] = ailego::Float16(float(i));
      holder->emplace(i, std::move(row));
    }
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, quantizer ? builder->init(meta, {}, quantizer)
                           : builder->init(meta, {}));
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
    ASSERT_EQ(0, dumper->close());
    builder.reset();
    holder.reset();
    IndexFlow flow;
    auto searcher = std::make_shared<BatchCountingFlatSearcher>();
    ASSERT_EQ(0,
              quantizer ? searcher->init({}, quantizer) : searcher->init({}));
    ASSERT_EQ(0, flow.set_searcher(searcher));
    ASSERT_EQ(0, flow.load(path));
    auto context = flow.create_context();
    ASSERT_NE(nullptr, context->searcher_context());
    context->set_topk(1);
    std::vector<float> queries(queries_count * dimension);
    for (size_t q = 0; q < queries_count; ++q)
      std::fill_n(queries.data() + q * dimension, dimension, float(q % rows));
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, dimension);
    ASSERT_EQ(0, pool.init(64 * 1024));
    EXPECT_EQ(IndexError_NoMemory,
              flow.search_impl(queries.data(), qmeta, queries_count, context));
    EXPECT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(1024 * 1024));
    for (bool brute_force : {false, true}) {
      searcher->batches = 0;
      context->mutable_features()->resize(1024 * 1024);
      ASSERT_EQ(0, brute_force ? flow.search_bf_impl(queries.data(), qmeta,
                                                     queries_count, context)
                               : flow.search_impl(queries.data(), qmeta,
                                                  queries_count, context));
      EXPECT_EQ(1u, searcher->batches);
      EXPECT_EQ(queries_count, searcher->last_count);
      EXPECT_TRUE(context->features().empty());
      EXPECT_LT(context->features().capacity(), 1024u);
      for (size_t q = 0; q < queries_count; ++q) {
        ASSERT_EQ(1u, context->result(q).size());
        EXPECT_EQ(q % rows, context->result(q)[0].key());
        EXPECT_FLOAT_EQ(0.0f, context->result(q)[0].score());
      }
    }
    ASSERT_EQ(0, flow.search_impl(queries.data(), qmeta, context));
    ASSERT_EQ(1u, context->result().size());
    EXPECT_EQ(0u, context->result()[0].key());
    context.reset();
    ASSERT_EQ(0, flow.unload());
  }
}
}  // namespace
}  // namespace zvec::core

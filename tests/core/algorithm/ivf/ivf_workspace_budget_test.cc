// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <filesystem>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "ailego/container/vector_array.h"
#include "ailego/pattern/defer.h"
#include "algorithm/ivf/ivf_dumper.h"
#include "algorithm/ivf/ivf_entity.h"
#include "algorithm/ivf/ivf_searcher_context.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_context.h"
#include "quantizer/rotator/rotator.h"
#include "turbo/algorithm/kmeans.h"
#include "turbo/preprocessor/fht_rotator/fht_rotator.h"
#include "turbo/quantizer/entropy_integer_quantizer/integer_quantizer.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

class IVFWorkspaceBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool.capacity();
    ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool.external_used());
    EXPECT_EQ(0, pool.init(previous_));
  }
  ailego::PoolPayloadBudget &pool = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
};

TEST_F(IVFWorkspaceBudgetTest, TrainingAndMetadataRejectBeforeAllocation) {
  ailego::NumericalVectorArray<float> centers(1536);
  IndexMeta meta(IndexMeta::DT_FP32, 32);
  IndexSearcher::Context::Pointer centroid_context;
  IVFSearcherContext context(nullptr, centroid_context);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  EXPECT_THROW(centers.resize(1024), std::bad_alloc);
  EXPECT_EQ(0u, centers.count());
  EXPECT_THROW((IVFDumper(meta, nullptr, 65536, 32)), std::bad_alloc);
  EXPECT_THROW(context.reset_results(100000), std::bad_alloc);
}

TEST_F(IVFWorkspaceBudgetTest, WorkerBudgetRejectionDoesNotEscapeThread) {
  using Algorithm = turbo::NumericalKmeans<float, IndexThreads>;
  class WorkerProbe : public Algorithm {
   public:
    WorkerProbe() : Algorithm(1, 1536) {}
    using Algorithm::cluster_matrix_features;
  } algorithm;
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  std::thread worker([&] { algorithm.cluster_matrix_features(0, 1); });
  worker.join();
  EXPECT_EQ(Algorithm::kWorkspaceNoMemory, algorithm.matrix_status());
}

TEST_F(IVFWorkspaceBudgetTest, QueryScratchReusesCapacityButRejectsGrowth) {
  IVFEntity::IVFReformerWrapper reformer;
  IndexMeta quantized(IndexMeta::DT_INT8, 1536);
  quantized.set_metric("InnerProduct", 0, ailego::Params{});
  ailego::Params params;
  params.set(INT8_QUANTIZER_REFORMER_BIAS, 0.0f);
  params.set(INT8_QUANTIZER_REFORMER_SCALE, 1.0f);
  quantized.set_reformer("Int8QuantizerReformer", 0, params);
  ASSERT_EQ(0, reformer.init(quantized));
  std::vector<float> queries(64 * 1536, 1.0f);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, 1536), output_meta;
  const void *out = nullptr;
  ASSERT_EQ(0,
            reformer.transform(queries.data(), qmeta, 32, &out, &output_meta));
  EXPECT_GE(pool.external_used(), 32u * 1536);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  const void *first = out;
  EXPECT_EQ(0,
            reformer.transform(queries.data(), qmeta, 32, &out, &output_meta));
  EXPECT_EQ(first, out);
  EXPECT_EQ(IndexError_NoMemory,
            reformer.transform(queries.data(), qmeta, 64, &out, &output_meta));
}

}  // namespace
}  // namespace zvec::core

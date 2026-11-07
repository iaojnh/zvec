// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstring>
#include <filesystem>
#include <tuple>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core_interface {
namespace {
class IndexBuildAdmissionTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    previous_capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(64UL << 20));
    std::filesystem::remove_all(path_);
    ASSERT_TRUE(std::filesystem::create_directory(path_));
  }
  void TearDown() override {
    std::filesystem::remove_all(path_);
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_capacity_));
  }
  size_t previous_capacity_{0};
  const std::string path_{"index_build_admission_test_data"};
};

TEST_P(IndexBuildAdmissionTest, RejectsExhaustedBudgetAndRetries) {
  const bool diskann = std::get<0>(GetParam());
  const bool fp16 = std::get<1>(GetParam());
  constexpr uint32_t dimension = 32;
  constexpr uint32_t count = 256;
  const QuantizerParam quantizer(fp16 ? QuantizerType::kFP16
                                      : QuantizerType::kNone);
  BaseIndexParam::Pointer param;
  if (diskann) {
    param = DiskAnnIndexParamBuilder()
                .with_dimension(dimension)
                .with_data_type(DataType::DT_FP32)
                .with_metric_type(MetricType::kL2sq)
                .with_quantizer_param(quantizer)
                .with_max_degree(16)
                .with_list_size(32)
                .with_pq_chunk_num(4)
                .build();
  } else {
    param = IVFIndexParamBuilder()
                .with_dimension(dimension)
                .with_data_type(DataType::DT_FP32)
                .with_metric_type(MetricType::kL2sq)
                .with_quantizer_param(quantizer)
                .with_n_list(4)
                .with_n_iters(3)
                .build();
  }
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path_ + "/index",
                           {StorageOptions::StorageType::kBufferPool, true}));
  std::vector<float> values(count * dimension);
  for (uint32_t id = 0; id < count; ++id) {
    for (uint32_t d = 0; d < dimension; ++d)
      values[id * dimension + d] = std::sin(float(id * 23 + d * 17));
    ASSERT_EQ(
        0, index->add(VectorData{DenseVector{values.data() + id * dimension}},
                      id));
  }
  ASSERT_EQ(0, index->flush());
  ASSERT_FALSE(index->is_trained());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    EXPECT_EQ(core::IndexError_NoMemory, index->train());
    EXPECT_FALSE(index->is_trained());
  }
  // A denied construction must keep every source row readable, and a fresh
  // attempt must succeed once the competing allocation has been released.
  ASSERT_EQ(count, index->get_doc_count());
  for (uint32_t id = 0; id < count; ++id) {
    VectorDataBuffer fetched;
    ASSERT_EQ(0, index->fetch(id, &fetched));
    const auto &data = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
    ASSERT_EQ(dimension * sizeof(float), data.size());
    for (uint32_t d = 0; d < dimension; ++d) {
      float actual;
      std::memcpy(&actual, data.data() + d * sizeof(float), sizeof(actual));
      ASSERT_NEAR(values[id * dimension + d], actual, fp16 ? 0.002f : 1e-6f);
    }
  }
  ASSERT_EQ(0, index->train());
  EXPECT_TRUE(index->is_trained());
  EXPECT_EQ(count, index->get_doc_count());
  BaseIndexQueryParam::Pointer query;
  if (diskann) {
    auto options = std::make_shared<DiskAnnQueryParam>();
    options->list_size = 256;
    query = std::move(options);
  } else {
    auto options = std::make_shared<IVFQueryParam>();
    options->nprobe = 4;
    query = std::move(options);
  }
  query->topk = 1;
  SearchResult result;
  ASSERT_EQ(
      0, index->search(VectorData{DenseVector{values.data()}}, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(0u, result.doc_list_[0].key());
  ASSERT_EQ(0, index->close());
}

INSTANTIATE_TEST_SUITE_P(
    IndexAndPrecision, IndexBuildAdmissionTest,
    testing::Combine(testing::Bool(), testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<bool, bool>> &info) {
      return std::string(std::get<0>(info.param) ? "Diskann" : "Ivf") +
             (std::get<1>(info.param) ? "Fp16" : "Fp32");
    });
}  // namespace
}  // namespace zvec::core_interface

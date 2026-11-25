// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <tuple>
#include <vector>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_streamer.h"

namespace zvec::core_interface {
namespace {

#if GTEST_HAS_DEATH_TEST
using StorageType = StorageOptions::StorageType;

void HnswOpenFailureAndRetry(StorageType type, bool pressure, bool fp16) {
  auto param = HNSWIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(16)
                   .with_m(17)
                   .with_ef_construction(65)
                   .with_quantizer_param(QuantizerParam(
                       fp16 ? QuantizerType::kFP16 : QuantizerType::kNone))
                   .build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  const std::string path = "hnsw_open_failure_retry.index";
  ailego::File::Delete(path);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(size_t{256} << 20));
  ASSERT_TRUE(pool.configure_process_rss_limit(size_t{1024} << 20));
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  if (pressure) pool.set_memory_pressure(Pressure::kCritical);
  size_t after_first = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    SCOPED_TRACE(attempt);
    EXPECT_NO_THROW(EXPECT_EQ(
        pressure ? core::IndexError_NoMemory : core::IndexError_OpenFile,
        index->open(path, {type, pressure})));
    EXPECT_FALSE(ailego::File::IsExist(path));
    if (attempt == 0) after_first = pool.used();
    EXPECT_EQ(after_first, pool.used());
  }
  pool.set_memory_pressure(Pressure::kNormal);
  ASSERT_NO_THROW(ASSERT_EQ(0, index->open(path, {type, true})));
  const auto retained =
      std::dynamic_pointer_cast<HNSWIndexParam>(index->get_param());
  ASSERT_NE(nullptr, retained);
  EXPECT_EQ(17, retained->m);
  EXPECT_EQ(65, retained->ef_construction);
  std::vector<float> row(16, 1.0F);
  ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 7));
  row.assign(16, 2.0F);
  ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 9));
  row.assign(16, 1.0F);
  auto query = HNSWQueryParamBuilder().with_topk(1).with_ef_search(16).build();
  SearchResult result;
  ASSERT_EQ(0,
            index->search(VectorData{DenseVector{row.data()}}, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(7u, result.doc_list_[0].key());
  EXPECT_TRUE(std::isfinite(result.doc_list_[0].score()));
  ASSERT_EQ(0, index->close());
  ailego::File::Delete(path);
}

class HnswOpenRetryTest
    : public ::testing::TestWithParam<std::tuple<StorageType, bool, bool>> {};

TEST_P(HnswOpenRetryTest, FailurePreservesConcreteParamsAndAllowsRetry) {
  const auto [type, pressure, fp16] = GetParam();
  ASSERT_EXIT(
      {
        HnswOpenFailureAndRetry(type, pressure, fp16);
        std::fflush(stdout);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    MmapAndBufferPool, HnswOpenRetryTest,
    ::testing::Combine(::testing::Values(StorageType::kMMAP,
                                         StorageType::kBufferPool),
                       ::testing::Bool(), ::testing::Bool()));

struct OpenFailure {
  int remaining{0};
  bool throws{false};
};

class FailingOpenHnswStreamer : public core::HnswStreamer {
 public:
  using core::HnswStreamer::open;

  explicit FailingOpenHnswStreamer(OpenFailure *failure) : failure_(failure) {}

  int open(core::IndexStorage::Pointer storage, bool read_only) override {
    if (failure_->remaining > 0) {
      --failure_->remaining;
      if (failure_->throws) throw std::bad_alloc();
      return core::IndexError_NoMemory;
    }
    return core::HnswStreamer::open(std::move(storage), read_only);
  }

 private:
  OpenFailure *failure_;
};

class ScopedOpenFailure {
 public:
  explicit ScopedOpenFailure(OpenFailure &failure) {
    ailego::Factory<core::IndexStreamer>::Register<FailingOpenHnswStreamer>
        replace("HnswStreamer", &failure);
  }

  ~ScopedOpenFailure() {
    ailego::Factory<core::IndexStreamer>::Register<core::HnswStreamer> restore(
        "HnswStreamer");
  }
};

void HnswFormatSwitchFailureAndRetry(StorageType type, bool from_legacy,
                                     bool throws) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(size_t{256} << 20));
  const std::string legacy_path = "hnsw_pipeline_retry_legacy.index";
  const std::string turbo_path = "hnsw_pipeline_retry_turbo.index";
  ailego::File::Delete(legacy_path);
  ailego::File::Delete(turbo_path);
  AILEGO_DEFER([&] {
    ailego::File::Delete(legacy_path);
    ailego::File::Delete(turbo_path);
  });

  auto legacy_param =
      HNSWIndexParamBuilder()
          .with_metric_type(MetricType::kL2sq)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(16)
          .with_quantizer_param(QuantizerParam(QuantizerType::kInt8, true))
          .build();
  auto turbo_param =
      HNSWIndexParamBuilder()
          .with_metric_type(MetricType::kL2sq)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(16)
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .build();
  OpenFailure failure;
  ScopedOpenFailure scoped_failure(failure);
  for (bool legacy : {true, false}) {
    auto index =
        IndexFactory::CreateAndInitIndex(legacy ? *legacy_param : *turbo_param);
    ASSERT_NE(nullptr, index);
    ASSERT_EQ(0, index->open(legacy ? legacy_path : turbo_path, {type, true}));
    std::vector<float> row(16, 1.0F);
    ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 7));
    row.assign(16, 3.0F);
    ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 9));
    ASSERT_EQ(0, index->close());
  }

  // The configured legacy encoding can read either persisted layout. Load
  // its rotator first when switching to Turbo, so restoring that reformer
  // alone would require a rotator segment that the Turbo file does not have.
  auto index = IndexFactory::CreateAndInitIndex(*legacy_param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(from_legacy ? legacy_path : turbo_path,
                           {type, false, true}));
  ASSERT_EQ(0, index->close());
  const std::string &next_path = from_legacy ? turbo_path : legacy_path;
  failure.remaining = 2;
  failure.throws = throws;
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_NO_THROW(EXPECT_EQ(core::IndexError_NoMemory,
                              index->open(next_path, {type, false, true})));
  }
  ASSERT_EQ(0, failure.remaining);
  ASSERT_EQ(0, index->open(next_path, {type, false, true}));
  const std::vector<float> row(16, 1.0F);
  auto query = HNSWQueryParamBuilder().with_topk(1).with_ef_search(16).build();
  query->fetch_vector = true;
  SearchResult result;
  ASSERT_EQ(0,
            index->search(VectorData{DenseVector{row.data()}}, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(7u, result.doc_list_[0].key());
  EXPECT_TRUE(std::isfinite(result.doc_list_[0].score()));
  ASSERT_EQ(1u, result.reverted_vector_list_.size());
  EXPECT_EQ(16 * sizeof(float), result.reverted_vector_list_[0].size());
  ASSERT_EQ(0, index->close());
}

class HnswFormatSwitchRetryTest
    : public ::testing::TestWithParam<std::tuple<StorageType, bool, bool>> {};

TEST_P(HnswFormatSwitchRetryTest, RestoresConsistentPipelineAfterOpenFailure) {
  const auto [type, from_legacy, throws] = GetParam();
  ASSERT_EXIT(
      {
        HnswFormatSwitchFailureAndRetry(type, from_legacy, throws);
        std::fflush(stdout);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    MmapAndBufferPool, HnswFormatSwitchRetryTest,
    ::testing::Combine(::testing::Values(StorageType::kMMAP,
                                         StorageType::kBufferPool),
                       ::testing::Bool(), ::testing::Bool()));

void VamanaOpenFailureAndRetry() {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(size_t{256} << 20));
  ASSERT_TRUE(pool.configure_process_rss_limit(size_t{1024} << 20));
  for (const auto type : {StorageOptions::StorageType::kMMAP,
                          StorageOptions::StorageType::kBufferPool}) {
    auto param = VamanaIndexParamBuilder()
                     .with_metric_type(MetricType::kL2sq)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(16)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);
    const std::string path = "vamana_open_failure_retry.index";
    ailego::File::Delete(path);
    EXPECT_EQ(core::IndexError_OpenFile, index->open(path, {type, false}));
    ASSERT_NO_THROW(ASSERT_EQ(0, index->open(path, {type, true})));
    EXPECT_NE(nullptr,
              std::dynamic_pointer_cast<VamanaIndexParam>(index->get_param()));
    const std::vector<float> row(16, 1.0F);
    ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 7));
    auto query = VamanaQueryParamBuilder().with_topk(1).build();
    SearchResult result;
    ASSERT_EQ(
        0, index->search(VectorData{DenseVector{row.data()}}, query, &result));
    ASSERT_EQ(1u, result.doc_list_.size());
    EXPECT_EQ(7u, result.doc_list_[0].key());
    ASSERT_EQ(0, index->close());
    ailego::File::Delete(path);
  }
}

TEST(IndexOpenRetry, VamanaRetainsConcreteParamsAfterMissingFile) {
  ASSERT_EXIT(
      {
        VamanaOpenFailureAndRetry();
        std::fflush(stdout);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}
#endif  // GTEST_HAS_DEATH_TEST

}  // namespace
}  // namespace zvec::core_interface

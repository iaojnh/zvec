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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/turbo/turbo.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {

constexpr uint32_t kDimension = 17;
constexpr uint32_t kCount = core::HnswEntity::kDefaultBruteForceThreshold + 257;
constexpr uint32_t kTopK = 10;

enum class Format { Fp32, NativeFp16, QuantizedFp16 };

struct Case {
  Format format;
  MetricType metric;
  bool contiguous;
  StorageOptions storage;
  std::string name;
};

std::vector<Case> Cases() {
  std::vector<Case> cases;
  const std::pair<Format, const char *> formats[] = {
      {Format::Fp32, "Fp32"},
      {Format::NativeFp16, "NativeFp16"},
      {Format::QuantizedFp16, "QuantizedFp16"}};
  const std::pair<MetricType, const char *> metrics[] = {
      {MetricType::kL2sq, "L2"},
      {MetricType::kInnerProduct, "IP"},
      {MetricType::kCosine, "Cosine"}};
  const std::pair<StorageOptions, const char *> stores[] = {
      {{StorageOptions::StorageType::kMMAP, false}, "Mmap"},
      {{StorageOptions::StorageType::kMMAP, false, true}, "ReadOnly"},
      {{StorageOptions::StorageType::kMMAP, false, false, true}, "CopyOnWrite"},
      {{StorageOptions::StorageType::kBufferPool, false, true}, "BufferPool"}};
  for (const auto &format : formats) {
    for (const auto &metric : metrics) {
      for (bool contiguous : {false, true}) {
        for (const auto &store : stores) {
          cases.push_back({format.first, metric.first, contiguous, store.first,
                           std::string(format.second) + "_" + metric.second +
                               (contiguous ? "_Contiguous_" : "_Regular_") +
                               store.second});
        }
      }
    }
  }
  return cases;
}

class HnswPlatformTest : public testing::TestWithParam<Case> {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(0,
              ailego::MemoryLimitPool::get_instance().init(256 * 1024 * 1024));
  }

  void SetUp() override {
    path_ = "hnsw_platform_" + GetParam().name + ".index";
    test_util::RemoveTestFiles(path_);
    // Dyadic coordinates avoid differences between platform FP16 converters.
    // mt19937 output is standardized; uniform_real_distribution is not.
    std::mt19937 random(20261029);
    vectors_.assign(kCount, std::vector<float>(kDimension));
    for (auto &vector : vectors_) {
      for (auto &value : vector) {
        value = (static_cast<int>(random() % 1025) - 512) / 1024.0f;
      }
    }
    queries_.assign(4, std::vector<float>(kDimension));
    for (auto &query : queries_) {
      for (auto &value : query) {
        value = (static_cast<int>(random() % 257) - 128) / 256.0f;
      }
    }
  }

  void TearDown() override {
    test_util::RemoveTestFiles(path_);
  }

  HNSWIndexParam::Pointer param() const {
    return HNSWIndexParamBuilder()
        .with_data_type(GetParam().format == Format::NativeFp16
                            ? DataType::DT_FP16
                            : DataType::DT_FP32)
        .with_dimension(kDimension)
        .with_metric_type(GetParam().metric)
        .with_m(24)
        .with_ef_construction(160)
        .with_use_contiguous_memory(GetParam().contiguous)
        .with_quantizer_param(QuantizerParam(
            GetParam().format == Format::QuantizedFp16 ? QuantizerType::kFP16
                                                       : QuantizerType::kNone))
        .build();
  }

  double score(const std::vector<float> &a, const std::vector<float> &b) const {
    double dot = 0, square = 0, norm_a = 0, norm_b = 0;
    for (size_t d = 0; d < kDimension; ++d) {
      const double x = a[d], y = b[d];
      dot += x * y;
      square += (x - y) * (x - y);
      norm_a += x * x;
      norm_b += y * y;
    }
    if (GetParam().metric == MetricType::kL2sq) return square;
    if (GetParam().metric == MetricType::kInnerProduct) return dot;
    return 1 - dot / std::sqrt(norm_a * norm_b);
  }

  void check(Index *index) const {
    auto context = index->index_searcher()->create_context();
    auto *hnsw = dynamic_cast<core::HnswContext *>(context.get());
    ASSERT_NE(nullptr, hnsw);
    ASSERT_GT(index->get_doc_count(), hnsw->get_bruteforce_threshold());
    for (bool filtered : {false, true, false}) {
      const std::array<uint64_t, 8> excluded = {{0, 3, 6, 9, 12, 15, 18, 21}};
      for (const auto &query : queries_) {
        std::vector<double> expected(kCount), ordered;
        for (uint32_t key = 0; key < kCount; ++key) {
          expected[key] = score(vectors_[key], query);
          if (!filtered || key % 3 != 0) ordered.push_back(expected[key]);
        }
        const bool ip = GetParam().metric == MetricType::kInnerProduct;
        std::sort(ordered.begin(), ordered.end(),
                  [ip](double a, double b) { return ip ? a > b : a < b; });
        std::vector<ailego::Float16> native_query(query.begin(), query.end());
        const void *query_data =
            GetParam().format == Format::NativeFp16
                ? static_cast<const void *>(native_query.data())
                : static_cast<const void *>(query.data());
        for (bool linear : {true, false}) {
          SCOPED_TRACE(testing::Message()
                       << "filtered=" << filtered << " linear=" << linear);
          // The DB creates a call-local predicate, then transfers it into the
          // context. Reuse the context across calls, not a consumed predicate.
          std::shared_ptr<IndexFilter> filter;
          if (filtered) {
            filter = std::make_shared<IndexFilter>();
            filter->set([excluded](uint64_t key) {
              return std::find(excluded.begin(), excluded.end(), key % 24) !=
                     excluded.end();
            });
          }
          auto params = HNSWQueryParamBuilder()
                            .with_topk(kTopK)
                            .with_ef_search(256)
                            .with_is_linear(linear)
                            .with_filter(filter)
                            .build();
          SearchResult result;
          ASSERT_EQ(0, index->search(VectorData{DenseVector{query_data}},
                                     params, &result));
          ASSERT_EQ(kTopK, result.doc_list_.size());
          std::unordered_set<uint64_t> seen;
          size_t relevant = 0;
          for (size_t rank = 0; rank < result.doc_list_.size(); ++rank) {
            const auto &hit = result.doc_list_[rank];
            ASSERT_LT(hit.key(), kCount);
            EXPECT_TRUE(seen.insert(hit.key()).second);
            if (filtered) EXPECT_NE(0U, hit.key() % 3);
            ASSERT_TRUE(std::isfinite(hit.score()));
            const double tolerance =
                GetParam().metric == MetricType::kCosine
                    ? 2e-3
                    : 1e-5 * std::max(1.0, std::abs(expected[hit.key()]));
            EXPECT_NEAR(expected[hit.key()], hit.score(), tolerance);
            if (rank) {
              const auto previous = result.doc_list_[rank - 1].score();
              EXPECT_TRUE(ip ? previous >= hit.score()
                             : previous <= hit.score());
            }
            relevant +=
                ip ? expected[hit.key()] >= ordered[kTopK - 1] - tolerance
                   : expected[hit.key()] <= ordered[kTopK - 1] + tolerance;
          }
          EXPECT_GE(relevant, linear ? kTopK : 9U);
        }
      }
    }
  }

  std::string path_;
  std::vector<std::vector<float>> vectors_, queries_;
};

TEST_P(HnswPlatformTest, GraphFilterAndReopenMatchIndependentScores) {
  auto writer = IndexFactory::CreateAndInitIndex(*param());
  ASSERT_NE(nullptr, writer);
  ASSERT_EQ(0, writer->open(path_, {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t key = 0; key < kCount; ++key) {
    std::vector<ailego::Float16> native(vectors_[key].begin(),
                                        vectors_[key].end());
    const void *data = GetParam().format == Format::NativeFp16
                           ? static_cast<const void *>(native.data())
                           : static_cast<const void *>(vectors_[key].data());
    ASSERT_EQ(0, writer->add(VectorData{DenseVector{data}}, key));
  }
  ASSERT_EQ(0, writer->train());
  check(writer.get());
  ASSERT_EQ(0, writer->close());
  writer.reset();

  // Read the same persisted graph through each supported storage/layout pair.
  for (int reopen = 0; reopen < 2; ++reopen) {
    auto reader = IndexFactory::CreateAndInitIndex(*param());
    ASSERT_NE(nullptr, reader);
    ASSERT_EQ(0, reader->open(path_, GetParam().storage));
    check(reader.get());
    ASSERT_EQ(0, reader->close());
  }
}

INSTANTIATE_TEST_SUITE_P(FormatsMetricsLayoutsAndStorage, HnswPlatformTest,
                         testing::ValuesIn(Cases()),
                         [](const testing::TestParamInfo<Case> &info) {
                           return info.param.name;
                         });

TEST(HnswPlatformDispatch, Fp16ExercisesEveryAvailableArchitecture) {
  const auto &flags = ailego::internal::CpuFeatures::static_flags_;
#if defined(__aarch64__) || defined(_M_ARM64)
  EXPECT_TRUE(flags.NEON);
#elif defined(__x86_64__) || defined(_M_X64)
  EXPECT_TRUE(flags.SSE2);
#endif
  struct Arch {
    turbo::CpuArchType type;
    const char *name;
    bool enabled;
  };
  const Arch arches[] = {
      {turbo::CpuArchType::kAVX512, "AVX512", flags.AVX512F && flags.F16C},
      {turbo::CpuArchType::kAVX2, "AVX2", flags.AVX2 && flags.F16C},
      {turbo::CpuArchType::kSSE2, "SSE2", flags.SSE2},
      {turbo::CpuArchType::kNEON, "NEON", flags.NEON},
      {turbo::CpuArchType::kScalar, "Scalar", true}};
  std::ostringstream coverage;
  for (const auto &arch : arches)
    coverage << arch.name << '=' << arch.enabled << ';';
  using RawDistance = void (*)(const void *, const void *, size_t, float *);
  const auto neon_raw = turbo::get_distance_kernels(
      turbo::MetricType::kCosine, turbo::DataType::kFp16,
      turbo::QuantizeType::kRaw, turbo::CpuArchType::kNEON);
  const auto neon_quantized = turbo::get_distance_kernels(
      turbo::MetricType::kCosine, turbo::DataType::kFp16,
      turbo::QuantizeType::kFp16, turbo::CpuArchType::kNEON);
  const bool native_neon = neon_raw.dist && neon_quantized.dist &&
                           *neon_raw.dist.target<RawDistance>() !=
                               *neon_quantized.dist.target<RawDistance>();
  coverage << "NEON_BASELINE=" << static_cast<bool>(neon_raw.dist)
           << ";NEON_FP16=" << native_neon << ';';
  coverage << "F16C=" << flags.F16C << ";FP16=" << flags.FP16 << ';';
  RecordProperty("fp16_cpu_architectures", coverage.str());
  std::cout << "FP16 architecture coverage: " << coverage.str() << std::endl;

  const std::pair<turbo::MetricType, const char *> metrics[] = {
      {turbo::MetricType::kSquaredEuclidean, "SquaredEuclidean"},
      {turbo::MetricType::kInnerProduct, "InnerProduct"},
      {turbo::MetricType::kCosine, "Cosine"}};
  for (const auto &metric : metrics) {
    for (const auto encoding :
         {turbo::QuantizeType::kFp16, turbo::QuantizeType::kRaw}) {
      // Raw storage has specialized L2/cosine dispatch; IP uses Fp16Quantizer.
      if (encoding == turbo::QuantizeType::kRaw &&
          metric.first == turbo::MetricType::kInnerProduct)
        continue;
      auto automatic = turbo::get_distance_kernels(
          metric.first, turbo::DataType::kFp16, encoding);
      ASSERT_TRUE(automatic.dist);
      ASSERT_NE(nullptr, automatic.dist.target<RawDistance>());
      bool checked_auto = false;
      for (const auto &arch : arches) {
        SCOPED_TRACE(arch.name);
        auto kernels = turbo::get_distance_kernels(
            metric.first, turbo::DataType::kFp16, encoding, arch.type);
        bool enabled = arch.enabled;
        if (encoding == turbo::QuantizeType::kRaw &&
            metric.first == turbo::MetricType::kSquaredEuclidean) {
          enabled = arch.type == turbo::CpuArchType::kScalar ||
                    (arch.type == turbo::CpuArchType::kNEON && flags.NEON) ||
                    (arch.type == turbo::CpuArchType::kAVX512 && arch.enabled &&
                     flags.AVX512DQ);
        }
        ASSERT_EQ(enabled, static_cast<bool>(kernels.dist));
        ASSERT_EQ(enabled, static_cast<bool>(kernels.batch));
        if (!enabled) continue;
        if (!checked_auto) {
          ASSERT_NE(nullptr, kernels.dist.target<RawDistance>());
          EXPECT_EQ(*kernels.dist.target<RawDistance>(),
                    *automatic.dist.target<RawDistance>());
          checked_auto = true;
        }
        for (size_t dim : {1, 7, 8, 17, 32, 65}) {
          for (float scale : {1.0f, 1.0f / 65536, 32768.0f}) {
            auto quantizer =
                core::IndexFactory::CreateQuantizer("Fp16Quantizer");
            ASSERT_NE(nullptr, quantizer);
            core::IndexMeta meta(core::IndexMeta::DT_FP32, dim);
            meta.set_metric(metric.second, 0, ailego::Params());
            ASSERT_EQ(0, quantizer->init(meta, {}));
            std::vector<std::string> codes(14);
            for (size_t row = 0; row < codes.size(); ++row) {
              std::vector<float> values(dim);
              for (size_t d = 0; d < dim; ++d) {
                values[d] = scale *
                            (static_cast<int>((row * 31 + d * 17) % 29) - 14) /
                            32.0f;
              }
              core::IndexQueryMeta output_meta;
              ASSERT_EQ(0,
                        quantizer->quantize(
                            values.data(),
                            core::IndexQueryMeta(core::IndexMeta::DT_FP32, dim),
                            &codes[row], &output_meta));
            }
            std::vector<const void *> rows;
            for (size_t row = 1; row < codes.size(); ++row)
              rows.push_back(codes[row].data());
            auto expected_score = [&](size_t row) {
              double dot = 0, squared = 0;
              for (size_t d = 0; d < dim; ++d) {
                uint16_t a, b;
                std::memcpy(&a, codes[row + 1].data() + d * sizeof(a),
                            sizeof(a));
                std::memcpy(&b, codes[0].data() + d * sizeof(b), sizeof(b));
                const double x = ailego::FloatHelper::ToFP32(a),
                             y = ailego::FloatHelper::ToFP32(b);
                dot += x * y;
                squared += (x - y) * (x - y);
              }
              return metric.first == turbo::MetricType::kSquaredEuclidean
                         ? squared
                     : metric.first == turbo::MetricType::kCosine ? 1 - dot
                                                                  : -dot;
            };
            for (size_t count : {1, 11, 12, 13}) {
              std::vector<float> actual(count);
              kernels.batch(rows.data(), codes[0].data(), count, dim,
                            actual.data(), nullptr);
              for (size_t row = 0; row < count; ++row) {
                float single = 0;
                kernels.dist(rows[row], codes[0].data(), dim, &single);
                const double expected = expected_score(row);
                const double tolerance =
                    metric.first == turbo::MetricType::kCosine
                        ? 2e-3
                        : 1e-5 * std::max(1e-12, std::abs(expected));
                EXPECT_NEAR(expected, single, tolerance);
                EXPECT_NEAR(expected, actual[row], tolerance);
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace zvec::core_interface

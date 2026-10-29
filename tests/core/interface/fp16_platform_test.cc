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
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "tests/test_util.h"
#include "fp16_test_util.h"

namespace zvec::core_interface {
namespace {

constexpr uint32_t kDimension = 17;
constexpr uint32_t kCount = 192;
constexpr uint32_t kTopK = 10;

#if defined(__ANDROID__) || defined(IOS)
static_assert(DISKANN_SUPPORTED == 1, "Mobile SDKs must include DiskANN");
#endif

struct Case {
  IndexType index;
  MetricType metric;
  bool quantized;
  std::string name;
};

std::vector<Case> Cases() {
  std::vector<Case> cases;
  for (auto index : {IndexType::kFlat, IndexType::kIVF, IndexType::kDiskAnn}) {
#if !DISKANN_SUPPORTED
    if (index == IndexType::kDiskAnn) continue;
#endif
    const char *kind = index == IndexType::kFlat  ? "Flat"
                       : index == IndexType::kIVF ? "IVF"
                                                  : "DiskAnn";
    for (auto metric :
         {MetricType::kL2sq, MetricType::kInnerProduct, MetricType::kCosine}) {
      const char *distance = metric == MetricType::kL2sq           ? "L2"
                             : metric == MetricType::kInnerProduct ? "IP"
                                                                   : "Cosine";
      for (bool quantized : {false, true}) {
        // DiskANN stores native FP16; it has no FP16 quantization option.
        if (index == IndexType::kDiskAnn && quantized) continue;
        cases.push_back({index, metric, quantized,
                         std::string(kind) + "_" + distance +
                             (quantized ? "_QuantizedFp16" : "_NativeFp16")});
      }
    }
  }
  return cases;
}

class Fp16PlatformTest : public testing::TestWithParam<Case> {
 protected:
  static void SetUpTestSuite() {
    RecordProperty("fp16_diskann_supported", DISKANN_SUPPORTED);
    std::cout << "FP16 index coverage: DiskAnn=" << DISKANN_SUPPORTED << ';'
              << std::endl;
  }
  void SetUp() override {
    path_ = "fp16_platform_" + GetParam().name + ".index";
    test_util::RemoveTestFiles(path_);
    std::mt19937 random(20261030);
    // Exactly representable values give the oracle the actual stored values
    // without calling the production converter or distance implementation.
    auto fill = [&](std::vector<std::vector<float>> &rows, size_t count) {
      rows.assign(count, std::vector<float>(kDimension));
      for (auto &row : rows)
        for (auto &value : row)
          value = (static_cast<int>(random() % 1025) - 512) / 1024.0f;
    };
    fill(vectors_, kCount);
    fill(queries_, 3);
  }

  void TearDown() override {
    test_util::RemoveTestFiles(path_);
  }

  BaseIndexParam::Pointer param() const {
    BaseIndexParam::Pointer result;
    if (GetParam().index == IndexType::kFlat) {
      result = FlatIndexParamBuilder().build();
    } else if (GetParam().index == IndexType::kIVF) {
      result = IVFIndexParamBuilder().with_n_list(4).with_n_iters(5).build();
    } else {
      result = DiskAnnIndexParamBuilder()
                   .with_max_degree(24)
                   .with_list_size(128)
                   .with_pq_chunk_num(4)
                   .build();
    }
    result->dimension = kDimension;
    result->metric_type = GetParam().metric;
    result->data_type =
        GetParam().quantized ? DataType::DT_FP32 : DataType::DT_FP16;
    result->quantizer_param = std::make_shared<QuantizerParam>(
        GetParam().quantized ? QuantizerType::kFP16 : QuantizerType::kNone);
    return result;
  }

  double score(const std::vector<float> &a, const std::vector<float> &b) const {
    double dot = 0, square = 0, aa = 0, bb = 0;
    for (size_t d = 0; d < kDimension; ++d) {
      const double x = a[d], y = b[d];
      dot += x * y;
      square += (x - y) * (x - y);
      aa += x * x;
      bb += y * y;
    }
    if (GetParam().metric == MetricType::kL2sq) return square;
    if (GetParam().metric == MetricType::kInnerProduct) return dot;
    return 1 - dot / std::sqrt(aa * bb);
  }

  void check(Index *index) const {
    const bool ip = GetParam().metric == MetricType::kInnerProduct;
    for (bool filtered : {false, true, false}) {
      const std::array<uint64_t, 8> excluded = {{0, 3, 6, 9, 12, 15, 18, 21}};
      for (const auto &query : queries_) {
        std::vector<double> expected, ordered;
        for (const auto &row : vectors_) expected.push_back(score(row, query));
        for (uint32_t key = 0; key < kCount; ++key)
          if (!filtered || key % 3 != 0) ordered.push_back(expected[key]);
        std::sort(ordered.begin(), ordered.end(),
                  [ip](double a, double b) { return ip ? a > b : a < b; });
        const std::vector<ailego::Float16> native(query.begin(), query.end());
        const void *data = GetParam().quantized
                               ? static_cast<const void *>(query.data())
                               : static_cast<const void *>(native.data());
        for (bool linear : {true, false}) {
          BaseIndexQueryParam::Pointer params;
          if (GetParam().index == IndexType::kFlat) {
            params = FlatQueryParamBuilder().with_topk(kTopK).build();
          } else if (GetParam().index == IndexType::kIVF) {
            params =
                IVFQueryParamBuilder().with_topk(kTopK).with_nprobe(4).build();
          } else {
            auto diskann = std::make_shared<DiskAnnQueryParam>();
            diskann->topk = kTopK;
            diskann->list_size = 256;
            params = diskann;
          }
          params->is_linear = linear;
          // Match the DB's per-request filter lifecycle. The core may move
          // the predicate into its reused search context.
          if (filtered) {
            params->filter = std::make_shared<IndexFilter>();
            params->filter->set([excluded](uint64_t key) {
              return std::find(excluded.begin(), excluded.end(), key % 24) !=
                     excluded.end();
            });
          }
          SCOPED_TRACE(testing::Message()
                       << "linear=" << linear << " filtered=" << filtered);
          SearchResult result;
          ASSERT_EQ(
              0, index->search(VectorData{DenseVector{data}}, params, &result));
          ASSERT_EQ(kTopK, result.doc_list_.size());
          std::unordered_set<uint64_t> seen;
          size_t relevant = 0;
          for (size_t rank = 0; rank < result.doc_list_.size(); ++rank) {
            const auto &hit = result.doc_list_[rank];
            ASSERT_LT(hit.key(), kCount);
            EXPECT_TRUE(seen.insert(hit.key()).second);
            if (filtered) EXPECT_NE(0U, hit.key() % 3);
            ASSERT_TRUE(std::isfinite(hit.score()));
            double tolerance =
                GetParam().metric == MetricType::kCosine
                    ? 2e-3
                    : 1e-4 * std::max(1.0, std::abs(expected[hit.key()]));
            if (ip && !GetParam().quantized) {
              tolerance = std::max(tolerance, test_util::NativeFp16DotTolerance(
                                                  vectors_[hit.key()], query));
            }
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
          // All IVF buckets are probed. Only DiskANN's graph is approximate.
          const bool exact = linear || GetParam().index != IndexType::kDiskAnn;
          EXPECT_GE(relevant, exact ? kTopK : 9U);
        }
      }
    }
    if (GetParam().index == IndexType::kDiskAnn &&
        GetParam().metric == MetricType::kL2sq) {
      // Exercise the batch visit-set regression on the mobile SDK too.
      auto searcher = index->index_searcher();
      ASSERT_NE(nullptr, searcher);
      auto context = searcher->create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(kTopK);
      ailego::Params options;
      options.set("zvec.diskann.searcher.list_size", 256);
      ASSERT_EQ(0, context->update(options));
      std::vector<ailego::Float16> batch;
      for (const auto &query : queries_)
        batch.insert(batch.end(), query.begin(), query.end());
      const core::IndexQueryMeta meta(core::IndexMeta::DT_FP16, kDimension);
      for (bool linear : {false, true}) {
        ASSERT_EQ(0, linear ? searcher->search_bf_impl(batch.data(), meta,
                                                       queries_.size(), context)
                            : searcher->search_impl(batch.data(), meta,
                                                    queries_.size(), context));
        for (size_t q = 0; q < queries_.size(); ++q) {
          std::vector<double> expected, ordered;
          for (const auto &row : vectors_)
            expected.push_back(score(row, queries_[q]));
          ordered = expected;
          std::sort(ordered.begin(), ordered.end());
          const auto &hits = context->result(q);
          ASSERT_EQ(kTopK, hits.size());
          std::unordered_set<uint64_t> seen;
          size_t relevant = 0;
          for (const auto &hit : hits) {
            ASSERT_LT(hit.key(), kCount);
            EXPECT_TRUE(seen.insert(hit.key()).second);
            EXPECT_NEAR(expected[hit.key()], hit.score(),
                        1e-4 * std::max(1.0, expected[hit.key()]));
            relevant += expected[hit.key()] <= ordered[kTopK - 1] + 1e-6;
          }
          EXPECT_GE(relevant, linear ? kTopK : 9U);
        }
      }
    }
  }

  std::string path_;
  std::vector<std::vector<float>> vectors_, queries_;
};

TEST_P(Fp16PlatformTest, IndependentScoresAndRecallSurviveReopen) {
  auto index = IndexFactory::CreateAndInitIndex(*param());
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path_, {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t key = 0; key < kCount; ++key) {
    const std::vector<ailego::Float16> native(vectors_[key].begin(),
                                              vectors_[key].end());
    const void *data = GetParam().quantized
                           ? static_cast<const void *>(vectors_[key].data())
                           : static_cast<const void *>(native.data());
    ASSERT_EQ(0, index->add(VectorData{DenseVector{data}}, key));
  }
  ASSERT_EQ(0, index->train());
  check(index.get());
  ASSERT_EQ(0, index->close());
  index.reset();
  for (int reopen = 0; reopen < 2; ++reopen) {
    auto reader = IndexFactory::CreateAndInitIndex(*param());
    ASSERT_NE(nullptr, reader);
    ASSERT_EQ(0,
              reader->open(path_, {StorageOptions::StorageType::kMMAP, false}));
    check(reader.get());
    ASSERT_EQ(0, reader->close());
  }
}

INSTANTIATE_TEST_SUITE_P(IndexesFormatsAndMetrics, Fp16PlatformTest,
                         testing::ValuesIn(Cases()),
                         [](const testing::TestParamInfo<Case> &info) {
                           return info.param.name;
                         });

}  // namespace
}  // namespace zvec::core_interface

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

#include "../core/interface/pq_buffered_training_test_helper.h"

namespace zvec::core_interface {
namespace {
class PqBufferedTrainingAcceptanceTest : public PqBufferedTrainingTest {
 protected:
  template <core::IndexMeta::DataType Type, class T,
            class Quantizer = turbo::PqInt8Quantizer, size_t Dimension = 17>
  void compare(bool opq = false, size_t workspace_limit = 0) {
    constexpr size_t kDim = Dimension;
    // Identical rows eliminate random cluster initialization as a confounder.
    // A non-batch-aligned count also covers the untransposed tail.
    auto input = std::make_shared<core::MultiPassIndexHolder<Type>>(kDim);
    // Keep even the FP16 corpus larger than the workspace allowance below.
    // This still forces paging after reserving enough mandatory scratch for
    // two workers and 16 KiB-page platforms.
    const bool fast_opq =
        opq && std::is_same_v<Quantizer, turbo::PqFastQuantizer>;
    const bool wide_opq =
        opq && (fast_opq || std::is_same_v<Quantizer, turbo::PqInt8Quantizer>);
    // OPQ overlaps rotation tiles with two Lloyd workers' centroid scratch.
    // Keep the corpus larger than this mandatory workspace allowance.
    const size_t workspace_bytes =
        workspace_limit != 0 ? workspace_limit : (wide_opq ? 512 : 256) * 1024;
    const size_t row_count = fast_opq ? 20003 : 10001;
    ASSERT_GT(row_count * kDim * sizeof(T), workspace_bytes);
    for (size_t i = 0; i < row_count; ++i) {
      ailego::NumericalVector<T> row(kDim);
      for (size_t d = 0; d < kDim; ++d) row[d] = T(float(d + 1) / 32.0f);
      ASSERT_TRUE(input->emplace(i, std::move(row)));
    }
    for (const char *metric : {"SquaredEuclidean", "Cosine"}) {
      for (bool center : {false, true}) {
        SCOPED_TRACE(metric);
        SCOPED_TRACE(center);
        auto &pool = ailego::PoolPayloadBudget::get_instance();
        ASSERT_EQ(pool.init(0), 0);
        core::IndexMeta meta(Type, kDim);
        meta.set_metric(metric, 0, ailego::Params());
        ailego::Params params;
        params.set("num_chunk", 4u);
        params.set("thread_count", 2u);
        params.set("use_zero_mean", center);
        params.set("build_sdc_table", false);
        if (opq) {
          params.set("rotate_type", std::string("opq"));
          params.set("opq_iter", 2u);
          params.set("opq_pq_iter", 2u);
        }
        Quantizer legacy;
        ASSERT_EQ(legacy.init(meta, params), 0);
        ASSERT_EQ(legacy.train(input), 0);
        std::string reference;
        ASSERT_EQ(legacy.serialize(&reference), 0);
        const size_t metadata =
            ailego::PoolPayloadBudget::storage_metadata_bytes(4096);
        ASSERT_EQ(pool.init((opq ? 2 : 1) * metadata + workspace_bytes), 0);
        Quantizer paged;
        ASSERT_EQ(paged.init(meta, params), 0);
        ASSERT_EQ(paged.train(input), 0);
        // Training scratch is released; the fitted model/optional rotation
        // remain charged until the quantizer is destroyed (checked in
        // teardown).
        const size_t retained = opq ? kDim * kDim * sizeof(float) : 0;
        EXPECT_GT(pool.used(), retained);
        EXPECT_EQ(pool.used(), pool.external_used());
        EXPECT_LT(pool.used(), 128u * 1024);
        std::string actual;
        ASSERT_EQ(paged.serialize(&actual), 0);
        std::vector<uint8_t> a(legacy.quantized_datapoint_vector_length()),
            b(a.size());
        auto row = input->create_iterator();
        legacy.quantize_data(row->data(), a.data());
        paged.quantize_data(row->data(), b.data());
        if (!opq) {
          EXPECT_EQ(actual, reference);
          EXPECT_EQ(a, b);
        } else {
          // Identical rows make the Procrustes covariance rank-deficient:
          // the rotation on its nullspace is not unique. Tiled/dense QR and
          // SVD need not produce identical model bytes or centroid labels.
          // Check the observable reconstruction instead, including finiteness,
          // and require exact persistence of each fitted model independently.
          core::IndexQueryMeta qmeta(Type, kDim);
          auto check_reconstruction = [&](const Quantizer &quantizer,
                                          const std::vector<uint8_t> &code) {
            if constexpr (std::is_same_v<Quantizer, turbo::PqFastQuantizer>) {
              // FastScan only exposes packed-block distances, not vector
              // reconstruction. Identical training rows have zero
              // self-distance.
              std::vector<uint8_t> packed(
                  turbo::fast_scan_packed_block_size(4));
              ASSERT_EQ(0, quantizer.pack_codes(code.data(), 1, code.size(),
                                                packed.data()));
              std::vector<uint8_t> query(
                  quantizer.quantized_query_vector_length());
              quantizer.quantize_query(row->data(), query.data());
              float distance = std::numeric_limits<float>::quiet_NaN();
              quantizer.calc_distance_packed_block(packed.data(), 1,
                                                   query.data(), &distance);
              ASSERT_TRUE(std::isfinite(distance));
              EXPECT_NEAR(0.0f, distance, 1e-4f);
              return;
            }
            std::string decoded;
            ASSERT_EQ(0, quantizer.dequantize(code.data(), qmeta, &decoded));
            ASSERT_EQ(kDim * sizeof(float), decoded.size());
            for (size_t d = 0; d < kDim; ++d) {
              float value;
              std::memcpy(&value, decoded.data() + d * sizeof(float),
                          sizeof(value));
              ASSERT_TRUE(std::isfinite(value)) << d;
              EXPECT_NEAR(float(d + 1) / 32.0f, value, 1e-4f) << d;
            }
          };
          ASSERT_NO_FATAL_FAILURE(check_reconstruction(legacy, a));
          ASSERT_NO_FATAL_FAILURE(check_reconstruction(paged, b));
          Quantizer restored;
          ASSERT_EQ(0, restored.init(meta, params));
          ASSERT_EQ(0, restored.deserialize(actual));
          std::string roundtrip;
          ASSERT_EQ(0, restored.serialize(&roundtrip));
          EXPECT_EQ(actual, roundtrip);
          std::vector<uint8_t> restored_code(b.size());
          restored.quantize_data(row->data(), restored_code.data());
          EXPECT_EQ(b, restored_code);
          ASSERT_NO_FATAL_FAILURE(
              check_reconstruction(restored, restored_code));
        }
      }
    }
  }
};

TEST_F(PqBufferedTrainingAcceptanceTest, FP32MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP32, float>();
}
TEST_F(PqBufferedTrainingAcceptanceTest, FP16MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP16, ailego::Float16>();
}
TEST_F(PqBufferedTrainingAcceptanceTest, Int4FP32MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP32, float, turbo::PqInt4Quantizer, 16>();
}
TEST_F(PqBufferedTrainingAcceptanceTest, Int4FP16MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP16, ailego::Float16, turbo::PqInt4Quantizer,
          16>();
}
TEST_F(PqBufferedTrainingAcceptanceTest, FastFP32MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP32, float, turbo::PqFastQuantizer, 16>();
}
TEST_F(PqBufferedTrainingAcceptanceTest, FastFP16MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP16, ailego::Float16, turbo::PqFastQuantizer,
          16>();
}
TEST_F(PqBufferedTrainingAcceptanceTest, OpqFP32MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP32, float>(true);
}
TEST_F(PqBufferedTrainingAcceptanceTest, OpqFP32MatchesHeapTrainingWithReducedWorkspace) {
  // Keep both workers and every result check, with less workspace than the
  // existing 512 KiB case while the corpus still requires paging.
  compare<core::IndexMeta::DT_FP32, float>(true, 320 * 1024);
}
TEST_F(PqBufferedTrainingAcceptanceTest, OpqInt4MatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP32, float, turbo::PqInt4Quantizer, 16>(true);
}
TEST_F(PqBufferedTrainingAcceptanceTest, OpqFastMatchesHeapTrainingUnderPressure) {
  compare<core::IndexMeta::DT_FP32, float, turbo::PqFastQuantizer, 16>(true);
}
}  // namespace
}  // namespace zvec::core_interface

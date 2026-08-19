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

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>
#include "distance/neon/fp16/cosine.h"
#include "distance/neon/fp16/inner_product.h"
#include "distance/neon/fp16/squared_euclidean.h"
#include "distance/neon/fp32/cosine.h"
#include "distance/neon/fp32/inner_product.h"
#include "distance/neon/fp32/squared_euclidean.h"
#include "distance/scalar/fp16/cosine.h"
#include "distance/scalar/fp16/inner_product.h"
#include "distance/scalar/fp16/squared_euclidean.h"
#include "distance/scalar/fp32/cosine.h"
#include "distance/scalar/fp32/inner_product.h"
#include "distance/scalar/fp32/squared_euclidean.h"

namespace zvec::turbo {
namespace {

using RawDistanceFn = void (*)(const void *, const void *, size_t, float *);

void CompareDistance(RawDistanceFn expected_fn, RawDistanceFn actual_fn,
                     const void *a, const void *b, size_t dim,
                     float tolerance) {
  float expected = 0.0f;
  float actual = 0.0f;
  expected_fn(a, b, dim, &expected);
  actual_fn(a, b, dim, &actual);
  EXPECT_NEAR(expected, actual, tolerance) << "dim=" << dim;
}

TEST(NeonDistance, Fp32MatchesScalarForRemainders) {
#if !defined(__ARM_NEON) || !defined(__aarch64__)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  std::vector<float> a(33);
  std::vector<float> b(33);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(scalar::inner_product_fp32_distance,
                    neon::inner_product_fp32_distance, a.data(), b.data(), dim,
                    1e-5f);
    CompareDistance(scalar::squared_euclidean_fp32_distance,
                    neon::squared_euclidean_fp32_distance, a.data(), b.data(),
                    dim, 1e-5f);
    CompareDistance(scalar::cosine_fp32_distance, neon::cosine_fp32_distance,
                    a.data(), b.data(), dim, 1e-5f);
  }
#endif
}

TEST(NeonDistance, Fp16MatchesScalarForRemainders) {
#if !defined(__ARM_NEON) || !defined(__aarch64__)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  std::vector<float> a_fp32(33);
  std::vector<float> b_fp32(33);
  for (size_t i = 0; i < a_fp32.size(); ++i) {
    a_fp32[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b_fp32[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }
  std::vector<uint16_t> a(a_fp32.size());
  std::vector<uint16_t> b(b_fp32.size());
  ailego::FloatHelper::ToFP16(a_fp32.data(), a_fp32.size(), a.data());
  ailego::FloatHelper::ToFP16(b_fp32.data(), b_fp32.size(), b.data());

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(scalar::inner_product_fp16_distance,
                    neon::inner_product_fp16_distance, a.data(), b.data(), dim,
                    2e-3f);
    CompareDistance(scalar::squared_euclidean_fp16_distance,
                    neon::squared_euclidean_fp16_distance, a.data(), b.data(),
                    dim, 2e-3f);
    CompareDistance(scalar::cosine_fp16_distance, neon::cosine_fp16_distance,
                    a.data(), b.data(), dim, 2e-3f);
  }
#endif
}

TEST(NeonDistance, DispatchProvidesAllFpKernels) {
#if !defined(__ARM_NEON) || !defined(__aarch64__)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  constexpr std::array<MetricType, 3> metrics = {MetricType::kSquaredEuclidean,
                                                 MetricType::kCosine,
                                                 MetricType::kInnerProduct};
  for (MetricType metric : metrics) {
    const auto fp32 = get_distance_kernels(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kNEON);
    EXPECT_TRUE(fp32.dist);
    EXPECT_TRUE(fp32.batch);

    const auto fp16 = get_distance_kernels(
        metric, DataType::kFp16, QuantizeType::kFp16, CpuArchType::kNEON);
    EXPECT_TRUE(fp16.dist);
    EXPECT_TRUE(fp16.batch);
  }
#endif
}

}  // namespace
}  // namespace zvec::turbo

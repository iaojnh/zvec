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

#include <ailego/internal/cpu_features.h>
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__AVX2__)
float InnerProductAndSquaredNormAVX2(const int8_t *lhs, const int8_t *rhs,
                                     size_t size, float *sql, float *sqr);
#endif

#if defined(__SSE__)
float InnerProductAndSquaredNormSSE(const int8_t *lhs, const int8_t *rhs,
                                    size_t size, float *sql, float *sqr);
#endif

#if defined(__SSE4_1__)
//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    sum = InnerProductAndSquaredNormAVX2(p, q, dim, &u2, &v2);
  } else
#endif
  {
    sum = InnerProductAndSquaredNormSSE(p, q, dim, &u2, &v2);
  }

  *out = ComputeSphericalInjection(sum, u2, v2, e2);
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    sum = InnerProductAndSquaredNormAVX2(p, q, dim, &u2, &v2);
  } else
#endif
  {
    sum = InnerProductAndSquaredNormSSE(p, q, dim, &u2, &v2);
  }

  sum = e2 * (u2 + v2 - 2 * sum);
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    sum += (u2 - v2) * (u2 - v2);
    u2 = u2 * u2;
    v2 = v2 * v2;
  }
  *out = sum;
}
#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec
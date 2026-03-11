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

#if defined(__ARM_NEON)
float InnerProductAndSquaredNormNEON(const Float16 *lhs, const Float16 *rhs,
                                     size_t size, float *sql, float *sqr);
#endif

#if defined(__AVX512F__)
float InnerProductAndSquaredNormAVX512(const Float16 *lhs, const Float16 *rhs,
                                       size_t size, float *sql, float *sqr);
#endif

#if defined(__AVX__)
float InnerProductAndSquaredNormAVX(const Float16 *lhs, const Float16 *rhs,
                                    size_t size, float *sql, float *sqr);
#endif

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

#if defined(__ARM_NEON)
  sum = InnerProductAndSquaredNormNEON(p, q, dim, &u2, &v2);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    sum = InnerProductAndSquaredNormAVX512(p, q, dim, &u2, &v2);
  } else
#endif  //__AVX512F__
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
      sum = InnerProductAndSquaredNormAVX(p, q, dim, &u2, &v2);
    }
#endif  //__ARM_NEON

  *out = ComputeSphericalInjection(sum, u2, v2, e2);
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

#if defined(__ARM_NEON)
  sum = InnerProductAndSquaredNormNEON(p, q, dim, &u2, &v2);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    sum = InnerProductAndSquaredNormAVX512(p, q, dim, &u2, &v2);
  } else
#endif  //__AVX512F__
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
      sum = InnerProductAndSquaredNormAVX(p, q, dim, &u2, &v2);
    }
#endif  //__ARM_NEON

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

#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec
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
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__ARM_NEON)
void SquaredEuclideanDistanceNEON(const float *lhs, const float *rhs,
                                  size_t size, float *out);
#endif

#if defined(__AVX512F__)
float SquaredEuclideanDistanceAVX512(const float *lhs, const float *rhs,
                                     size_t size);
float EuclideanDistanceAVX512(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__AVX__)
float SquaredEuclideanDistanceAVX(const float *lhs, const float *rhs,
                                  size_t size);
float EuclideanDistanceAVX(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__SSE__)
float SquaredEuclideanDistanceSSE(const float *lhs, const float *rhs,
                                  size_t size);
float EuclideanDistanceSSE(const float *lhs, const float *rhs, size_t size);
#endif

//-----------------------------------------------------------
//  SquaredEuclideanDistance
//-----------------------------------------------------------
#if defined(__SSE__) || defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  SquaredEuclideanDistanceNEON(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = SquaredEuclideanDistanceAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = SquaredEuclideanDistanceAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = SquaredEuclideanDistanceSSE(m, q, dim);
#endif  // __ARM_NEON
}
#endif  // __SSE__ || __ARM_NEON


//-----------------------------------------------------------
//  EuclideanDistance
//-----------------------------------------------------------
#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void EuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
  SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(m, q, dim, out);
  *out = std::sqrt(*out);
}
#endif  // __SSE__ || __ARM_NEON && __aarch64__

}  // namespace ailego
}  // namespace zvec
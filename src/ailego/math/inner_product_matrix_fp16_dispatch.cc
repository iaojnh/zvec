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
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__ARM_NEON)
float InnerProductNEON(const Float16 *lhs, const Float16 *rhs, size_t size);
float MinusInnerProductNEON(const Float16 *lhs, const Float16 *rhs,
                            size_t size);
#endif

#if defined(__AVX__)
void InnerProductAVX(const Float16 *lhs, const Float16 *rhs, size_t size,
                     float *out);
void MinusInnerProductAVX(const Float16 *lhs, const Float16 *rhs, size_t size,
                          float *out);
float InnerProductSparseInSegmentAVX(uint32_t m_sparse_count,
                                     const uint16_t *m_sparse_index,
                                     const Float16 *m_sparse_value,
                                     uint32_t q_sparse_count,
                                     const uint16_t *q_sparse_index,
                                     const Float16 *q_sparse_value);
#endif

#if defined(__AVX512F__)
void InnerProductAVX512(const Float16 *lhs, const Float16 *rhs, size_t size,
                        float *out);
void MinusInnerProductAVX512(const Float16 *lhs, const Float16 *rhs,
                             size_t size, float *out);
#endif

#if defined(__AVX512FP16__)
float InnerProductAVX512FP16(const Float16 *lhs, const Float16 *rhs,
                             size_t size);
float InnerProductSparseInSegmentAVX512FP16(uint32_t m_sparse_count,
                                            const uint16_t *m_sparse_index,
                                            const Float16 *m_sparse_value,
                                            uint32_t q_sparse_count,
                                            const uint16_t *q_sparse_index,
                                            const Float16 *q_sparse_value);
#endif

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the distance between matrix and query (FP16, M=1, N=1)
void InnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON(m, q, dim);
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = InnerProductAVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512(m, q, dim, out);
    return;
  }
#endif  //__AVX512F__
  InnerProductAVX(m, q, dim, out);
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void MinusInnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON(m, q, dim);
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = -InnerProductAVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512(m, q, dim, out);
    return;
  }
#endif  //__AVX512F__

  MinusInnerProductAVX(m, q, dim, out);

#endif  //__ARM_NEON
}

#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

// sparse
float InnerProductSparseInSegment(uint32_t m_sparse_count,
                                  const uint16_t *m_sparse_index,
                                  const Float16 *m_sparse_value,
                                  uint32_t q_sparse_count,
                                  const uint16_t *q_sparse_index,
                                  const Float16 *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;
  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      sum += m_sparse_value[m_i] * q_sparse_value[q_i];

      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      ++m_i;
    } else {
      ++q_i;
    }
  }

  return sum;
}

template <>
float MinusInnerProductSparseMatrix<Float16>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value) {
#if defined(__AVX512FP16__)
  return InnerProductSparseInSegmentAVX512FP16(m_sparse_count, m_sparse_index,
                                               m_sparse_value, q_sparse_count,
                                               q_sparse_index, q_sparse_value);
#elif defined(__AVX__)
  return InnerProductSparseInSegmentAVX(m_sparse_count, m_sparse_index,
                                        m_sparse_value, q_sparse_count,
                                        q_sparse_index, q_sparse_value);

#else
  return InnerProductSparseInSegment(m_sparse_count, m_sparse_index,
                                     m_sparse_value, q_sparse_count,
                                     q_sparse_index, q_sparse_value);
#endif
}

}  // namespace ailego
}  // namespace zvec
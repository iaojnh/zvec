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

// Shared AVX2 inner product kernel for record_quantized_int8 distance
// implementations (inner_product, squared_euclidean, cosine).

#pragma once

#if defined(__AVX2__)
#include <immintrin.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>

namespace zvec::turbo::avx2::internal {

// For 16..64 signed codes, every partial sum and the final dot product fit
// exactly in FP32. Keep the distance kernels' floating score epilogues intact.
template <size_t Batch>
inline void raw_inner_product_low_dim_batch(
    const int8_t *query, const void *const *records,
    std::array<const int8_t *, Batch> &prefetch_ptrs, size_t dim,
    float *results) {
  __m256i accs[Batch];
  for (size_t i = 0; i < Batch; ++i) {
    accs[i] = _mm256_setzero_si256();
    if (prefetch_ptrs[i] != nullptr) {
      ailego_prefetch(prefetch_ptrs[i]);
      ailego_prefetch(prefetch_ptrs[i] + dim - 1);
    }
  }
  auto accumulate = [&](const __m128i query_bytes, size_t offset) {
    const __m256i q = _mm256_cvtepi8_epi16(query_bytes);
    for (size_t i = 0; i < Batch; ++i) {
      const auto *record = static_cast<const int8_t *>(records[i]);
      const __m256i data = _mm256_cvtepi8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(record + offset)));
      accs[i] = _mm256_add_epi32(accs[i], _mm256_madd_epi16(q, data));
    }
  };
  size_t offset = 0;
  for (; offset + 16 <= dim; offset += 16) {
    accumulate(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(query + offset)),
        offset);
  }
  if (offset < dim) {
    // Reload only valid bytes; zero the already accumulated query prefix.
    // Bare vectors ending at a page boundary need no padding or metadata.
    const __m128i lanes =
        _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i mask = _mm_cmpgt_epi8(
        lanes, _mm_set1_epi8(static_cast<char>(15 - (dim - offset))));
    const __m128i q = _mm_and_si128(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(query + dim - 16)),
        mask);
    accumulate(q, dim - 16);
  }
  for (size_t i = 0; i < Batch; ++i) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(accs[i]),
                                _mm256_extracti128_si256(accs[i], 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    results[i] = static_cast<float>(_mm_cvtsi128_si32(sum));
  }
}

// Raw integer inner product of two int8 code arrays of length `dim`.
inline float raw_inner_product(const int8_t *a, const int8_t *b, size_t dim) {
  if (dim >= 16 && dim <= 64) {
    const void *records[] = {a};
    std::array<const int8_t *, 1> prefetch_ptrs{};
    float result;
    raw_inner_product_low_dim_batch(b, records, prefetch_ptrs, dim, &result);
    return result;
  }
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i accumulator = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m128i lhs_bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    const __m128i rhs_bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    const __m256i lhs = _mm256_cvtepi8_epi16(lhs_bytes);
    const __m256i rhs = _mm256_cvtepi8_epi16(rhs_bytes);
    const __m256i products = _mm256_mullo_epi16(lhs, rhs);
    accumulator =
        _mm256_add_epi32(accumulator, _mm256_madd_epi16(products, ones));
  }

  alignas(32) int32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), accumulator);
  int64_t sum = 0;
  for (int32_t lane : lanes) {
    sum += lane;
  }
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
}

// One-to-many raw inner product kernel: the query block is loaded and
// widened to int16 once per iteration and reused across all `dp_batch`
// records.
template <size_t dp_batch>
inline void raw_inner_product_batch_impl(
    const int8_t *query, const void *const *records,
    std::array<const int8_t *, dp_batch> &prefetch_ptrs, size_t dim,
    float *results) {
  if (dim >= 16 && dim <= 64) {
    return raw_inner_product_low_dim_batch(query, records, prefetch_ptrs, dim,
                                           results);
  }
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_si256();
  }

  size_t d = 0;
  for (; d + 16 <= dim; d += 16) {
    const __m256i rhs = _mm256_cvtepi8_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(query + d)));

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + d);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      const __m256i lhs = _mm256_cvtepi8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(
              static_cast<const int8_t *>(records[i]) + d)));
      const __m256i products = _mm256_mullo_epi16(lhs, rhs);
      accs[i] = _mm256_add_epi32(accs[i], _mm256_madd_epi16(products, ones));
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    const int8_t *record = static_cast<const int8_t *>(records[i]);
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), accs[i]);
    int64_t sum = 0;
    for (int32_t lane : lanes) {
      sum += lane;
    }
    for (size_t j = d; j < dim; ++j) {
      sum += static_cast<int32_t>(record[j]) * static_cast<int32_t>(query[j]);
    }
    results[i] = static_cast<float>(sum);
  }
}

// Dispatch batched raw inner products over all `n` records with prefetching.
// The `int8_t` query type keeps this overload distinct from the int4 one.
inline void raw_inner_product_batch(const void *const *vectors,
                                    const int8_t *query, size_t n, size_t dim,
                                    float *results) {
  if (dim >= 16 && dim <= 64) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      std::array<const int8_t *, 4> prefetch_ptrs{};
      for (size_t j = 0; j < 4; ++j) {
        // Keep the legacy lookahead of four records, independent of batching.
        if (i + j + 4 < n) {
          prefetch_ptrs[j] = static_cast<const int8_t *>(vectors[i + j + 4]);
        }
      }
      raw_inner_product_batch_impl<4>(query, vectors + i, prefetch_ptrs, dim,
                                      results + i);
    }
    if (i + 2 <= n) {
      std::array<const int8_t *, 2> prefetch_ptrs{};
      raw_inner_product_batch_impl<2>(query, vectors + i, prefetch_ptrs, dim,
                                      results + i);
      i += 2;
    }
    if (i < n) {
      std::array<const int8_t *, 1> prefetch_ptrs{};
      raw_inner_product_batch_impl<1>(query, vectors + i, prefetch_ptrs, dim,
                                      results + i);
    }
    return;
  }
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const int8_t *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = static_cast<const int8_t *>(
            vectors[i + j + batch_size * prefetch_step]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    raw_inner_product_batch_impl<batch_size>(query, &vectors[i], prefetch_ptrs,
                                             dim, results + i);
  }
  for (; i < n; i++) {
    std::array<const int8_t *, 1> prefetch_ptrs{nullptr};
    raw_inner_product_batch_impl<1>(query, &vectors[i], prefetch_ptrs, dim,
                                    results + i);
  }
}

}  // namespace zvec::turbo::avx2::internal

#endif  // defined(__AVX2__)

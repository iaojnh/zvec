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

#pragma once
#include <limits>
#include <type_traits>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include "turbo/quantizer/quantizer.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

struct FlatMatrixScratch {
  template <typename T>
  auto &values() {
    if constexpr (std::is_same_v<T, float>) {
      return fp32;
    } else {
      static_assert(std::is_same_v<T, ailego::Float16>);
      return fp16;
    }
  }

  ailego::BudgetedVector<float> fp32;
  ailego::BudgetedVector<ailego::Float16> fp16;
  ailego::BudgetedVector<const void *> pointers;
};

// Flat stores complete blocks in column order. Quantizers consume encoded
// row vectors; preserve every stored component, including any cosine norm.
// Queries have already been converted by IndexFlow or the IVF reformer.
template <typename T>
inline void QuantizedFlatMatrixDistance(const turbo::Quantizer &quantizer,
                                        const void *m, const void *q,
                                        size_t rows_count, size_t queries_count,
                                        float *out,
                                        FlatMatrixScratch *scratch = nullptr) {
  if (rows_count == 0 || queries_count == 0) return;
  if (rows_count == 1 && queries_count == 1) {
    *out = quantizer.calc_distance_dp_dp(m, q);
    return;
  }
  const size_t dim = quantizer.quantized_datapoint_vector_length() / sizeof(T);
  FlatMatrixScratch local;
  if (!scratch) scratch = &local;
  auto &values = scratch->template values<T>();
  auto &pointers = scratch->pointers;
  if (rows_count > static_cast<size_t>((std::numeric_limits<int>::max)()) ||
      dim > values.max_size() / (rows_count + 1) ||
      rows_count > pointers.max_size()) {
    throw std::bad_alloc();
  }
  values.resize((rows_count + 1) * dim);
  pointers.resize(rows_count);
  auto *rows = values.data();
  auto *query = rows + rows_count * dim;
  const auto *a = static_cast<const T *>(m);
  const auto *b = static_cast<const T *>(q);
  for (size_t i = 0; i < rows_count; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      rows[i * dim + d] = a[d * rows_count + i];
    }
    pointers[i] = rows + i * dim;
  }
  for (size_t j = 0; j < queries_count; ++j) {
    for (size_t d = 0; d < dim; ++d) {
      query[d] = b[d * queries_count + j];
    }
    quantizer.calc_distance_dp_query_batch(pointers.data(),
                                           static_cast<int>(rows_count), query,
                                           out + j * rows_count);
  }
}

/*! Brute Force Distance Tuple
 */
template <size_t K, typename = void>
class FlatDistanceTuple;

/*! Brute Force Distance Tuple
 */
template <>
class FlatDistanceTuple<1> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return !!distance_;
  }

  //! Retrieve non-zero if a distance is valid.
  bool is_valid(size_t m) const {
    return m == 1 && !!distance_;
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    distance_ = measure.distance_matrix(1, 1);
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure, size_t m) {
    distance_ = measure.distance_matrix(m, 1);
  }

  //! Compute the distance between matrix and query
  template <size_t M>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<M == 1>::type {
    distance_(m, q, dim, out);
  }

 private:
  IndexMetric::MatrixDistance distance_{};
};

/*! Brute Force Distance Tuple
 */
template <size_t K>
class FlatDistanceTuple<
    K, typename std::enable_if<IsEqualPowerofTwo<K>::value>::type> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return (distance_tuple_.is_valid() && !!distance_);
  }

  //! Retrieve non-zero if a distance is valid.
  bool is_valid(size_t m) const {
    return (m == K ? (!!distance_)
                   : (m < K ? distance_tuple_.is_valid(m) : false));
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    distance_tuple_.initialize(measure);
    distance_ = measure.distance_matrix(K, 1);
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure, size_t m) {
    distance_tuple_.initialize(measure, m);
    distance_ = measure.distance_matrix(m, K);
  }

  //! Compute the distance between matrix and query
  template <size_t M>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<K == M>::type {
    distance_(m, q, dim, out);
  }

  //! Compute the distance between matrix and query
  template <size_t M>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<(K > M) && IsEqualPowerofTwo<M>::value>::type {
    distance_tuple_.template distance<M>(m, q, dim, out);
  }

 private:
  FlatDistanceTuple<(K >> 1)> distance_tuple_{};
  IndexMetric::MatrixDistance distance_{};
};

/*! Brute Force Distance Matrix
 */
template <size_t K, typename = void>
class FlatDistanceMatrix;

/*! Brute Force Distance Matrix
 */
template <>
class FlatDistanceMatrix<1> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return quantizer_ || !!distance_;
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    quantizer_.reset();
    distance_ = measure.distance_matrix(1, 1);
  }

  void initialize(turbo::Quantizer::Pointer quantizer) {
    quantizer_ = std::move(quantizer);
  }

  //! Compute the distance between matrix and query
  template <size_t M, size_t N = 1u>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<M == 1u && N == 1u>::type {
    if (quantizer_) {
      *out = quantizer_->calc_distance_dp_dp(m, q);
    } else {
      distance_(m, q, dim, out);
    }
  }

 private:
  IndexMetric::MatrixDistance distance_{};
  turbo::Quantizer::Pointer quantizer_{};
};

/*! Brute Force Distance Matrix
 */
template <size_t K>
class FlatDistanceMatrix<
    K, typename std::enable_if<IsEqualPowerofTwo<K>::value>::type> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return quantizer_ || (tuple_h_.is_valid() && tuple_v_.is_valid());
  }

  //! Retrieve non-zero if a distance is valid.
  bool is_valid(size_t m, size_t n) const {
    if (quantizer_) {
      return m > 0 && n > 0 && m <= K && n <= K && (m & (m - 1)) == 0 &&
             (n & (n - 1)) == 0 && (m == K || n == 1);
    }
    return (m == K ? tuple_h_.is_valid(n)
                   : (m < K && n == 1 ? tuple_v_.is_valid(m) : false));
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    quantizer_.reset();
    tuple_h_.initialize(measure, K);
    tuple_v_.initialize(measure);
  }

  void initialize(turbo::Quantizer::Pointer quantizer) {
    quantizer_ = std::move(quantizer);
  }

  //! Compute the distance between matrix and query
  template <size_t M, size_t N>
  auto distance(const void *m, const void *q, size_t dim, float *out,
                FlatMatrixScratch *scratch = nullptr) const ->
      typename std::enable_if<(K == M) && (K >= N)>::type {
    if (quantizer_) {
      if (quantizer_->meta().data_type() == IndexMeta::DT_FP16) {
        QuantizedFlatMatrixDistance<ailego::Float16>(*quantizer_, m, q, M, N,
                                                     out, scratch);
      } else {
        QuantizedFlatMatrixDistance<float>(*quantizer_, m, q, M, N, out,
                                           scratch);
      }
      return;
    }
    tuple_h_.template distance<N>(m, q, dim, out);
  }

  //! Compute the distance between matrix and query
  template <size_t M, size_t N = 1u>
  auto distance(const void *m, const void *q, size_t dim, float *out,
                FlatMatrixScratch *scratch = nullptr) const ->
      typename std::enable_if<(K > M) && (N == 1u)>::type {
    if (quantizer_) {
      if (quantizer_->meta().data_type() == IndexMeta::DT_FP16) {
        QuantizedFlatMatrixDistance<ailego::Float16>(*quantizer_, m, q, M, N,
                                                     out, scratch);
      } else {
        QuantizedFlatMatrixDistance<float>(*quantizer_, m, q, M, N, out,
                                           scratch);
      }
      return;
    }
    tuple_v_.template distance<M>(m, q, dim, out);
  }

 private:
  turbo::Quantizer::Pointer quantizer_{};
  FlatDistanceTuple<K> tuple_h_{};
  FlatDistanceTuple<(K >> 1)> tuple_v_{};
};

}  // namespace core
}  // namespace zvec

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

#include "quantizer/quantizer.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>
#include <zvec/ailego/buffer/budgeted_allocator.h>

namespace zvec::turbo {

namespace {

using ScratchBytes = std::vector<uint8_t, ailego::BudgetedAllocator<uint8_t>>;
using ScratchPointers =
    std::vector<const void *, ailego::BudgetedAllocator<const void *>>;

// Keep common single-vector distances allocation-free. Larger dimensions
// use charged storage, released at the end of the call rather than in TLS.
class ScratchVector {
 public:
  explicit ScratchVector(size_t bytes) {
    if (bytes > inline_.size()) overflow_.resize(bytes);
  }
  uint8_t *data() {
    return overflow_.empty() ? inline_.data() : overflow_.data();
  }

 private:
  alignas(std::max_align_t) std::array<uint8_t, 4096> inline_;
  ScratchBytes overflow_;
};

void InputDistanceBatch(const Quantizer &quantizer, const void *const *dp_list,
                        int dp_num, const void *query, float *dist_list) {
  const size_t length = quantizer.quantized_datapoint_vector_length();
  if (length == 0 ||
      length > std::numeric_limits<size_t>::max() - sizeof(void *))
    throw std::bad_alloc();
  // Only one bounded batch is live. Never retain the largest request in TLS.
  size_t batch = std::min(
      static_cast<size_t>(dp_num),
      std::max(size_t{1}, size_t{64 * 1024} / (length + sizeof(void *))));
  ScratchBytes data;
  ScratchPointers pointers;
  for (;;) {
    try {
      ScratchBytes next_data(batch * length);
      ScratchPointers next_pointers(batch);
      data.swap(next_data);
      pointers.swap(next_pointers);
      break;
    } catch (const std::bad_alloc &) {
      if (batch == 1) throw;
      batch = std::max(size_t{1}, batch / 2);
    }
  }
  for (size_t first = 0; first < static_cast<size_t>(dp_num); first += batch) {
    const size_t count = std::min(batch, static_cast<size_t>(dp_num) - first);
    for (size_t i = 0; i < count; ++i) {
      void *output = data.data() + length * i;
      quantizer.quantize_data(dp_list[first + i], output);
      pointers[i] = output;
    }
    quantizer.calc_distance_dp_query_batch(
        pointers.data(), static_cast<int>(count), query, dist_list + first);
  }
}

}  // namespace

float Quantizer::calc_distance_input_query(const void *dp,
                                           const void *query) const {
  ScratchVector datapoint(quantized_datapoint_vector_length());
  quantize_data(dp, datapoint.data());
  return calc_distance_dp_query(datapoint.data(), query);
}

void Quantizer::calc_distance_input_query_batch(const void *const *dp_list,
                                                int dp_num, const void *query,
                                                float *dist_list) const {
  if (dp_num <= 0) {
    return;
  }
  InputDistanceBatch(*this, dp_list, dp_num, query, dist_list);
}

float Quantizer::calc_distance_input_input(const void *dp1,
                                           const void *dp2) const {
  ScratchVector datapoint(quantized_datapoint_vector_length());
  ScratchVector query(quantized_query_vector_length());
  quantize_data(dp1, datapoint.data());
  quantize_query(dp2, query.data());
  return calc_distance_dp_query(datapoint.data(), query.data());
}

void Quantizer::calc_distance_input_input_batch(const void *const *dp_list,
                                                int dp_num, const void *query,
                                                float *dist_list) const {
  if (dp_num <= 0) {
    return;
  }
  ScratchVector encoded_query(quantized_query_vector_length());
  quantize_query(query, encoded_query.data());
  InputDistanceBatch(*this, dp_list, dp_num, encoded_query.data(), dist_list);
}

}  // namespace zvec::turbo

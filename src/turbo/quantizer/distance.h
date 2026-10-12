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

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace turbo {

//! A callable distance handle bound to a quantized query vector.
//!
//! DistanceImpl owns the quantized query bytes and a dispatched
//! DistanceFunc. Invoking `operator()(candidate)` computes the distance
//! between the stored query and the given candidate vector, which is
//! expected to already be in the same quantized layout.
class DistanceImpl {
 public:
  using QueryBuffer = std::vector<uint8_t, ailego::BudgetedAllocator<uint8_t>>;
  DistanceImpl() = default;

  DistanceImpl(DistanceFunc func, BatchDistanceFunc batch_func,
               std::shared_ptr<QueryBuffer> query, size_t dim)
      : func_(std::move(func)),
        batch_func_(std::move(batch_func)),
        owned_query_(std::move(query)),
        dim_(dim) {}

  // Reuse a private workspace; a copied handle must keep its old query intact.
  std::shared_ptr<QueryBuffer> take_query_buffer() {
    if (owned_query_ && owned_query_.use_count() == 1)
      return std::move(owned_query_);
    return std::make_shared<QueryBuffer>();
  }
  const void *query_data() const {
    return owned_query_ ? static_cast<const void *>(owned_query_->data())
                        : static_cast<const void *>(query_storage_.data());
  }
  size_t query_bytes() const {
    return owned_query_ ? owned_query_->size() : query_storage_.size();
  }

  DistanceImpl(DistanceFunc func, std::string quantized_query, size_t dim)
      : func_(std::move(func)),
        query_storage_(std::move(quantized_query)),
        dim_(dim) {}

  DistanceImpl(DistanceFunc func, BatchDistanceFunc batch_func,
               std::string quantized_query, size_t dim)
      : func_(std::move(func)),
        batch_func_(std::move(batch_func)),
        query_storage_(std::move(quantized_query)),
        dim_(dim) {}

  //! Whether the handle is ready to compute distances.
  bool valid() const {
    return static_cast<bool>(func_);
  }

  //! Whether a batch distance function is available.
  bool batch_valid() const {
    return static_cast<bool>(batch_func_);
  }

  //! Compute the distance between the stored query and `candidate`.
  float operator()(const void *candidate) const {
    float d = 0.0f;
    func_(candidate, query_data(), dim_, &d);
    return d;
  }

  //! Compute distances for a batch of `num` candidates against the
  //! stored query. Falls back to the scalar path when no batch function
  //! is bound.
  void batch(const void **candidates, size_t num, float *out) const {
    if (batch_func_) {
      batch_func_(candidates, query_data(), num, dim_, out, nullptr);
      return;
    }
    for (size_t i = 0; i < num; ++i) {
      out[i] = 0.0f;
      func_(candidates[i], query_data(), dim_, out + i);
    }
  }

  //! Access the quantized query bytes (for pairwise helpers).
  std::string_view query_storage() const {
    return {static_cast<const char *>(query_data()), query_bytes()};
  }

  size_t dim() const {
    return dim_;
  }

  //! Raw scalar distance function (operates on already-quantized
  //! candidates). Useful for pairwise node-vs-node distance where no
  //! stored query is involved.
  const DistanceFunc &func() const {
    return func_;
  }

  //! Raw batch distance function.
  const BatchDistanceFunc &batch_func() const {
    return batch_func_;
  }

 private:
  DistanceFunc func_{};
  BatchDistanceFunc batch_func_{};
  std::string query_storage_{};
  std::shared_ptr<QueryBuffer> owned_query_;
  size_t dim_{0};
};

}  // namespace turbo
}  // namespace zvec

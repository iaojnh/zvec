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

#include "fht_rotator.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace core {

namespace {

//! Largest power of 2 <= n (e.g. floor_pow2(97) = 64, floor_pow2(128) = 128).
size_t floor_pow2(size_t n) {
  if (n == 0) return 0;
  size_t p = 1;
  while (p <= n / 2) p *= 2;
  return p;
}

}  // anonymous namespace

// ============================================================================
// FhtRotator method implementations
// ============================================================================

void FhtRotator::init_context(size_t dim) {
  const size_t flip_offset = dim / kByteLen + (dim % kByteLen != 0);
  if (flip_offset >
      (std::numeric_limits<size_t>::max() - sizeof(turbo::FhtCtx)) / 4)
    throw std::bad_alloc();
  // Admit replacement while the old context is still live. Turbo's kernels
  // use caller-owned output as scratch, so no unbudgeted inverse copy remains.
  ailego::BudgetedVector<uint8_t> replacement(sizeof(turbo::FhtCtx) +
                                              4 * flip_offset);
  auto *ctx = new (replacement.data()) turbo::FhtCtx{};
  ctx->flip_offset = flip_offset;
  ctx->trunc_dim = floor_pow2(dim);
  ctx->fac = 1.0f / std::sqrt(static_cast<float>(ctx->trunc_dim));
  context_.swap(replacement);
  kernels_ = turbo::get_rotator_kernels(turbo::RotateType::kFht);
}

int FhtRotator::init_impl(size_t dim) {
  if (dim == 0) {
    return IndexError_InvalidArgument;
  }
  init_context(dim);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < blob_bytes(); ++i) {
    context()->flip[i] = static_cast<uint8_t>(dist(gen));
  }
  return 0;
}

void FhtRotator::rotate(const float *in, float *out) const {
  kernels_.rotate(in, out, dimension_, dimension_,
                   const_cast<turbo::FhtCtx *>(context()));
}

void FhtRotator::unrotate(const float *in, float *out) const {
  kernels_.unrotate(in, out, dimension_, dimension_,
                     const_cast<turbo::FhtCtx *>(context()));
}

RotatorType FhtRotator::rotator_type() const {
  return RotatorType::FhtKac;
}

void FhtRotator::save_blob(char *data) const {
  std::memcpy(data, context()->flip, blob_bytes());
}

void FhtRotator::load_blob(const char *data) {
  init_context(dimension_);
  std::memcpy(context()->flip, data, blob_bytes());
}

size_t FhtRotator::blob_bytes() const {
  return context_.empty() ? 0 : 4 * context()->flip_offset;
}

}  // namespace core
}  // namespace zvec

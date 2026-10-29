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

#include <algorithm>
#include <cmath>
#include <vector>

namespace zvec::test_util {

inline double NativeFp16DotTolerance(const std::vector<float> &a,
                                     const std::vector<float> &b) {
  // Legacy NEON/AVX512-FP16 inner products accumulate in half precision.
  // Bound rounding by gamma_n * sum(|a_i*b_i|), not by |dot(a,b)|:
  // cancellation can make the latter arbitrarily small. Eight NEON lanes
  // give an upper bound of ceil(dimension/8) half FMAs per lane; the final
  // reduction is FP32. Include half subnormal and FP32 reduction errors.
  constexpr double unit_roundoff = 1.0 / 2048;
  const double steps = (a.size() + 7) / 8;
  const double gamma = steps * unit_roundoff / (1 - steps * unit_roundoff);
  double magnitude = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    magnitude += std::abs(static_cast<double>(a[i]) * b[i]);
  }
  return gamma * magnitude + a.size() * std::ldexp(1.0, -25) +
         1e-6 * std::max(1.0, magnitude);
}

}  // namespace zvec::test_util

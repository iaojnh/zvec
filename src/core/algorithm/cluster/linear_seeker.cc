
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
#include "linear_seeker.h"

namespace zvec {
namespace core {

int LinearSeeker::seek(const void *query, size_t len, Document *out) {
  return seek(query, len, out, nullptr, nullptr);
}

int LinearSeeker::seek(const void *query, size_t len, Document *out,
                       const IndexCluster::CentroidList *stored_centroids,
                       ailego::BudgetedAlignedBuffer *scratch) {
  if (ailego_unlikely(!query || !out || meta_.element_size() != len)) {
    return IndexError_InvalidArgument;
  }

  float sel_score = std::numeric_limits<float>::max();
  uint32_t sel_column = 0;
  uint32_t total = static_cast<uint32_t>(features_->count());
  constexpr uint32_t batch_size = 32;
  if (stored_centroids) {
    const size_t rows = quantizer_ ? std::min(batch_size, total) : 1;
    if (rows && len > std::numeric_limits<size_t>::max() / rows)
      return IndexError_Overflow;
    scratch->resize(rows * len, 64);
  }

  auto read_feature = [&](uint32_t id, size_t slot, const void **ptr) -> int {
    if (!stored_centroids) {
      *ptr = features_->element(id);
      return 0;
    }
    auto *destination = static_cast<char *>(scratch->data()) + slot * len;
    int ret = (*stored_centroids)[id].read_feature(destination, len);
    if (ret == 0) *ptr = destination;
    return ret;
  };

  if (quantizer_) {
    const void *vectors[batch_size];
    float scores[batch_size];
    for (uint32_t base = 0; base < total;) {
      uint32_t count = std::min(batch_size, total - base);
      for (uint32_t j = 0; j < count; ++j) {
        int ret = read_feature(base + j, j, &vectors[j]);
        if (ret != 0) return ret;
      }
      quantizer_->calc_distance_dp_query_batch(vectors, count, query, scores);
      for (uint32_t j = 0; j < count; ++j) {
        if (scores[j] < sel_score) {
          sel_score = scores[j];
          sel_column = base + j;
        }
      }
      base += count;
    }
  } else {
    for (uint32_t i = 0; i < total; ++i) {
      float score = 0.0f;
      const void *feature = nullptr;
      int ret = read_feature(i, 0, &feature);
      if (ret != 0) return ret;
      distance_func_(feature, query, meta_.dimension(), &score);
      if (score < sel_score) {
        sel_score = score;
        sel_column = i;
      }
    }
  }

  out->index = sel_column;
  out->score = sel_score;
  return 0;
}

}  // namespace core
}  // namespace zvec

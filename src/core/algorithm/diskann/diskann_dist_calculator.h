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

#include <array>
#include <memory>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_meta.h>
#include "diskann_entity.h"
#include "diskann_visit_filter.h"

namespace zvec {
namespace core {

class DistCalculator {
 public:
  typedef std::shared_ptr<DistCalculator> Pointer;

 public:
  //! Constructor
  DistCalculator(const DiskAnnEntity *entity, const IndexMeta &meta,
                 const IndexMetric::Pointer &measure,
                 const turbo::Quantizer::Pointer &data_quantizer = nullptr)
      : entity_(entity),
        query_(nullptr),
        dim_(meta.dimension()),
        compare_cnt_(0) {
    bind_distance(meta, measure, data_quantizer);
  }

  void update(const DiskAnnEntity *entity, const IndexMeta &meta,
              const IndexMetric::Pointer &measure,
              const turbo::Quantizer::Pointer &data_quantizer = nullptr) {
    entity_ = entity;
    bind_distance(meta, measure, data_quantizer);
    dim_ = meta.dimension();
  }

  inline void update_distance(const IndexMetric::MatrixDistance &distance) {
    distance_ = distance;
  }

  //! Reset query vector data
  inline void reset_query(const void *query) {
    error_code_ = 0;
    query_ = query;
  }

  //! Returns distance
  inline dist_t dist(const void *vec_lhs, const void *vec_rhs) {
    if (ailego_unlikely(vec_lhs == nullptr || vec_rhs == nullptr)) {
      LOG_ERROR("Nullptr of dense vector");

      error_code_ = IndexError_ReadData;
      return 0.0f;
    }

    float score{0.0f};
    distance_(vec_lhs, vec_rhs, dim_, &score);

    return score;
  }

  //! Returns distance between query and vec.
  inline dist_t dist(const void *vec) {
    compare_cnt_++;

    return dist(vec, query_);
  }

  inline dist_t dist(diskann_id_t id) {
    compare_cnt_++;

    IndexStorage::MemoryBlock block;
    const int ret = entity_->read_vector(id, block);
    if (ailego_unlikely(ret != 0 || block.data() == nullptr)) {
      LOG_ERROR("Read vector failed, id=%u, ret=%d", id, ret);
      error_code_ = ret != 0 ? ret : IndexError_ReadData;
      return 0.0f;
    }

    return dist(block.data(), query_);
  }

  inline dist_t dist(diskann_id_t lhs, diskann_id_t rhs) {
    compare_cnt_++;

    IndexStorage::MemoryBlock lhs_block;
    int ret = entity_->read_vector(lhs, lhs_block);
    if (ailego_unlikely(ret != 0 || lhs_block.data() == nullptr)) {
      LOG_ERROR("Read vector failed, lhs id=%u, ret=%d", lhs, ret);
      error_code_ = ret != 0 ? ret : IndexError_ReadData;
      return 0.0f;
    }

    IndexStorage::MemoryBlock rhs_block;
    ret = entity_->read_vector(rhs, rhs_block);
    if (ailego_unlikely(ret != 0 || rhs_block.data() == nullptr)) {
      LOG_ERROR("Read vector failed, rhs id=%u, ret=%d", rhs, ret);
      error_code_ = ret != 0 ? ret : IndexError_ReadData;
      return 0.0f;
    }

    return dist(lhs_block.data(), rhs_block.data());
  }

  void dist_batch(const diskann_id_t *ids, size_t count, dist_t *distances) {
    std::array<IndexStorage::MemoryBlock, DiskAnnEntity::kVectorReadBatchSize>
        blocks;
    error_code_ = entity_->read_vectors(ids, blocks.data(), count);
    if (error_code_ != 0) return;
    for (size_t i = 0; i < count; ++i) {
      distances[i] = dist(blocks[i].data());
      if (error()) return;
    }
  }

  dist_t operator()(const void *vec) {
    return dist(vec);
  }

  inline void bind_quantizer(
      const turbo::Quantizer *quantizer, const uint8_t *codes,
      uint32_t code_size,
      IndexStorage::Segment::Pointer paged_codes = nullptr) {
    quantizer_ = quantizer;
    quant_codes_ = codes;
    quant_code_size_ = code_size;
    paged_quant_codes_ = std::move(paged_codes);
  }

  int quantize_query(const void *query_rotated) try {
    error_code_ =
        quantizer_->prepare_distance(query_rotated, &quant_dist_impl_);
    return error_code_;
  } catch (const std::bad_alloc &) {
    return error_code_ = IndexError_NoMemory;
  } catch (const std::exception &) {
    return error_code_ = IndexError_Runtime;
  }

  void quantized_dist(const diskann_id_t *ids, uint32_t n, float *dists) try {
    auto release =
        ailego::ScopeGuard::Make([&]() { quant_code_blocks_.clear(); });
    quant_dp_list_.resize(n);
    if (paged_quant_codes_) {
      quant_code_blocks_.resize(n);
      quant_code_reads_.clear();
      quant_code_reads_.reserve(n);
    }
    for (uint32_t i = 0; i < n; ++i) {
      if (ids[i] >= entity_->doc_cnt()) {
        error_code_ = IndexError_InvalidFormat;
        return;
      }
      const size_t offset = static_cast<size_t>(ids[i]) * quant_code_size_;
      if (paged_quant_codes_) {
        quant_code_reads_.emplace_back(paged_quant_codes_.get(), offset,
                                       quant_code_size_,
                                       &quant_code_blocks_[i]);
      } else {
        if (!quant_codes_) {
          error_code_ = IndexError_NoReady;
          return;
        }
        quant_dp_list_[i] = quant_codes_ + offset;
      }
    }
    if (paged_quant_codes_) {
      if (!paged_quant_codes_->read_borrowed_batch(quant_code_reads_.data(),
                                                   n)) {
        error_code_ = paged_quant_codes_->read_error();
        return;
      }
      for (size_t i = 0; i < n; ++i) {
        quant_dp_list_[i] = quant_code_blocks_[i].data();
        if (!quant_dp_list_[i]) {
          error_code_ = IndexError_ReadData;
          return;
        }
      }
    }
    quant_dist_impl_.batch(quant_dp_list_.data(), n, dists);
  } catch (const std::bad_alloc &) {
    error_code_ = IndexError_NoMemory;
  } catch (const std::exception &) {
    error_code_ = IndexError_ReadData;
  }

  // Keep first-visit order, but avoid reading PQ pages for neighbors that
  // cannot enter the candidate queue. Scratch is bounded by one graph row;
  // quantized_dist releases all page pins before returning.
  template <typename IdBuffer, typename DistanceBuffer>
  void quantized_dist_unvisited(const diskann_id_t *ids, uint32_t n,
                                VisitFilter &visited, IdBuffer &unvisited,
                                DistanceBuffer &distances) try {
    unvisited.clear();
    distances.clear();
    for (uint32_t i = 0; i < n; ++i) {
      // Validate before indexing the visit bitmap, including skipped IDs.
      if (ids[i] >= entity_->doc_cnt()) {
        error_code_ = IndexError_InvalidFormat;
        return;
      }
      if (!visited.visited(ids[i])) {
        visited.set_visited(ids[i]);
        unvisited.push_back(ids[i]);
      }
    }
    if (unvisited.empty()) return;
    distances.resize(unvisited.size());
    quantized_dist(unvisited.data(), static_cast<uint32_t>(unvisited.size()),
                   distances.data());
  } catch (const std::bad_alloc &) {
    error_code_ = IndexError_NoMemory;
  }

  inline void clear() {
    compare_cnt_ = 0;
    error_code_ = 0;
  }

  inline void clear_compare_cnt() {
    compare_cnt_ = 0;
  }

  inline bool error() const {
    return error_code_ != 0;
  }

  inline int error_code() const {
    return error_code_;
  }

  //! Get distances compute times
  inline uint32_t compare_cnt() const {
    return compare_cnt_;
  }

  inline uint32_t dimension() const {
    return dim_;
  }

  DistCalculator(const DistCalculator &) = delete;
  DistCalculator &operator=(const DistCalculator &) = delete;

 private:
  void bind_distance(const IndexMeta &meta, const IndexMetric::Pointer &measure,
                     const turbo::Quantizer::Pointer &external_quantizer) {
    data_quantizer_.reset();

    // load(storage, metric) and IndexFlow can override the persisted metric.
    // A quantizer chosen for that persisted metric must not replace the
    // caller's distance function (including an unnamed custom metric).
    if (measure->name() != meta.metric_name()) {
      distance_ = measure->distance();
      return;
    }

    // An externally constructed quantizer (already initialized by the caller)
    // takes precedence over the internal factory selection below.
    if (external_quantizer) {
      turbo::DistanceImpl impl =
          external_quantizer->distance("", IndexQueryMeta());
      if (impl.valid()) {
        data_quantizer_ = external_quantizer;
        distance_ = [func = impl.func(),
                     quant_dim = static_cast<size_t>(data_quantizer_->dim())](
                        const void *m, const void *q, size_t /*dim*/,
                        float *out) { func(m, q, quant_dim, out); };
        return;
      }
      LOG_WARN("External data quantizer has no valid distance, falling back");
    }

    const char *name = nullptr;
    if (meta.data_type() == IndexMeta::DataType::DT_FP16) {
      name = "Fp16Quantizer";
    } else if (meta.data_type() == IndexMeta::DataType::DT_FP32) {
      name = "Fp32Quantizer";
    }
    if (name != nullptr) {
      turbo::Quantizer::Pointer quantizer = IndexFactory::CreateQuantizer(name);
      if (quantizer) {
        IndexMeta quant_meta = meta;
        // Legacy Cosine layout folds the stored norm into an inflated
        // dimension; the turbo-quantizer layout keeps the raw dimension and
        // accounts the norm with extra_meta_size instead.
        if (meta.metric_name() == "Cosine" && meta.extra_meta_size() == 0) {
          quant_meta.set_dimension(meta.dimension() -
                                   sizeof(float) / meta.unit_size());
        }
        if (quantizer->init(quant_meta, quant_meta.metric_params()) == 0) {
          turbo::DistanceImpl impl = quantizer->distance("", IndexQueryMeta());
          if (impl.valid()) {
            data_quantizer_ = std::move(quantizer);
            distance_ = [func = impl.func(), quant_dim = static_cast<size_t>(
                                                 data_quantizer_->dim())](
                            const void *m, const void *q, size_t /*dim*/,
                            float *out) { func(m, q, quant_dim, out); };
            return;
          }
        }
      }
    }
    distance_ = measure->distance();
  }

 private:
  const DiskAnnEntity *entity_;

  turbo::Quantizer::Pointer data_quantizer_{};

  IndexMetric::MatrixDistance distance_;
  const void *query_;
  uint32_t dim_;

  uint32_t compare_cnt_;
  int error_code_{0};

  const turbo::Quantizer *quantizer_{nullptr};
  const uint8_t *quant_codes_{nullptr};
  IndexStorage::Segment::Pointer paged_quant_codes_;
  uint32_t quant_code_size_{0};

  turbo::DistanceImpl quant_dist_impl_{};

  ailego::BudgetedVector<const void *> quant_dp_list_;
  ailego::BudgetedVector<IndexStorage::MemoryBlock> quant_code_blocks_;
  ailego::BudgetedVector<IndexStorage::Segment::BorrowedRead> quant_code_reads_;
};

}  // namespace core
}  // namespace zvec

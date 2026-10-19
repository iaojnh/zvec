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

#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/core/framework/index_document.h>
#include <zvec/core/framework/index_error.h>
#include "utility/group_candidate_collector.h"
#include "utility/group_topk_heap.h"
#include "flat_index_format.h"
#include "flat_searcher.h"
#include "flat_utility.h"


namespace zvec {
namespace core {

/*! Brute Force Searcher Context
 */
template <size_t BATCH_SIZE>
class FlatSearcherContext : public IndexSearcher::Context {
 public:
  //! Constructor
  FlatSearcherContext(const FlatSearcher<BATCH_SIZE> *owner) {
    this->reset(owner);
  }

  //! Destructor
  ~FlatSearcherContext() override = default;

  //! Set topk of search result
  void set_topk(uint32_t topk) override {
    topk_ = topk;
  }

  //! Retrieve search result
  const IndexDocumentList &result() const override {
    return result(0);
  }

  //! Retrieve search result with index
  const IndexDocumentList &result(size_t index) const override {
    if (results_.empty()) {
      // Retain candidate heaps until the entire batch is materialized. A
      // denied later result must not leave an apparently complete prefix.
      auto pending = std::move(results_);
      try {
        pending.resize(result_heaps_.size());
        for (size_t i = 0; i < result_heaps_.size(); ++i) {
          const auto &items = result_heaps_[i].container();
          pending[i].assign(items.begin(), items.end());
        }
      } catch (...) {
        pending.clear();
        results_ = std::move(pending);
        throw;
      }
      results_ = std::move(pending);
      for (auto &heap : result_heaps_) heap.release();
    }
    return results_.at(index);
  }

  //! Retrieve result object for output
  IndexDocumentList *mutable_result(size_t idx) override {
    (void)result(idx);
    return &results_.at(idx);
  }

  //! Retrieve search group result with index
  const IndexGroupDocumentList &group_result() const override {
    return group_results_[0];
  }

  //! Retrieve search group result with index
  const IndexGroupDocumentList &group_result(size_t idx) const override {
    return group_results_[idx];
  }

  IndexGroupDocumentList *mutable_group_result() override {
    return &group_results_[0];
  }

  IndexGroupDocumentList *mutable_group_result(size_t idx) override {
    return &group_results_[idx];
  }

  //! Update the parameters of context
  int update(const ailego::Params & /*params*/) override {
    return 0;
  }

  //! Retrieve magic number
  uint32_t magic() const override {
    return magic_;
  }

  //! Get group topk
  inline uint32_t group_topk() const {
    return group_topk_;
  }

  //! Get group num
  inline uint32_t group_num() const {
    return group_num_;
  }

  inline GroupTopkMap<node_id_t, dist_t> &group_topk_heaps() {
    return group_topk_heaps_;
  }

  void set_fetch_vector(bool v) override {
    fetch_vector_ = v;
  }

  bool fetch_vector() const override {
    return fetch_vector_;
  }

  inline void resize_group_results(size_t size) {
    if (group_by_search()) {
      group_results_.resize(size);
    }
  }

  void topk_to_group_result(uint32_t idx) {
    ailego_assert_with(idx < group_results_.size(), "invalid idx");

    group_results_[idx].clear();

    auto group_topk_list = SelectGroupHeaps(group_topk_heaps_, group_num());

    group_results_[idx].resize(group_topk_list.size());
    auto provider = fetch_vector_ ? owner_->create_provider() : nullptr;
    if (fetch_vector_ && !provider)
      throw typename FlatSearcher<BATCH_SIZE>::MetadataReadError{
          IndexError_ReadData};

    for (uint32_t i = 0; i < group_topk_list.size(); ++i) {
      const std::string &group_id = group_topk_list[i]->first;
      group_results_[idx][i].set_group_id(group_id);

      uint32_t size =
          std::min(group_topk_,
                   static_cast<uint32_t>(group_topk_list[i]->second.size()));

      for (uint32_t j = 0; j < size; ++j) {
        auto score = group_topk_list[i]->second[j].second;
        if (score > this->threshold()) {
          break;
        }

        node_id_t id = group_topk_list[i]->second[j].first;

        const auto key = owner_->key(id);
        if (fetch_vector_) {
          IndexStorage::MemoryBlock block;
          const int ret = provider->get_vector(key, block);
          if (ret != 0)
            throw typename FlatSearcher<BATCH_SIZE>::MetadataReadError{ret};
          group_results_[idx][i].mutable_docs()->emplace_back(key, score, id,
                                                              std::move(block));
        } else {
          group_results_[idx][i].mutable_docs()->emplace_back(key, score, id);
        }
      }
    }
  }

  //! Get if group by search
  bool group_by_search() {
    return group_num_ > 0;
  }

  //! Set group params
  void set_group_params(uint32_t group_num, uint32_t group_topk) override {
    group_num_ = group_num;
    group_topk_ = group_topk;
    group_topk_heaps_.clear();
  }

  void reset() override {
    results_.clear();
    for (auto &heap : result_heaps_) heap.release();
    decltype(result_heaps_)().swap(result_heaps_);
    decltype(batch_queries_)().swap(batch_queries_);
    group_topk_heaps_.clear();
    group_results_.clear();
  }

  //! Reset the context
  void reset(const FlatSearcher<BATCH_SIZE> *owner) {
    magic_ = owner->magic();
    feature_size_ = owner->meta().element_size();

    uint32_t block_size = feature_size_ * BATCH_SIZE;
    actual_read_size_ =
        (owner->read_block_size() + block_size - 1) / block_size * block_size;
    features_segment_ = owner->clone_features_segment();
    quantizer_ = owner->quantizer();
    owner_ = owner;
    if (quantizer_) {
      this->update_index_quantizer(quantizer_);
    } else {
      this->update_index_metric(owner->metric());
    }
  }

  //! Similarity search
  int search_row(const void *query, const IndexQueryMeta &qmeta) {
    return (this->filter().is_valid()
                ? this->search_row_filter(query, qmeta)
                : this->search_row_nofilter(query, qmeta));
  }

  //! Similarity search
  int search_row(const void *query, const IndexQueryMeta &qmeta, size_t count) {
    return (this->filter().is_valid()
                ? this->batch_search_row_filter(query, qmeta, count)
                : this->batch_search_row_nofilter(query, qmeta, count));
  }

  //! Similarity search
  int search_column(const void *query, const IndexQueryMeta &qmeta) {
    return (this->filter().is_valid()
                ? this->search_column_filter(query, qmeta)
                : this->search_column_nofilter(query, qmeta));
  }

  //! Similarity search
  int search_column(const void *query, const IndexQueryMeta &qmeta,
                    size_t count) {
    return (this->filter().is_valid()
                ? this->batch_search_column_filter(query, qmeta, count)
                : this->batch_search_column_nofilter(query, qmeta, count));
  }

  int group_by_search_impl(const void *query, const IndexQueryMeta &qmeta,
                           uint32_t count);

  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count);

 protected:
  //! Compute the distance between a stored feature and a query
  inline void feature_distance(const void *feature, const void *query,
                               size_t dim, float *out) const {
    if (quantizer_) {
      *out = quantizer_->calc_distance_dp_query(feature, query);
    } else {
      owner_->distance_matrix().template distance<1>(feature, query, dim, out);
    }
  }

  //! Compute the quantizer distances of contiguous row features
  inline void quantized_batch_distance(const void *features, size_t num,
                                       const void *query, float *out) const {
    const void *dp_list[BATCH_SIZE];
    for (size_t i = 0; i != num; ++i) {
      dp_list[i] = static_cast<const char *>(features) + i * feature_size_;
    }
    quantizer_->calc_distance_dp_query_batch(dp_list, static_cast<int>(num),
                                             query, out);
  }

  //! Enqueue a chunk of contiguous row features into the heap (no filter)
  void enqueue_row_chunk(const void *data, size_t size, const void *query,
                         const IndexQueryMeta &qmeta, size_t *feature_index,
                         BudgetedIndexCandidateHeap *heap) {
    if (quantizer_) {
      size_t count = size / feature_size_;
      for (size_t i = 0; i < count; i += BATCH_SIZE) {
        size_t num = std::min<size_t>(count - i, BATCH_SIZE);
        this->quantized_batch_distance(
            static_cast<const char *>(data) + i * feature_size_, num, query,
            scores_);
        for (size_t j = 0; j != num; ++j) {
          heap->emplace(0, scores_[j], (*feature_index)++);
        }
      }
    } else {
      auto matrix = owner_->distance_matrix();
      for (size_t offset = 0; offset < size; offset += feature_size_) {
        float score;
        matrix.template distance<1>(static_cast<const char *>(data) + offset,
                                    query, qmeta.dimension(), &score);
        heap->emplace(0, score, (*feature_index)++);
      }
    }
  }

  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto batch_enqueue_nofilter(const void *block, size_t block_index,
                              size_t query_index, const IndexQueryMeta &qmeta,
                              size_t query_count) ->
      typename std::enable_if<K != 1 && IsEqualPowerofTwo<K>::value>::type {
    size_t query_batch_count = query_count / K;

    for (size_t i = 0; i != query_batch_count; ++i) {
      owner_->distance_matrix().template distance<BATCH_SIZE, K>(
          block,
          &batch_queries_[(query_index - query_base_) * qmeta.element_size()],
          qmeta.dimension(), scores_);

      for (size_t k = 0; k != K; ++k) {
        BudgetedIndexCandidateHeap *heap = &result_heaps_[query_index++];
        for (size_t j = 0; j != BATCH_SIZE; ++j) {
          heap->emplace(0, scores_[k * BATCH_SIZE + j], block_index + j);
        }
      }  // end of for
    }  // end of for

    size_t query_left_count = query_count % K;
    if (query_left_count != 0) {
      this->batch_enqueue_nofilter<(K >> 1)>(block, block_index, query_index,
                                             qmeta, query_left_count);
    }
  }

  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto batch_enqueue_nofilter(const void *block, size_t block_index,
                              size_t query_index, const IndexQueryMeta &qmeta,
                              size_t query_count) ->
      typename std::enable_if<K == 1>::type {
    ailego_assert(query_count == 1);
    (void)query_count;

    owner_->distance_matrix().template distance<BATCH_SIZE, 1>(
        block,
        &batch_queries_[(query_index - query_base_) * qmeta.element_size()],
        qmeta.dimension(), scores_);

    BudgetedIndexCandidateHeap *heap = &result_heaps_[query_index];
    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      heap->emplace(0, scores_[i], block_index + i);
    }
  }

  //! Enqueue items into the search heaps (with filter)
  template <size_t K>
  auto batch_enqueue_filter(const void *block, size_t block_index,
                            size_t block_mask, size_t query_index,
                            const IndexQueryMeta &qmeta, size_t query_count) ->
      typename std::enable_if<K != 1 && IsEqualPowerofTwo<K>::value>::type {
    size_t query_batch_count = query_count / K;

    for (size_t i = 0; i != query_batch_count; ++i) {
      owner_->distance_matrix().template distance<BATCH_SIZE, K>(
          block,
          &batch_queries_[(query_index - query_base_) * qmeta.element_size()],
          qmeta.dimension(), scores_);

      for (size_t k = 0; k != K; ++k) {
        BudgetedIndexCandidateHeap *heap = &result_heaps_[query_index++];
        for (size_t j = 0; j != BATCH_SIZE; ++j) {
          if ((block_mask & (1 << j)) != 0) {
            heap->emplace(0, scores_[k * BATCH_SIZE + j], block_index + j);
          }
        }
      }  // end of for
    }  // end of for

    size_t query_left_count = query_count % K;
    if (query_left_count != 0) {
      this->batch_enqueue_filter<(K >> 1)>(
          block, block_index, block_mask, query_index, qmeta, query_left_count);
    }
  }

  //! Enqueue items into the search heaps (with filter)
  template <size_t K>
  auto batch_enqueue_filter(const void *block, size_t block_index,
                            size_t block_mask, size_t query_index,
                            const IndexQueryMeta &qmeta, size_t query_count) ->
      typename std::enable_if<K == 1>::type {
    ailego_assert(query_count == 1);
    (void)query_count;

    owner_->distance_matrix().template distance<BATCH_SIZE, 1>(
        block,
        &batch_queries_[(query_index - query_base_) * qmeta.element_size()],
        qmeta.dimension(), scores_);

    BudgetedIndexCandidateHeap *heap = &result_heaps_[query_index];
    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      if ((block_mask & (1 << i)) != 0) {
        heap->emplace(0, scores_[i], block_index + i);
      }
    }
  }

  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto single_enqueue_nofilter(const void *feature, size_t feature_index,
                               size_t query_index, const IndexQueryMeta &qmeta,
                               size_t query_count) ->
      typename std::enable_if<K != 1 && IsEqualPowerofTwo<K>::value>::type {
    size_t query_batch_count = query_count / K;

    for (size_t i = 0; i != query_batch_count; ++i) {
      owner_->distance_matrix().template distance<K, 1>(
          &batch_queries_[(query_index - query_base_) * qmeta.element_size()],
          feature, qmeta.dimension(), scores_);

      for (size_t k = 0; k != K; ++k) {
        result_heaps_[query_index++].emplace(0, scores_[k], feature_index);
      }
    }
    size_t query_left_count = query_count % K;
    if (query_left_count != 0) {
      this->single_enqueue_nofilter<(K >> 1)>(
          feature, feature_index, query_index, qmeta, query_left_count);
    }
  }

  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto single_enqueue_nofilter(const void *feature, size_t feature_index,
                               size_t query_index, const IndexQueryMeta &qmeta,
                               size_t query_count) ->
      typename std::enable_if<K == 1>::type {
    ailego_assert(query_count == 1);
    (void)query_count;

    owner_->distance_matrix().template distance<1>(
        feature,
        &batch_queries_[(query_index - query_base_) * qmeta.element_size()],
        qmeta.dimension(), scores_);
    result_heaps_[query_index].emplace(0, scores_[0], feature_index);
  }

 protected:
  void prepare_result_heaps(size_t count) {
    // Do not retain a rare batch's array of heap objects after a small query.
    if (result_heaps_.capacity() > 64 && result_heaps_.capacity() / 2 > count)
      decltype(result_heaps_)().swap(result_heaps_);
    result_heaps_.resize(count);
  }

  int batch_search_column(const void *query, const IndexQueryMeta &qmeta,
                          size_t query_count, bool filtered);

  //! Similarity search (1 column without filter)
  int search_column_nofilter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (1 column with filter)
  int search_column_filter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (1 row without filter)
  int search_row_nofilter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (1 row with filter)
  int search_row_filter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (n columns without filter)
  int batch_search_column_nofilter(const void *query,
                                   const IndexQueryMeta &qmeta,
                                   size_t query_count);

  //! Similarity search (n columns with filter)
  int batch_search_column_filter(const void *query, const IndexQueryMeta &qmeta,
                                 size_t query_count);

  //! Similarity search (n rows without filter)
  int batch_search_row_nofilter(const void *query, const IndexQueryMeta &qmeta,
                                size_t query_count);

  //! Similarity search (n rows with filter)
  int batch_search_row_filter(const void *query, const IndexQueryMeta &qmeta,
                              size_t query_count);

 private:
  // Query-scoped leases must not survive in a worker's pointer-read TLS after
  // the owning centroid index is destroyed. Keep mmap's direct read path.
  size_t read_features(size_t offset, const void **out, size_t bytes,
                       IndexStorage::MemoryBlock &guard) const {
    guard.reset();
    const size_t count = features_segment_->read(offset, guard, bytes);
    *out = guard.data();
    return count;
  }

  const FlatSearcher<BATCH_SIZE> *owner_{nullptr};
  uint32_t magic_{0};
  uint32_t topk_{0};
  uint32_t feature_size_{0};
  uint32_t actual_read_size_{0};
  IndexStorage::Segment::Pointer features_segment_{};
  std::shared_ptr<zvec::turbo::Quantizer> quantizer_{};
  mutable std::vector<BudgetedIndexCandidateHeap,
                      ailego::BudgetedAllocator<BudgetedIndexCandidateHeap>>
      result_heaps_{};
  mutable ailego::BudgetedVector<IndexDocumentList> results_;
  std::vector<char, ailego::BudgetedAllocator<char>> batch_queries_{};
  size_t query_base_{0};
  float scores_[BATCH_SIZE * BATCH_SIZE];
  bool fetch_vector_{false};

  // group
  uint32_t group_num_{0}, group_topk_{0};
  GroupTopkMap<node_id_t, dist_t> group_topk_heaps_{};
  ailego::BudgetedVector<IndexGroupDocumentList> group_results_{};
};

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_column_nofilter(
    const void *query, const IndexQueryMeta &qmeta) {
  prepare_result_heaps(1);
  BudgetedIndexCandidateHeap *heap = &result_heaps_[0];
  results_.clear();
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t block_size = feature_size_ * BATCH_SIZE;
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = this->owner_->distance_matrix();

  while (left_size >= actual_read_size_) {
    IndexStorage::MemoryBlock read_guard;
    const void *data = nullptr;
    if (read_features(read_offset, &data, actual_read_size_, read_guard) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_; offset += block_size) {
      matrix.template distance<BATCH_SIZE, 1>(
          (const char *)data + offset, query, qmeta.dimension(), scores_);

      for (size_t i = 0; i != BATCH_SIZE; ++i) {
        heap->emplace(0, scores_[i], feature_index++);
      }
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  IndexStorage::MemoryBlock read_guard;
  const void *data = nullptr;
  if (read_features(read_offset, &data, left_size, read_guard) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left block features
  size_t left_size_aligned = left_size / block_size * block_size;
  for (size_t offset = 0; offset != left_size_aligned; offset += block_size) {
    matrix.template distance<BATCH_SIZE, 1>((const char *)data + offset, query,
                                            qmeta.dimension(), scores_);

    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      heap->emplace(0, scores_[i], feature_index++);
    }
  }

  // Process left single features
  for (size_t offset = left_size_aligned; offset < left_size;
       offset += feature_size_) {
    float score;
    matrix.template distance<1>((const char *)data + offset, query,
                                qmeta.dimension(), &score);
    heap->emplace(0, score, feature_index++);
  }

  for (auto &it : heap->mutable_container()) {
    it.set_key(owner_->key(it.index()));
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_column_filter(
    const void *query, const IndexQueryMeta &qmeta) {
  prepare_result_heaps(1);
  BudgetedIndexCandidateHeap *heap = &result_heaps_[0];
  results_.clear();
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t block_size = feature_size_ * BATCH_SIZE;
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = owner_->distance_matrix();

  while (left_size >= actual_read_size_) {
    IndexStorage::MemoryBlock read_guard;
    const void *data = nullptr;
    if (read_features(read_offset, &data, actual_read_size_, read_guard) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_; offset += block_size) {
      matrix.template distance<BATCH_SIZE, 1>(
          (const char *)data + offset, query, qmeta.dimension(), scores_);

      for (size_t i = 0; i != BATCH_SIZE; ++i) {
        uint64_t feature_key = owner_->key(feature_index);

        if (!this->filter()(feature_key)) {
          if (group_by_search()) {
          }
          heap->emplace(feature_key, scores_[i], feature_index);
        }
        feature_index += 1;
      }
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  IndexStorage::MemoryBlock read_guard;
  const void *data = nullptr;
  if (read_features(read_offset, &data, left_size, read_guard) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left block features
  size_t left_size_aligned = left_size / block_size * block_size;
  for (size_t offset = 0; offset != left_size_aligned; offset += block_size) {
    matrix.template distance<BATCH_SIZE, 1>((const char *)data + offset, query,
                                            qmeta.dimension(), scores_);

    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      uint64_t feature_key = owner_->key(feature_index);

      if (!this->filter()(feature_key)) {
        heap->emplace(feature_key, scores_[i], feature_index);
      }
      feature_index += 1;
    }
  }

  // Process left single features
  for (size_t offset = left_size_aligned; offset < left_size;
       offset += feature_size_) {
    uint64_t feature_key = owner_->key(feature_index);
    if (!this->filter()(feature_key)) {
      float score;
      matrix.template distance<1>((const char *)data + offset, query,
                                  qmeta.dimension(), &score);
      heap->emplace(feature_key, score, feature_index);
    }
    feature_index += 1;
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_row_nofilter(
    const void *query, const IndexQueryMeta &qmeta) {
  prepare_result_heaps(1);
  BudgetedIndexCandidateHeap *heap = &result_heaps_[0];
  results_.clear();
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;

  while (left_size >= actual_read_size_) {
    IndexStorage::MemoryBlock read_guard;
    const void *data = nullptr;
    if (read_features(read_offset, &data, actual_read_size_, read_guard) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    this->enqueue_row_chunk(data, actual_read_size_, query, qmeta,
                            &feature_index, heap);
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  IndexStorage::MemoryBlock read_guard;
  const void *data = nullptr;
  if (read_features(read_offset, &data, left_size, read_guard) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  this->enqueue_row_chunk(data, left_size, query, qmeta, &feature_index, heap);
  for (auto &it : heap->mutable_container()) {
    it.set_key(owner_->key(it.index()));
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_row_filter(
    const void *query, const IndexQueryMeta &qmeta) {
  prepare_result_heaps(1);
  BudgetedIndexCandidateHeap *heap = &result_heaps_[0];
  results_.clear();
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;

  while (left_size >= actual_read_size_) {
    IndexStorage::MemoryBlock read_guard;
    const void *data = nullptr;
    if (read_features(read_offset, &data, actual_read_size_, read_guard) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_;
         offset += feature_size_) {
      uint64_t feature_key = owner_->key(feature_index);
      if (!this->filter()(feature_key)) {
        float score;
        this->feature_distance((const char *)data + offset, query,
                               qmeta.dimension(), &score);
        heap->emplace(feature_key, score, feature_index);
      }
      feature_index += 1;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  IndexStorage::MemoryBlock read_guard;
  const void *data = nullptr;
  if (read_features(read_offset, &data, left_size, read_guard) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  for (size_t offset = 0; offset < left_size; offset += feature_size_) {
    uint64_t feature_key = owner_->key(feature_index);
    if (!this->filter()(feature_key)) {
      float score;
      this->feature_distance((const char *)data + offset, query,
                             qmeta.dimension(), &score);
      heap->emplace(feature_key, score, feature_index);
    }
    feature_index += 1;
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_column_nofilter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  return batch_search_column(query, qmeta, query_count, false);
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_column_filter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  return batch_search_column(query, qmeta, query_count, true);
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_column(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count,
    bool filtered) {
  if (!qmeta.element_size() || query_count > SIZE_MAX / qmeta.element_size())
    return IndexError_InvalidArgument;
  results_.clear();
  prepare_result_heaps(query_count);
  for (auto &heap : result_heaps_) {
    heap.clear();
    heap.limit(topk_);
    heap.set_threshold(this->threshold());
  }
  auto release_queries = ailego::ScopeGuard::Make([&] {
    decltype(batch_queries_)().swap(batch_queries_);
    query_base_ = 0;
  });
  // Keep the storage scan outside the query tiles: each feature byte is read
  // once, even when the transposed query workspace is much smaller than the
  // batch. Only the CPU transpose is repeated for subsequent read blocks.
  const size_t tile_size = std::max<size_t>(
      BATCH_SIZE, (64 * 1024 / qmeta.element_size() / BATCH_SIZE) * BATCH_SIZE);
  size_t offset = 0;
  const size_t total = features_segment_->data_size();
  while (offset < total) {
    const size_t bytes = std::min<size_t>(actual_read_size_, total - offset);
    if (!bytes) return IndexError_ReadData;
    IndexStorage::MemoryBlock guard;
    const void *data = nullptr;
    if (read_features(offset, &data, bytes, guard) != bytes)
      return IndexError_ReadData;
    for (size_t base = 0; base < query_count; base += tile_size) {
      const size_t count = std::min(tile_size, query_count - base);
      query_base_ = base;
      batch_queries_.clear();
      TransposeQueries<BATCH_SIZE>(
          static_cast<const char *>(query) + base * qmeta.element_size(), qmeta,
          count, &batch_queries_);
      size_t pos = 0;
      const size_t block_bytes = feature_size_ * BATCH_SIZE;
      for (; pos + block_bytes <= bytes; pos += block_bytes) {
        const size_t index = (offset + pos) / feature_size_;
        const void *block = static_cast<const char *>(data) + pos;
        if (!filtered) {
          batch_enqueue_nofilter<BATCH_SIZE>(block, index, base, qmeta, count);
        } else {
          size_t mask = 0;
          for (size_t i = 0; i < BATCH_SIZE; ++i)
            if (!this->filter()(owner_->key(index + i))) mask |= size_t{1} << i;
          if (mask)
            batch_enqueue_filter<BATCH_SIZE>(block, index, mask, base, qmeta,
                                             count);
        }
      }
      for (; pos < bytes; pos += feature_size_) {
        const size_t index = (offset + pos) / feature_size_;
        if (!filtered || !this->filter()(owner_->key(index)))
          single_enqueue_nofilter<BATCH_SIZE>(
              static_cast<const char *>(data) + pos, index, base, qmeta, count);
      }
    }
    offset += bytes;
  }
  for (auto &heap : result_heaps_) {
    for (auto &item : heap.mutable_container())
      item.set_key(owner_->key(item.index()));
    heap.sort();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_row_nofilter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  // Initialize resources
  prepare_result_heaps(query_count);
  for (auto &heap : result_heaps_) {
    results_.clear();
    heap.clear();
    heap.limit(topk_);
    heap.set_threshold(this->threshold());
  }

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;

  // Process feature blocks
  while (left_size >= actual_read_size_) {
    IndexStorage::MemoryBlock read_guard;
    const void *data = nullptr;
    if (read_features(read_offset, &data, actual_read_size_, read_guard) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_;
         offset += feature_size_) {
      size_t query_offset = 0;
      const void *feature = (const char *)data + offset;

      for (auto &heap : result_heaps_) {
        float score;
        this->feature_distance(feature, (const char *)query + query_offset,
                               qmeta.dimension(), &score);
        heap.emplace(0, score, feature_index);
        query_offset += qmeta.element_size();
      }
      feature_index += 1;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  IndexStorage::MemoryBlock read_guard;
  const void *data = nullptr;
  if (read_features(read_offset, &data, left_size, read_guard) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left features
  for (size_t offset = 0; offset < left_size; offset += feature_size_) {
    size_t query_offset = 0;
    const void *feature = (const char *)data + offset;

    for (auto &heap : result_heaps_) {
      float score;
      this->feature_distance(feature, (const char *)query + query_offset,
                             qmeta.dimension(), &score);
      heap.emplace(0, score, feature_index);
      query_offset += qmeta.element_size();
    }
    feature_index += 1;
  }

  // Normalize results
  for (auto &heap : result_heaps_) {
    for (auto &it : heap.mutable_container()) {
      it.set_key(owner_->key(it.index()));
    }
    heap.sort();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_row_filter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  // Initialize resources
  prepare_result_heaps(query_count);
  for (auto &heap : result_heaps_) {
    results_.clear();
    heap.clear();
    heap.limit(topk_);
    heap.set_threshold(this->threshold());
  }

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;

  // Process feature blocks
  while (left_size >= actual_read_size_) {
    IndexStorage::MemoryBlock read_guard;
    const void *data = nullptr;
    if (read_features(read_offset, &data, actual_read_size_, read_guard) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_;
         offset += feature_size_) {
      uint64_t feature_key = owner_->key(feature_index);

      if (!this->filter()(feature_key)) {
        size_t query_offset = 0;
        const void *feature = (const char *)data + offset;

        for (auto &heap : result_heaps_) {
          float score;
          this->feature_distance(feature, (const char *)query + query_offset,
                                 qmeta.dimension(), &score);
          heap.emplace(feature_key, score, feature_index);
          query_offset += qmeta.element_size();
        }
      }
      feature_index += 1;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  IndexStorage::MemoryBlock read_guard;
  const void *data = nullptr;
  if (read_features(read_offset, &data, left_size, read_guard) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left features
  for (size_t offset = 0; offset < left_size; offset += feature_size_) {
    uint64_t feature_key = owner_->key(feature_index);

    if (!this->filter()(feature_key)) {
      size_t query_offset = 0;
      const void *feature = (const char *)data + offset;

      for (auto &heap : result_heaps_) {
        float score;
        this->feature_distance(feature, (const char *)query + query_offset,
                               qmeta.dimension(), &score);
        heap.emplace(feature_key, score, feature_index);
        query_offset += qmeta.element_size();
      }
    }
    feature_index += 1;
  }

  // Normalize results
  for (auto &heap : result_heaps_) {
    heap.sort();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::group_by_search_impl(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count) {
  this->resize_group_results(count);
  if (!this->group_by().is_valid()) {
    LOG_ERROR("Invalid group-by function");
    return IndexError_InvalidArgument;
  }

  std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
    return this->group_by()(key);
  };

  auto provider = owner_->create_provider();
  if (!provider) return IndexError_ReadData;

  for (size_t q = 0; q < count; ++q) {
    this->group_topk_heaps().clear();
    GroupCandidateCollector groups(this->group_num(), this->group_topk());

    for (node_id_t id = 0; id < provider->count(); ++id) {
      if (!this->filter().is_valid() || !this->filter()(owner_->key(id))) {
        const void *vector = provider->get_vector(owner_->key(id));
        if (!vector) return IndexError_ReadData;
        dist_t dist = 0;
        if (quantizer_) {
          dist = quantizer_->calc_distance_dp_query(vector, query);
        } else {
          owner_->distance_matrix().template distance<1>(
              query, vector, provider->dimension(), &dist);
        }

        std::string group_id = group_by(owner_->key(id));
        int ret = groups.add(group_id, id, dist);
        if (ret != 0) return ret;
      }
    }
    int ret = groups.finish_heaps(this->group_topk_heaps(), this->group_topk());
    if (ret != 0) return ret;
    this->topk_to_group_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count) {
  auto provider = owner_->create_provider();
  if (!provider) return IndexError_ReadData;
  if (this->group_by_search()) {
    this->resize_group_results(count);
    if (!this->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
      return this->group_by()(key);
    };

    for (size_t q = 0; q < count; ++q) {
      this->group_topk_heaps().clear();
      GroupCandidateCollector groups(this->group_num(), this->group_topk());
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!this->filter().is_valid() || !this->filter()(pk)) {
          const void *vector = provider->get_vector(pk);
          if (!vector) return IndexError_ReadData;
          dist_t dist = 0;
          if (quantizer_) {
            dist = quantizer_->calc_distance_dp_query(vector, query);
          } else {
            owner_->distance_matrix().template distance<1>(
                query, vector, provider->dimension(), &dist);
          }

          std::string group_id = group_by(pk);
          int ret = groups.add(group_id, owner_->get_id(pk), dist);
          if (ret != 0) return ret;
        }
      }
      int ret =
          groups.finish_heaps(this->group_topk_heaps(), this->group_topk());
      if (ret != 0) return ret;
      this->topk_to_group_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    prepare_result_heaps(count);
    for (auto &heap : result_heaps_) {
      results_.clear();
      heap.clear();
      heap.limit(topk_);
      heap.set_threshold(this->threshold());
    }
    for (size_t q = 0; q < count; ++q) {
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!this->filter().is_valid() || !this->filter()(pk)) {
          const void *vector = provider->get_vector(pk);
          if (!vector) return IndexError_ReadData;
          dist_t dist = 0;
          if (quantizer_) {
            dist = quantizer_->calc_distance_dp_query(vector, query);
          } else {
            owner_->distance_matrix().template distance<1>(
                query, vector, provider->dimension(), &dist);
          }
          result_heaps_[q].emplace(pk, dist, owner_->get_id(pk));
        }
      }
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
    for (auto &heap : result_heaps_) {
      heap.sort();
    }
  }
  return 0;
}

}  // namespace core
}  // namespace zvec

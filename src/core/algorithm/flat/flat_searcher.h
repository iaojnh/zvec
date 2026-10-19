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

#include <memory>
#include <unordered_map>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_searcher.h>
#include "utility/immutable_segment_view.h"
#include "flat_distance_matrix.h"
#include "flat_index_format.h"

namespace zvec {
namespace core {

/*! Flat Searcher
 */
template <size_t BATCH_SIZE>
class FlatSearcher : public IndexSearcher {
 public:
  // Internal query/iterator boundary converts metadata I/O failures into the
  // existing error-code API. Never turn an unreadable key into a real key.
  struct MetadataReadError {
    int status;
  };
  //! Destructor
  ~FlatSearcher() override = default;

  //! Initialize Searcher
  int init(const ailego::Params &index_params) override {
    quantizer_.reset();
    params_ = index_params;
    read_block_size_ = FLAT_DEFAULT_READ_BLOCK_SIZE;
    index_params.get(PARAM_FLAT_READ_BLOCK_SIZE, &read_block_size_);
    return 0;
  }

  // The caller supplies vectors/queries in the quantizer's encoded layout.
  // Query conversion remains the responsibility of IndexFlow/the IVF reformer.
  int init(const ailego::Params &params,
           const turbo::Quantizer::Pointer &quantizer) override {
    int ret = init(params);
    if (ret == 0) quantizer_ = quantizer;
    return ret;
  }

  //! Cleanup Searcher
  int cleanup() override {
    return this->unload();
  }

  //! Load index from container
  int load(IndexStorage::Pointer cntr, IndexMetric::Pointer measure) override;

  //! Unload index
  int unload() override {
    std::lock_guard<std::mutex> lock(vector_provider_mutex_);
    vector_provider_.reset();
    distance_matrix_ = {};
    quantizer_.reset();
    container_ = nullptr;
    measure_ = nullptr;
    legacy_quantizer_layout_ = false;
    features_segment_ = nullptr;
    keys_ = {};
    mapping_ = {};
    mapping_loaded_ = false;
    paged_metadata_ = false;
    decltype(key_id_mapping_)().swap(key_id_mapping_);
    return 0;
  }

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const override {
    return this->search_impl(query, qmeta, context);
  }

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     uint32_t count, Context::Pointer &context) const override {
    return this->search_impl(query, qmeta, count, context);
  }

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const override;

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               Context::Pointer &context) const override {
    return search_bf_by_p_keys_impl(query, p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               Context::Pointer &context) const override;

  //! Retrieve statistics
  const IndexSearcher::Stats &stats() const override {
    return stats_;
  }

  //! Retrieve meta of index
  const IndexMeta &meta() const override {
    return meta_;
  }

  //! Retrieve params of index
  const ailego::Params &params() const override {
    return params_;
  }

  //! Create a searcher context
  IndexSearcher::Context::Pointer create_context() const override;

  //! Create a searcher provider
  IndexProvider::Pointer create_provider() const override;

  //! Retrieve magic number
  uint32_t magic() const {
    return magic_;
  }

  //! Retrieve block size of data read
  uint32_t read_block_size() const {
    return read_block_size_;
  }

  //! Retrieve primary key via index id
  uint64_t key(size_t i) const {
    uint64_t value = 0;
    if (i >= keys_.size() / sizeof(value))
      throw MetadataReadError{IndexError_OutOfRange};
    const int ret = keys_.read(i * sizeof(value), &value, sizeof(value));
    if (ret != 0) throw MetadataReadError{ret};
    return value;
  }

  // Retrieve index id via primary key
  node_id_t get_id(key_t key) const {
    if (paged_metadata_) {
      // The index already persists ids sorted by key. Reuse that table rather
      // than building an unbudgeted hash with one node per document.
      size_t lo = 0, hi = mapping_.size() / sizeof(uint32_t);
      while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (this->key(local_index(mid)) < key)
          lo = mid + 1;
        else
          hi = mid;
      }
      node_id_t result = kInvalidNodeId;
      for (; lo < mapping_.size() / sizeof(uint32_t); ++lo) {
        const auto id = local_index(lo);
        if (this->key(id) != key) break;
        // Preserve the old hash's last-id-wins behavior for duplicate keys.
        if (result == kInvalidNodeId || id > result) result = id;
      }
      return result;
    }
    auto it = key_id_mapping_.find(key);
    if (it != key_id_mapping_.end()) {
      return it->second;
    } else {
      return kInvalidNodeId;
    }
  }

  //! Retrieve primary key via index id
  uint32_t local_index(size_t i) const {
    uint32_t value = 0;
    if (i >= mapping_.size() / sizeof(value))
      throw MetadataReadError{IndexError_OutOfRange};
    const int ret = mapping_.read(i * sizeof(value), &value, sizeof(value));
    if (ret != 0) throw MetadataReadError{ret};
    if (value >= keys_.size() / sizeof(uint64_t))
      throw MetadataReadError{IndexError_InvalidFormat};
    return value;
  }

  //! Retrieve primary key via index id
  inline bool column_major_order() const {
    return column_major_order_;
  }

  //! Retrieve the distance matrix
  const FlatDistanceMatrix<BATCH_SIZE> &distance_matrix() const {
    return distance_matrix_;
  }

  //! Retrieve the turbo quantizer
  const std::shared_ptr<zvec::turbo::Quantizer> &quantizer() const {
    return quantizer_;
  }

  const IndexMetric::Pointer &metric() const {
    return measure_;
  }

  //! Clone a features segment
  IndexStorage::Segment::Pointer clone_features_segment() const {
    return features_segment_->clone();
  }

  const void *get_vector(key_t key) const override {
    std::lock_guard<std::mutex> lock(vector_provider_mutex_);
    if (!vector_provider_) vector_provider_ = this->create_provider();
    return vector_provider_ ? vector_provider_->get_vector(key) : nullptr;
  }

  int get_vector(key_t key, IndexStorage::MemoryBlock &block) const override {
    if (!container_) {
      block.reset();
      return IndexError_NoReady;
    }
    auto provider = this->create_provider();
    if (!provider) {
      block.reset();
      return IndexError_NoReady;
    }
    return provider->get_vector(key, block);
  }

 private:
  //! Validate turbo query records, preserving legacy metric assertions.
  int check_query_meta(const IndexQueryMeta &qmeta) const;

  //! Members
  mutable std::mutex vector_provider_mutex_;
  mutable IndexProvider::Pointer vector_provider_;
  ImmutableSegmentView keys_;
  std::unordered_map<
      key_t, node_id_t, std::hash<key_t>, std::equal_to<key_t>,
      ailego::BudgetedAllocator<std::pair<const key_t, node_id_t>>>
      key_id_mapping_;
  uint32_t magic_{IndexContext::GenerateMagic()};
  uint32_t read_block_size_{FLAT_DEFAULT_READ_BLOCK_SIZE};
  bool column_major_order_{false};
  bool legacy_quantizer_layout_{false};
  IndexMeta meta_{};
  IndexStorage::Pointer container_{};
  IndexMetric::Pointer measure_{};
  turbo::Quantizer::Pointer quantizer_{};
  ailego::Params params_{};
  IndexStorage::Segment::Pointer features_segment_{};
  mutable ImmutableSegmentView mapping_;
  mutable bool mapping_loaded_{false};
  bool paged_metadata_{false};
  mutable std::mutex mapping_mutex_{};
  FlatDistanceMatrix<BATCH_SIZE> distance_matrix_{};
  IndexSearcher::Stats stats_{};
};

}  // namespace core
}  // namespace zvec

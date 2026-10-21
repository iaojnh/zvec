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

#include <cstdint>
#include <string>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_heap.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

/*! Index Sparse Document
 */
class IndexSparseDocument {
 public:
  //! Constructor
  IndexSparseDocument() = default;
  IndexSparseDocument(const IndexSparseDocument &other)
      : sparse_count_(other.sparse_count_) {
    reserve_sparse_buffers(other.sparse_indices_.size(),
                           other.sparse_values_.size());
    sparse_indices_ = other.sparse_indices_;
    sparse_values_ = other.sparse_values_;
  }
  IndexSparseDocument(IndexSparseDocument &&) noexcept = default;
  IndexSparseDocument &operator=(IndexSparseDocument other) noexcept {
    memory_.swap(other.memory_);
    std::swap(sparse_count_, other.sparse_count_);
    sparse_indices_.swap(other.sparse_indices_);
    sparse_values_.swap(other.sparse_values_);
    return *this;
  }

  // Public string accessors retain their standard allocator. Engine decoders
  // must admit their capacity before filling them; retained capacity stays
  // charged across reuse and the reservation follows copies/moves.
  void reserve_sparse_buffers(size_t indices_bytes, size_t values_bytes) {
    if (indices_bytes <= sparse_indices_.capacity() &&
        values_bytes <= sparse_values_.capacity())
      return;
    indices_bytes = std::max(indices_bytes, sparse_indices_.size());
    values_bytes = std::max(values_bytes, sparse_values_.size());
    ailego::MemoryReservation replacement_memory;
    if (indices_bytes == SIZE_MAX || values_bytes == SIZE_MAX)
      throw std::bad_alloc();
    replacement_memory.add_array(indices_bytes + 1, 2);
    replacement_memory.add_array(values_bytes + 1, 2);
    std::string indices, values;
    indices.reserve(indices_bytes);
    values.reserve(values_bytes);
    indices = sparse_indices_;
    values = sparse_values_;
    replacement_memory.retain_at_most(indices.capacity() + values.capacity() +
                                      2);
    sparse_indices_.swap(indices);
    sparse_values_.swap(values);
    memory_.swap(replacement_memory);
  }

  //! Retrieve mutable sparse count
  uint32_t *mutable_sparse_count() {
    return &sparse_count_;
  }

  //! Retrieve mutable sparse indices
  std::string *mutable_sparse_indices() {
    return &sparse_indices_;
  }

  //! Retrieve mutable sparse values
  std::string *mutable_sparse_values() {
    return &sparse_values_;
  }

  //! Retrieve sparse count
  uint32_t sparse_count() const {
    return sparse_count_;
  }

  //! Retrieve sparse indices
  const std::string &sparse_indices() const {
    return sparse_indices_;
  }

  //! Retrieve sparse values
  const std::string &sparse_values() const {
    return sparse_values_;
  }

  // Transfer public string outputs together with their capacity reservation.
  void swap_buffers(uint32_t &count, std::string &indices, std::string &values,
                    ailego::MemoryReservation &memory) noexcept {
    std::swap(sparse_count_, count);
    sparse_indices_.swap(indices);
    sparse_values_.swap(values);
    memory_.swap(memory);
  }

 private:
  ailego::MemoryReservation memory_;
  uint32_t sparse_count_{0};
  std::string sparse_indices_{};
  std::string sparse_values_{};
};

/*! Index Document
 */
class IndexDocument {
 public:
  // Immutable after publication. Candidate heaps and returned documents share
  // one payload and one reservation, including copies retained by the caller.
  struct VectorPayload {
    VectorPayload(const void *src, size_t size) {
      if (size == SIZE_MAX) throw std::bad_alloc();
      auto &pool = ailego::MemoryLimitPool::get_instance();
      charge_ = pool.capacity() ? size + 1 : 0;
      if (charge_ && !pool.try_charge_workspace(charge_))
        throw std::bad_alloc();
      try {
        data.assign(static_cast<const char *>(src), size);
      } catch (...) {
        if (charge_) pool.release_external(charge_);
        throw;
      }
    }
    ~VectorPayload() {
      // Release the backing string before making its budget available again.
      std::string().swap(data);
      if (charge_)
        ailego::MemoryLimitPool::get_instance().release_external(charge_);
    }
    VectorPayload(const VectorPayload &) = delete;
    VectorPayload &operator=(const VectorPayload &) = delete;
    std::string data;

   private:
    size_t charge_{0};
  };

  //! Constructor
  IndexDocument() = default;

  //! Constructor
  IndexDocument(uint64_t k, float v) : key_(k), score_(v) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i)
      : key_(k), score_(v), index_(i) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, const void *vector)
      : key_(k), score_(v), index_(i), vector_(vector) {}

  IndexDocument(uint64_t k, float v, uint32_t i,
                IndexStorage::MemoryBlock vec_block)
      : key_(k), score_(v), index_(i), vec_mem_block_(std::move(vec_block)) {
    vector_ = vec_mem_block_.data();
    has_vec_mem_block_ = true;
  }

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, const void *vector,
                IndexSparseDocument sparse_doc)
      : key_(k),
        score_(v),
        index_(i),
        vector_(vector),
        sparse_doc_(std::move(sparse_doc)) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, std::string vector_string)
      : key_(k),
        score_(v),
        index_(i),
        vector_string_(std::move(vector_string)) {}

  IndexDocument(uint64_t k, float v, uint32_t i,
                std::shared_ptr<const VectorPayload> payload)
      : key_(k), score_(v), index_(i), vector_payload_(std::move(payload)) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, std::string vector_string,
                IndexSparseDocument sparse_doc)
      : key_(k),
        score_(v),
        index_(i),
        vector_{nullptr},
        vector_string_(std::move(vector_string)),
        sparse_doc_(std::move(sparse_doc)) {}

  IndexDocument(uint64_t k, float v, uint32_t i,
                IndexStorage::MemoryBlock vec_block,
                IndexSparseDocument sparse_doc)
      : key_(k),
        score_(v),
        index_(i),
        vec_mem_block_(std::move(vec_block)),
        sparse_doc_(std::move(sparse_doc)) {
    has_vec_mem_block_ = true;
    vector_ = vec_mem_block_.data();
  }

  //! Constructor
  IndexDocument(const IndexDocument &rhs)
      : key_(rhs.key_),
        score_(rhs.score_),
        index_(rhs.index_),
        vector_(rhs.vector_),
        vector_string_(rhs.vector_string_),
        vector_payload_(rhs.vector_payload_),
        sparse_doc_{rhs.sparse_doc_} {
    if (rhs.has_vec_mem_block_) {
      vec_mem_block_ = rhs.vec_mem_block_;
      has_vec_mem_block_ = true;
      vector_ = vec_mem_block_.data();
    }
  }

  IndexDocument(IndexDocument &&) noexcept = default;
  IndexDocument &operator=(IndexDocument &&) noexcept = default;

  //! Assignment
  IndexDocument &operator=(const IndexDocument &rhs) {
    if (this != &rhs) {
      key_ = rhs.key_;
      score_ = rhs.score_;
      index_ = rhs.index_;
      vector_ = rhs.vector_;
      vector_string_ = rhs.vector_string_;
      vector_payload_ = rhs.vector_payload_;
      vec_mem_block_ = rhs.vec_mem_block_;
      has_vec_mem_block_ = rhs.has_vec_mem_block_;
      if (has_vec_mem_block_) vector_ = vec_mem_block_.data();
      sparse_doc_ = rhs.sparse_doc_;
    }
    return *this;
  }

  //! Less than
  bool operator<(const IndexDocument &rhs) const {
    return (this->score_ < rhs.score_);
  }

  //! Greater than
  bool operator>(const IndexDocument &rhs) const {
    return (this->score_ > rhs.score_);
  }

  //! Retrieve primary key
  uint64_t key() const {
    return key_;
  }

  //! Retrieve score value
  float score() const {
    return score_;
  }

  //! Retrieve index id
  uint32_t index() const {
    return index_;
  }

  //! Retrieve vec
  const void *vector() const {
    return vector_;
  }

  //! Retrieve vec string
  const std::string &vector_string() const {
    return vector_payload_ ? vector_payload_->data : vector_string_;
  }

  //! Retrieve vec
  const IndexSparseDocument &sparse_doc() const {
    return sparse_doc_;
  }

  //! Retrieve mutable primary key
  uint64_t *mutable_key() {
    return &key_;
  }

  //! Retrieve mutable score value
  float *mutable_score() {
    return &score_;
  }

  //! Retrieve mutable index id
  uint32_t *mutable_index() {
    return &index_;
  }

  //! Retrieve primary key
  void set_key(uint64_t val) {
    key_ = val;
  }

  //! Retrieve score value
  void set_score(float val) {
    score_ = val;
  }

  //! Retrieve index id
  void set_index(uint32_t val) {
    index_ = val;
  }

 private:
  //! Data members
  uint64_t key_{0u};
  float score_{0.0f};
  uint32_t index_{0u};
  const void *vector_{nullptr};
  std::string vector_string_{};
  std::shared_ptr<const VectorPayload> vector_payload_{};
  bool has_vec_mem_block_{false};
  mutable IndexStorage::MemoryBlock vec_mem_block_{};
  IndexSparseDocument sparse_doc_{};
};

/*! Index Document Heap
 */
class IndexDocumentHeap : public ailego::Heap<IndexDocument> {
 public:
  //! Constructor
  IndexDocumentHeap() = default;

  //! Constructor
  IndexDocumentHeap(size_t max) : ailego::Heap<IndexDocument>(max) {}

  //! Constructor
  IndexDocumentHeap(size_t max, float val)
      : ailego::Heap<IndexDocument>(max), threshold_(val) {}

  //! Constructor
  IndexDocumentHeap(const IndexDocumentHeap &rhs) = default;

  //! Constructor
  IndexDocumentHeap(IndexDocumentHeap &&rhs)
      : ailego::Heap<IndexDocument>(std::move(rhs)),
        threshold_(rhs.threshold_) {}

  //! Constructor
  IndexDocumentHeap(const std::vector<IndexDocument> &rhs)
      : ailego::Heap<IndexDocument>(rhs) {}

  //! Constructor
  IndexDocumentHeap(std::vector<IndexDocument> &&rhs)
      : ailego::Heap<IndexDocument>(std::move(rhs)) {}

  //! Insert a document into the heap
  void emplace(uint64_t key, float score) {
    if (score <= threshold_) {
      ailego::Heap<IndexDocument>::emplace(key, score);
    }
  }

  //! Insert a document into the heap
  void emplace(uint64_t key, float score, uint32_t index) {
    if (score <= threshold_) {
      ailego::Heap<IndexDocument>::emplace(key, score, index);
    }
  }

  //! Set threshold for RNN
  void set_threshold(float val) {
    threshold_ = val;
  }

  //! Retrieve value of threshold for RNN
  float threshold() const {
    return threshold_;
  }

 private:
  //! members
  float threshold_{std::numeric_limits<float>::max()};
};

// Search scratch does not own vectors. Materialize IndexDocument only at the
// result boundary, rather than carrying its payload holders in every heap slot.
class IndexCandidate {
 public:
  IndexCandidate() = default;
  IndexCandidate(uint64_t key, float score, uint32_t index = 0)
      : key_(key), score_(score), index_(index) {}
  uint64_t key() const {
    return key_;
  }
  float score() const {
    return score_;
  }
  uint32_t index() const {
    return index_;
  }
  uint64_t *mutable_key() {
    return &key_;
  }
  float *mutable_score() {
    return &score_;
  }
  uint32_t *mutable_index() {
    return &index_;
  }
  void set_key(uint64_t key) {
    key_ = key;
  }
  bool operator<(const IndexCandidate &rhs) const {
    return score_ < rhs.score_;
  }
  operator IndexDocument() const {
    return IndexDocument(key_, score_, index_);
  }

 private:
  uint64_t key_{0};
  float score_{0};
  uint32_t index_{0};
};
static_assert(sizeof(IndexCandidate) == 16,
              "Candidate payload must stay compact");

template <typename Base>
class BasicIndexCandidateHeap : public Base {
 public:
  using Base::mutable_container;
  using Container = std::remove_reference_t<
      decltype(std::declval<Base &>().mutable_container())>;
  BasicIndexCandidateHeap() = default;
  explicit BasicIndexCandidateHeap(size_t max) {
    limit(max);
  }
  void emplace(uint64_t key, float score, uint32_t index = 0) {
    if (score <= threshold_) Base::emplace(key, score, index);
  }
  void set_threshold(float value) {
    threshold_ = value;
  }
  float threshold() const {
    return threshold_;
  }
  void clear() {
    auto &items = mutable_container();
    items.clear();
    // Retain small scratch for the hot path, not a rare query's high-water
    // mark.
    if (items.capacity() > 4096) Container().swap(items);
  }
  void release() {
    Container().swap(mutable_container());
  }
  void limit(size_t max) {
    // Do not reserve user-supplied topk before seeing any candidates.
    Base::limit(max, false);
  }

 private:
  float threshold_{std::numeric_limits<float>::max()};
};

using IndexCandidateHeap =
    BasicIndexCandidateHeap<ailego::BudgetedHeap<IndexCandidate>>;
using BudgetedIndexCandidateHeap = IndexCandidateHeap;

// Capacity belongs to the container, including empty reserved slots. Keep its
// reservation with the allocation across clear(), moves and caller handoff.
using IndexDocumentList = ailego::BudgetedVector<IndexDocument>;

class GroupIndexDocument {
 public:
  GroupIndexDocument() = default;

  GroupIndexDocument(const std::string &group_id, IndexDocumentList docs)
      : docs_(std::move(docs)) {
    set_group_id(group_id);
  }
  GroupIndexDocument(const GroupIndexDocument &other)
      : GroupIndexDocument(other.group_id_, other.docs_) {}
  GroupIndexDocument(GroupIndexDocument &&) noexcept = default;
  GroupIndexDocument &operator=(GroupIndexDocument other) noexcept {
    group_memory_.swap(other.group_memory_);
    group_id_.swap(other.group_id_);
    docs_.swap(other.docs_);
    return *this;
  }

  const std::string &group_id() const {
    return group_id_;
  }

  const IndexDocumentList &docs() const {
    return docs_;
  }

  std::string *mutable_group_id() {
    return &group_id_;
  }

  IndexDocumentList *mutable_docs() {
    return &docs_;
  }

  void set_group_id(const std::string &group_id) {
    ailego::MemoryReservation replacement_memory;
    replacement_memory.add_array(group_id.size() + 1, 2);
    std::string replacement(group_id);
    replacement_memory.retain_at_most(replacement.capacity() + 1);
    group_id_.swap(replacement);
    group_memory_.swap(replacement_memory);
  }

 private:
  ailego::MemoryReservation group_memory_;
  std::string group_id_;
  IndexDocumentList docs_;
};

/*! Index Document List
 */
using IndexGroupDocumentList = ailego::BudgetedVector<GroupIndexDocument>;

}  // namespace core
}  // namespace zvec

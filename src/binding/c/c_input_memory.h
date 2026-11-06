// Copyright 2026-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>
#include <zvec/db/doc_iterator.h>
#include <zvec/db/query.h>
#include "db/common/document_memory.h"
#include "db/common/result_memory_allowance.h"

namespace zvec::detail {

inline size_t CStringStorage(std::string_view text) {
  return text.empty() ? 0 : BindingMemoryBytes(1, text.size(), 32);
}

inline size_t CFieldsStorage(
    const std::optional<std::vector<std::string>> &fields) {
  if (!fields) return 0;
  size_t bytes = BindingMemoryBytes(fields->capacity(), sizeof(std::string));
  for (const auto &field : *fields)
    bytes = BindingMemoryBytes(1, CStringStorage(field), bytes);
  return bytes;
}

inline size_t CFtsStorage(const FtsClause &fts) {
  return BindingMemoryBytes(1, CStringStorage(fts.query_string_),
                            CStringStorage(fts.match_string_));
}

inline size_t CTargetStorage(const QueryTarget &target) {
  size_t bytes = CStringStorage(target.field_name_);
  if (const auto *vector = target.get_vector_clause()) {
    bytes = BindingMemoryBytes(1, CStringStorage(vector->query_vector_), bytes);
    bytes = BindingMemoryBytes(1, CStringStorage(vector->sparse_indices_), bytes);
    bytes = BindingMemoryBytes(1, CStringStorage(vector->sparse_values_), bytes);
  } else if (const auto *fts = target.get_fts_clause()) {
    bytes = BindingMemoryBytes(1, CFtsStorage(*fts), bytes);
  }
  return bytes;
}

inline size_t CInputPayload(const Doc &doc) {
  return DocumentMemoryAccess::copy_bytes(doc) - sizeof(Doc) - 256;
}

inline size_t CInputPayload(const FtsClause &fts) {
  return CFtsStorage(fts);
}

inline size_t CInputPayload(const SearchQuery &query) {
  return BindingMemoryBytes(
      1, CTargetStorage(query.target_),
      BindingMemoryBytes(1, CStringStorage(query.filter_),
                         CFieldsStorage(query.output_fields_)));
}

inline size_t CInputPayload(const GroupByVectorQuery &query) {
  return BindingMemoryBytes(
      1, CStringStorage(query.group_by_field_name_),
      BindingMemoryBytes(
          1, CTargetStorage(query.target_),
          BindingMemoryBytes(1, CStringStorage(query.filter_),
                             CFieldsStorage(query.output_fields_))));
}

inline size_t CInputPayload(const SubQuery &query) {
  return CTargetStorage(query.target_);
}

inline size_t CInputPayload(const MultiQuery &query) {
  size_t bytes = BindingMemoryBytes(
      query.queries.capacity(), sizeof(SubQuery),
      BindingMemoryBytes(1, CStringStorage(query.filter),
                         CFieldsStorage(query.output_fields)));
  for (const auto &sub : query.queries)
    bytes = BindingMemoryBytes(1, CInputPayload(sub), bytes);
  if (const auto *weights = std::get_if<reranker::WeightedParams>(&query.rerank))
    bytes = BindingMemoryBytes(weights->weights.capacity(), sizeof(double), bytes);
  return bytes;
}

inline size_t CInputPayload(const IteratorOptions &options) {
  return CFieldsStorage(options.output_fields_);
}

struct CInputMemory {
  ResultMemoryAllowance::Lease lease{&SharedResultMemoryAllowance()};
};

// The pointer passed through the C ABI is the native base subobject. Destroy
// its payload before the budget owner. Only owned handles use this wrapper.
template <typename T>
struct CInput final : CInputMemory, T {
  static constexpr size_t base_bytes = sizeof(T) + 256;

  template <typename... Args>
  explicit CInput(Args &&...args) : T(std::forward<Args>(args)...) {}

  size_t bytes() const {
    return BindingMemoryBytes(1, CInputPayload(*this), base_bytes);
  }

  template <typename... Args>
  static std::unique_ptr<CInput> make(size_t bytes, Args &&...args) {
    ResultMemoryAllowance::Lease admission(&SharedResultMemoryAllowance());
    admission.add(bytes);
    auto result = std::make_unique<CInput>(std::forward<Args>(args)...);
    result->lease.absorb(admission);
    return result;
  }

  static void destroy(T *value) {
    delete static_cast<CInput *>(value);
  }
};

template <typename T>
class CInputMutation {
 public:
  CInputMutation(T &value, size_t bytes) : input_(static_cast<CInput<T> &>(value)) {
    const size_t before = input_.bytes();
    const size_t tracked = input_.lease.bytes();
    const size_t existing = before > tracked ? before - tracked : 0;
    admission_.add(BindingMemoryBytes(1, existing, bytes), bytes);
  }

  void commit() {
    input_.lease.absorb(admission_);
    input_.lease.retain_at_most(input_.bytes());
  }

 private:
  CInput<T> &input_;
  ResultMemoryAllowance::Lease admission_{&SharedResultMemoryAllowance()};
};

}  // namespace zvec::detail

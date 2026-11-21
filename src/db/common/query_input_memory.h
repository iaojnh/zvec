// Copyright 2026-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>
#include <zvec/db/query.h>
#include "binding_memory.h"

namespace zvec::detail {

// Size the owned request copies that the native query paths keep while
// planning and executing. Vector views may become owning during sanitization.
inline size_t QueryInputCopyBytes(
    const QueryTarget &target, std::string_view filter,
    const std::optional<std::vector<std::string>> &fields,
    size_t bytes = 256, bool materialize_view = false) {
  const auto add = [&](std::string_view value) {
    if (!value.empty())
      bytes = BindingMemoryBytes(1, value.size(),
                                 BindingMemoryBytes(1, bytes, 32));
  };
  add(target.field_name_);
  add(filter);
  if (const auto *fts = target.get_fts_clause()) {
    add(fts->query_string_);
    add(fts->match_string_);
  } else if (materialize_view || target.get_vector_clause()) {
    if (const auto vector = target.get_vector_view()) {
      add(vector->query_vector_);
      add(vector->sparse_indices_);
      add(vector->sparse_values_);
    }
  }
  if (fields) {
    bytes = BindingMemoryBytes(fields->size(), sizeof(std::string), bytes);
    for (const auto &field : *fields) add(field);
  }
  return bytes;
}

}  // namespace zvec::detail

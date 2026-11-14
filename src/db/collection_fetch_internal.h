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

#include <zvec/db/collection.h>
#include "db/common/constants.h"

namespace arrow {
class MemoryPool;
class Table;
}  // namespace arrow

namespace zvec::internal {

// Arrays store raw allocator pointers. Keep both allocator layers alive until
// after every table is destroyed. This owner must accompany the tables; they
// must not escape it. Only selected rows, not pinned full row groups, are kept.
struct ForwardFetchResult {
  ForwardFetchResult() = default;
  ForwardFetchResult(const ForwardFetchResult &) = delete;
  ForwardFetchResult &operator=(const ForwardFetchResult &) = delete;

  std::shared_ptr<arrow::MemoryPool> parent;
  std::shared_ptr<arrow::MemoryPool> workspace;
  std::vector<std::shared_ptr<arrow::Table>> tables;
  bool batch_exceeded{false};
};

// Internal binding bridge, not part of the public Collection API. A binding
// can convert and release one small batch before fetching the next. Never run
// language-runtime callbacks while holding collection/segment locks.
struct FetchResultSnapshot {
  // The routing allowance outlives the native map during binding conversion.
  std::shared_ptr<void> routing_memory;
  DocPtrMap docs;
  std::shared_ptr<const CollectionSchema> schema;
  std::shared_ptr<ForwardFetchResult> forward;
};

// Capture values and their schema under the same read lock. For subsequent
// batches, retain the first snapshot's schema and pass its address here. A
// concurrent DDL fails the request instead of mixing schemas across batches.
Result<FetchResultSnapshot> fetch_result_snapshot(
    const Collection &collection, const std::vector<std::string> &pks,
    const std::optional<std::vector<std::string>> &output_fields,
    bool include_vector, const CollectionSchema *expected_schema = nullptr,
    bool prefer_columns = false);

}  // namespace zvec::internal

namespace zvec::detail {

// Keep fetch's has()-based error semantics: a failed point read is a missing
// result, not a failed fetch. A batch error can leave partially filled IDs, so
// retry every key using the original point lookup instead of consuming them.
template <typename Lookup>
std::vector<uint64_t> FetchPrimaryKeyIds(const Lookup &lookup,
                                         const std::vector<std::string> &keys) {
  std::vector<uint64_t> ids;
  if (keys.empty()) return ids;
  if (keys.size() > 1 && lookup.multi_get(keys, &ids).ok()) return ids;

  ids.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t id;
    ids[i] = lookup.has(keys[i], &id) ? id : INVALID_DOC_ID;
  }
  return ids;
}

// Preserve explicit lookup errors in the budgeted integration API.
template <typename Lookup>
Result<std::vector<uint64_t>> FetchPrimaryKeyIdsChecked(
    const Lookup &lookup, const std::vector<std::string> &keys) {
  std::vector<uint64_t> ids;
  if (keys.empty()) return ids;
  if (keys.size() > 1 && lookup.multi_get(keys, &ids).ok()) return ids;
  ids.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t id;
    auto found = lookup.lookup(keys[i], &id);
    if (!found) return tl::make_unexpected(found.error());
    ids[i] = *found ? id : INVALID_DOC_ID;
  }
  return ids;
}

}  // namespace zvec::detail

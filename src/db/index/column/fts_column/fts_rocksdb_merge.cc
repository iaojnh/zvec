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

#include "fts_rocksdb_merge.h"
#include <cstring>
#include <roaring/roaring.h>
#include <zvec/ailego/logger/logger.h>
#include "db/index/column/fts_column/posting/bitpacked_posting_list.h"
#include "db/index/column/posting_merge.h"

namespace zvec::fts {

bool FtsPostingsMerge::FullMergeV2(const MergeOperationInput &in,
                                   MergeOperationOutput *out) const {
  if (!in.existing_value && in.operand_list.size() == 1) {
    out->existing_operand = in.operand_list.front();
    return true;
  }
  if (in.existing_value && in.operand_list.empty()) {
    out->existing_operand = *in.existing_value;
    return true;
  }
  return detail::PostingMerge::merge(in.operand_list.data(),
                                     in.operand_list.size(), in.existing_value,
                                     false, &out->new_value);
}

bool FtsPostingsMerge::PartialMerge(const rocksdb::Slice &,
                                    const rocksdb::Slice &left,
                                    const rocksdb::Slice &right,
                                    std::string *out, rocksdb::Logger *) const {
  const rocksdb::Slice inputs[]{left, right};
  return detail::PostingMerge::merge(inputs, 2, nullptr, false, out);
}

// ============================================================
// FtsMaxTfMerge: uint32_t max merge
// ============================================================

bool FtsMaxTfMerge::FullMergeV2(const MergeOperationInput &merge_in,
                                MergeOperationOutput *merge_out) const {
  uint32_t max_tf = 0;

  if (merge_in.existing_value != nullptr &&
      merge_in.existing_value->size() >= sizeof(uint32_t)) {
    std::memcpy(&max_tf, merge_in.existing_value->data(), sizeof(uint32_t));
  }

  for (const auto &operand : merge_in.operand_list) {
    if (operand.size() >= sizeof(uint32_t)) {
      uint32_t operand_tf = 0;
      std::memcpy(&operand_tf, operand.data(), sizeof(uint32_t));
      if (operand_tf > max_tf) {
        max_tf = operand_tf;
      }
    }
  }

  merge_out->new_value.resize(sizeof(uint32_t));
  std::memcpy(merge_out->new_value.data(), &max_tf, sizeof(uint32_t));
  return true;
}

bool FtsMaxTfMerge::PartialMerge(const rocksdb::Slice & /*key*/,
                                 const rocksdb::Slice &left_operand,
                                 const rocksdb::Slice &right_operand,
                                 std::string *new_value,
                                 rocksdb::Logger * /*logger*/) const {
  if (left_operand.size() < sizeof(uint32_t) ||
      right_operand.size() < sizeof(uint32_t)) {
    LOG_ERROR(
        "FtsMaxTfMerge::PartialMerge: operand too small. "
        "left_size[%zu] right_size[%zu] expected[%zu]",
        left_operand.size(), right_operand.size(), sizeof(uint32_t));
    return false;
  }

  uint32_t left_tf = 0;
  uint32_t right_tf = 0;
  std::memcpy(&left_tf, left_operand.data(), sizeof(uint32_t));
  std::memcpy(&right_tf, right_operand.data(), sizeof(uint32_t));

  uint32_t max_tf = (left_tf > right_tf) ? left_tf : right_tf;
  new_value->resize(sizeof(uint32_t));
  std::memcpy(new_value->data(), &max_tf, sizeof(uint32_t));
  return true;
}

}  // namespace zvec::fts

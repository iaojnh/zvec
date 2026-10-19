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


#include <ailego/pattern/defer.h>
#include <rocksdb/merge_operator.h>
#include "db/index/column/posting_merge.h"
#include "inverted_codec.h"


namespace zvec {


class InvertedRocksdbValueMerger : public rocksdb::MergeOperator {
 public:
  bool FullMergeV2(const MergeOperationInput &in,
                   MergeOperationOutput *out) const override {
    if (!in.existing_value && in.operand_list.size() == 1) {
      out->existing_operand = in.operand_list.front();
      return true;
    }
    if (in.existing_value && in.operand_list.empty()) {
      out->existing_operand = *in.existing_value;
      return true;
    }
    return detail::PostingMerge::merge(
        in.operand_list.data(), in.operand_list.size(), in.existing_value, true,
        &out->new_value);
  }

  bool PartialMerge(const rocksdb::Slice &, const rocksdb::Slice &left,
                    const rocksdb::Slice &right, std::string *out,
                    rocksdb::Logger *) const override {
    const rocksdb::Slice inputs[]{left, right};
    return detail::PostingMerge::merge(inputs, 2, nullptr, true, out);
  }

  const char *Name() const override {
    return "InvertedRocksdbValueMerger";
  }
};


}  // namespace zvec

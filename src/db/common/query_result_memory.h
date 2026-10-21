// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <memory>
#include <arrow/api.h>
#include <zvec/db/doc.h>
#include "result_memory_allowance.h"

namespace zvec::detail {

// Arrow buffers are already managed. This separate allowance covers the
// ordinary-heap Doc copies, field containers and result-vector capacity.
// Keep its backing high water across calls, as with Python fetch results.
class QueryResultEstimate {
 public:
  void add(size_t bytes, size_t count = 1) {
    if (count && bytes > (std::numeric_limits<size_t>::max() - bytes_) / count)
      throw std::bad_alloc();
    bytes_ += bytes * count;
  }
  void value(const arrow::Array &array, int64_t row) {
    if (array.IsNull(row)) return;
    switch (array.type_id()) {
      case arrow::Type::STRING:
      case arrow::Type::BINARY: {
        const auto &binary = static_cast<const arrow::BinaryArray &>(array);
        add(static_cast<size_t>(binary.value_length(row)));
        break;
      }
      case arrow::Type::LIST: {
        const auto &list = static_cast<const arrow::ListArray &>(array);
        add(64);
        add(32, static_cast<size_t>(list.value_length(row)));
        const auto end = list.value_offset(row) + list.value_length(row);
        for (auto i = list.value_offset(row); i < end; ++i)
          value(*list.values(), i);
        break;
      }
      case arrow::Type::STRUCT: {
        const auto &fields = static_cast<const arrow::StructArray &>(array);
        for (const auto &field : fields.fields()) value(*field, row);
        break;
      }
      default:
        break;  // Fixed-width scalars live inside Doc::Value.
    }
  }
  void value(const arrow::Scalar &scalar) {
    if (!scalar.is_valid) return;
    switch (scalar.type->id()) {
      case arrow::Type::STRING:
      case arrow::Type::BINARY:
        add(static_cast<const arrow::BaseBinaryScalar &>(scalar).view().size());
        break;
      case arrow::Type::LIST: {
        const auto &array =
            *static_cast<const arrow::ListScalar &>(scalar).value;
        add(64);
        add(32, static_cast<size_t>(array.length()));
        for (int64_t i = 0; i < array.length(); ++i) value(array, i);
        break;
      }
      default:
        break;
    }
  }
  size_t bytes() const {
    return bytes_;
  }

 private:
  size_t bytes_{0};
};

inline size_t EstimateQueryResult(const arrow::RecordBatch &batch) {
  QueryResultEstimate estimate;
  // Include shared ownership, growing DocPtrList capacity and hash buckets.
  estimate.add(sizeof(Doc) + 256, static_cast<size_t>(batch.num_rows()));
  for (int column = 0; column < batch.num_columns(); ++column) {
    estimate.add(192 + batch.schema()->field(column)->name().size() * 2,
                 static_cast<size_t>(batch.num_rows()));
    for (int64_t row = 0; row < batch.num_rows(); ++row)
      estimate.value(*batch.column(column), row);
  }
  return estimate.bytes();
}

// Aliasing ownership keeps admission alive even if a C++ caller retains just
// one document after discarding the list. Destroy Docs before releasing it.
struct BudgetedQueryDocument {
  std::shared_ptr<ResultMemoryAllowance::Lease> memory;
  Doc doc;
};

inline Doc::Ptr MakeQueryDocument(
    const std::shared_ptr<ResultMemoryAllowance::Lease> &memory) {
  if (!memory) return std::make_shared<Doc>();
  auto owner = std::make_shared<BudgetedQueryDocument>();
  owner->memory = memory;
  auto *doc = &owner->doc;
  return Doc::Ptr(std::move(owner), doc);
}

}  // namespace zvec::detail

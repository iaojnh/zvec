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

#include "db/sqlengine/planner/doc_filter.h"
#include <optional>
#include <arrow/acero/exec_plan.h>
#include <arrow/table.h>
#include <zvec/ailego/logger/logger.h>
#include "db/index/storage/parquet_memory_pool.h"
#include "db/sqlengine/planner/invert_search.h"

namespace zvec::sqlengine {

Status DocFilter::compute_filter() {
  error_.store(StatusCode::OK, std::memory_order_relaxed);
  if (invert_cond_) {
    InvertSearch search(segment_.get());
    auto invert_res = search.exec_invert_cond_tree(invert_cond_.get());
    if (!invert_res) {
      return Status::InternalError("Execute invert search failed: ",
                                   invert_res.error().message());
    }
    invert_result_ = invert_res.value();
    invert_filter_ = invert_result_->make_filter();
  }

  if (forward_plan_) {
    auto *pool = segment_->uses_buffer_storage()
                     ? detail::GetParquetMemoryPool().get()
                     : arrow::default_memory_pool();
    auto forward_result =
        arrow::acero::DeclarationToTable(*forward_plan_, true, pool);
    if (!forward_result.ok()) {
      if (forward_result.status().IsOutOfMemory())
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      forward_result.status().ToString());
      return Status::InternalError("Execute filter bitmap failed: ",
                                   forward_result.status().ToString());
    }
    // has only one column with boolean type
    auto &forward_table = forward_result.ValueUnsafe();
    if (forward_table->num_columns() != 1 ||
        forward_table->column(0)->type() != arrow::boolean()) {
      return Status::InternalError("Filter bitmap is not boolean type");
    }
    forward_bitmap_ = forward_table->column(0);
  }

  if (forward_filter_expr_) {
    // get schema to bind to Expression
    auto table = segment_->fetch(query_info_->get_forward_filter_field_names(),
                                 std::vector<int>{});
    if (!table) {
      return Status::InternalError("Fetch forward failed");
    }
    auto bind_res = forward_filter_expr_->Bind(*table->schema());
    if (!bind_res.ok()) {
      return Status::InternalError("Bind forward filter expression failed",
                                   bind_res.status().ToString());
    }
    *forward_filter_expr_ = bind_res.MoveValueUnsafe();
  }
  return Status::OK();
}

bool DocFilter::empty() const {
  return !(delete_filter_ || invert_filter_ || forward_plan_ ||
           forward_filter_expr_);
}

Status DocFilter::status() const {
  const auto code = error_.load(std::memory_order_relaxed);
  return code == StatusCode::OK
             ? Status::OK()
             : Status(code, "Forward filter evaluation failed");
}

bool DocFilter::is_filtered(uint64_t id) const try {
  if (error_.load(std::memory_order_relaxed) != StatusCode::OK) return true;
  if (delete_filter_ && delete_filter_->is_filtered(id)) {
    return true;
  }
  if (invert_filter_ && invert_filter_->is_filtered(id)) {
    return true;
  }
  auto forward_bit = get_forward_bit(id);
  if (!forward_bit) {
    if (forward_filter_expr_ || forward_bitmap_) {
      if (error_.load(std::memory_order_relaxed) == StatusCode::OK)
        error_.store(StatusCode::INTERNAL_ERROR, std::memory_order_relaxed);
      return true;
    }
    return false;
  }
  // revert to return false if forward filter is matched
  return !forward_bit.value();
} catch (const std::bad_alloc &) {
  error_.store(StatusCode::RESOURCE_EXHAUSTED, std::memory_order_relaxed);
  return true;
}

std::optional<bool> DocFilter::get_forward_bit(uint64_t id) const {
  if (forward_filter_expr_) {
    return is_matched_by_forward_filter(id);
  }
  if (!forward_bitmap_) {
    return std::nullopt;
  }
  uint64_t rows_seen = 0;
  for (int c = 0; c < forward_bitmap_->num_chunks(); c++) {
    const auto &arr = forward_bitmap_->chunk(c);
    if (id < rows_seen + arr->length()) {
      auto *bool_array = static_cast<arrow::BooleanArray *>(arr.get());
      return (*bool_array)[id - rows_seen].value_or(false);
    }
    rows_seen += arr->length();
  }
  LOG_ERROR("ID is out or range: id[%zu] count[%zu]", (size_t)id,
            (size_t)rows_seen);
  return std::nullopt;
}

std::optional<std::vector<uint64_t>> DocFilter::get_bf_by_keys_and_update(
    float ratio, ailego::MemoryReservation &keys_memory) {
  // TODO: support forward
  if (!invert_result_) {
    return std::nullopt;
  }
  // Insert/flush() rewrite the block behind doc_count(), so read it locked.
  size_t doc_count = segment_->doc_count_snapshot();
  uint64_t bf_by_keys_threshold = static_cast<uint64_t>(doc_count * ratio);

  // decide to use brute force by keys or not
  if (size_t match_count = invert_result_->count();
      match_count <= bf_by_keys_threshold) {
    ailego::MemoryReservation reservation;
    std::vector<uint64_t> ids;
    try {
      reservation.add_array(match_count, sizeof(uint64_t));
      ids.reserve(match_count);
      // Widen directly into the sole owned candidate list, without a
      // temporary uint32 list that duplicates all matching document IDs.
      auto iter = invert_result_->create_iterator();
      for (; iter->valid(); iter->next()) ids.push_back(iter->doc_id());
      if (ids.size() != match_count) return std::nullopt;
    } catch (const std::bad_alloc &) {
      return std::nullopt;
    }
    keys_memory.swap(reservation);
    invert_filter_.reset();
    invert_result_.reset();
    LOG_INFO(
        "Use brute force by keys, doc_count[%zu] invert_result_count[%zu] "
        "ratio[%.4f]",
        doc_count, match_count, ratio);
    return ids;
  } else {
    LOG_DEBUG(
        "Not use brute force by keys, doc_count[%zu] invert_result_count[%zu] "
        "ratio[%.4f]",
        doc_count, match_count, ratio);
  }
  return std::nullopt;
}

std::optional<bool> DocFilter::is_matched_by_forward_filter(uint64_t id) const {
  auto exec_batch =
      segment_->fetch(query_info_->get_forward_filter_field_names(), id);
  if (!exec_batch) {
    LOG_ERROR("Fetch forward failed, id[%zu]", (size_t)id);
    return std::nullopt;
  }
  auto *pool = segment_->uses_buffer_storage()
                   ? detail::GetParquetMemoryPool().get()
                   : arrow::default_memory_pool();
  arrow::compute::ExecContext context(pool);
  auto maybe_result = arrow::compute::ExecuteScalarExpression(
      *forward_filter_expr_, *exec_batch, &context);
  if (!maybe_result.ok()) {
    error_.store(maybe_result.status().IsOutOfMemory()
                     ? StatusCode::RESOURCE_EXHAUSTED
                     : StatusCode::INTERNAL_ERROR,
                 std::memory_order_relaxed);
    LOG_ERROR("Execute scalar expression failed, id[%zu] err[%s]", (size_t)id,
              maybe_result.status().ToString().c_str());
    return std::nullopt;
  }
  arrow::Datum datum = maybe_result.MoveValueUnsafe();
  if (datum.is_scalar()) {
    const auto &scalar = datum.scalar_as<arrow::BooleanScalar>();
    return scalar.is_valid && scalar.value;
  }
  LOG_ERROR("Datum is not scalar, id[%zu] type[%s]", (size_t)id,
            datum.type()->ToString().c_str());
  return std::nullopt;
}


}  // namespace zvec::sqlengine

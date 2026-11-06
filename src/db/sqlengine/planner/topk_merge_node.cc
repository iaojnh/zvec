// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "topk_merge_node.h"
#include <array>
#include <numeric>
#include <arrow/compute/api.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include "db/index/segment/segment_fetch_internal.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/ops/fetch_vector_op.h"

namespace zvec::sqlengine {

TopKMergeNode::TopKMergeNode(std::vector<PlanInfo::Ptr> plans,
                             std::vector<Segment::Ptr> segments,
                             QueryInfo::Ptr output, bool descending,
                             ailego::ThreadPool *pool)
    : plans_(std::move(plans)),
      segments_(std::move(segments)),
      output_(std::move(output)),
      candidates_(output_->query_topn(), descending),
      pool_(pool) {
  if (output_->group_by()) {
    groups_ = std::make_unique<GroupTopKCandidates>(
        output_->group_by()->group_count, output_->group_by()->group_topk,
        descending);
  }
  auto table = internal::fetch_table(
      *segments_.front(), output_->get_selected_scalar_field_names(), {});
  if (table.status().IsOutOfMemory()) throw std::bad_alloc();
  if (!table.ok()) throw std::runtime_error(table.status().ToString());
  schema_ =
      Util::append_field(*(*table)->schema(), FIELD_SCORE, arrow::float32());
  // Match the existing vector-recall output contract. FTS has scalar output.
  if (output_->vector_cond_info() && output_->is_include_vector()) {
    for (const auto &field : output_->selected_vector_fields()) {
      schema_ = Util::append_field(*schema_, field.field_name,
                                   field.field_schema_ptr->is_dense_vector()
                                       ? arrow::binary()
                                       : Util::sparse_type());
    }
  }
  if (groups_)
    schema_ = Util::append_field(*schema_, FIELD_GROUP_ID, arrow::utf8());
}

arrow::Status TopKMergeNode::collect(size_t segment,
                                     arrow::RecordBatchReader *reader) {
  uint64_t ordinal = 0;
  for (;;) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
    if (!batch) break;
    auto rows = batch->GetColumnByName(LOCAL_ROW_ID);
    auto scores = batch->GetColumnByName(FIELD_SCORE);
    if (!rows || !scores || rows->type_id() != arrow::Type::INT32 ||
        scores->type_id() != arrow::Type::FLOAT || rows->null_count() ||
        scores->null_count())
      return arrow::Status::Invalid("Invalid rank-only result batch");
    const auto &row_array = static_cast<const arrow::Int32Array &>(*rows);
    const auto &score_array = static_cast<const arrow::FloatArray &>(*scores);
    auto names = batch->GetColumnByName(FIELD_GROUP_ID);
    if (groups_ && (!names || names->type_id() != arrow::Type::STRING))
      return arrow::Status::Invalid("Missing candidate group ID");
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (row_array.Value(i) < 0)
        return arrow::Status::Invalid("Negative local row ID");
      RankedDoc row{segment, ordinal++, row_array.Value(i),
                    score_array.Value(i)};
      if (groups_) {
        const auto &strings = static_cast<const arrow::StringArray &>(*names);
        if (!strings.IsNull(i)) groups_->offer(strings.GetView(i), row);
      } else
        candidates_.offer(row);
    }
  }
  return reader->Close();
}

arrow::Status TopKMergeNode::prepare() {
  if (!groups_ && !output_->query_topn()) return arrow::Status::OK();
  // Keep only a window of searches in flight. Each child reader has explicit
  // backpressure; completed plans are released, including their index results.
  for (size_t end = plans_.size(); end > 0;) {
    const size_t count = std::min(std::max(size_t{1}, pool_->count()), end);
    ailego::BudgetedVector<Result<std::unique_ptr<arrow::RecordBatchReader>>>
        readers(count);
    auto start = [&](size_t i) {
      try {
        readers[i] = plans_[end - 1 - i]->execute_to_reader(64 * 1024);
      } catch (const std::bad_alloc &) {
        readers[i] = tl::make_unexpected(Status(
            StatusCode::RESOURCE_EXHAUSTED, "TopK reader exceeds budget"));
      } catch (const std::exception &error) {
        readers[i] = tl::make_unexpected(Status::InternalError(error.what()));
      }
    };
    // Preserve per-segment search parallelism without launching every segment.
    // A caller already on this pool must not block waiting for its own workers.
    if (pool_->indexof_this() >= 0) {
      for (size_t i = 0; i < count; ++i) start(i);
    } else {
      auto group = pool_->make_group();
      try {
        for (size_t i = 0; i < count; ++i) group->execute(start, i);
      } catch (...) {
        group->wait_finish();
        throw;
      }
      group->wait_finish();
    }
    for (size_t i = 0; i < count; ++i) {
      const size_t segment = end - 1 - i;
      if (!readers[i]) {
        const auto &error = readers[i].error();
        if (error.code() == StatusCode::RESOURCE_EXHAUSTED)
          return arrow::Status::OutOfMemory(error.message());
        return arrow::Status::ExecutionError(error.message());
      }
      ARROW_RETURN_NOT_OK(collect(segment, readers[i].value().get()));
      readers[i].value().reset();
      plans_[segment].reset();
    }
    end -= count;
  }
  if (groups_)
    groups_->finish();
  else
    candidates_.finish();
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> TopKMergeNode::fetch(
    size_t segment, const std::vector<int> &rows,
    std::shared_ptr<arrow::Array> scores) const {
  ARROW_ASSIGN_OR_RAISE(
      auto table, internal::fetch_table(
                      *segments_[segment],
                      output_->get_selected_scalar_field_names(), rows));
  ARROW_ASSIGN_OR_RAISE(
      auto batch,
      table->CombineChunksToBatch(segments_[segment]->query_memory_pool()));
  ARROW_ASSIGN_OR_RAISE(
      batch,
      batch->AddColumn(batch->num_columns(), FIELD_SCORE, std::move(scores)));
  if (output_->vector_cond_info() && output_->is_include_vector()) {
    for (const auto &field : output_->selected_vector_fields()) {
      auto values = field.field_schema_ptr->is_dense_vector()
                        ? FetchVectorOp::fetch_dense_vector(
                              *segments_[segment], field.field_name, rows)
                        : FetchVectorOp::fetch_sparse_vector(
                              *segments_[segment], field.field_name, rows);
      if (!values) {
        if (values.error().code() == StatusCode::RESOURCE_EXHAUSTED)
          return arrow::Status::OutOfMemory(values.error().message());
        return arrow::Status::ExecutionError(values.error().message());
      }
      ARROW_ASSIGN_OR_RAISE(
          batch, batch->AddColumn(batch->num_columns(), field.field_name,
                                  std::move(values.value())));
    }
  }
  return batch;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
TopKMergeNode::materialize() {
  auto *memory_pool = segments_.front()->query_memory_pool();
  constexpr size_t kBatchRows = 1024;
  const auto &winners = this->winners();
  const size_t count = std::min(kBatchRows, winners.size() - offset_);
  std::array<size_t, kBatchRows> order;
  std::iota(order.begin(), order.begin() + count, 0);
  std::sort(order.begin(), order.begin() + count, [&](size_t a, size_t b) {
    return winners[offset_ + a].segment < winners[offset_ + b].segment;
  });
  std::array<uint32_t, kBatchRows> inverse;
  ailego::MemoryReservation batch_memory;
  ailego::MemoryReservation row_memory;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::vector<int> rows;
  ailego::ReserveVectorCapacity(batches, batch_memory,
                                std::min(count, segments_.size()));
  ailego::ReserveVectorCapacity(rows, row_memory, count);
  for (size_t begin = 0; begin < count;) {
    const size_t segment = winners[offset_ + order[begin]].segment;
    rows.clear();
    arrow::FloatBuilder scores(memory_pool);
    arrow::StringBuilder group_ids(memory_pool);
    size_t end = begin;
    while (end < count && winners[offset_ + order[end]].segment == segment) {
      const auto &winner = winners[offset_ + order[end]];
      rows.push_back(winner.row);
      ARROW_RETURN_NOT_OK(scores.Append(winner.score));
      if (groups_)
        ARROW_RETURN_NOT_OK(group_ids.Append(groups_->group(winner.group)));
      inverse[order[end]] = static_cast<uint32_t>(end);
      ++end;
    }
    ARROW_ASSIGN_OR_RAISE(auto score_array, scores.Finish());
    ARROW_ASSIGN_OR_RAISE(auto batch,
                          fetch(segment, rows, std::move(score_array)));
    if (groups_) {
      ARROW_ASSIGN_OR_RAISE(auto labels, group_ids.Finish());
      ARROW_ASSIGN_OR_RAISE(batch, batch->AddColumn(batch->num_columns(),
                                                    FIELD_GROUP_ID, labels));
    }
    batches.push_back(std::move(batch));
    begin = end;
  }
  ARROW_ASSIGN_OR_RAISE(auto table,
                        arrow::Table::FromRecordBatches(schema_, batches));
  arrow::UInt32Builder positions(memory_pool);
  ARROW_RETURN_NOT_OK(positions.AppendValues(inverse.data(), count));
  ARROW_ASSIGN_OR_RAISE(auto indices, positions.Finish());
  arrow::compute::ExecContext context(memory_pool);
  ARROW_ASSIGN_OR_RAISE(
      auto sorted,
      arrow::compute::Take(table, indices,
                           arrow::compute::TakeOptions::Defaults(), &context));
  ARROW_ASSIGN_OR_RAISE(auto result,
                        sorted.table()->CombineChunksToBatch(memory_pool));
  offset_ += count;
  return result;
}

arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>>
TopKMergeNode::gen() {
  return [self = shared_from_this()]()
             -> arrow::Future<std::optional<arrow::compute::ExecBatch>> {
    using Batch = std::optional<arrow::compute::ExecBatch>;
    try {
      if (!self->prepared_) {
        auto status = self->prepare();
        self->plans_.clear();
        if (!status.ok()) return arrow::Future<Batch>::MakeFinished(status);
        self->prepared_ = true;
      }
      if (self->offset_ == self->winners().size())
        return arrow::Future<Batch>::MakeFinished(std::nullopt);
      auto result = self->materialize();
      if (!result.ok())
        return arrow::Future<Batch>::MakeFinished(result.status());
      return arrow::Future<Batch>::MakeFinished(
          arrow::compute::ExecBatch(**result));
    } catch (const std::bad_alloc &) {
      self->plans_.clear();
      return arrow::Future<Batch>::MakeFinished(
          arrow::Status::OutOfMemory("Global topK workspace exceeds budget"));
    } catch (const std::exception &error) {
      self->plans_.clear();
      return arrow::Future<Batch>::MakeFinished(
          arrow::Status::ExecutionError(error.what()));
    }
  };
}

}  // namespace zvec::sqlengine

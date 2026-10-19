// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <arrow/util/async_generator.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include "db/index/segment/segment.h"
#include "db/sqlengine/analyzer/query_info.h"
#include "db/sqlengine/planner/group_topk_candidates.h"
#include "db/sqlengine/planner/plan_info.h"
#include "db/sqlengine/planner/topk_candidates.h"

namespace zvec::sqlengine {

// Rank-only per-segment plans feed a bounded global heap. Output fields are
// fetched only for the winners, in small batches, after selection is complete.
class TopKMergeNode : public std::enable_shared_from_this<TopKMergeNode> {
 public:
  TopKMergeNode(std::vector<PlanInfo::Ptr> plans,
                std::vector<Segment::Ptr> segments, QueryInfo::Ptr output,
                bool descending, ailego::ThreadPool *pool);

  std::shared_ptr<arrow::Schema> schema() const {
    return schema_;
  }
  arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>> gen();

 private:
  arrow::Status prepare();
  arrow::Status collect(size_t segment, arrow::RecordBatchReader *reader);
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> materialize();
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> fetch(
      size_t segment, const std::vector<int> &rows,
      std::shared_ptr<arrow::Array> scores) const;

  std::vector<PlanInfo::Ptr> plans_;
  std::vector<Segment::Ptr> segments_;
  QueryInfo::Ptr output_;
  TopKCandidates candidates_;
  std::unique_ptr<GroupTopKCandidates> groups_;
  const auto &winners() const {
    return groups_ ? groups_->rows() : candidates_.rows();
  }
  ailego::ThreadPool *pool_;
  std::shared_ptr<arrow::Schema> schema_;
  size_t offset_{0};
  bool prepared_{false};
};

}  // namespace zvec::sqlengine

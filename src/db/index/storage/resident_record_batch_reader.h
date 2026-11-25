// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <arrow/record_batch.h>
#include <arrow/result.h>

namespace zvec {

// Optional preparation may consume decoded buffers, but must not start I/O or
// decoding. A false result leaves the next logical batch for normal ReadNext;
// wrappers may retain already obtained child batches until then. True with a
// null batch means EOF. The usual memory admission still applies to new views.
class ResidentRecordBatchReader : public arrow::RecordBatchReader {
 public:
  virtual arrow::Result<bool> ReadNextResident(
      std::shared_ptr<arrow::RecordBatch> *batch) = 0;
};

inline arrow::Result<bool> ReadNextResident(
    arrow::RecordBatchReader &reader,
    std::shared_ptr<arrow::RecordBatch> *batch) {
  *batch = nullptr;
  auto *resident = dynamic_cast<ResidentRecordBatchReader *>(&reader);
  if (!resident) return false;
  return resident->ReadNextResident(batch);
}

}  // namespace zvec

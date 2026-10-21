// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "db/common/metadata_memory.h"
#include "db/common/result_memory_allowance.h"
#include "db/common/write_buffer_reservation.h"

namespace zvec::detail {

// Unit tests deliberately republish different process budgets. Production
// retains these allowances across close/reopen and must never call this.
// Only discard idle planning allowances, never active result/metadata owners.
class MemoryBudgetTestPeer {
 public:
  static bool reset_idle_allowances() {
    auto &results = SharedResultMemoryAllowance();
    auto &metadata = RetainedMetadataBudget::instance();
    auto &writers = SharedWriteBufferMemoryAllowance();
    std::scoped_lock lock(results.materialization_mutex_, results.mutex_,
                          metadata.mutex_, writers.mutex_);
    if (results.active_ || metadata.active_ || writers.active_ ||
        writers.active_backing_)
      return false;
    results.reservation_.retain_at_most(0);
    writers.reservation_.retain_at_most(0);
    writers.retained_ = 0;
    if (metadata.reserved_) {
      ailego::MemoryLimitPool::get_instance().release_metadata(
          metadata.reserved_);
      metadata.reserved_ = 0;
    }
    return true;
  }
};

}  // namespace zvec::detail

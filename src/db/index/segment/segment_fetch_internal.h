// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "segment.h"

namespace zvec::internal {

// Write-side pre-reads must distinguish missing documents from read failures.
// Keep the nullable public fetch API unchanged.
Result<Doc::Ptr> fetch_document(Segment &segment, uint64_t doc_id);

// Preserve the failure category when SQL materializes a forward table.
arrow::Result<TablePtr> fetch_table(const Segment &segment,
                                    const std::vector<std::string> &columns,
                                    const std::vector<int> &segment_doc_ids);

// Internal bridge without changing Segment's virtual/public interface. The
// caller must retain memory_pool until the returned table is destroyed.
Result<TablePtr> fetch_forward_batch(
    Segment &segment, const std::vector<uint64_t> &doc_ids,
    const std::optional<std::vector<std::string>> &output_fields,
    arrow::MemoryPool *memory_pool);

}  // namespace zvec::internal

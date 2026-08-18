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

#include <cstdint>
#include <vector>
#include "diskann_file_reader.h"

namespace zvec {
namespace core {

// Trace capture is opt-in. Set ZVEC_DISKANN_IO_TRACE to an output CSV path.
// Records are buffered in memory and written once when the process exits so
// file logging does not perturb the measured search I/O path.
bool diskann_io_trace_enabled();

uint64_t diskann_io_trace_begin_query();

void diskann_io_trace_record_batch(
    uint64_t query_id, uint32_t batch_id,
    const std::vector<AlignedRead> &read_requests);

#if defined(_WIN32) || defined(_WIN64)
// Diagnostic-only control run. When ZVEC_DISKANN_IO_CONTEXT_REPLAY names a
// captured trace, the first completed real search is followed by a pure I/O
// replay through that search's reader, IOContext, file handle, and sector
// buffer. The real search diagnostics are restored afterwards.
int diskann_io_context_replay_once(AlignedFileReader &reader, IOContext &io_ctx,
                                   void *sector_buffer,
                                   size_t sector_buffer_size,
                                   size_t read_stride);
#endif

}  // namespace core
}  // namespace zvec

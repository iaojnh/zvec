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

#include "diskann_io_trace.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace zvec {
namespace core {
namespace {

constexpr uint64_t kDefaultMaxRecords = 2000000;

struct TraceRecord {
  uint64_t query_id;
  uint32_t batch_id;
  uint64_t offset;
  uint64_t length;
};

uint64_t trace_record_limit() {
  const char *value = std::getenv("ZVEC_DISKANN_IO_TRACE_MAX_RECORDS");
  if (value == nullptr || value[0] == '\0') {
    return kDefaultMaxRecords;
  }

  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    return kDefaultMaxRecords;
  }
  return static_cast<uint64_t>(parsed);
}

class TraceWriter {
 public:
  TraceWriter() : max_records_(trace_record_limit()) {
    const char *path = std::getenv("ZVEC_DISKANN_IO_TRACE");
    if (path != nullptr && path[0] != '\0') {
      path_ = path;
      records_.reserve(static_cast<size_t>(
          std::min<uint64_t>(max_records_, static_cast<uint64_t>(100000))));
    }
  }

  ~TraceWriter() {
    if (!path_.empty()) {
      flush();
    }
  }

  bool enabled() const {
    return !path_.empty();
  }

  uint64_t begin_query() {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_query_id_++;
  }

  void record_batch(uint64_t query_id, uint32_t batch_id,
                    const std::vector<AlignedRead> &read_requests) {
    if (read_requests.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (truncated_) {
      return;
    }
    const uint64_t record_count = static_cast<uint64_t>(records_.size());
    const uint64_t remaining =
        max_records_ - std::min(max_records_, record_count);
    if (static_cast<uint64_t>(read_requests.size()) > remaining) {
      truncated_ = true;
      return;
    }

    for (const AlignedRead &request : read_requests) {
      records_.push_back(
          TraceRecord{query_id, batch_id, request.offset, request.len});
    }
  }

 private:
  void flush() {
    std::ofstream output(path_, std::ios::out | std::ios::trunc);
    if (!output) {
      std::cerr << "Failed to write DiskANN I/O trace: " << path_ << '\n';
      return;
    }

    output << "# zvec_diskann_io_trace_v1\n"
           << "# truncated=" << (truncated_ ? "true" : "false") << '\n'
           << "query_id,batch_id,offset,length\n";
    for (const TraceRecord &record : records_) {
      output << record.query_id << ',' << record.batch_id << ','
             << record.offset << ',' << record.length << '\n';
    }
  }

  std::string path_;
  uint64_t max_records_;
  uint64_t next_query_id_{0};
  bool truncated_{false};
  std::mutex mutex_;
  std::vector<TraceRecord> records_;
};

TraceWriter &trace_writer() {
  static TraceWriter writer;
  return writer;
}

}  // namespace

bool diskann_io_trace_enabled() {
  return trace_writer().enabled();
}

uint64_t diskann_io_trace_begin_query() {
  return trace_writer().begin_query();
}

void diskann_io_trace_record_batch(
    uint64_t query_id, uint32_t batch_id,
    const std::vector<AlignedRead> &read_requests) {
  trace_writer().record_batch(query_id, batch_id, read_requests);
}

}  // namespace core
}  // namespace zvec

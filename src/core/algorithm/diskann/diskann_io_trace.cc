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
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <zvec/ailego/logger/logger.h>

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

#if defined(_WIN32) || defined(_WIN64)

struct ReplayRead {
  uint64_t offset;
  uint64_t length;
};

struct ReplayBatch {
  uint64_t query_id;
  uint64_t batch_id;
  std::vector<ReplayRead> reads;
};

bool parse_unsigned(const std::string &text, uint64_t &value) {
  if (text.empty()) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    return false;
  }
  value = static_cast<uint64_t>(parsed);
  return true;
}

uint32_t replay_seconds(const char *name, uint32_t default_value) {
  const char *text = std::getenv(name);
  if (text == nullptr || text[0] == '\0') {
    return default_value;
  }
  uint64_t parsed = 0;
  if (!parse_unsigned(text, parsed) || parsed > 3600) {
    LOG_WARN("Ignoring invalid %s=%s", name, text);
    return default_value;
  }
  return static_cast<uint32_t>(parsed);
}

int load_replay_trace(const std::string &path,
                      std::vector<ReplayBatch> &batches,
                      size_t sector_buffer_size, size_t read_stride) {
  std::ifstream input(path);
  if (!input) {
    LOG_ERROR("Failed to open DiskAnn context replay trace: %s", path.c_str());
    return IndexError_InvalidArgument;
  }
  if (read_stride == 0 || sector_buffer_size < read_stride) {
    LOG_ERROR("Invalid DiskAnn context replay buffer geometry");
    return IndexError_InvalidArgument;
  }

  const size_t max_batch_size = sector_buffer_size / read_stride;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#' ||
        line == "query_id,batch_id,offset,length") {
      continue;
    }

    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
      fields.push_back(field);
    }
    uint64_t query_id = 0;
    uint64_t batch_id = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    if (fields.size() != 4 || !parse_unsigned(fields[0], query_id) ||
        !parse_unsigned(fields[1], batch_id) ||
        !parse_unsigned(fields[2], offset) ||
        !parse_unsigned(fields[3], length)) {
      LOG_ERROR("Invalid DiskAnn context replay trace row at line %zu",
                line_number);
      return IndexError_InvalidArgument;
    }
    if (length == 0 || length > read_stride ||
        offset % DiskAnnUtil::kSectorSize != 0 ||
        length % DiskAnnUtil::kSectorSize != 0) {
      LOG_ERROR("Invalid DiskAnn context replay read at line %zu", line_number);
      return IndexError_InvalidArgument;
    }

    if (batches.empty() || batches.back().query_id != query_id ||
        batches.back().batch_id != batch_id) {
      batches.push_back(ReplayBatch{query_id, batch_id, {}});
    }
    ReplayBatch &batch = batches.back();
    if (batch.reads.size() >= max_batch_size) {
      LOG_ERROR("DiskAnn context replay batch exceeds the sector buffer");
      return IndexError_InvalidArgument;
    }
    batch.reads.push_back(ReplayRead{offset, length});
  }

  if (batches.empty()) {
    LOG_ERROR("DiskAnn context replay trace contains no reads");
    return IndexError_InvalidArgument;
  }
  return 0;
}

int replay_until(AlignedFileReader &reader, IOContext &io_ctx,
                 std::vector<std::vector<AlignedRead>> &batches,
                 std::chrono::steady_clock::time_point deadline,
                 size_t &batch_index) {
  std::vector<uint32_t> completed;
  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<AlignedRead> &requests = batches[batch_index];
    batch_index = (batch_index + 1) % batches.size();
    PendingBatch pending;
    int ret = reader.submit(pending, requests, io_ctx);
    if (ret != 0) {
      LOG_ERROR("DiskAnn context replay submit failed, ret=%d", ret);
      return ret;
    }
    while (pending.n_reaped < pending.n_submitted) {
      ret = reader.get_completed(pending, io_ctx, 1, completed);
      if (ret < 0) {
        LOG_ERROR("DiskAnn context replay completion failed, ret=%d", ret);
        return ret;
      }
    }
  }
  return 0;
}

int measure_context_replay(AlignedFileReader &reader, IOContext &io_ctx,
                           std::vector<std::vector<AlignedRead>> &batches,
                           size_t sector_buffer_size, size_t read_stride,
                           const char *context_name) {
  if (io_ctx == nullptr || io_ctx == reinterpret_cast<IOContext>(-1) ||
      context_name == nullptr) {
    LOG_ERROR("DiskAnn context replay received an invalid search context");
    return IndexError_InvalidArgument;
  }

  const bool saved_diagnostics_enabled = io_ctx->diagnostics_enabled;
  const IoBackend::IocpDiagnostics saved_diagnostics = io_ctx->diagnostics;
  auto restore_diagnostics = [&]() {
    io_ctx->diagnostics = saved_diagnostics;
    io_ctx->diagnostics_enabled = saved_diagnostics_enabled;
  };

  const uint32_t warmup_seconds =
      replay_seconds("ZVEC_DISKANN_IO_CONTEXT_REPLAY_WARMUP_SECONDS", 2);
  const uint32_t measurement_seconds =
      replay_seconds("ZVEC_DISKANN_IO_CONTEXT_REPLAY_SECONDS", 10);
  size_t batch_index = 0;
  io_ctx->diagnostics_enabled = false;
  int ret = replay_until(
      reader, io_ctx, batches,
      std::chrono::steady_clock::now() + std::chrono::seconds(warmup_seconds),
      batch_index);
  if (ret != 0) {
    restore_diagnostics();
    return ret;
  }

  io_ctx->diagnostics = IoBackend::IocpDiagnostics{};
  io_ctx->diagnostics_enabled = true;
  const auto measurement_start = std::chrono::steady_clock::now();
  ret = replay_until(
      reader, io_ctx, batches,
      measurement_start + std::chrono::seconds(measurement_seconds),
      batch_index);
  const auto measurement_end = std::chrono::steady_clock::now();
  if (ret != 0) {
    restore_diagnostics();
    return ret;
  }

  const IoBackend::IocpDiagnostics replay = io_ctx->diagnostics;
  restore_diagnostics();
  const double elapsed_seconds =
      std::chrono::duration<double>(measurement_end - measurement_start)
          .count();
  const auto average = [](uint64_t total, uint64_t count) {
    return count == 0 ? 0.0
                      : static_cast<double>(total) / static_cast<double>(count);
  };
  const double reads_per_batch =
      average(replay.submitted_reads, replay.batch_count);
  const double iops =
      elapsed_seconds == 0.0
          ? 0.0
          : static_cast<double>(replay.submitted_reads) / elapsed_seconds;
  const double pending_ratio =
      100.0 * average(replay.pending_reads, replay.submitted_reads);

  LOG_INFO(
      "DiskAnn in-process IOCP replay: phase=after_search, context=%s, "
      "batches=%llu, reads=%llu, reads/batch=%.2f, iops=%.2f, "
      "pending_ratio=%.2f%%, max_outstanding=%u, submit_us/batch=%.2f, "
      "first_completion_us=%.2f, batch_duration_us=%.2f, "
      "iocp_wait_us/batch=%.2f, readfile_submit_us/read=%.2f, "
      "get_overlapped_us/read=%.2f, completions/dequeue=%.2f, "
      "max_dequeued_once=%u, buffer_bytes=%zu, read_stride=%zu",
      context_name, static_cast<unsigned long long>(replay.batch_count),
      static_cast<unsigned long long>(replay.submitted_reads), reads_per_batch,
      iops, pending_ratio, replay.max_outstanding,
      average(replay.batch_submit_ns, replay.batch_count) / 1000.0,
      average(replay.batch_first_completion_ns, replay.batch_count) / 1000.0,
      average(replay.batch_duration_ns, replay.batch_count) / 1000.0,
      average(replay.wait_us, replay.batch_count),
      average(replay.readfile_submit_ns, replay.submitted_reads) / 1000.0,
      average(replay.get_overlapped_ns, replay.submitted_reads) / 1000.0,
      average(replay.dequeued_reads, replay.dequeue_calls),
      replay.max_dequeued_once, sector_buffer_size, read_stride);
  return 0;
}

int run_context_replay(AlignedFileReader &reader, IOContext &io_ctx,
                       void *sector_buffer, size_t sector_buffer_size,
                       size_t read_stride, const std::string &trace_path) {
  if (io_ctx == nullptr || io_ctx == reinterpret_cast<IOContext>(-1) ||
      sector_buffer == nullptr) {
    LOG_ERROR("DiskAnn context replay received an invalid search context");
    return IndexError_InvalidArgument;
  }

  std::vector<ReplayBatch> trace_batches;
  int ret = load_replay_trace(trace_path, trace_batches, sector_buffer_size,
                              read_stride);
  if (ret != 0) {
    return ret;
  }

  auto *buffer = static_cast<uint8_t *>(sector_buffer);
  std::vector<std::vector<AlignedRead>> batches;
  batches.reserve(trace_batches.size());
  for (const ReplayBatch &trace_batch : trace_batches) {
    std::vector<AlignedRead> requests;
    requests.reserve(trace_batch.reads.size());
    for (size_t index = 0; index < trace_batch.reads.size(); ++index) {
      const ReplayRead &read = trace_batch.reads[index];
      requests.emplace_back(read.offset, read.length,
                            buffer + index * read_stride);
    }
    batches.push_back(std::move(requests));
  }

  ret = measure_context_replay(reader, io_ctx, batches, sector_buffer_size,
                               read_stride, "used");
  if (ret != 0) {
    return ret;
  }

  IOContext fresh_io_ctx = nullptr;
  ret = setup_io_ctx(fresh_io_ctx);
  if (ret != 0) {
    LOG_ERROR("Failed to create fresh DiskAnn replay context, ret=%d", ret);
    return ret;
  }
  ret = measure_context_replay(reader, fresh_io_ctx, batches,
                               sector_buffer_size, read_stride, "fresh");
  const int destroy_ret = destroy_io_ctx(fresh_io_ctx);
  if (ret != 0) {
    return ret;
  }
  return destroy_ret;
}

#endif

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

#if defined(_WIN32) || defined(_WIN64)
int diskann_io_context_replay_once(AlignedFileReader &reader, IOContext &io_ctx,
                                   void *sector_buffer,
                                   size_t sector_buffer_size,
                                   size_t read_stride) {
  const char *trace_path = std::getenv("ZVEC_DISKANN_IO_CONTEXT_REPLAY");
  if (trace_path == nullptr || trace_path[0] == '\0') {
    return 0;
  }

  static std::once_flag once;
  static int replay_result = 0;
  std::call_once(once, [&]() {
    replay_result =
        run_context_replay(reader, io_ctx, sector_buffer, sector_buffer_size,
                           read_stride, trace_path);
  });
  return replay_result;
}
#endif

}  // namespace core
}  // namespace zvec

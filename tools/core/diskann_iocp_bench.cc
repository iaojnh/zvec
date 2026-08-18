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

#include <windows.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "diskann_file_reader.h"
#include "diskann_util.h"

namespace {

constexpr uint32_t kRequiredAlignment =
    static_cast<uint32_t>(zvec::core::DiskAnnUtil::kSectorSize);
constexpr size_t kDiskAnnContextBufferSize =
    static_cast<size_t>(zvec::core::DiskAnnUtil::kMaxSectorReadNum) *
    zvec::core::DiskAnnUtil::kSectorSize;

class UniqueHandle {
 public:
  explicit UniqueHandle(HANDLE handle = INVALID_HANDLE_VALUE)
      : handle_(handle) {}

  ~UniqueHandle() {
    close();
  }

  UniqueHandle(const UniqueHandle &) = delete;
  UniqueHandle &operator=(const UniqueHandle &) = delete;

  bool valid() const {
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
  }

  HANDLE get() const {
    return handle_;
  }

  void close() {
    if (valid()) {
      ::CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
  }

 private:
  HANDLE handle_;
};

class VirtualBuffer {
 public:
  explicit VirtualBuffer(size_t size)
      : data_(::VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE,
                             PAGE_READWRITE)) {
    if (data_ == nullptr) {
      throw std::runtime_error("VirtualAlloc failed with error " +
                               std::to_string(::GetLastError()));
    }
  }

  ~VirtualBuffer() {
    if (data_ != nullptr) {
      ::VirtualFree(data_, 0, MEM_RELEASE);
    }
  }

  VirtualBuffer(const VirtualBuffer &) = delete;
  VirtualBuffer &operator=(const VirtualBuffer &) = delete;

  unsigned char *data() {
    return static_cast<unsigned char *>(data_);
  }

 private:
  void *data_;
};

enum class BufferAllocator { kVirtual, kAlignedHeap };

class ReaderBuffer {
 public:
  ReaderBuffer(size_t size, BufferAllocator allocator)
      : allocator_(allocator), data_(nullptr) {
    if (allocator_ == BufferAllocator::kVirtual) {
      data_ = ::VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE,
                             PAGE_READWRITE);
      if (data_ == nullptr) {
        throw std::runtime_error("VirtualAlloc failed with error " +
                                 std::to_string(::GetLastError()));
      }
      return;
    }

    zvec::core::DiskAnnUtil::alloc_aligned(&data_, size, kRequiredAlignment);
    if (data_ == nullptr) {
      throw std::runtime_error("_aligned_malloc failed");
    }
  }

  ~ReaderBuffer() {
    if (data_ == nullptr) {
      return;
    }
    if (allocator_ == BufferAllocator::kVirtual) {
      ::VirtualFree(data_, 0, MEM_RELEASE);
    } else {
      zvec::core::DiskAnnUtil::free_aligned(data_);
    }
  }

  ReaderBuffer(const ReaderBuffer &) = delete;
  ReaderBuffer &operator=(const ReaderBuffer &) = delete;

  unsigned char *data() {
    return static_cast<unsigned char *>(data_);
  }

 private:
  BufferAllocator allocator_;
  void *data_;
};

struct Request {
  Request()
      : overlapped(), submitted_at(0), expected_length(0), buffer(nullptr) {}

  OVERLAPPED overlapped;
  uint64_t submitted_at;
  uint32_t expected_length;
  unsigned char *buffer;
};

static_assert(std::is_standard_layout<Request>::value,
              "Request must have standard layout");
static_assert(offsetof(Request, overlapped) == 0,
              "OVERLAPPED must be the first Request member");

enum class RandomAccessHint { kOff, kOn, kBoth };
enum class TraceMode { kContinuous, kBatched, kBoth };

struct ReadSpec {
  uint64_t offset;
  uint32_t length;
};

struct TraceBatch {
  uint64_t query_id;
  uint32_t batch_id;
  std::vector<ReadSpec> reads;
};

struct TraceData {
  std::vector<ReadSpec> reads;
  std::vector<TraceBatch> batches;
  uint32_t max_length = 0;
  uint32_t max_batch_size = 0;
  bool truncated = false;
};

struct Options {
  std::wstring file_path;
  std::vector<uint32_t> queue_depths{1, 2, 4, 8, 20};
  std::vector<uint32_t> batch_gaps_us{0};
  uint32_t duration_seconds = 10;
  uint32_t warmup_seconds = 2;
  uint32_t block_size = kRequiredAlignment;
  uint64_t seed = 0x9e3779b97f4a7c15ULL;
  RandomAccessHint random_access_hint = RandomAccessHint::kOn;
  std::wstring trace_file;
  TraceMode trace_mode = TraceMode::kBoth;
  bool reader_replay = false;
  bool random_batched = false;
  bool cached_handle_abba = false;
};

struct RunResult {
  std::string mode = "uniform";
  bool random_access_hint = false;
  uint32_t queue_depth = 0;
  uint32_t max_batch_size = 0;
  uint32_t batch_gap_us = 0;
  uint64_t completed = 0;
  uint64_t completed_bytes = 0;
  uint64_t immediate_submissions = 0;
  uint64_t pending_submissions = 0;
  uint64_t dequeue_calls = 0;
  uint32_t max_dequeued = 0;
  double elapsed_seconds = 0.0;
  uint64_t batch_count = 0;
  uint64_t gap_count = 0;
  double gap_duration_us = 0.0;
  double batch_submit_us = 0.0;
  double batch_first_completion_us = 0.0;
  double batch_duration_us = 0.0;
  double iocp_wait_us = 0.0;
  double readfile_submit_us = 0.0;
  double get_overlapped_us = 0.0;
  std::vector<double> latency_ms;
};

class XorShift64Star {
 public:
  explicit XorShift64Star(uint64_t seed)
      : state_(seed == 0 ? 0x2545f4914f6cdd1dULL : seed) {}

  uint64_t next() {
    uint64_t value = state_;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state_ = value;
    return value * 0x2545f4914f6cdd1dULL;
  }

 private:
  uint64_t state_;
};

class ReadGenerator {
 public:
  ReadGenerator(uint64_t block_count, uint32_t block_size, uint64_t seed)
      : block_count_(block_count), block_size_(block_size), random_(seed) {}

  ReadGenerator(const std::vector<ReadSpec> &trace, uint64_t seed)
      : trace_(&trace), random_(seed) {
    if (trace.empty()) {
      throw std::invalid_argument("Trace contains no read requests");
    }
  }

  ReadSpec next() {
    if (trace_ != nullptr) {
      const ReadSpec spec = (*trace_)[trace_index_];
      trace_index_ = (trace_index_ + 1) % trace_->size();
      return spec;
    }
    return ReadSpec{(random_.next() % block_count_) * block_size_, block_size_};
  }

 private:
  const std::vector<ReadSpec> *trace_{nullptr};
  size_t trace_index_{0};
  uint64_t block_count_{0};
  uint32_t block_size_{0};
  XorShift64Star random_;
};

uint64_t counter_now() {
  LARGE_INTEGER counter;
  if (!::QueryPerformanceCounter(&counter)) {
    throw std::runtime_error("QueryPerformanceCounter failed with error " +
                             std::to_string(::GetLastError()));
  }
  return static_cast<uint64_t>(counter.QuadPart);
}

uint64_t counter_frequency() {
  LARGE_INTEGER frequency;
  if (!::QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
    throw std::runtime_error("QueryPerformanceFrequency failed");
  }
  return static_cast<uint64_t>(frequency.QuadPart);
}

std::string wide_to_utf8(const std::wstring &value) {
  if (value.empty()) {
    return std::string();
  }
  if (value.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    throw std::invalid_argument("Path is too long to convert to UTF-8");
  }
  const int length = static_cast<int>(value.size());
  const int required =
      ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), length,
                            nullptr, 0, nullptr, nullptr);
  if (required <= 0) {
    throw std::runtime_error("WideCharToMultiByte failed with error " +
                             std::to_string(::GetLastError()));
  }
  std::string converted(static_cast<size_t>(required), '\0');
  const int written =
      ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), length,
                            &converted[0], required, nullptr, nullptr);
  if (written != required) {
    throw std::runtime_error("WideCharToMultiByte failed with error " +
                             std::to_string(::GetLastError()));
  }
  return converted;
}

uint32_t parse_uint32(const std::wstring &text, const char *name,
                      bool allow_zero) {
  size_t parsed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &parsed, 10);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("Invalid value for ") + name);
  }
  if (parsed != text.size() || value > (std::numeric_limits<uint32_t>::max)() ||
      (!allow_zero && value == 0)) {
    throw std::invalid_argument(std::string("Invalid value for ") + name);
  }
  return static_cast<uint32_t>(value);
}

uint64_t parse_uint64(const std::wstring &text, const char *name) {
  size_t parsed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &parsed, 10);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("Invalid value for ") + name);
  }
  if (parsed != text.size()) {
    throw std::invalid_argument(std::string("Invalid value for ") + name);
  }
  return static_cast<uint64_t>(value);
}

std::vector<uint32_t> parse_queue_depths(const std::wstring &text) {
  std::vector<uint32_t> depths;
  std::wstringstream stream(text);
  std::wstring item;
  while (std::getline(stream, item, L',')) {
    uint32_t depth = parse_uint32(item, "--queue-depths", false);
    if (depth > 4096) {
      throw std::invalid_argument("Queue depth cannot exceed 4096");
    }
    depths.push_back(depth);
  }
  if (depths.empty()) {
    throw std::invalid_argument("--queue-depths cannot be empty");
  }
  return depths;
}

std::vector<uint32_t> parse_batch_gaps(const std::wstring &text) {
  std::vector<uint32_t> gaps;
  std::wstringstream stream(text);
  std::wstring item;
  while (std::getline(stream, item, L',')) {
    uint32_t gap = parse_uint32(item, "--batch-gaps-us", true);
    if (gap > 1000000) {
      throw std::invalid_argument("Batch gap cannot exceed 1000000 us");
    }
    gaps.push_back(gap);
  }
  if (gaps.empty()) {
    throw std::invalid_argument("--batch-gaps-us cannot be empty");
  }
  return gaps;
}

std::wstring require_value(int argc, wchar_t **argv, int &index,
                           const char *name) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string("Missing value for ") + name);
  }
  ++index;
  return argv[index];
}

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  diskann_iocp_bench --file PATH [options]\n\n"
      << "Options:\n"
      << "  --queue-depths LIST          Comma-separated QDs (default: "
         "1,2,4,8,20)\n"
      << "  --duration SECONDS           Measurement time per run (default: "
         "10)\n"
      << "  --warmup SECONDS             Warmup time per run (default: 2)\n"
      << "  --block-size BYTES           Aligned read size (default: 4096)\n"
      << "  --random-access-hint MODE    on, off, or both (default: on)\n"
      << "  --trace-file PATH            Replay a captured DiskANN I/O trace\n"
      << "  --trace-mode MODE            continuous, batched, or both "
         "(default: both)\n"
      << "  --batch-gaps-us LIST         Comma-separated busy gaps between "
         "trace batches (default: 0)\n"
      << "  --reader-replay              Replay trace through the project's "
         "WindowsAlignedFileReader with four buffer allocations\n"
      << "  --cached-handle-abba         Run only the cached-handle A-B-B-A "
         "reader comparison\n"
      << "  --random-batched             Replay trace batch shapes with fresh "
         "random offsets\n"
      << "  --seed VALUE                 Random seed\n"
      << "  --help                       Show this help\n";
}

Options parse_options(int argc, wchar_t **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::wstring argument = argv[index];
    if (argument == L"--help" || argument == L"-h") {
      print_usage();
      std::exit(0);
    } else if (argument == L"--file") {
      options.file_path = require_value(argc, argv, index, "--file");
    } else if (argument == L"--queue-depths") {
      options.queue_depths = parse_queue_depths(
          require_value(argc, argv, index, "--queue-depths"));
    } else if (argument == L"--duration") {
      options.duration_seconds = parse_uint32(
          require_value(argc, argv, index, "--duration"), "--duration", false);
    } else if (argument == L"--warmup") {
      options.warmup_seconds = parse_uint32(
          require_value(argc, argv, index, "--warmup"), "--warmup", true);
    } else if (argument == L"--block-size") {
      options.block_size =
          parse_uint32(require_value(argc, argv, index, "--block-size"),
                       "--block-size", false);
    } else if (argument == L"--seed") {
      options.seed =
          parse_uint64(require_value(argc, argv, index, "--seed"), "--seed");
    } else if (argument == L"--random-access-hint") {
      const std::wstring mode =
          require_value(argc, argv, index, "--random-access-hint");
      if (mode == L"on") {
        options.random_access_hint = RandomAccessHint::kOn;
      } else if (mode == L"off") {
        options.random_access_hint = RandomAccessHint::kOff;
      } else if (mode == L"both") {
        options.random_access_hint = RandomAccessHint::kBoth;
      } else {
        throw std::invalid_argument(
            "--random-access-hint must be on, off, or both");
      }
    } else if (argument == L"--trace-file") {
      options.trace_file = require_value(argc, argv, index, "--trace-file");
    } else if (argument == L"--trace-mode") {
      const std::wstring mode =
          require_value(argc, argv, index, "--trace-mode");
      if (mode == L"continuous") {
        options.trace_mode = TraceMode::kContinuous;
      } else if (mode == L"batched") {
        options.trace_mode = TraceMode::kBatched;
      } else if (mode == L"both") {
        options.trace_mode = TraceMode::kBoth;
      } else {
        throw std::invalid_argument(
            "--trace-mode must be continuous, batched, or both");
      }
    } else if (argument == L"--batch-gaps-us") {
      options.batch_gaps_us =
          parse_batch_gaps(require_value(argc, argv, index, "--batch-gaps-us"));
    } else if (argument == L"--reader-replay") {
      options.reader_replay = true;
    } else if (argument == L"--cached-handle-abba") {
      options.cached_handle_abba = true;
    } else if (argument == L"--random-batched") {
      options.random_batched = true;
    } else {
      throw std::invalid_argument("Unknown argument");
    }
  }

  if (options.file_path.empty()) {
    throw std::invalid_argument("--file is required");
  }
  if (options.block_size % kRequiredAlignment != 0) {
    throw std::invalid_argument(
        "--block-size must be a multiple of 4096 bytes");
  }
  return options;
}

uint64_t parse_trace_integer(const std::string &text, const char *field_name,
                             size_t line_number) {
  size_t parsed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &parsed, 10);
  } catch (const std::exception &) {
    throw std::runtime_error("Invalid trace " + std::string(field_name) +
                             " on line " + std::to_string(line_number));
  }
  if (parsed != text.size()) {
    throw std::runtime_error("Invalid trace " + std::string(field_name) +
                             " on line " + std::to_string(line_number));
  }
  return static_cast<uint64_t>(value);
}

TraceData load_trace(const std::wstring &path) {
  std::ifstream input(wide_to_utf8(path));
  if (!input) {
    throw std::runtime_error("Failed to open trace file");
  }

  TraceData trace;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line == "query_id,batch_id,offset,length") {
      continue;
    }
    if (line[0] == '#') {
      if (line == "# truncated=true") {
        trace.truncated = true;
      }
      continue;
    }

    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
      fields.push_back(field);
    }
    if (fields.size() != 4) {
      throw std::runtime_error("Invalid trace row on line " +
                               std::to_string(line_number));
    }

    const uint64_t query_id =
        parse_trace_integer(fields[0], "query_id", line_number);
    const uint64_t batch_value =
        parse_trace_integer(fields[1], "batch_id", line_number);
    const uint64_t offset =
        parse_trace_integer(fields[2], "offset", line_number);
    const uint64_t length_value =
        parse_trace_integer(fields[3], "length", line_number);
    if (batch_value > (std::numeric_limits<uint32_t>::max)() ||
        length_value == 0 ||
        length_value > (std::numeric_limits<uint32_t>::max)()) {
      throw std::runtime_error("Trace value is out of range on line " +
                               std::to_string(line_number));
    }
    const uint32_t batch_id = static_cast<uint32_t>(batch_value);
    const uint32_t length = static_cast<uint32_t>(length_value);
    if (offset % kRequiredAlignment != 0 || length % kRequiredAlignment != 0) {
      throw std::runtime_error("Unaligned trace read on line " +
                               std::to_string(line_number));
    }

    const ReadSpec spec{offset, length};
    trace.reads.push_back(spec);
    if (trace.batches.empty() || trace.batches.back().query_id != query_id ||
        trace.batches.back().batch_id != batch_id) {
      trace.batches.push_back(TraceBatch{query_id, batch_id, {}});
    }
    trace.batches.back().reads.push_back(spec);
    trace.max_length = std::max(trace.max_length, length);
    trace.max_batch_size =
        std::max(trace.max_batch_size,
                 static_cast<uint32_t>(trace.batches.back().reads.size()));
  }

  if (trace.reads.empty()) {
    throw std::runtime_error("Trace file contains no read requests");
  }
  return trace;
}

uint64_t file_size(HANDLE file_handle) {
  LARGE_INTEGER size;
  if (!::GetFileSizeEx(file_handle, &size) || size.QuadPart <= 0) {
    throw std::runtime_error("GetFileSizeEx failed with error " +
                             std::to_string(::GetLastError()));
  }
  return static_cast<uint64_t>(size.QuadPart);
}

void prime_cached_handle(HANDLE file_handle, uint64_t size) {
  const DWORD probe_size = static_cast<DWORD>(
      std::min<uint64_t>(size, static_cast<uint64_t>(kRequiredAlignment)));
  std::vector<unsigned char> probe(probe_size);
  DWORD bytes_read = 0;
  if (!::ReadFile(file_handle, probe.data(), probe_size, &bytes_read,
                  nullptr)) {
    throw std::runtime_error("Cached-handle probe read failed with error " +
                             std::to_string(::GetLastError()));
  }
  if (bytes_read != probe_size) {
    throw std::runtime_error(
        "Cached-handle probe returned an unexpected byte count");
  }
}

void set_overlapped_offset(OVERLAPPED &overlapped, uint64_t offset) {
  std::memset(&overlapped, 0, sizeof(overlapped));
  overlapped.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
}

bool submit_read(HANDLE file_handle, Request &request, const ReadSpec &spec,
                 uint32_t &outstanding, RunResult *result, std::string &error) {
  set_overlapped_offset(request.overlapped, spec.offset);
  request.submitted_at = counter_now();
  request.expected_length = spec.length;

  const BOOL completed = ::ReadFile(file_handle, request.buffer, spec.length,
                                    nullptr, &request.overlapped);
  if (!completed) {
    const DWORD last_error = ::GetLastError();
    if (last_error != ERROR_IO_PENDING) {
      error = "ReadFile failed with error " + std::to_string(last_error);
      return false;
    }
    if (result != nullptr) {
      ++result->pending_submissions;
    }
  } else if (result != nullptr) {
    ++result->immediate_submissions;
  }
  ++outstanding;
  return true;
}

bool validate_completion(const OVERLAPPED_ENTRY &entry, Request *&request,
                         std::string &error) {
  if (entry.lpOverlapped == nullptr) {
    error = "IOCP returned a null OVERLAPPED pointer";
    return false;
  }
  if (entry.lpOverlapped->Internal != 0) {
    std::ostringstream stream;
    stream << "Asynchronous read failed with NTSTATUS 0x" << std::hex
           << entry.lpOverlapped->Internal;
    error = stream.str();
    return false;
  }
  request = reinterpret_cast<Request *>(entry.lpOverlapped);
  if (entry.dwNumberOfBytesTransferred != request->expected_length) {
    error = "Short read: expected " + std::to_string(request->expected_length) +
            " bytes, received " +
            std::to_string(entry.dwNumberOfBytesTransferred);
    return false;
  }
  return true;
}

bool process_until(HANDLE file_handle, HANDLE completion_port,
                   uint64_t deadline, uint64_t frequency,
                   ReadGenerator &generator,
                   std::vector<OVERLAPPED_ENTRY> &entries,
                   uint32_t &outstanding, bool keep_queue_full,
                   RunResult *result, std::string &error) {
  while (counter_now() < deadline) {
    ULONG removed = 0;
    const uint64_t wait_started = counter_now();
    const BOOL dequeued = ::GetQueuedCompletionStatusEx(
        completion_port, entries.data(), static_cast<ULONG>(entries.size()),
        &removed, INFINITE, FALSE);
    const uint64_t wait_finished = counter_now();
    if (!dequeued || removed == 0) {
      error = "GetQueuedCompletionStatusEx failed with error " +
              std::to_string(::GetLastError());
      return false;
    }

    if (result != nullptr) {
      ++result->dequeue_calls;
      result->iocp_wait_us +=
          static_cast<double>(wait_finished - wait_started) * 1000000.0 /
          static_cast<double>(frequency);
      result->max_dequeued =
          std::max(result->max_dequeued, static_cast<uint32_t>(removed));
    }
    if (removed > outstanding) {
      error = "IOCP completion count exceeded submitted read count";
      return false;
    }
    outstanding -= removed;

    for (ULONG entry_index = 0; entry_index < removed; ++entry_index) {
      const OVERLAPPED_ENTRY &entry = entries[entry_index];
      Request *request = nullptr;
      if (!validate_completion(entry, request, error)) {
        return false;
      }
      if (result != nullptr) {
        const uint64_t completed_at = counter_now();
        const double latency =
            static_cast<double>(completed_at - request->submitted_at) * 1000.0 /
            static_cast<double>(frequency);
        result->latency_ms.push_back(latency);
        ++result->completed;
        result->completed_bytes += request->expected_length;
      }

      if ((keep_queue_full || counter_now() < deadline) &&
          !submit_read(file_handle, *request, generator.next(), outstanding,
                       result, error)) {
        return false;
      }
    }
  }
  return true;
}

bool drain_requests(HANDLE completion_port,
                    std::vector<OVERLAPPED_ENTRY> &entries,
                    uint32_t &outstanding, std::string &error) {
  while (outstanding != 0) {
    ULONG removed = 0;
    const BOOL dequeued = ::GetQueuedCompletionStatusEx(
        completion_port, entries.data(), static_cast<ULONG>(entries.size()),
        &removed, INFINITE, FALSE);
    if (!dequeued || removed == 0) {
      error = "Failed to drain IOCP requests with error " +
              std::to_string(::GetLastError());
      return false;
    }
    if (removed > outstanding) {
      error = "Drained completion count exceeded outstanding read count";
      return false;
    }
    outstanding -= removed;
  }
  return true;
}

RunResult run_continuous_benchmark(const Options &options, uint32_t queue_depth,
                                   bool random_access_hint,
                                   const TraceData *trace) {
  RunResult result;
  result.mode = trace == nullptr ? "uniform" : "trace_continuous";
  result.random_access_hint = random_access_hint;
  result.queue_depth = queue_depth;
  const size_t expected_completions = std::min<size_t>(
      static_cast<size_t>(options.duration_seconds) * 50000U, 10000000U);
  result.latency_ms.reserve(expected_completions);
  const uint32_t buffer_stride =
      trace == nullptr ? options.block_size : trace->max_length;
  VirtualBuffer buffer(static_cast<size_t>(queue_depth) * buffer_stride);

  DWORD flags =
      FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
  if (random_access_hint) {
    flags |= FILE_FLAG_RANDOM_ACCESS;
  }

  UniqueHandle file_handle(
      ::CreateFileW(options.file_path.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, flags, nullptr));
  if (!file_handle.valid()) {
    throw std::runtime_error("CreateFileW failed with error " +
                             std::to_string(::GetLastError()));
  }

  const uint64_t size = file_size(file_handle.get());
  const uint64_t block_count = size / options.block_size;
  if (block_count == 0) {
    throw std::runtime_error("Input file is smaller than one aligned block");
  }
  if (trace != nullptr) {
    for (const ReadSpec &spec : trace->reads) {
      if (spec.offset > size || spec.length > size - spec.offset) {
        throw std::runtime_error("Trace read extends beyond the input file");
      }
    }
  }
  UniqueHandle completion_port(
      ::CreateIoCompletionPort(file_handle.get(), nullptr, 0, 1));
  if (!completion_port.valid()) {
    throw std::runtime_error("CreateIoCompletionPort failed with error " +
                             std::to_string(::GetLastError()));
  }
  if (!::SetFileCompletionNotificationModes(file_handle.get(),
                                            FILE_SKIP_SET_EVENT_ON_HANDLE)) {
    std::cerr << "Warning: SetFileCompletionNotificationModes failed with "
                 "error "
              << ::GetLastError() << '\n';
  }

  std::vector<Request> requests(queue_depth);
  std::vector<OVERLAPPED_ENTRY> entries(queue_depth);
  for (uint32_t index = 0; index < queue_depth; ++index) {
    requests[index].buffer =
        buffer.data() + static_cast<size_t>(index) * buffer_stride;
  }

  const uint64_t frequency = counter_frequency();
  ReadGenerator generator(block_count, options.block_size,
                          options.seed + queue_depth);
  if (trace != nullptr) {
    generator = ReadGenerator(trace->reads, options.seed + queue_depth);
  }
  uint32_t outstanding = 0;
  std::string error;
  for (uint32_t index = 0; index < queue_depth; ++index) {
    if (!submit_read(file_handle.get(), requests[index], generator.next(),
                     outstanding, nullptr, error)) {
      ::CancelIoEx(file_handle.get(), nullptr);
      drain_requests(completion_port.get(), entries, outstanding, error);
      throw std::runtime_error(error);
    }
  }

  const uint64_t warmup_deadline =
      counter_now() + static_cast<uint64_t>(options.warmup_seconds) * frequency;
  if (!process_until(file_handle.get(), completion_port.get(), warmup_deadline,
                     frequency, generator, entries, outstanding, true, nullptr,
                     error)) {
    ::CancelIoEx(file_handle.get(), nullptr);
    drain_requests(completion_port.get(), entries, outstanding, error);
    throw std::runtime_error(error);
  }

  const uint64_t measurement_start = counter_now();
  const uint64_t measurement_deadline =
      measurement_start +
      static_cast<uint64_t>(options.duration_seconds) * frequency;
  if (!process_until(file_handle.get(), completion_port.get(),
                     measurement_deadline, frequency, generator, entries,
                     outstanding, false, &result, error)) {
    ::CancelIoEx(file_handle.get(), nullptr);
    drain_requests(completion_port.get(), entries, outstanding, error);
    throw std::runtime_error(error);
  }
  const uint64_t measurement_end = counter_now();
  result.elapsed_seconds =
      static_cast<double>(measurement_end - measurement_start) /
      static_cast<double>(frequency);

  if (!drain_requests(completion_port.get(), entries, outstanding, error)) {
    throw std::runtime_error(error);
  }
  return result;
}

bool wait_for_batch(HANDLE completion_port, uint64_t frequency,
                    std::vector<OVERLAPPED_ENTRY> &entries,
                    uint32_t &outstanding, RunResult *result,
                    uint64_t &first_completion_at, std::string &error) {
  while (outstanding != 0) {
    ULONG removed = 0;
    const uint64_t wait_started = counter_now();
    const BOOL dequeued = ::GetQueuedCompletionStatusEx(
        completion_port, entries.data(), static_cast<ULONG>(entries.size()),
        &removed, INFINITE, FALSE);
    const uint64_t wait_finished = counter_now();
    if (!dequeued || removed == 0) {
      error = "GetQueuedCompletionStatusEx failed with error " +
              std::to_string(::GetLastError());
      return false;
    }
    if (removed > outstanding) {
      error = "IOCP completion count exceeded submitted read count";
      return false;
    }
    outstanding -= removed;

    if (result != nullptr) {
      ++result->dequeue_calls;
      result->iocp_wait_us +=
          static_cast<double>(wait_finished - wait_started) * 1000000.0 /
          static_cast<double>(frequency);
      result->max_dequeued =
          std::max(result->max_dequeued, static_cast<uint32_t>(removed));
    }
    if (first_completion_at == 0) {
      first_completion_at = wait_finished;
    }
    for (ULONG entry_index = 0; entry_index < removed; ++entry_index) {
      Request *request = nullptr;
      if (!validate_completion(entries[entry_index], request, error)) {
        return false;
      }
      if (result != nullptr) {
        const uint64_t completed_at = counter_now();
        const double latency =
            static_cast<double>(completed_at - request->submitted_at) * 1000.0 /
            static_cast<double>(frequency);
        result->latency_ms.push_back(latency);
        ++result->completed;
        result->completed_bytes += request->expected_length;
      }
    }
  }
  return true;
}

bool replay_batches_until(
    HANDLE file_handle, HANDLE completion_port, uint64_t deadline,
    uint64_t frequency, const TraceData &trace, std::vector<Request> &requests,
    std::vector<OVERLAPPED_ENTRY> &entries, size_t &batch_index,
    uint32_t batch_gap_us, uint64_t input_size, XorShift64Star *random,
    bool &has_previous_batch, RunResult *result, std::string &error) {
  while (counter_now() < deadline) {
    if (has_previous_batch && batch_gap_us != 0) {
      const uint64_t gap_started = counter_now();
      const uint64_t gap_ticks =
          (static_cast<uint64_t>(batch_gap_us) * frequency + 999999ULL) /
          1000000ULL;
      const uint64_t gap_deadline = gap_started + gap_ticks;
      while (counter_now() < gap_deadline) {
      }
      if (result != nullptr) {
        ++result->gap_count;
        result->gap_duration_us +=
            static_cast<double>(counter_now() - gap_started) * 1000000.0 /
            static_cast<double>(frequency);
      }
    }
    const TraceBatch &batch = trace.batches[batch_index];
    batch_index = (batch_index + 1) % trace.batches.size();
    const uint64_t batch_started = counter_now();
    uint32_t outstanding = 0;
    for (size_t index = 0; index < batch.reads.size(); ++index) {
      ReadSpec spec = batch.reads[index];
      if (random != nullptr) {
        const uint64_t eligible_blocks =
            (input_size - spec.length) / kRequiredAlignment + 1;
        spec.offset = (random->next() % eligible_blocks) * kRequiredAlignment;
      }
      if (!submit_read(file_handle, requests[index], spec, outstanding, result,
                       error)) {
        ::CancelIoEx(file_handle, nullptr);
        drain_requests(completion_port, entries, outstanding, error);
        return false;
      }
    }
    const uint64_t batch_submitted = counter_now();
    uint64_t first_completion_at = 0;
    if (!wait_for_batch(completion_port, frequency, entries, outstanding,
                        result, first_completion_at, error)) {
      ::CancelIoEx(file_handle, nullptr);
      drain_requests(completion_port, entries, outstanding, error);
      return false;
    }
    if (result != nullptr) {
      const uint64_t batch_finished = counter_now();
      ++result->batch_count;
      result->batch_submit_us +=
          static_cast<double>(batch_submitted - batch_started) * 1000000.0 /
          static_cast<double>(frequency);
      result->batch_first_completion_us +=
          static_cast<double>(first_completion_at - batch_started) * 1000000.0 /
          static_cast<double>(frequency);
      result->batch_duration_us +=
          static_cast<double>(batch_finished - batch_started) * 1000000.0 /
          static_cast<double>(frequency);
    }
    has_previous_batch = true;
  }
  return true;
}

RunResult run_batched_benchmark(const Options &options, bool random_access_hint,
                                const TraceData &trace, uint32_t batch_gap_us,
                                bool random_offsets) {
  RunResult result;
  if (random_offsets) {
    result.mode = batch_gap_us == 0 ? "random_batched" : "random_gapped";
  } else {
    result.mode = batch_gap_us == 0 ? "trace_batched" : "trace_gapped";
  }
  result.random_access_hint = random_access_hint;
  result.max_batch_size = trace.max_batch_size;
  result.batch_gap_us = batch_gap_us;
  const size_t expected_completions = std::min<size_t>(
      static_cast<size_t>(options.duration_seconds) * 50000U, 10000000U);
  result.latency_ms.reserve(expected_completions);

  DWORD flags =
      FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
  if (random_access_hint) {
    flags |= FILE_FLAG_RANDOM_ACCESS;
  }

  VirtualBuffer buffer(static_cast<size_t>(trace.max_batch_size) *
                       trace.max_length);
  UniqueHandle file_handle(
      ::CreateFileW(options.file_path.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, flags, nullptr));
  if (!file_handle.valid()) {
    throw std::runtime_error("CreateFileW failed with error " +
                             std::to_string(::GetLastError()));
  }
  const uint64_t size = file_size(file_handle.get());
  for (const ReadSpec &spec : trace.reads) {
    if (spec.offset > size || spec.length > size - spec.offset) {
      throw std::runtime_error("Trace read extends beyond the input file");
    }
  }

  UniqueHandle completion_port(
      ::CreateIoCompletionPort(file_handle.get(), nullptr, 0, 1));
  if (!completion_port.valid()) {
    throw std::runtime_error("CreateIoCompletionPort failed with error " +
                             std::to_string(::GetLastError()));
  }
  if (!::SetFileCompletionNotificationModes(file_handle.get(),
                                            FILE_SKIP_SET_EVENT_ON_HANDLE)) {
    std::cerr << "Warning: SetFileCompletionNotificationModes failed with "
                 "error "
              << ::GetLastError() << '\n';
  }

  std::vector<Request> requests(trace.max_batch_size);
  std::vector<OVERLAPPED_ENTRY> entries(trace.max_batch_size);
  for (uint32_t index = 0; index < trace.max_batch_size; ++index) {
    requests[index].buffer =
        buffer.data() + static_cast<size_t>(index) * trace.max_length;
  }

  const uint64_t frequency = counter_frequency();
  size_t batch_index = 0;
  XorShift64Star random(options.seed);
  XorShift64Star *random_ptr = random_offsets ? &random : nullptr;
  bool has_previous_batch = false;
  std::string error;
  const uint64_t warmup_deadline =
      counter_now() + static_cast<uint64_t>(options.warmup_seconds) * frequency;
  if (!replay_batches_until(file_handle.get(), completion_port.get(),
                            warmup_deadline, frequency, trace, requests,
                            entries, batch_index, batch_gap_us, size,
                            random_ptr, has_previous_batch, nullptr, error)) {
    throw std::runtime_error(error);
  }

  const uint64_t measurement_start = counter_now();
  const uint64_t measurement_deadline =
      measurement_start +
      static_cast<uint64_t>(options.duration_seconds) * frequency;
  if (!replay_batches_until(file_handle.get(), completion_port.get(),
                            measurement_deadline, frequency, trace, requests,
                            entries, batch_index, batch_gap_us, size,
                            random_ptr, has_previous_batch, &result, error)) {
    throw std::runtime_error(error);
  }
  const uint64_t measurement_end = counter_now();
  result.elapsed_seconds =
      static_cast<double>(measurement_end - measurement_start) /
      static_cast<double>(frequency);
  return result;
}

class ScopedDiskAnnIoContext {
 public:
  ScopedDiskAnnIoContext() {
    const int result = zvec::core::setup_io_ctx(context_);
    if (result != 0 || context_ == nullptr) {
      throw std::runtime_error("setup_io_ctx failed with result " +
                               std::to_string(result));
    }
  }

  ~ScopedDiskAnnIoContext() {
    zvec::core::destroy_io_ctx(context_);
  }

  ScopedDiskAnnIoContext(const ScopedDiskAnnIoContext &) = delete;
  ScopedDiskAnnIoContext &operator=(const ScopedDiskAnnIoContext &) = delete;

  zvec::core::IOContext &get() {
    return context_;
  }

 private:
  zvec::core::IOContext context_{nullptr};
};

void replay_reader_batches_until(
    zvec::core::WindowsAlignedFileReader &reader,
    zvec::core::IOContext &context, uint64_t deadline, uint64_t frequency,
    const TraceData &trace, unsigned char *buffer, size_t &batch_index,
    RunResult *result,
    size_t max_batches = std::numeric_limits<size_t>::max()) {
  std::vector<zvec::core::AlignedRead> read_requests;
  read_requests.reserve(trace.max_batch_size);
  std::vector<uint32_t> completed_indices;
  completed_indices.reserve(trace.max_batch_size);

  size_t processed_batches = 0;
  while (processed_batches < max_batches && counter_now() < deadline) {
    const TraceBatch &batch = trace.batches[batch_index];
    batch_index = (batch_index + 1) % trace.batches.size();
    ++processed_batches;
    read_requests.clear();
    for (size_t index = 0; index < batch.reads.size(); ++index) {
      const ReadSpec &spec = batch.reads[index];
      read_requests.emplace_back(
          spec.offset, spec.length,
          buffer + static_cast<size_t>(index) * trace.max_length);
    }

    const uint64_t batch_started = counter_now();
    zvec::core::PendingBatch pending;
    const int submit_result = reader.submit(pending, read_requests, context);
    if (submit_result != 0) {
      throw std::runtime_error(
          "WindowsAlignedFileReader::submit failed with "
          "result " +
          std::to_string(submit_result));
    }

    while (pending.n_reaped < pending.n_submitted) {
      completed_indices.clear();
      const int completed =
          reader.get_completed(pending, context, 1, completed_indices);
      if (completed < 0) {
        throw std::runtime_error(
            "WindowsAlignedFileReader::get_completed failed with result " +
            std::to_string(completed));
      }
      if (result == nullptr) {
        continue;
      }

      const uint64_t completed_at = counter_now();
      const double latency_ms =
          static_cast<double>(completed_at - batch_started) * 1000.0 /
          static_cast<double>(frequency);
      for (uint32_t index : completed_indices) {
        const size_t request_index = static_cast<size_t>(index);
        if (request_index >= batch.reads.size()) {
          throw std::runtime_error(
              "WindowsAlignedFileReader returned an invalid request index");
        }
        result->latency_ms.push_back(latency_ms);
        ++result->completed;
        result->completed_bytes += batch.reads[request_index].length;
      }
    }
  }
}

RunResult run_reader_batched_benchmark(
    const Options &options, const TraceData &trace, const std::string &mode,
    BufferAllocator allocator, size_t buffer_size, bool keep_cached_handle) {
  RunResult result;
  result.mode = mode;
  result.random_access_hint = true;
  result.max_batch_size = trace.max_batch_size;
  const size_t expected_completions = std::min<size_t>(
      static_cast<size_t>(options.duration_seconds) * 50000U, 10000000U);
  result.latency_ms.reserve(expected_completions);

  const DWORD cached_handle_attributes = options.cached_handle_abba
                                             ? FILE_ATTRIBUTE_NORMAL
                                             : FILE_ATTRIBUTE_READONLY;
  UniqueHandle cached_handle(
      ::CreateFileW(options.file_path.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, cached_handle_attributes, nullptr));
  if (!cached_handle.valid()) {
    throw std::runtime_error("CreateFileW failed with error " +
                             std::to_string(::GetLastError()));
  }
  const uint64_t size = file_size(cached_handle.get());
  if (options.cached_handle_abba) {
    prime_cached_handle(cached_handle.get(), size);
  }
  for (const ReadSpec &spec : trace.reads) {
    if (spec.offset > size || spec.length > size - spec.offset) {
      throw std::runtime_error("Trace read extends beyond the input file");
    }
  }
  if (!keep_cached_handle) {
    cached_handle.close();
  }

  if (trace.max_length == 0 || trace.max_batch_size == 0 ||
      static_cast<size_t>(trace.max_batch_size) >
          buffer_size / static_cast<size_t>(trace.max_length)) {
    throw std::runtime_error(
        "Reader replay buffer is smaller than the largest "
        "captured batch");
  }
  ReaderBuffer buffer(buffer_size, allocator);
  zvec::core::WindowsAlignedFileReader reader;
  reader.open(wide_to_utf8(options.file_path));
  ScopedDiskAnnIoContext scoped_context;
  zvec::core::IOContext &context = scoped_context.get();
  context->diagnostics_enabled = true;

  const uint64_t frequency = counter_frequency();
  size_t batch_index = 0;
  if (options.cached_handle_abba) {
    replay_reader_batches_until(
        reader, context, std::numeric_limits<uint64_t>::max(), frequency, trace,
        buffer.data(), batch_index, nullptr, trace.batches.size());
  } else {
    const uint64_t warmup_deadline =
        counter_now() +
        static_cast<uint64_t>(options.warmup_seconds) * frequency;
    replay_reader_batches_until(reader, context, warmup_deadline, frequency,
                                trace, buffer.data(), batch_index, nullptr);
  }

  // Restart measurement at the first captured batch after either warmup mode.
  batch_index = 0;
  context->diagnostics = zvec::core::IoBackend::IocpDiagnostics{};
  context->diagnostics_enabled = true;
  const uint64_t measurement_start = counter_now();
  const uint64_t measurement_deadline =
      measurement_start +
      static_cast<uint64_t>(options.duration_seconds) * frequency;
  replay_reader_batches_until(reader, context, measurement_deadline, frequency,
                              trace, buffer.data(), batch_index, &result);
  const uint64_t measurement_end = counter_now();
  result.elapsed_seconds =
      static_cast<double>(measurement_end - measurement_start) /
      static_cast<double>(frequency);

  const auto &io = context->diagnostics;
  result.immediate_submissions = io.immediate_reads;
  result.pending_submissions = io.pending_reads;
  result.dequeue_calls = io.dequeue_calls;
  result.max_dequeued = io.max_dequeued_once;
  result.batch_count = io.batch_count;
  result.batch_submit_us = static_cast<double>(io.batch_submit_ns) / 1000.0;
  result.batch_first_completion_us =
      static_cast<double>(io.batch_first_completion_ns) / 1000.0;
  result.batch_duration_us = static_cast<double>(io.batch_duration_ns) / 1000.0;
  result.iocp_wait_us = static_cast<double>(io.wait_us);
  result.readfile_submit_us =
      static_cast<double>(io.readfile_submit_ns) / 1000.0;
  result.get_overlapped_us = static_cast<double>(io.get_overlapped_ns) / 1000.0;
  return result;
}

double percentile(const std::vector<double> &sorted_values, double fraction) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double position =
      fraction * static_cast<double>(sorted_values.size() - 1);
  return sorted_values[static_cast<size_t>(position)];
}

void print_result(RunResult result) {
  std::sort(result.latency_ms.begin(), result.latency_ms.end());
  const double latency_sum =
      std::accumulate(result.latency_ms.begin(), result.latency_ms.end(), 0.0);
  const double average_latency =
      result.latency_ms.empty()
          ? 0.0
          : latency_sum / static_cast<double>(result.latency_ms.size());
  const double iops =
      static_cast<double>(result.completed) / result.elapsed_seconds;
  const double mib_per_second = static_cast<double>(result.completed_bytes) /
                                (result.elapsed_seconds * 1024.0 * 1024.0);
  const double effective_queue_depth = iops * average_latency / 1000.0;
  const double completions_per_dequeue =
      result.dequeue_calls == 0 ? 0.0
                                : static_cast<double>(result.completed) /
                                      static_cast<double>(result.dequeue_calls);
  const uint64_t submissions =
      result.immediate_submissions + result.pending_submissions;
  const double pending_ratio =
      submissions == 0 ? 0.0
                       : static_cast<double>(result.pending_submissions) *
                             100.0 / static_cast<double>(submissions);
  const double submit_us_per_batch =
      result.batch_count == 0
          ? 0.0
          : result.batch_submit_us / static_cast<double>(result.batch_count);
  const double actual_gap_us =
      result.gap_count == 0
          ? 0.0
          : result.gap_duration_us / static_cast<double>(result.gap_count);
  const double first_completion_ms =
      result.batch_count == 0
          ? 0.0
          : result.batch_first_completion_us /
                static_cast<double>(result.batch_count) / 1000.0;
  const double batch_duration_ms =
      result.batch_count == 0
          ? 0.0
          : result.batch_duration_us / static_cast<double>(result.batch_count) /
                1000.0;
  const double iocp_wait_ms_per_batch =
      result.batch_count == 0
          ? 0.0
          : result.iocp_wait_us / static_cast<double>(result.batch_count) /
                1000.0;
  const double readfile_submit_us_per_read =
      result.completed == 0
          ? 0.0
          : result.readfile_submit_us / static_cast<double>(result.completed);
  const double get_overlapped_us_per_read =
      result.completed == 0
          ? 0.0
          : result.get_overlapped_us / static_cast<double>(result.completed);

  std::cout << result.mode << ',' << (result.random_access_hint ? "on" : "off")
            << ',' << result.queue_depth << ',' << result.max_batch_size << ','
            << result.batch_gap_us << ',' << std::fixed << std::setprecision(2)
            << actual_gap_us << ',' << iops << ',' << mib_per_second << ','
            << average_latency << ',' << percentile(result.latency_ms, 0.50)
            << ',' << percentile(result.latency_ms, 0.95) << ','
            << percentile(result.latency_ms, 0.99) << ','
            << result.latency_ms.front() << ',' << result.latency_ms.back()
            << ',' << effective_queue_depth << ',' << result.batch_count << ','
            << submit_us_per_batch << ',' << first_completion_ms << ','
            << batch_duration_ms << ',' << iocp_wait_ms_per_batch << ','
            << readfile_submit_us_per_read << ',' << get_overlapped_us_per_read
            << ',' << completions_per_dequeue << ',' << result.max_dequeued
            << ',' << pending_ratio << ',' << result.completed << '\n';
}

std::vector<bool> hint_modes(RandomAccessHint mode) {
  if (mode == RandomAccessHint::kBoth) {
    return std::vector<bool>{false, true};
  }
  return std::vector<bool>{mode == RandomAccessHint::kOn};
}

}  // namespace

int wmain(int argc, wchar_t **argv) {
  try {
    const Options options = parse_options(argc, argv);
    TraceData trace;
    const bool has_trace = !options.trace_file.empty();
    if (has_trace) {
      trace = load_trace(options.trace_file);
    }
    if (options.reader_replay && !has_trace) {
      throw std::invalid_argument("--reader-replay requires --trace-file");
    }
    if (options.cached_handle_abba && !has_trace) {
      throw std::invalid_argument("--cached-handle-abba requires --trace-file");
    }
    if (options.random_batched && !has_trace) {
      throw std::invalid_argument("--random-batched requires --trace-file");
    }
    std::cout << "DiskANN Windows IOCP microbenchmark\n"
              << "file: " << wide_to_utf8(options.file_path) << '\n'
              << "block_size: " << options.block_size << " bytes\n";
    if (options.cached_handle_abba) {
      std::cout << "warmup: one complete trace cycle/run\n";
    } else {
      std::cout << "warmup: " << options.warmup_seconds << " seconds/run\n";
    }
    std::cout << "duration: " << options.duration_seconds << " seconds/run\n";
    if (has_trace) {
      std::cout << "trace_file: " << wide_to_utf8(options.trace_file) << '\n'
                << "trace_reads: " << trace.reads.size() << '\n'
                << "trace_batches: " << trace.batches.size() << '\n'
                << "trace_max_batch: " << trace.max_batch_size << '\n';
      if (trace.truncated) {
        std::cout << "WARNING: trace capture was truncated\n";
      }
    }
    std::cout
        << '\n'
        << "mode,random_access_hint,queue_depth,max_batch_size,batch_gap_us,"
           "actual_gap_us,iops,mib_per_s,"
           "avg_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,min_"
           "latency_ms,max_latency_ms,effective_qd,batch_count,submit_us_per_"
           "batch,first_completion_ms,batch_duration_ms,iocp_wait_ms_per_"
           "batch,readfile_submit_us_per_read,get_overlapped_us_per_read,"
           "completions_per_dequeue,max_dequeued,pending_ratio_pct,completed_"
           "reads\n";

    if (options.cached_handle_abba) {
      std::vector<RunResult> results;
      results.reserve(4);
      results.emplace_back(run_reader_batched_benchmark(
          options, trace, "cached_closed_a1", BufferAllocator::kAlignedHeap,
          kDiskAnnContextBufferSize, false));
      results.emplace_back(run_reader_batched_benchmark(
          options, trace, "cached_held_b1", BufferAllocator::kAlignedHeap,
          kDiskAnnContextBufferSize, true));
      results.emplace_back(run_reader_batched_benchmark(
          options, trace, "cached_held_b2", BufferAllocator::kAlignedHeap,
          kDiskAnnContextBufferSize, true));
      results.emplace_back(run_reader_batched_benchmark(
          options, trace, "cached_closed_a2", BufferAllocator::kAlignedHeap,
          kDiskAnnContextBufferSize, false));
      for (const RunResult &result : results) {
        print_result(result);
      }
      return 0;
    }

    const std::vector<bool> modes = hint_modes(options.random_access_hint);
    for (bool random_access_hint : modes) {
      if (!has_trace || options.trace_mode == TraceMode::kContinuous ||
          options.trace_mode == TraceMode::kBoth) {
        for (uint32_t queue_depth : options.queue_depths) {
          print_result(run_continuous_benchmark(options, queue_depth,
                                                random_access_hint,
                                                has_trace ? &trace : nullptr));
        }
      }
      if (has_trace && (options.trace_mode == TraceMode::kBatched ||
                        options.trace_mode == TraceMode::kBoth)) {
        for (uint32_t batch_gap_us : options.batch_gaps_us) {
          print_result(run_batched_benchmark(options, random_access_hint, trace,
                                             batch_gap_us, false));
        }
        if (options.random_batched) {
          for (uint32_t batch_gap_us : options.batch_gaps_us) {
            print_result(run_batched_benchmark(options, random_access_hint,
                                               trace, batch_gap_us, true));
          }
        }
      }
    }
    if (options.reader_replay) {
      const size_t compact_size = static_cast<size_t>(trace.max_batch_size) *
                                  static_cast<size_t>(trace.max_length);
      if (compact_size > kDiskAnnContextBufferSize) {
        throw std::runtime_error(
            "Captured batch exceeds the DiskAnnContext 512 KiB buffer");
      }
      print_result(run_reader_batched_benchmark(
          options, trace, "reader_virtual_compact", BufferAllocator::kVirtual,
          compact_size, false));
      print_result(run_reader_batched_benchmark(
          options, trace, "reader_virtual_context", BufferAllocator::kVirtual,
          kDiskAnnContextBufferSize, false));
      print_result(run_reader_batched_benchmark(
          options, trace, "reader_aligned_compact",
          BufferAllocator::kAlignedHeap, compact_size, false));
      print_result(run_reader_batched_benchmark(
          options, trace, "reader_aligned_context",
          BufferAllocator::kAlignedHeap, kDiskAnnContextBufferSize, false));
    }
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

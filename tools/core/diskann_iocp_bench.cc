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
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr uint32_t kRequiredAlignment = 4096;

class UniqueHandle {
 public:
  explicit UniqueHandle(HANDLE handle = INVALID_HANDLE_VALUE)
      : handle_(handle) {}

  ~UniqueHandle() {
    if (valid()) {
      ::CloseHandle(handle_);
    }
  }

  UniqueHandle(const UniqueHandle &) = delete;
  UniqueHandle &operator=(const UniqueHandle &) = delete;

  bool valid() const {
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
  }

  HANDLE get() const {
    return handle_;
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

struct Request {
  Request() : overlapped(), submitted_at(0), buffer(nullptr) {}

  OVERLAPPED overlapped;
  uint64_t submitted_at;
  unsigned char *buffer;
};

static_assert(std::is_standard_layout<Request>::value,
              "Request must have standard layout");
static_assert(offsetof(Request, overlapped) == 0,
              "OVERLAPPED must be the first Request member");

enum class RandomAccessHint { kOff, kOn, kBoth };

struct Options {
  std::wstring file_path;
  std::vector<uint32_t> queue_depths{1, 2, 4, 8, 20};
  uint32_t duration_seconds = 10;
  uint32_t warmup_seconds = 2;
  uint32_t block_size = kRequiredAlignment;
  uint64_t seed = 0x9e3779b97f4a7c15ULL;
  RandomAccessHint random_access_hint = RandomAccessHint::kOn;
};

struct RunResult {
  bool random_access_hint = false;
  uint32_t queue_depth = 0;
  uint64_t completed = 0;
  uint64_t immediate_submissions = 0;
  uint64_t pending_submissions = 0;
  uint64_t dequeue_calls = 0;
  uint32_t max_dequeued = 0;
  double elapsed_seconds = 0.0;
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

uint64_t file_size(HANDLE file_handle) {
  LARGE_INTEGER size;
  if (!::GetFileSizeEx(file_handle, &size) || size.QuadPart <= 0) {
    throw std::runtime_error("GetFileSizeEx failed with error " +
                             std::to_string(::GetLastError()));
  }
  return static_cast<uint64_t>(size.QuadPart);
}

void set_overlapped_offset(OVERLAPPED &overlapped, uint64_t offset) {
  std::memset(&overlapped, 0, sizeof(overlapped));
  overlapped.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
}

bool submit_read(HANDLE file_handle, Request &request, uint32_t block_size,
                 uint64_t block_count, XorShift64Star &random,
                 uint32_t &outstanding, RunResult *result, std::string &error) {
  const uint64_t offset = (random.next() % block_count) * block_size;
  set_overlapped_offset(request.overlapped, offset);
  request.submitted_at = counter_now();

  const BOOL completed = ::ReadFile(file_handle, request.buffer, block_size,
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

bool validate_completion(const OVERLAPPED_ENTRY &entry, uint32_t block_size,
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
  if (entry.dwNumberOfBytesTransferred != block_size) {
    error = "Short read: expected " + std::to_string(block_size) +
            " bytes, received " +
            std::to_string(entry.dwNumberOfBytesTransferred);
    return false;
  }
  return true;
}

bool process_until(HANDLE file_handle, HANDLE completion_port,
                   uint64_t deadline, uint64_t frequency, uint32_t block_size,
                   uint64_t block_count, XorShift64Star &random,
                   std::vector<OVERLAPPED_ENTRY> &entries,
                   uint32_t &outstanding, bool keep_queue_full,
                   RunResult *result, std::string &error) {
  while (counter_now() < deadline) {
    ULONG removed = 0;
    const BOOL dequeued = ::GetQueuedCompletionStatusEx(
        completion_port, entries.data(), static_cast<ULONG>(entries.size()),
        &removed, INFINITE, FALSE);
    if (!dequeued || removed == 0) {
      error = "GetQueuedCompletionStatusEx failed with error " +
              std::to_string(::GetLastError());
      return false;
    }

    if (result != nullptr) {
      ++result->dequeue_calls;
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
      if (!validate_completion(entry, block_size, error)) {
        return false;
      }
      Request *request = reinterpret_cast<Request *>(entry.lpOverlapped);
      if (result != nullptr) {
        const uint64_t completed_at = counter_now();
        const double latency =
            static_cast<double>(completed_at - request->submitted_at) * 1000.0 /
            static_cast<double>(frequency);
        result->latency_ms.push_back(latency);
        ++result->completed;
      }

      if ((keep_queue_full || counter_now() < deadline) &&
          !submit_read(file_handle, *request, block_size, block_count, random,
                       outstanding, result, error)) {
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

RunResult run_benchmark(const Options &options, uint32_t queue_depth,
                        bool random_access_hint) {
  RunResult result;
  result.random_access_hint = random_access_hint;
  result.queue_depth = queue_depth;
  const size_t expected_completions = std::min<size_t>(
      static_cast<size_t>(options.duration_seconds) * 50000U, 10000000U);
  result.latency_ms.reserve(expected_completions);

  DWORD flags =
      FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
  if (random_access_hint) {
    flags |= FILE_FLAG_RANDOM_ACCESS;
  }

  VirtualBuffer buffer(static_cast<size_t>(queue_depth) * options.block_size);
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
        buffer.data() + static_cast<size_t>(index) * options.block_size;
  }

  const uint64_t frequency = counter_frequency();
  XorShift64Star random(options.seed + queue_depth);
  uint32_t outstanding = 0;
  std::string error;
  for (uint32_t index = 0; index < queue_depth; ++index) {
    if (!submit_read(file_handle.get(), requests[index], options.block_size,
                     block_count, random, outstanding, nullptr, error)) {
      ::CancelIoEx(file_handle.get(), nullptr);
      drain_requests(completion_port.get(), entries, outstanding, error);
      throw std::runtime_error(error);
    }
  }

  const uint64_t warmup_deadline =
      counter_now() + static_cast<uint64_t>(options.warmup_seconds) * frequency;
  if (!process_until(file_handle.get(), completion_port.get(), warmup_deadline,
                     frequency, options.block_size, block_count, random,
                     entries, outstanding, true, nullptr, error)) {
    ::CancelIoEx(file_handle.get(), nullptr);
    drain_requests(completion_port.get(), entries, outstanding, error);
    throw std::runtime_error(error);
  }

  const uint64_t measurement_start = counter_now();
  const uint64_t measurement_deadline =
      measurement_start +
      static_cast<uint64_t>(options.duration_seconds) * frequency;
  if (!process_until(file_handle.get(), completion_port.get(),
                     measurement_deadline, frequency, options.block_size,
                     block_count, random, entries, outstanding, false, &result,
                     error)) {
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

double percentile(const std::vector<double> &sorted_values, double fraction) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double position =
      fraction * static_cast<double>(sorted_values.size() - 1);
  return sorted_values[static_cast<size_t>(position)];
}

void print_result(RunResult result, uint32_t block_size) {
  std::sort(result.latency_ms.begin(), result.latency_ms.end());
  const double latency_sum =
      std::accumulate(result.latency_ms.begin(), result.latency_ms.end(), 0.0);
  const double average_latency =
      result.latency_ms.empty()
          ? 0.0
          : latency_sum / static_cast<double>(result.latency_ms.size());
  const double iops =
      static_cast<double>(result.completed) / result.elapsed_seconds;
  const double mib_per_second =
      iops * static_cast<double>(block_size) / (1024.0 * 1024.0);
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

  std::cout << (result.random_access_hint ? "on" : "off") << ','
            << result.queue_depth << ',' << std::fixed << std::setprecision(2)
            << iops << ',' << mib_per_second << ',' << average_latency << ','
            << percentile(result.latency_ms, 0.50) << ','
            << percentile(result.latency_ms, 0.95) << ','
            << percentile(result.latency_ms, 0.99) << ','
            << result.latency_ms.front() << ',' << result.latency_ms.back()
            << ',' << effective_queue_depth << ',' << completions_per_dequeue
            << ',' << result.max_dequeued << ',' << pending_ratio << ','
            << result.completed << '\n';
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
    std::cout << "DiskANN Windows IOCP microbenchmark\n"
              << "file: " << wide_to_utf8(options.file_path) << '\n'
              << "block_size: " << options.block_size << " bytes\n"
              << "warmup: " << options.warmup_seconds << " seconds/run\n"
              << "duration: " << options.duration_seconds << " seconds/run\n\n"
              << "random_access_hint,queue_depth,iops,mib_per_s,avg_latency_"
                 "ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,min_latency_"
                 "ms,max_latency_ms,effective_qd,completions_per_dequeue,max_"
                 "dequeued,pending_ratio_pct,completed_reads\n";

    const std::vector<bool> modes = hint_modes(options.random_access_hint);
    for (bool random_access_hint : modes) {
      for (uint32_t queue_depth : options.queue_depths) {
        print_result(run_benchmark(options, queue_depth, random_access_hint),
                     options.block_size);
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

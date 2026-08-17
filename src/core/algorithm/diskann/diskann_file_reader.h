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

#define MAX_IO_DEPTH 128

#include <fcntl.h>

#if (defined(__linux) || defined(__linux__))
#include <ailego/io/iouring_loader.h>  // raw-syscall io_uring wrapper (IoUringRing)
#include <ailego/io/libaio_loader.h>  // dlopen-based libaio wrapper
#elif defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <Windows.h>
#endif

#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#endif
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/core/framework/index_context.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

// IoBackend holds the selected backend for each thread. On Linux it also owns
// the resources required by the asynchronous backends. The priority is:
//   1. io_uring  (raw kernel syscalls — zero dependency)
//   2. libaio    (dlopen — soft dependency)
//   3. pread     (always available — synchronous fallback)
//
// macOS uses a real context with type kPread so that the active backend can be
// inspected and reported consistently instead of using an opaque placeholder.
// Windows stores a private file handle, completion port, and stable OVERLAPPED
// request slots in each thread context. Keeping completion ports private
// prevents one search thread from consuming another thread's completions.
// IOContext is a pointer to IoBackend, which preserves the existing
// sentinel conventions: nullptr means uninitialised and (IOContext)-1 is
// the invalid-handle sentinel returned by get_ctx() for unregistered
// threads.
struct IoBackend {
  ailego::IOBackendType type{ailego::IOBackendType::kPread};

#if (defined(__linux) || defined(__linux__))
  IoUringRing ring{};
  io_context_t aio_ctx{nullptr};
#elif defined(_WIN32) || defined(_WIN64)
  enum class RequestMode {
    kIdle,
    kBatch,
    kStreaming,
  };

  struct IocpDiagnostics {
    uint64_t submit_calls{0};
    uint64_t submitted_reads{0};
    uint64_t immediate_reads{0};
    uint64_t pending_reads{0};
    uint64_t dequeue_calls{0};
    uint64_t dequeued_reads{0};
    uint64_t wait_us{0};
    uint64_t readfile_submit_ns{0};
    uint64_t get_overlapped_ns{0};
    uint64_t batch_count{0};
    uint64_t batch_submit_ns{0};
    uint64_t batch_first_completion_ns{0};
    uint64_t batch_duration_ns{0};
    uint32_t max_outstanding{0};
    uint32_t max_dequeued_once{0};
    uint64_t active_batch_started_ns{0};
    bool active_batch_first_completion_recorded{false};
  };

  std::vector<OVERLAPPED> reqs;
  std::vector<uint64_t> slot_expected_lengths;
  std::vector<uint64_t> slot_tags;
  std::vector<uint8_t> slot_active;
  HANDLE file_handle{INVALID_HANDLE_VALUE};
  HANDLE completion_port{nullptr};
  std::wstring file_path;
  uint32_t submitted_count{0};
  uint32_t outstanding_count{0};
  uint64_t generation{0};
  RequestMode request_mode{RequestMode::kIdle};
  bool diagnostics_enabled{false};
  IocpDiagnostics diagnostics;
#endif
};

typedef IoBackend *IOContext;

int setup_io_ctx(IOContext &ctx);
int destroy_io_ctx(IOContext &ctx);

// Diagnostics are opt-in so production searches do not pay for timing and
// histogram collection. Set ZVEC_DISKANN_IO_DIAGNOSTICS=1 before starting the
// process to enable them.
bool diskann_io_diagnostics_enabled();

// Experimental Windows-only rolling pipeline. It is deliberately opt-in
// because processing nodes in completion order can change approximate-search
// traversal and recall. Set ZVEC_DISKANN_IO_PIPELINE=1 to enable it.
bool diskann_io_pipeline_enabled();

// Experimental Windows-only batched completion mode. The default search path
// processes each group of completed reads before dequeuing more completions.
// Set ZVEC_DISKANN_IO_DRAIN_FIRST=1 to drain the active batch first and process
// its nodes afterwards, preserving the observed completion order.
bool diskann_io_drain_first_enabled();

// Log the current DiskAnn I/O backend (io_uring, libaio, or pread). Probes the
// backend on first call. No-op outside Linux and macOS.
void log_diskann_io_backend();

struct AlignedRead {
  uint64_t offset;
  uint64_t len;
  void *buf;

  AlignedRead() : offset(0), len(0), buf(nullptr) {}

  AlignedRead(uint64_t offset, uint64_t len, void *buf)
      : offset(offset), len(len), buf(buf) {
#if defined(__linux__) || defined(__linux)
    // O_DIRECT requires 512-byte alignment on Linux.
    ailego_assert(static_cast<size_t>(offset) % 512 == 0);
    ailego_assert(static_cast<size_t>(len) % 512 == 0);
    ailego_assert(reinterpret_cast<size_t>(buf) % 512 == 0);
#endif
  }
};

struct PendingBatch {
#if (defined(__linux) || defined(__linux__))
  std::vector<struct iocb> cbs;
  std::vector<struct iocb *> cb_ptrs;
#elif defined(_WIN32) || defined(_WIN64)
  std::vector<uint64_t> expected_lengths;
  std::vector<uint8_t> completed;
  uint64_t generation{0};
#endif
  uint32_t n_submitted{0};
  uint32_t n_reaped{0};
  bool used_pread{false};
};

class AlignedFileReader {
 protected:
  std::map<std::thread::id, IOContext> ctx_map;
  std::mutex ctx_mut;

 public:
  virtual IOContext &get_ctx() = 0;

  virtual ~AlignedFileReader() {}

  virtual void register_thread() = 0;
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;

  virtual void open(const std::string &fname) = 0;
  virtual void close() = 0;

  virtual int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
                   bool async = false) = 0;

  virtual int submit(PendingBatch &batch, std::vector<AlignedRead> &read_reqs,
                     IOContext &ctx) = 0;

  virtual int get_completed(PendingBatch &batch, IOContext &ctx,
                            int min_completed,
                            std::vector<uint32_t> &completed_indices) = 0;
};

// POSIX reader implementation. Linux selects io_uring, libaio, or pread;
// macOS ARM64 uses synchronous pread.
#if !defined(_WIN32) && !defined(_WIN64)
class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  int file_desc;

  IOContext bad_ctx = (IOContext)-1;

 public:
  LinuxAlignedFileReader();
  LinuxAlignedFileReader(int file_desc);
  ~LinuxAlignedFileReader();

 public:
  IOContext &get_ctx();

  void register_thread();
  void deregister_thread();
  void deregister_all_threads();
  void open(const std::string &fname);
  void close();

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false);

  int submit(PendingBatch &batch, std::vector<AlignedRead> &read_reqs,
             IOContext &ctx);

  int get_completed(PendingBatch &batch, IOContext &ctx, int min_completed,
                    std::vector<uint32_t> &completed_indices);
};
#else
class WindowsAlignedFileReader : public AlignedFileReader {
 private:
  std::wstring file_path_;
  IOContext bad_ctx{reinterpret_cast<IOContext>(-1)};

  int prepare_io_ctx(IOContext &ctx);
  void reset_io_ctx(IOContext &ctx);

 public:
  WindowsAlignedFileReader();
  ~WindowsAlignedFileReader() override;

  IOContext &get_ctx() override;
  void register_thread() override;
  void deregister_thread() override;
  void deregister_all_threads() override;
  void open(const std::string &fname) override;
  void close() override;

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false) override;
  int submit(PendingBatch &batch, std::vector<AlignedRead> &read_reqs,
             IOContext &ctx) override;
  int get_completed(PendingBatch &batch, IOContext &ctx, int min_completed,
                    std::vector<uint32_t> &completed_indices) override;

  // Streaming requests may be added while earlier requests are still in
  // flight. The caller-provided tag is returned with the completion so search
  // buffers can be safely reused without exposing native OVERLAPPED slots.
  int submit_streaming(const AlignedRead &read_req, uint64_t tag,
                       IOContext &ctx);
  int get_streaming_completed(IOContext &ctx, int min_completed,
                              std::vector<uint64_t> &completed_tags);
};
#endif

#if defined(_WIN32) || defined(_WIN64)
using PlatformAlignedFileReader = WindowsAlignedFileReader;
#else
using PlatformAlignedFileReader = LinuxAlignedFileReader;
#endif

}  // namespace core
}  // namespace zvec

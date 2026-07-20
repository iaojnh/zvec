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
#include <ailego/io/libaio_loader.h>  // dlopen-based libaio wrapper
#endif

#include <unistd.h>
#include <atomic>
#include <vector>
#include <zvec/core/framework/index_context.h>
#include "diskann_util.h"

namespace zvec {
namespace ailego {
// Forward declaration only: including vector_page_table.h here would pull in
// the <zvec/ailego/io/libaio_loader.h> definitions, which clash with the
// <ailego/io/libaio_loader.h> copy this header already relies on. The buffer
// pool is used only as an opaque pointer here; the concrete calls live in the
// shim TU (diskann_buffer_pool_shim.cc).
class VecBufferPool;
}  // namespace ailego
}  // namespace zvec

namespace zvec {
namespace core {

#if (defined(__linux) || defined(__linux__))
typedef io_context_t IOContext;
#else
typedef uint32_t IOContext;
#endif

int setup_io_ctx(IOContext &ctx);
int destroy_io_ctx(IOContext &ctx);

// Log the current DiskAnn I/O backend status (async vs. synchronous pread).
// Probes the backend on first call.  No-op on non-Linux platforms.
void log_diskann_io_backend();

struct AlignedRead {
  uint64_t offset;
  uint64_t len;
  void *buf;

  AlignedRead() : offset(0), len(0), buf(nullptr) {}

  AlignedRead(uint64_t offset, uint64_t len, void *buf)
      : offset(offset), len(len), buf(buf) {
    ailego_assert(static_cast<size_t>(offset) % 512 == 0);
    ailego_assert(static_cast<size_t>(len) % 512 == 0);
    ailego_assert(reinterpret_cast<size_t>(buf) % 512 == 0);
  }
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
};

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
};

// AlignedFileReader that satisfies DiskAnn sector reads through a
// VecBufferPool paged cache instead of O_DIRECT + libaio.  A DiskAnn sector
// (4KB) maps 1:1 to a VecBufferPool page, so each AlignedRead is expanded into
// one or more page ids, acquired in a single batch (preserving the pool's
// batched miss / AIO path), copied into the caller's buffer, then released.
// The pool manages its own AIO context, so thread registration / file open /
// IOContext are all no-ops here.
class BufferPoolAlignedFileReader : public AlignedFileReader {
 public:
  explicit BufferPoolAlignedFileReader(ailego::VecBufferPool *pool);
  ~BufferPoolAlignedFileReader() override;

  IOContext &get_ctx() override;

  void register_thread() override;
  void deregister_thread() override;
  void deregister_all_threads() override;
  void open(const std::string &fname) override;
  void close() override;

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false) override;

 private:
  ailego::VecBufferPool *pool_{nullptr};
  IOContext unused_ctx_{};
};

}  // namespace core
}  // namespace zvec

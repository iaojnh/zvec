// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <limits>
#include <memory>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

// Read-only storage for immutable files. All payload capacity is charged to
// MemoryLimitPool before allocation. Atomic chunk pins protect readers while
// the process-wide BlockEvictionQueue controls cache residency.
class ChunkReadStorage : public IndexStorage {
 public:
  struct Stats {
    uint64_t cache_hits{0}, cache_misses{0}, fills{0}, replacements{0};
    uint64_t io_calls{0}, io_bytes{0}, copied_bytes{0}, workspace_waits{0};
    uint64_t trims{0}, cache_bytes{0}, workspace_bytes{0}, metadata_bytes{0};
    uint64_t prefetch_submitted{0}, prefetch_consumed{0}, prefetch_failed{0};
    uint64_t prefetch_wait_ns{0}, prefetch_peak_requests{0},
        prefetch_peak_bytes{0};
    uint64_t prefetch_depth{0}, prefetch_limit_bytes{0};
    uint64_t prefetch_pending_requests{0}, prefetch_pending_bytes{0};
    uint64_t prefetch_read_bytes{0}, prefetch_submit_batches{0};
  };

  ChunkReadStorage();
  ~ChunkReadStorage() override;
  int init(const ailego::Params &) override;
  int cleanup() override;
  int open(const std::string &, bool) override;
  int close() override;
  int flush() override;
  int append(const std::string &, size_t) override;
  void refresh(uint64_t) override;
  uint64_t check_point() const override;
  Segment::Pointer get(const std::string &, int = 0) override;
  std::map<std::string, Segment::Pointer> get_all() const override;
  bool has(const std::string &) const override;
  uint32_t magic() const override;
  MemoryBlock::MemoryBlockType memory_block_type() const override;
  std::string file_path() const override;
  Stats stats() const;
  // Disabling hints does not discard an already submitted read. Demand reads
  // still join it, and teardown drains it before releasing any backing.
  void enable_prefetch(bool enabled);

 private:
  struct State;
  class ReadSegment;
  std::shared_ptr<State> state_;
  size_t chunk_bytes_{0};
  size_t cache_bytes_{std::numeric_limits<size_t>::max()};
  size_t max_read_bytes_{4UL << 20};
  size_t workspace_count_{8};
  size_t wait_ms_{100};
  size_t prefetch_depth_{0}, prefetch_bytes_{16UL << 20};
  int64_t header_offset_{0}, footer_offset_{0};
  bool checksum_{false}, direct_io_{true}, adaptive_{false};
};

// Factory name: ChunkReadStorage. Existing BufferReadStorage parameters for
// container offsets and checksum validation also apply.
inline constexpr const char *CHUNK_READ_STORAGE_CHUNK_BYTES =
    "proxima.chunk_read_storage.chunk_bytes";
inline constexpr const char *CHUNK_READ_STORAGE_CACHE_BYTES =
    "proxima.chunk_read_storage.cache_bytes";
inline constexpr const char *CHUNK_READ_STORAGE_MAX_READ_BYTES =
    "proxima.chunk_read_storage.max_read_bytes";
inline constexpr const char *CHUNK_READ_STORAGE_WORKSPACE_COUNT =
    "proxima.chunk_read_storage.workspace_count";
inline constexpr const char *CHUNK_READ_STORAGE_WAIT_MS =
    "proxima.chunk_read_storage.wait_ms";
inline constexpr const char *CHUNK_READ_STORAGE_DIRECT_IO =
    "proxima.chunk_read_storage.direct_io";
// Size initial workspaces to the file and shared budget. Larger reads
// may grow a leased execution slot through normal workspace admission.
inline constexpr const char *CHUNK_READ_STORAGE_ADAPTIVE =
    "proxima.chunk_read_storage.adaptive";
// Linux direct I/O lookahead. Zero depth preserves synchronous operation.
// Both limits apply across all readers sharing this storage, including
// completed reads whose buffers have not yet been consumed.
inline constexpr const char *CHUNK_READ_STORAGE_PREFETCH_DEPTH =
    "proxima.chunk_read_storage.prefetch_depth";
inline constexpr const char *CHUNK_READ_STORAGE_PREFETCH_BYTES =
    "proxima.chunk_read_storage.prefetch_bytes";
}  // namespace core
}  // namespace zvec

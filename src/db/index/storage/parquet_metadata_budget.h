// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <arrow/io/interfaces.h>
#include <parquet/file_reader.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/common/metadata_memory.h"

namespace zvec::detail {

// Thrift metadata uses standard containers, not Arrow's MemoryPool. This is
// an explicit planning allowance, not exact allocator accounting or an RSS
// limit. Keep the concurrent high water charged after close: free() need not
// return backing pages, and reopening can reuse the allowance.
class ParquetMetadataBudget : public RetainedMetadataBudget {
 public:
  static size_t estimate(size_t footer_bytes) {
    // Encoded footer, decoded strings/nodes, reader directories and allocator
    // growth. This policy must be revisited if the Parquet schema changes;
    // it is deliberately not advertised as a bound for arbitrary Thrift data.
    constexpr size_t kReaderOverhead = 64 * 1024;
    constexpr size_t kFooterMultiplier = 4;
    if (footer_bytes > (std::numeric_limits<size_t>::max() - kReaderOverhead) /
                           kFooterMultiplier) {
      throw std::bad_alloc();
    }
    return footer_bytes * kFooterMultiplier + kReaderOverhead;
  }
};

inline arrow::Result<std::unique_ptr<parquet::ParquetFileReader>>
OpenBudgetedParquetReader(
    const std::shared_ptr<arrow::io::RandomAccessFile> &file,
    const parquet::ReaderProperties &properties,
    std::shared_ptr<parquet::FileMetaData> metadata = nullptr) {
  try {
    if (metadata) {
      return parquet::ParquetFileReader::Open(file, properties,
                                              std::move(metadata));
    }
    ARROW_ASSIGN_OR_RAISE(const int64_t file_size, file->GetSize());
    if (file_size < 12) return arrow::Status::Invalid("Invalid Parquet size");
    std::array<uint8_t, 8> footer{};
    ARROW_ASSIGN_OR_RAISE(const auto read,
                          file->ReadAt(file_size - 8, 8, footer.data()));
    if (read != 8 || std::memcmp(footer.data() + 4, "PAR1", 4) != 0) {
      return arrow::Status::Invalid("Invalid Parquet footer");
    }
    const uint32_t size = uint32_t{footer[0]} | (uint32_t{footer[1]} << 8) |
                          (uint32_t{footer[2]} << 16) |
                          (uint32_t{footer[3]} << 24);
    if (size > static_cast<uint64_t>(file_size - 12)) {
      return arrow::Status::Invalid("Parquet footer exceeds file size");
    }
    struct Owner {
      ParquetMetadataBudget::Lease lease{ParquetMetadataBudget::instance()};
      std::shared_ptr<parquet::FileMetaData> metadata;
    };
    auto owner = std::make_shared<Owner>();
    owner->lease.acquire(ParquetMetadataBudget::estimate(size));
    auto reader = parquet::ParquetFileReader::Open(file, properties);
    owner->metadata = reader->metadata();
    // All readers (including scans and cache misses) share the same lease;
    // closing the forward store must not retire a still-borrowed footer.
    auto *raw_metadata = owner->metadata.get();
    auto owned_metadata =
        std::shared_ptr<parquet::FileMetaData>(owner, raw_metadata);
    return parquet::ParquetFileReader::Open(file, properties,
                                            std::move(owned_metadata));
  } catch (const std::bad_alloc &) {
    return arrow::Status::OutOfMemory("Parquet metadata budget exhausted");
  } catch (const parquet::ParquetException &error) {
    return arrow::Status::Invalid(error.what());
  }
}

}  // namespace zvec::detail

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

#include <cstring>
#include <memory>
#include <new>
#include <string_view>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_format.h>

namespace zvec {
namespace core {
namespace index_metadata {

struct BufferDeleter {
  size_t charge{0};

  void operator()(char *data) const {
    delete[] data;
    if (charge != 0) {
      ailego::MemoryLimitPool::get_instance().release_external(charge);
    }
  }
};

using Buffer = std::unique_ptr<char[], BufferDeleter>;

inline Buffer AllocateBuffer(size_t size) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  // Capacity zero disables the shared pool. File-backed readers still work.
  const size_t charge = pool.capacity() == 0 ? 0 : size;
  using Pending = ailego::MemoryLimitPool::PendingAllocation;
  Pending pending(pool, Pending::Kind::kWorkspace);
  if (!pending.reserve(charge)) {
    throw std::bad_alloc();
  }
  Buffer buffer(new char[size], BufferDeleter{charge});
  pending.commit();
  return buffer;
}

// A container can hold either a packed index (footer after content, relative
// to the end) or a chain of mutable sections (footer before content).
inline int FooterOffset(const IndexFormat::MetaHeader &header, size_t remaining,
                        size_t *offset) {
  if (header.meta_header_size != sizeof(header) ||
      header.meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
    return IndexError_InvalidLength;
  }
  if (ailego::Crc32c::Hash(&header, sizeof(header), header.header_crc) !=
      header.header_crc) {
    return IndexError_InvalidChecksum;
  }
  const auto relative = static_cast<int32_t>(header.meta_footer_offset);
  const uint64_t distance = relative < 0 ? -static_cast<int64_t>(relative) : 0;
  if (distance > remaining) {
    return IndexError_InvalidLength;
  }
  *offset = relative < 0 ? remaining - distance : header.meta_footer_offset;
  if (*offset < sizeof(header) || *offset > remaining ||
      sizeof(IndexFormat::MetaFooter) > remaining - *offset) {
    return IndexError_InvalidLength;
  }
  return 0;
}

inline int CheckFooter(const IndexFormat::MetaHeader &header,
                       const IndexFormat::MetaFooter &footer,
                       size_t footer_offset, size_t remaining) {
  if (ailego::Crc32c::Hash(&footer, sizeof(footer), footer.footer_crc) !=
      footer.footer_crc) {
    return IndexError_InvalidChecksum;
  }
  if (footer.total_size > remaining ||
      footer_offset + sizeof(footer) > footer.total_size ||
      footer.segments_meta_size > footer_offset - sizeof(header) ||
      footer.segment_count >
          footer.segments_meta_size / sizeof(IndexFormat::SegmentMeta)) {
    return IndexError_InvalidLength;
  }
  const bool trailing = static_cast<int32_t>(header.meta_footer_offset) < 0;
  const auto content_end =
      trailing ? footer_offset - footer.segments_meta_size : footer.total_size;
  const auto content_start =
      trailing ? sizeof(header) : footer_offset + sizeof(footer);
  if (header.content_offset < content_start ||
      header.content_offset > content_end ||
      footer.content_size > content_end - header.content_offset ||
      footer.content_padding_size >
          content_end - header.content_offset - footer.content_size) {
    return IndexError_InvalidLength;
  }
  return 0;
}

inline int NextSection(const IndexFormat::MetaFooter &footer, size_t start,
                       size_t total, size_t *next) {
  const auto offset = footer.next_meta_header_offset;
  if (offset != 0 && (offset < start + footer.total_size || offset > total ||
                      sizeof(IndexFormat::MetaHeader) > total - offset)) {
    return IndexError_InvalidLength;
  }
  *next = static_cast<size_t>(offset);
  return 0;
}

inline int CheckTable(const IndexFormat::MetaFooter &footer, const void *data) {
  const auto crc =
      footer.segments_meta_size == 0
          ? 0u
          : ailego::Crc32c::Hash(data, footer.segments_meta_size, 0u);
  return crc == footer.segments_meta_crc ? 0 : IndexError_InvalidChecksum;
}

inline int SegmentId(const IndexFormat::MetaFooter &footer, const void *table,
                     const IndexFormat::SegmentMeta &segment,
                     std::string_view *id) {
  const size_t records = sizeof(segment) * footer.segment_count;
  if (segment.segment_id_offset < records ||
      segment.segment_id_offset >= footer.segments_meta_size) {
    return IndexError_InvalidValue;
  }
  if (segment.data_index > footer.content_size ||
      segment.data_size > footer.content_size - segment.data_index ||
      segment.padding_size >
          footer.content_size - segment.data_index - segment.data_size) {
    return IndexError_InvalidLength;
  }
  const auto *begin =
      static_cast<const char *>(table) + segment.segment_id_offset;
  const auto *end = static_cast<const char *>(std::memchr(
      begin, 0, footer.segments_meta_size - segment.segment_id_offset));
  if (!end) {
    return IndexError_InvalidValue;
  }
  *id = std::string_view(begin, end - begin);
  return 0;
}

// Offset of either endpoint of an index embedded in a larger file.
inline bool ResolveContainerOffset(size_t file_size, int64_t configured_offset,
                                   bool zero_from_end, size_t *resolved) {
  if (configured_offset > 0 || (configured_offset == 0 && !zero_from_end)) {
    const auto offset = static_cast<uint64_t>(configured_offset);
    if (offset > file_size) {
      return false;
    }
    *resolved = static_cast<size_t>(offset);
  } else {
    const auto distance = static_cast<uint64_t>(-(configured_offset + 1)) + 1;
    if (distance > file_size) {
      return false;
    }
    *resolved = file_size - static_cast<size_t>(distance);
  }
  return true;
}

}  // namespace index_metadata
}  // namespace core
}  // namespace zvec

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

#include <map>
#include <string_view>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/container/lazy_map.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/type_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_format.h>

namespace zvec {
namespace core {

/*! Index Unpacker
 */
class IndexUnpacker {
 public:
  /*! Index Unpacker Segment Meta
   */
  class SegmentMeta {
   public:
    //! Constructor
    SegmentMeta(size_t offset, size_t dsz, size_t psz, uint32_t crc)
        : data_offset_(offset),
          data_size_(dsz),
          padding_size_(psz),
          data_crc_(crc) {}

    //! Retrieve offset of data
    size_t data_offset() const {
      return data_offset_;
    }

    //! Retrieve size of data
    size_t data_size() const {
      return data_size_;
    }

    //! Retrieve crc of data
    uint32_t data_crc() const {
      return data_crc_;
    }

    //! Retrieve size of padding
    size_t padding_size() const {
      return padding_size_;
    }

   private:
    size_t data_offset_{0};
    size_t data_size_{0};
    size_t padding_size_{0};
    uint32_t data_crc_{0};
  };

  struct SegmentNameLess {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const {
      return a < b;
    }
  };
  using SegmentMap = ailego::LazyMap<
      std::map<ailego::BudgetedString, SegmentMeta, SegmentNameLess,
               ailego::BudgetedAllocator<
                   std::pair<const ailego::BudgetedString, SegmentMeta>>>>;

  //! Reset the unpacker
  void reset() {
    header_ = {};
    footer_ = {};
    current_header_start_offset_ = 0;
    segments_.clear();
    version_ = ailego::BudgetedStringBuffer{};
  }

  //! Retrieve segments of index package
  const SegmentMap &segments() const {
    return segments_;
  }

  //! Retrieve magic number of index
  uint32_t magic() const {
    return header_.magic;
  }

  //! Retrieve header of index package
  const IndexFormat::MetaHeader &header() const {
    return header_;
  }

  //! Retrieve footer of index package
  const IndexFormat::MetaFooter &footer() const {
    return footer_;
  }

  //! Retrieve version information
  const std::string &version() const {
    return version_.data;
  }

  //! Retrieve mutable segments of index package
  SegmentMap *mutable_segments() {
    return &segments_;
  }

  //! Unpack index data
  template <typename TFunc>
  bool unpack(TFunc read_data, size_t total, bool checksum,
              bool report_errors = true, bool retry_private = false) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");

    while (true) {
      if (current_header_start_offset_ > total ||
          sizeof(header_) > total - current_header_start_offset_ ||
          !this->unpack_header(read_data)) {
        if (report_errors) LOG_ERROR("Failed to unpack index header");
        return false;
      }
      if (!this->unpack_footer(read_data, total)) {
        if (report_errors) LOG_ERROR("Failed to unpack index footer");
        return false;
      }
      if (!this->unpack_segments(read_data, total, retry_private)) {
        if (report_errors) LOG_ERROR("Failed to unpack index segments' meta");
        return false;
      }
      if (checksum && !this->validate_checksum(read_data)) {
        if (report_errors)
          LOG_ERROR("Failed to validate checksum of index content");
        return false;
      }
      if (footer_.next_meta_header_offset == 0) {
        break;
      }
      if (footer_.next_meta_header_offset <
              current_header_start_offset_ + footer_.total_size ||
          footer_.next_meta_header_offset >= total)
        return false;
      current_header_start_offset_ = footer_.next_meta_header_offset;
    }

    if (!this->unpack_version(read_data, retry_private)) {
      if (report_errors) LOG_ERROR("Failed to unpack index version");
      return false;
    }
    return true;
  }

  //! Unpack index header
  template <typename TFunc>
  bool unpack_header(TFunc read_data) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    const void *data = nullptr;
    if (read_data(current_header_start_offset_, &data, sizeof(header_)) !=
        sizeof(header_)) {
      return false;
    }

    memcpy(&header_, data, sizeof(header_));
    if (header_.meta_header_size != sizeof(header_)) {
      return false;
    }
    if (ailego::Crc32c::Hash(&header_, sizeof(header_), header_.header_crc) !=
        header_.header_crc) {
      return false;
    }
    return true;
  }

  //! Unpack index footer
  template <typename TFunc>
  bool unpack_footer(TFunc read_data, size_t total) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    if (header_.meta_footer_size != sizeof(footer_)) {
      return false;
    }

    size_t footer_offset = 0;
    if (!resolve_footer_offset(total, &footer_offset)) return false;

    const void *data = nullptr;
    if (read_data(current_header_start_offset_ + footer_offset, &data,
                  sizeof(footer_)) != sizeof(footer_)) {
      return false;
    }

    memcpy(&footer_, data, sizeof(footer_));
    if (footer_.total_size > total - current_header_start_offset_ ||
        footer_offset + sizeof(footer_) > footer_.total_size ||
        header_.content_offset < sizeof(header_) ||
        header_.content_offset > footer_.total_size ||
        footer_.content_size > footer_.total_size - header_.content_offset ||
        footer_.content_padding_size > footer_.total_size -
                                           header_.content_offset -
                                           footer_.content_size) {
      return false;
    }
    if (ailego::Crc32c::Hash(&footer_, sizeof(footer_), footer_.footer_crc) !=
        footer_.footer_crc) {
      return false;
    }
    return true;
  }

  //! Unpack segments' meta
  template <typename TFunc>
  bool unpack_segments(TFunc read_data, size_t total,
                       bool retry_private = false) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    using Entry = IndexFormat::SegmentMeta;
    if (footer_.segment_count > footer_.segments_meta_size / sizeof(Entry))
      return false;
    size_t offset = 0;
    if (!resolve_footer_offset(total, &offset) ||
        footer_.segments_meta_size > offset - sizeof(header_))
      return false;
    const size_t footer_end = offset + sizeof(footer_);
    offset -= footer_.segments_meta_size;
    // Packed files place the directory after the payload. Writable files
    // reserve the directory before it. Neither layout permits overlap.
    const uint64_t payload_extent =
        footer_.content_size + footer_.content_padding_size;
    if (header_.content_offset < footer_end &&
        header_.content_offset + payload_extent > offset)
      return false;

    const void *data = nullptr;
    if (read_data(current_header_start_offset_ + offset, &data,
                  footer_.segments_meta_size) != footer_.segments_meta_size) {
      return false;
    }
    if (ailego::Crc32c::Hash(data, footer_.segments_meta_size, 0u) !=
        footer_.segments_meta_crc) {
      return false;
    }

    for (size_t i = 0; i < footer_.segment_count;) {
      // A package embedded in another segment can start at any byte offset.
      // Decode the entry before interpreting its fields.
      Entry entry;
      const auto *bytes = static_cast<const char *>(data);
      std::memcpy(&entry, bytes + i * sizeof(entry), sizeof(entry));
      if (entry.segment_id_offset < sizeof(Entry) * footer_.segment_count ||
          entry.segment_id_offset >= footer_.segments_meta_size ||
          entry.data_index > footer_.content_size ||
          entry.data_size > footer_.content_size - entry.data_index ||
          entry.padding_size >
              payload_extent - entry.data_index - entry.data_size) {
        return false;
      }
      const char *name = bytes + entry.segment_id_offset;
      const size_t max_length =
          footer_.segments_meta_size - entry.segment_id_offset;
      const size_t length = ::strnlen(name, max_length);
      if (length == max_length) return false;
      try {
        segments_.emplace(
            ailego::BudgetedString(name, length),
            SegmentMeta(entry.data_index + header_.content_offset +
                            current_header_start_offset_,
                        entry.data_size, entry.padding_size, entry.data_crc));
        ++i;
      } catch (const std::bad_alloc &) {
        // Keep the already-required directory prefix. Change only this
        // input's representation, then rebuild all pointers into its bytes.
        if (!retry_input(read_data, retry_private,
                         current_header_start_offset_ + offset, &data,
                         footer_.segments_meta_size) ||
            ailego::Crc32c::Hash(data, footer_.segments_meta_size, 0u) !=
                footer_.segments_meta_crc)
          return false;
      }
    }
    return true;
  }

  //! Unpack index version
  template <typename TFunc>
  bool unpack_version(TFunc read_data, bool retry_private = false) {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");

    auto it = segments_.find("IndexVersion");
    if (it == segments_.end()) {
      return false;
    }

    const SegmentMeta &segment = it->second;
    const void *data = nullptr;

    if (read_data(0 + segment.data_offset(), &data, segment.data_size()) !=
        segment.data_size()) {
      return false;
    }
    if (segment.data_crc() != 0u &&
        ailego::Crc32c::Hash(data, segment.data_size(), 0u) !=
            segment.data_crc()) {
      return false;
    }
    for (;;) {
      try {
        version_.reserve(segment.data_size());
        version_.data.assign(reinterpret_cast<const char *>(data),
                             segment.data_size());
        return true;
      } catch (const std::bad_alloc &) {
        if (!retry_input(read_data, retry_private, segment.data_offset(), &data,
                         segment.data_size()) ||
            (segment.data_crc() != 0u &&
             ailego::Crc32c::Hash(data, segment.data_size(), 0u) !=
                 segment.data_crc()))
          return false;
      }
    }
  }

  //! Validate checksum of content
  template <typename TFunc>
  bool validate_checksum(TFunc read_data) const {
    static_assert(ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t>::value,
                  "Invocable function type");
    if (footer_.content_crc == 0) {
      return true;
    }
    const size_t block_size = 4096u;
    const void *data = nullptr;
    uint32_t checksum = 0u;
    size_t total = footer_.content_size;
    size_t offset = header_.content_offset;

    while (total >= block_size) {
      if (read_data(current_header_start_offset_ + offset, &data, block_size) !=
          block_size) {
        return false;
      }
      checksum = ailego::Crc32c::Hash(data, block_size, checksum);
      total -= block_size;
      offset += block_size;
    }
    if (read_data(current_header_start_offset_ + offset, &data, total) !=
        total) {
      return false;
    }
    checksum = ailego::Crc32c::Hash(data, total, checksum);
    return (checksum == footer_.content_crc);
  }

 private:
  bool resolve_footer_offset(size_t total, size_t *offset) const {
    if (current_header_start_offset_ > total) return false;
    const size_t remaining = total - current_header_start_offset_;
    const int64_t encoded = static_cast<int32_t>(header_.meta_footer_offset);
    if (encoded < 0) {
      const size_t distance = static_cast<size_t>(-encoded);
      if (distance > remaining) return false;
      *offset = remaining - distance;
    } else {
      *offset = static_cast<size_t>(encoded);
    }
    return *offset >= sizeof(header_) && *offset <= remaining &&
           sizeof(footer_) <= remaining - *offset;
  }

  // Existing three-argument readers retain their throwing allocation
  // behavior. A parser explicitly opts into per-input recovery in unpack(),
  // with a fourth callback argument requesting a cheaper representation.
  template <typename TFunc>
  static bool retry_input(TFunc &read_data, bool enabled, size_t offset,
                          const void **data, size_t len) {
    if constexpr (ailego::IsInvocableWithResult<size_t, TFunc, size_t,
                                                const void **, size_t,
                                                bool>::value) {
      if (enabled) return read_data(offset, data, len, true) == len;
    }
    throw;
  }

  IndexFormat::MetaHeader header_{};
  IndexFormat::MetaFooter footer_{};
  ailego::BudgetedStringBuffer version_{};
  SegmentMap segments_{};
  uint64_t current_header_start_offset_{0u};
};

}  // namespace core
}  // namespace zvec

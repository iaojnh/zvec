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

#include <algorithm>
#include <map>
#include <memory>
#include <new>
#include <zvec/core/framework/index_metadata.h>

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

  //! Reset the unpacker
  void reset() {
    *this = IndexUnpacker();
  }

  //! Retrieve segments of index package
  const std::map<std::string, SegmentMeta> &segments() const {
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
    return version_;
  }

  //! Retrieve mutable segments of index package
  std::map<std::string, SegmentMeta> *mutable_segments() {
    return &segments_;
  }

  //! Parse a stable view held by the caller for the duration of the call.
  int unpack_view(const void *data, size_t total, bool checksum) {
    return unpack_from(
        [data](size_t offset, const void **out, size_t size) {
          *out = static_cast<const char *>(data) + offset;
          return size;
        },
        total, checksum);
  }

  //! Parse using a positional reader that copies into caller-owned memory.
  template <typename Fetch>
  int unpack_fetch(Fetch fetch, size_t total, bool checksum) {
    index_metadata::Buffer scratch;
    size_t capacity = 0;
    return unpack_from(
        [&](size_t offset, const void **out, size_t size) {
          if (size > capacity) {
            // All previous views have been consumed. Release the old charge
            // before reserving a larger buffer.
            scratch.reset();
            scratch = index_metadata::AllocateBuffer(size);
            capacity = size;
          }
          *out = scratch.get();
          return fetch(offset, scratch.get(), size);
        },
        total, checksum);
  }

  //! Parse with a view valid until the next invocation of the reader.
  template <typename Read>
  bool unpack(Read read, size_t total, bool checksum) {
    return unpack_from(read, total, checksum) == 0;
  }

 private:
  template <typename Read>
  int unpack_from(const Read &read, size_t total, bool checksum) try {
    IndexUnpacker candidate;
    int ret = candidate.parse(read, total, checksum);
    if (ret == 0) {
      *this = std::move(candidate);
    }
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  template <typename Read>
  int parse(const Read &read, size_t total, bool checksum) {
    size_t start = 0;
    do {
      const size_t remaining = total - start;
      const void *data = nullptr;
      if (sizeof(header_) > remaining) {
        return IndexError_InvalidLength;
      }
      if (read(start, &data, sizeof(header_)) != sizeof(header_) || !data) {
        return IndexError_ReadData;
      }
      std::memcpy(&header_, data, sizeof(header_));

      size_t footer_offset;
      int ret =
          index_metadata::FooterOffset(header_, remaining, &footer_offset);
      if (ret != 0) {
        return ret;
      }
      if (read(start + footer_offset, &data, sizeof(footer_)) !=
              sizeof(footer_) ||
          !data) {
        return IndexError_ReadData;
      }
      std::memcpy(&footer_, data, sizeof(footer_));
      ret = index_metadata::CheckFooter(header_, footer_, footer_offset,
                                        remaining);
      if (ret != 0) {
        return ret;
      }
      size_t next;
      ret = index_metadata::NextSection(footer_, start, total, &next);
      if (ret != 0) {
        return ret;
      }

      const size_t table_offset =
          start + footer_offset - footer_.segments_meta_size;
      if (footer_.segments_meta_size != 0 &&
          (read(table_offset, &data, footer_.segments_meta_size) !=
               footer_.segments_meta_size ||
           !data)) {
        return IndexError_ReadData;
      }
      ret = index_metadata::CheckTable(footer_, data);
      if (ret != 0) {
        return ret;
      }
      const auto *table = static_cast<const char *>(data);
      const size_t content_offset = start + header_.content_offset;
      for (size_t i = 0; i < footer_.segment_count; ++i) {
        IndexFormat::SegmentMeta segment;
        std::memcpy(&segment, table + i * sizeof(segment), sizeof(segment));
        std::string_view id;
        ret = index_metadata::SegmentId(footer_, table, segment, &id);
        if (ret != 0) {
          return ret;
        }
        if (!segments_
                 .try_emplace(
                     std::string(id), content_offset + segment.data_index,
                     segment.data_size, segment.padding_size, segment.data_crc)
                 .second) {
          return IndexError_Duplicate;
        }
      }
      if (checksum && footer_.content_crc != 0) {
        uint32_t crc = 0;
        for (size_t done = 0; done < footer_.content_size;) {
          const size_t len =
              std::min<size_t>(4096, footer_.content_size - done);
          if (read(content_offset + done, &data, len) != len || !data) {
            return IndexError_ReadData;
          }
          crc = ailego::Crc32c::Hash(data, len, crc);
          done += len;
        }
        if (crc != footer_.content_crc) {
          return IndexError_InvalidChecksum;
        }
      }
      start = next;
    } while (start != 0);

    auto version = segments_.find("IndexVersion");
    if (version == segments_.end()) {
      return IndexError_UnpackIndex;
    }
    const SegmentMeta &segment = version->second;
    if (segment.data_size() == 0) {
      return segment.data_crc() == 0 ? 0 : IndexError_InvalidChecksum;
    }
    const void *data = nullptr;
    if (read(segment.data_offset(), &data, segment.data_size()) !=
            segment.data_size() ||
        !data) {
      return IndexError_ReadData;
    }
    if (segment.data_crc() != 0 &&
        ailego::Crc32c::Hash(data, segment.data_size(), 0u) !=
            segment.data_crc()) {
      return IndexError_InvalidChecksum;
    }
    version_.assign(static_cast<const char *>(data), segment.data_size());
    return 0;
  }

  IndexFormat::MetaHeader header_{};
  IndexFormat::MetaFooter footer_{};
  std::string version_{};
  std::map<std::string, SegmentMeta> segments_{};
};

}  // namespace core
}  // namespace zvec

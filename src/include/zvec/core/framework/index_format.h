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
#include <random>
#include <string>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/disk_format.h>

namespace zvec {
namespace core {

/*! Index Format
 */
struct IndexFormat {
  /*! Version Number
   */
  enum { FORMAT_VERSION = 0x0003 };
  static constexpr uint64_t kFormatMagic = 0x335844494345565aULL;  // ZVECIDX3
  static constexpr uint64_t kLittleEndianIeee754 = 1;

  /*! Index Format Meta Header
   */
  struct MetaHeader {
    DiskUInt32 header_crc;
    DiskUInt16 reserved1_;
    DiskUInt16 version;
    DiskUInt32 revision;
    DiskUInt32 magic;
    DiskUInt16 meta_header_size;
    DiskUInt16 meta_footer_size;
    DiskUInt32 meta_footer_offset;
    DiskUInt64 content_offset;
    DiskUInt64 setup_time;
    DiskUInt64 format_magic;
    DiskUInt64 required_features;
    DiskUInt64 disk_alignment;
    uint8_t reserved_[4032];
  };

  static_assert(sizeof(MetaHeader) == kDiskAlignment &&
                    offsetof(MetaHeader, format_magic) == 40 &&
                    offsetof(MetaHeader, reserved_) == 64,
                "The canonical file header is exactly 4 KiB without padding");

  /*! Index Format Meta Footer
   */
  struct MetaFooter {
    DiskUInt32 footer_crc;
    DiskUInt32 segments_meta_crc;
    DiskUInt32 content_crc;
    DiskUInt32 segment_count;
    // meta section size
    DiskUInt32 segments_meta_size;
    DiskUInt32 reserved1_;
    // segments' data section size
    DiskUInt64 content_size;
    DiskUInt64 content_padding_size;

    DiskUInt64 check_point;
    DiskUInt64 update_time;
    DiskUInt64 reserved2_[7];
    DiskUInt64 next_meta_header_offset;
    DiskUInt64 total_size;
  };

  static_assert(sizeof(MetaFooter) == 128 &&
                    offsetof(MetaFooter, content_size) == 24 &&
                    offsetof(MetaFooter, next_meta_header_offset) == 112,
                "Canonical directory footer offsets must not change");

  /*! Index Format Segment Meta
   */
  struct SegmentMeta {
    DiskUInt32 segment_id_offset;
    // used only by immutable segments, e.g., IndexMeta, or searcher
    DiskUInt32 data_crc;
    DiskUInt64 data_index;
    DiskUInt64 data_size;
    DiskUInt64 padding_size;
  };

  static_assert(sizeof(SegmentMeta) == 32 &&
                    offsetof(SegmentMeta, data_index) == 8 &&
                    offsetof(SegmentMeta, padding_size) == 24,
                "Canonical segment records must not contain padding");

  static bool Supported(const MetaHeader &header) {
    return NativeDiskPayloadSupported() && header.version == FORMAT_VERSION &&
           header.format_magic == kFormatMagic &&
           header.required_features == kLittleEndianIeee754 &&
           header.disk_alignment == kDiskAlignment &&
           header.meta_header_size == sizeof(MetaHeader) &&
           header.meta_footer_size == sizeof(MetaFooter);
  }

  /*! Index Format Segment Meta Buffer
   */
  class SegmentMetaBuffer {
   public:
    //! Constructor
    SegmentMetaBuffer(uint32_t count) : capacity_(count) {
      buffer_.clear();
      buffer_.resize(sizeof(SegmentMeta) * capacity_);
    }

    //! Append a segment meta into buffer
    bool append(const std::string &id, size_t data_size, size_t padding_size,
                uint32_t data_crc) {
      if (count_ >= capacity_ || id.find('\0') != std::string::npos ||
          id.size() >= UINT32_MAX ||
          buffer_.size() > UINT32_MAX - id.size() - 1 ||
          data_size > SIZE_MAX - padding_size ||
          offset_ > SIZE_MAX - data_size - padding_size) {
        return false;
      }
      SegmentMeta *meta = (SegmentMeta *)buffer_.data() + count_;
      meta->segment_id_offset = static_cast<uint32_t>(buffer_.size());
      meta->data_index = offset_;
      meta->data_size = data_size;
      meta->data_crc = data_crc;
      meta->padding_size = padding_size;
      buffer_.append(id.c_str(), std::strlen(id.c_str()) + 1);
      count_ += 1;
      offset_ += data_size + padding_size;
      return true;
    }

    //! Resize the buffer
    void resize(size_t val) {
      buffer_.resize(val);
    }

    //! Retrieve pointer of data
    const void *data() const {
      return buffer_.data();
    }

    //! Retrieve size of data
    size_t size() const {
      return buffer_.size();
    }

    //! Retrieve crc of buffer
    uint32_t crc() const {
      return ailego::Crc32c::Hash(buffer_.data(), buffer_.size(), 0);
    }

    //! Disable them
    SegmentMetaBuffer() = delete;

   private:
    //! Members
    std::string buffer_{};
    size_t offset_{0u};
    uint32_t capacity_{0u};
    uint32_t count_{0u};
  };

  //! Setup meta header structure
  static void SetupMetaHeader(MetaHeader *header, uint32_t footer_offset,
                              uint32_t content_offset) {
    memset(header, 0, sizeof(MetaHeader));
    header->version = IndexFormat::FORMAT_VERSION;
    header->format_magic = kFormatMagic;
    header->required_features = kLittleEndianIeee754;
    header->disk_alignment = kDiskAlignment;
    header->revision = 0;
    header->magic = std::random_device()();
    header->meta_header_size = sizeof(MetaHeader);
    header->meta_footer_size = sizeof(MetaFooter);
    header->meta_footer_offset = footer_offset;
    header->content_offset = content_offset;
    header->setup_time = ailego::Realtime::Seconds();
    header->header_crc = ailego::Crc32c::Hash(header, sizeof(MetaHeader), 0);
  }

  static void UpdateMetaHeader(MetaHeader *header) {
    header->header_crc = 0;
    header->header_crc = ailego::Crc32c::Hash(header, sizeof(MetaHeader), 0);
  }

  //! Setup meta footer structure
  static void SetupMetaFooter(MetaFooter *footer) {
    memset(footer, 0, sizeof(MetaFooter));
  }

  //! Update meta footer structure
  static void UpdateMetaFooter(MetaFooter *footer, uint64_t check_point) {
    if (check_point != 0) {
      footer->check_point = check_point;
    }
    footer->update_time = ailego::Realtime::Seconds();
    footer->footer_crc = 0;
    footer->footer_crc = ailego::Crc32c::Hash(footer, sizeof(MetaFooter), 0);
  }
};

}  // namespace core
}  // namespace zvec

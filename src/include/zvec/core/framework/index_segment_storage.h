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

#include <limits>
#include <new>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_module.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/framework/index_unpacker.h>

namespace zvec {
namespace core {

/*! Index Segment Container
 */
class IndexSegmentStorage : public IndexStorage {
 public:
  //! Index Segment Container Pointer
  typedef std::shared_ptr<IndexSegmentStorage> Pointer;

  /*! Index Container Segment
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Container Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(const Segment &rhs) = delete;

    //! Constructor
    Segment(const IndexStorage::Segment::Pointer &parent,
            const IndexUnpacker::SegmentMeta &segment)
        : data_offset_(segment.data_offset()),
          data_size_(segment.data_size()),
          padding_size_(segment.padding_size()),
          region_size_(segment.data_size() + segment.padding_size()),
          data_crc_(segment.data_crc()),
          parent_(parent->clone()) {}

    //! Constructor (for clone)
    Segment(const IndexStorage::Segment::Pointer &cloned_parent,
            size_t data_offset, size_t data_size, size_t padding_size,
            uint32_t data_crc)
        : data_offset_(data_offset),
          data_size_(data_size),
          padding_size_(padding_size),
          region_size_(data_size + padding_size),
          data_crc_(data_crc),
          parent_(cloned_parent) {}

    //! Destructor
    ~Segment() override = default;

    //! Retrieve size of data
    size_t data_size() const override {
      return data_size_;
    }

    //! Retrieve crc of data
    uint32_t data_crc() const override {
      return data_crc_;
    }

    //! Retrieve size of padding
    size_t padding_size() const override {
      return padding_size_;
    }

    size_t capacity() const override {
      return region_size_;
    }

    //! Fetch data from segment (with own buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      return parent_->fetch(data_offset_ + offset, buf, len);
    }

    size_t fetch_for_parsing(size_t offset, void *buf,
                             size_t len) const override {
      if (offset > region_size_) return 0;
      len = std::min(len, region_size_ - offset);
      return parent_->fetch_for_parsing(data_offset_ + offset, buf, len);
    }

    int read_error() const override {
      return parent_->read_error();
    }

    size_t read_for_parsing_with_strategy(size_t offset, MemoryBlock &data,
                                          size_t len,
                                          ParsingReadStrategy strategy,
                                          int &error) override {
      if (offset > region_size_) {
        data.reset();
        error = IndexError_ReadData;
        return 0;
      }
      return parent_->read_for_parsing_with_strategy(
          data_offset_ + offset, data, std::min(len, region_size_ - offset),
          strategy, error);
    }

    bool supports_read_chunks() const override {
      return parent_->supports_read_chunks();
    }
    int set_read_chunks(const ReadChunks &chunks) override try {
      for (const auto &chunk : chunks) {
        if (chunk.offset > data_size_ ||
            chunk.size > data_size_ - chunk.offset ||
            chunk.offset > std::numeric_limits<size_t>::max() - data_offset_)
          return IndexError_InvalidArgument;
      }
      ReadChunks translated;
      translated.reserve(chunks.size());
      for (const auto &chunk : chunks) {
        translated.push_back({data_offset_ + chunk.offset, chunk.size});
      }
      return parent_->set_read_chunks(translated);
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }

    size_t read_scatter(size_t offset, ScatterBlock &data,
                        size_t len) override {
      if (offset > region_size_) {
        data.reset();
        return 0;
      }
      return parent_->read_scatter(data_offset_ + offset, data,
                                   std::min(len, region_size_ - offset));
    }

    //! Read data from segment
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      return parent_->read(data_offset_ + offset, data, len);
    }

    size_t read_for_parsing(size_t offset, MemoryBlock &data,
                            size_t len) override {
      if (offset > region_size_) {
        data.reset();
        return 0;
      }
      return parent_->read_for_parsing(data_offset_ + offset, data,
                                       std::min(len, region_size_ - offset));
    }

    //! Read data from segment
    bool read(SegmentData *iovec, size_t count) override {
      for (SegmentData *it = iovec, *end = iovec + count; it != end; ++it) {
        it->offset += data_offset_;
      }
      bool success = parent_->read(iovec, count);
      for (SegmentData *it = iovec, *end = iovec + count; it != end; ++it) {
        it->offset -= data_offset_;
      }
      return success;
    }

    size_t write(size_t, const void *, size_t) override {
      return IndexError_NotImplemented;
    }

    size_t resize(size_t) override {
      return IndexError_NotImplemented;
    }

    //! Retrieve offset of data
    size_t data_offset() const override {
      return 0;
    }

    void update_data_crc(uint32_t) override {
      return;
    }

    //! Clone the segment (each clone gets an independent parent buffer
    //! for thread safety — concurrent reads require separate buffers).
    IndexStorage::Segment::Pointer clone() override {
      return std::make_shared<Segment>(parent_->clone(), data_offset_,
                                       data_size_, padding_size_, data_crc_);
    }

   private:
    size_t data_offset_{0u};
    size_t data_size_{0u};
    size_t padding_size_{0u};
    size_t region_size_{0u};
    uint32_t data_crc_{0u};
    IndexStorage::Segment::Pointer parent_{nullptr};
  };

  //! Constructor
  IndexSegmentStorage(IndexStorage::Segment::Pointer &&seg)
      : parent_(std::move(seg)) {}

  //! Constructor
  IndexSegmentStorage(const IndexStorage::Segment::Pointer &seg)
      : parent_(seg) {}

  //! Destructor
  ~IndexSegmentStorage() override = default;

  //! Initialize container
  int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup container
  int cleanup() override {
    return 0;
  }

  //! Load the current segment, ignore path
  int open(const std::string &, bool) override try {
    if (!parent_) {
      LOG_ERROR("Failed to load an empty segment");
      return IndexError_NoReady;
    }

    using Strategy = IndexStorage::Segment::ParsingReadStrategy;
    constexpr Strategy strategies[] = {Strategy::kReuse, Strategy::kBorrow,
                                       Strategy::kGrow, Strategy::kOwned};
    unsigned excluded = 0;
    // Select per callback: a copied header can precede a borrowed directory.
    // On private refusal retire only a releasable input and exclude its failed
    // strategy. Each restart excludes a new strategy, so there are at most
    // four whole attempts and four representation probes per callback.
    for (size_t attempt = 0; attempt < 4; ++attempt) {
      IndexStorage::MemoryBlock metadata;
      ailego::BudgetedMappedBuffer fallback;
      IndexUnpacker unpacker;
      const size_t total = parent_->data_size();
      int read_error = IndexError_ReadData;
      Strategy active = Strategy::kOwned;
      auto read_data = [&, this](size_t offset, const void **data, size_t len) {
        metadata.reset();
        fallback.clear();
        if (offset > total || len > total - offset) return size_t{0};
        for (const auto strategy : strategies) {
          if (excluded & (1u << static_cast<unsigned>(strategy))) continue;
          if (strategy != Strategy::kOwned) {
            size_t got = 0;
            try {
              got = parent_->read_for_parsing_with_strategy(
                  offset, metadata, len, strategy, read_error);
            } catch (const std::bad_alloc &) {
              read_error = IndexError_NoMemory;
            }
            if (got == len) {
              read_error = 0;
              active = strategy;
              *data = metadata.data();
              return got;
            }
            metadata.reset();
            if (read_error != IndexError_Unsupported &&
                read_error != IndexError_NoMemory)
              return size_t{0};
            continue;
          }
          if (fallback.resize(len) != 0) {
            read_error = IndexError_NoMemory;
            return size_t{0};
          }
          active = strategy;
          const size_t got =
              parent_->fetch_for_parsing(offset, fallback.data(), len);
          read_error = got == len ? 0 : parent_->read_error();
          *data = fallback.data();
          return got;
        }
        read_error = IndexError_NoMemory;
        return size_t{0};
      };
      try {
        if (!unpacker.unpack(read_data, total, false,
                             /*report_errors=*/false)) {
          if (read_error == IndexError_NoMemory) return read_error;
          LOG_ERROR("Failed to unpack segment data");
          return IndexError_UnpackIndex;
        }
        segments_ = std::move(*unpacker.mutable_segments());
        magic_ = unpacker.magic();
        return 0;
      } catch (const std::bad_alloc &) {
        // Permanent execution workspaces and escaped inputs cannot be made
        // cheaper by another copy. Their owners keep the final veto.
        if (active == Strategy::kOwned || !metadata.release_for_parsing_retry())
          return IndexError_NoMemory;
        excluded |= 1u << static_cast<unsigned>(active);
      }
      // Destroy all unpublished private state before the next attempt.
    }
    return IndexError_NoMemory;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    if (!parent_) {
      return IndexStorage::Segment::Pointer();
    }
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return IndexStorage::Segment::Pointer();
    }
    return std::make_shared<IndexSegmentStorage::Segment>(parent_, it->second);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return (segments_.find(id) != segments_.end());
  }

  //! Retrieve all segments
  std::map<std::string, IndexStorage::Segment::Pointer> get_all()
      const override {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    if (parent_) {
      for (const auto &it : segments_) {
        result.emplace(it.first, std::make_shared<IndexSegmentStorage::Segment>(
                                     parent_, it.second));
      }
    }
    return result;
  }

  //! Unload all indexes
  int close() override {
    parent_ = nullptr;
    segments_.clear();
    return 0;
  }

  //! Retrieve magic number of index
  uint32_t magic() const override {
    return magic_;
  }

  int flush() override {
    return IndexError_NotImplemented;
  }

  int append(const std::string & /*id*/, size_t /*size*/) override {
    return IndexError_NotImplemented;
  }

  void refresh(uint64_t) override {
    return;
  }

  uint64_t check_point() const override {
    return 0;
  }

 private:
  uint32_t magic_{0};
  IndexUnpacker::SegmentMap segments_{};
  IndexStorage::Segment::Pointer parent_{};
};

}  // namespace core
}  // namespace zvec

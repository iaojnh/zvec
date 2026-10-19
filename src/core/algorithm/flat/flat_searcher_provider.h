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

#include <memory>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include "utility/legacy_read_buffer.h"
#include "flat_distance_matrix.h"
#include "flat_searcher.h"
// #include "flat_streamer.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

/*! Brute Force Searcher Provider
 */
template <size_t BATCH_SIZE>
class FlatSearcherProvider : public IndexProvider {
 public:
  //! Constructor
  FlatSearcherProvider(const FlatSearcher<BATCH_SIZE> *owner) {
    feature_size_ = owner->meta().element_size();
    features_segment_ = owner->clone_features_segment();
    if (!features_segment_) throw std::bad_alloc();
    total_vector_count_ =
        features_segment_->data_size() / owner->meta().element_size();
    owner_ = owner;
  }

  //! Create a new iterator
  IndexProvider::Iterator::Pointer create_iterator() override try {
    return IndexProvider::Iterator::Pointer(
        new (std::nothrow) FlatSearcherProvider::Iterator(owner_));
  } catch (const std::bad_alloc &) {
    return nullptr;
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return total_vector_count_;
  }

  //! Retrieve dimension of vector
  size_t dimension() const override {
    return owner_->meta().dimension();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return owner_->meta().data_type();
  }

  //! Retrieve vector size in bytes
  size_t element_size() const override {
    return owner_->meta().element_size();
  }

  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override try {
    IndexStorage::MemoryBlock block;
    return get_vector(key, block) == 0
               ? legacy_buffer_.assign(block, feature_size_)
               : nullptr;
  } catch (const std::bad_alloc &) {
    return nullptr;
  } catch (const typename FlatSearcher<BATCH_SIZE>::MetadataReadError &) {
    return nullptr;
  }

  int get_vector(uint64_t key, IndexStorage::MemoryBlock &block) const override
      try {
    block.reset();
    const auto index = owner_->get_id(key);
    const int ret = get_vector_by_index(index, block);
    if (ret != 0) return ret;
    if (block.type_ == IndexStorage::MemoryBlock::MBT_MMAP) {
      // Storage close can unmap a region even while its segment is retained.
      // Returned documents need their own allocation; pooled blocks already
      // carry a pin and keep their zero-copy path.
      auto owned = IndexStorage::MemoryBlock::AllocateOwned(feature_size_);
      std::memcpy(const_cast<void *>(owned.data()), block.data(),
                  feature_size_);
      block = std::move(owned);
    }
    return 0;
  } catch (const typename FlatSearcher<BATCH_SIZE>::MetadataReadError &error) {
    block.reset();
    return error.status;
  } catch (const std::bad_alloc &) {
    block.reset();
    return IndexError_NoMemory;
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_->name();
  }

 protected:
  /*! Brute Force Provider Iterator
   */
  class Iterator : public IndexProvider::Iterator {
   public:
    //! Constructor
    Iterator(const FlatSearcher<BATCH_SIZE> *owner) {
      feature_size_ = owner->meta().element_size();
      features_segment_ = owner->clone_features_segment();
      if (!features_segment_) throw std::bad_alloc();
      total_vector_count_ =
          features_segment_->data_size() / owner->meta().element_size();
      owner_ = owner;
      cursor_index_ = 0;
      offset_ = 0;
      this->next_block();
    }

    //! Retrieve pointer of data
    //! NOTICE: the vec feature will be changed after iterating to next, so
    //! the caller need to keep a copy of it before iterator to next vector
    const void *data() const override {
      return data_;
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      return (status_ == 0 && !invalid_ && cursor_index_ < total_vector_count_);
    }

    //! Retrieve primary key
    uint64_t key() const override try {
      return owner_->key(cursor_index_);
    } catch (
        const typename FlatSearcher<BATCH_SIZE>::MetadataReadError &error) {
      status_ = error.status;
      return 0;
    }

    int status() const override {
      return status_;
    }

    //! Next iterator
    void next() override {
      ++cursor_index_;

      if ((cursor_index_ % BATCH_SIZE) != 0) {
        data_ += feature_size_;
      } else {
        this->next_block();
      }
    }

   protected:
    //! Read a block of data
    void next_block() try {
      const void *read_data = nullptr;
      size_t read_size = 0;

      if (cursor_index_ >= total_vector_count_) {
        read_block_.reset();
        data_ = nullptr;
        invalid_ = true;
        return;
      }

      if (cursor_index_ + BATCH_SIZE < total_vector_count_) {
        read_size = BATCH_SIZE * feature_size_;
      } else {
        read_size = (total_vector_count_ - cursor_index_) * feature_size_;
      }
      read_block_.reset();
      if (features_segment_->read(offset_, read_block_, read_size) !=
          read_size) {
        LOG_ERROR("Failed to read data (%zu bytes) from features segment",
                  read_size);
        invalid_ = true;
        status_ = IndexError_ReadData;
        return;
      }
      read_data = read_block_.data();
      offset_ += read_size;

      // The order of data may be a column format, convert it to the row format.
      if (owner_->column_major_order() &&
          read_size == BATCH_SIZE * feature_size_) {
        uint32_t align_size =
            IndexMeta::AlignSizeof(owner_->meta().data_type());
        block_buffer_.resize(read_size);
        ReverseTranspose<BATCH_SIZE>(align_size, read_data,
                                     feature_size_ / align_size,
                                     &block_buffer_[0]);
        data_ = block_buffer_.data();
        read_block_.reset();
      } else {
        data_ = reinterpret_cast<const uint8_t *>(read_data);
      }
    } catch (const std::bad_alloc &) {
      read_block_.reset();
      invalid_ = true;
      data_ = nullptr;
      status_ = IndexError_NoMemory;
    }

   private:
    const FlatSearcher<BATCH_SIZE> *owner_{nullptr};
    IndexStorage::Segment::Pointer features_segment_{};
    uint32_t total_vector_count_{0};
    uint32_t feature_size_{0};
    ailego::BudgetedVector<uint8_t> block_buffer_{};
    IndexStorage::MemoryBlock read_block_;
    const uint8_t *data_{nullptr};
    uint64_t offset_{0};
    uint32_t cursor_index_{0};
    bool invalid_{false};
    mutable int status_{0};
  };

  //! Retrieve a vector via local index
  int get_vector_by_index(uint32_t index,
                          IndexStorage::MemoryBlock &block) const {
    block.reset();
    if (index >= total_vector_count_) {
      LOG_ERROR("Failed to get vector by Invalid Id.");
      return IndexError_NoExist;
    }

    if (owner_->column_major_order() &&
        index < (total_vector_count_ - (total_vector_count_ % BATCH_SIZE))) {
      uint32_t block_size = feature_size_ * BATCH_SIZE;
      uint64_t offset = (index - (index % BATCH_SIZE)) * feature_size_;

      IndexStorage::MemoryBlock source_block;
      if (features_segment_->read(offset, source_block, block_size) !=
          block_size) {
        LOG_ERROR("Failed to read data (%u bytes) from features segment",
                  block_size);
        return IndexError_ReadData;
      }

      const size_t align_size =
          IndexMeta::AlignSizeof(owner_->meta().data_type());
      auto owned = IndexStorage::MemoryBlock::AllocateOwned(feature_size_);
      auto *destination =
          static_cast<uint8_t *>(const_cast<void *>(owned.data()));
      const auto *source = static_cast<const uint8_t *>(source_block.data()) +
                           (index % BATCH_SIZE) * align_size;
      // Keep only this row, not a transposed batch for every returned vector.
      for (size_t pos = 0; pos < feature_size_; pos += align_size) {
        std::memcpy(destination + pos, source + pos * BATCH_SIZE, align_size);
      }
      block = std::move(owned);

    } else {
      if (features_segment_->read(static_cast<size_t>(index) * feature_size_,
                                  block, feature_size_) != feature_size_) {
        LOG_ERROR("Failed to read data (%u bytes) from features segment",
                  feature_size_);
        block.reset();
        return IndexError_ReadData;
      }
    }
    return 0;
  }

 private:
  //! Members
  const FlatSearcher<BATCH_SIZE> *owner_{nullptr};
  IndexStorage::Segment::Pointer features_segment_{};
  uint32_t feature_size_{0};
  uint32_t total_vector_count_{0};
  mutable LegacyReadBuffer legacy_buffer_;
};

}  // namespace core
}  // namespace zvec

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
#include <cstring>
#include <list>
#include <mutex>
#include <new>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/core/framework/index_searcher.h>
#include "ivf_entity.h"

namespace zvec {
namespace core {

/*! IVF IndexProvider
 */
class IVFIndexProvider : public IndexProvider {
 public:
  IVFIndexProvider(const IndexMeta &meta, const IVFEntity::Pointer &entity,
                   const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  IVFIndexProvider(const IVFIndexProvider &) = delete;
  IVFIndexProvider &operator=(const IVFIndexProvider &) = delete;

 public:
  //! Create a new iterator
  Iterator::Pointer create_iterator() override {
    return Iterator::Pointer(new (std::nothrow) SortedIterator(
        entity_, read_mutex_, element_size()));
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return entity_->vector_count();
  }

  //! Retrieve dimension of vector
  size_t dimension() const override {
    return meta_.dimension();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

  //! Retrieve vector size in bytes
  size_t element_size() const override {
    return meta_.element_size();
  }

  // Returned bytes survive reads on other threads/providers. They remain valid
  // until this thread's next pointer-returning get_vector() on this provider,
  // provider destruction, or thread exit.
  const void *get_vector(uint64_t key) const override {
    try {
      std::lock_guard<std::mutex> lock(*read_mutex_);
      auto &buffer = thread_result_buffer();
      IndexStorage::MemoryBlock data;
      if (entity_->get_vector_by_key(key, data) != 0) {
        return nullptr;
      }
      return CopyVector(entity_, data.data(), element_size(), &buffer);
    } catch (const std::bad_alloc &) {
      return nullptr;
    }
  }

  //! Return independently owned bytes, including for decoded Turbo postings.
  int get_vector(uint64_t key, IndexStorage::MemoryBlock &block) const override
      try {
    block.reset();
    auto buffer = std::make_shared<ailego::BudgetedStringWorkspace>();
    std::lock_guard<std::mutex> lock(*read_mutex_);
    IndexStorage::MemoryBlock data;
    int ret = entity_->get_vector_by_key(key, data);
    if (ret != 0) return ret;
    if (!CopyVector(entity_, data.data(), element_size(), buffer.get()))
      return IndexError_ReadData;
    block =
        IndexStorage::MemoryBlock::MakeSharedView(buffer->value.data(), buffer);
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  struct ThreadResultBuffer {
    std::thread::id thread;
    ailego::BudgetedStringWorkspace data;
  };
  struct ThreadResultRef {
    std::weak_ptr<std::mutex> owner;
    std::weak_ptr<ThreadResultBuffer> buffer;
    ~ThreadResultRef() {
      if (auto live = buffer.lock()) live->data = {};
    }
  };

  ailego::BudgetedStringWorkspace &thread_result_buffer() const {
    // TLS only references the buffer; provider destruction releases all
    // workers.
    static thread_local std::list<ThreadResultRef> buffers;
    for (auto it = buffers.begin(); it != buffers.end();) {
      auto owner = it->owner.lock();
      if (!owner) {
        it = buffers.erase(it);
      } else if (owner == read_mutex_) {
        if (auto buffer = it->buffer.lock()) return buffer->data;
        it = buffers.erase(it);
      } else {
        ++it;
      }
    }
    const auto thread = std::this_thread::get_id();
    std::shared_ptr<ThreadResultBuffer> buffer;
    for (auto &slot : thread_buffers_) {
      if (slot->thread == thread) {
        buffer = slot;
        break;
      }
    }
    if (!buffer) {
      buffer = std::make_shared<ThreadResultBuffer>();
      buffer->thread = thread;
      thread_buffers_.push_back(buffer);
    }
    buffers.emplace_back();
    buffers.back().owner = read_mutex_;
    buffers.back().buffer = buffer;
    return buffer->data;
  }

  // Copy storage-owned and column-major scratch bytes before releasing the
  // shared read lock. Decoded bytes already belong to the output buffer.
  static const void *CopyVector(const IVFEntity::Pointer &entity,
                                const void *data, size_t size,
                                ailego::BudgetedStringWorkspace *workspace) {
    workspace->reserve(std::max(
        size, entity->quantizer() ? entity->quantizer()->dim() * sizeof(float)
                                  : size));
    auto *buffer = &workspace->value;
    const void *decoded = DecodeVector(entity, data, buffer);
    if (!decoded) {
      return nullptr;
    }
    if (decoded != buffer->data()) {
      buffer->assign(static_cast<const char *>(decoded), size);
    }
    return buffer->data();
  }

  // Providers feed clustering/reducers, which need vectors in the original
  // input space even when postings use a different encoded layout.
  static const void *DecodeVector(const IVFEntity::Pointer &entity,
                                  const void *data, std::string *buffer) {
    if (!data || !entity->quantizer() || entity->has_orignal_feature()) {
      return data;
    }
    const auto &meta = entity->meta();
    IndexQueryMeta qmeta(meta.data_type(), meta.dimension());
    qmeta.set_meta(meta.data_type(), meta.dimension(),
                   static_cast<uint32_t>(entity->quantizer()->type()),
                   meta.extra_meta_size());
    if (entity->quantizer()->dequantize(data, qmeta, buffer) != 0) {
      return nullptr;
    }
    const size_t dim = entity->quantizer()->dim();
    if (entity->quantizer()->input_data_type() == turbo::DataType::kFp16 &&
        buffer->size() == dim * sizeof(float)) {
      // PQ reconstruction is FP32 even when its training input was FP16.
      ailego::BudgetedStringWorkspace half_workspace;
      half_workspace.reserve(dim * sizeof(ailego::Float16));
      auto &half = half_workspace.value;
      half.resize(dim * sizeof(ailego::Float16));
      for (size_t i = 0; i < dim; ++i) {
        float value;
        std::memcpy(&value, buffer->data() + i * sizeof(float), sizeof(value));
        ailego::Float16 encoded(value);
        std::memcpy(&half[i * sizeof(encoded)], &encoded, sizeof(encoded));
      }
      buffer->assign(half);
    }
    return buffer->data();
  }

  class SortedIterator : public IndexProvider::Iterator {
   public:
    SortedIterator(const IVFEntity::Pointer &entity,
                   const std::shared_ptr<std::mutex> &read_mutex,
                   size_t element_size)
        : entity_(entity),
          read_mutex_(read_mutex),
          element_size_(element_size) {
      std::lock_guard<std::mutex> lock(*read_mutex_);
      count_ = entity_->vector_count();
      // load() requires this segment. Do not turn a malformed/partially loaded
      // entity into an O(N) heap sort and potentially dereference missing keys.
      if (!entity_->has_key_order_mapping()) status_ = IndexError_InvalidFormat;
    }

    //! Retrieve pointer of data
    // Concurrent reads of the current position are supported. The owned bytes
    // stay immutable until next(); finish using them before
    // advancing/destroying the iterator. Other iterators and provider reads
    // cannot invalidate them.
    const void *data() const override try {
      std::lock_guard<std::mutex> lock(*read_mutex_);
      size_t local_id = current_local_id();
      if (local_id >= count_) {
        return nullptr;
      }
      if (!vector_loaded_) {
        IndexStorage::MemoryBlock data;
        if (entity_->get_vector(local_id, data) != 0 ||
            !CopyVector(entity_, data.data(), element_size_, &vector_)) {
          status_ = IndexError_ReadData;
          return nullptr;
        }
        vector_loaded_ = true;
      }
      return vector_.value.data();
    } catch (const std::bad_alloc &) {
      status_ = IndexError_NoMemory;
      return nullptr;
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      std::lock_guard<std::mutex> lock(*read_mutex_);
      return status_ == 0 && pos_ < count_ && ensure_mapping_chunk();
    }

    int status() const override {
      std::lock_guard<std::mutex> lock(*read_mutex_);
      return status_;
    }

    //! Retrieve primary key
    uint64_t key() const override {
      std::lock_guard<std::mutex> lock(*read_mutex_);
      size_t local_id = current_local_id();
      if (local_id >= count_) {
        return kInvalidKey;
      }
      const uint64_t result = entity_->get_key(local_id);
      if (result == kInvalidKey) {
        status_ = IndexError_ReadData;
      }
      return result;
    }

    //! Next iterator
    void next() override {
      std::lock_guard<std::mutex> lock(*read_mutex_);
      if (status_ == 0 && pos_ < count_) {
        ++pos_;
        vector_loaded_ = false;
        vector_.value.clear();
      }
    }

   private:
    bool ensure_mapping_chunk() const {
      if (status_ != 0 || pos_ >= count_) {
        return false;
      }
      if (pos_ >= mapping_chunk_begin_ &&
          pos_ - mapping_chunk_begin_ < mapping_chunk_.size()) {
        return true;
      }
      mapping_chunk_begin_ = pos_;
      const size_t chunk_count =
          std::min(kMappingChunkEntries, count_ - mapping_chunk_begin_);
      try {
        mapping_chunk_.resize(chunk_count);
      } catch (const std::bad_alloc &) {
        status_ = IndexError_NoMemory;
        return false;
      }
      if (entity_->get_key_order_mapping(mapping_chunk_begin_,
                                         mapping_chunk_.data(),
                                         chunk_count) != chunk_count) {
        mapping_chunk_.clear();
        status_ = IndexError_ReadData;
        return false;
      }
      if (std::any_of(mapping_chunk_.begin(), mapping_chunk_.end(),
                      [this](uint32_t id) { return id >= count_; })) {
        mapping_chunk_.clear();
        status_ = IndexError_InvalidFormat;
        return false;
      }
      return true;
    }

    size_t current_local_id() const {
      if (status_ != 0 || pos_ >= count_) {
        return count_;
      }
      if (!ensure_mapping_chunk()) {
        return count_;
      }
      return static_cast<size_t>(mapping_chunk_[pos_ - mapping_chunk_begin_]);
    }

    //! Members
    static constexpr size_t kMappingChunkEntries = 4096;
    IVFEntity::Pointer entity_;
    std::shared_ptr<std::mutex> read_mutex_;
    size_t element_size_;
    mutable ailego::BudgetedStringWorkspace vector_;
    mutable bool vector_loaded_{false};
    mutable int status_{0};
    mutable std::vector<uint32_t, ailego::BudgetedAllocator<uint32_t>>
        mapping_chunk_;
    mutable size_t mapping_chunk_begin_{0};
    size_t count_{0};
    size_t pos_{0};
  };

 private:
  //! Members
  mutable ailego::BudgetedVector<std::shared_ptr<ThreadResultBuffer>>
      thread_buffers_;
  IndexMeta meta_;
  IVFEntity::Pointer entity_;
  std::string owner_class_;
  // All iterators share the entity's storage and column-major scratch buffer.
  std::shared_ptr<std::mutex> read_mutex_{std::make_shared<std::mutex>()};
};

}  // namespace core
}  // namespace zvec

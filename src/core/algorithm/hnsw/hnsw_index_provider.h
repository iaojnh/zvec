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

#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_streamer.h>
#include "utility/legacy_read_buffer.h"
#include "utility/stable_vector_holder.h"
#include "hnsw_streamer_entity.h"

namespace zvec {
namespace core {

class HnswIndexProvider : public IndexProvider, public StableVectorHolder {
 public:
  HnswIndexProvider(const IndexMeta &meta,
                    const std::shared_ptr<HnswStreamerEntity> &entity,
                    const std::string &owner,
                    std::shared_ptr<const void> lifetime, bool writable)
      : dimension_(meta.dimension()),
        data_type_(meta.data_type()),
        element_size_(meta.element_size()),
        entity_(entity),
        owner_class_(owner),
        lifetime_(std::move(lifetime)),
        snapshot_vectors_(writable && entity->buffered_graph_mutex()) {}

  HnswIndexProvider(const HnswIndexProvider &) = delete;
  HnswIndexProvider &operator=(const HnswIndexProvider &) = delete;

 public:  // holder interface
  int retain_stable_data(std::shared_ptr<const void> *lifetime) override {
    if (!lifetime) return IndexError_InvalidArgument;
    if (!lifetime_) return IndexError_NotImplemented;
    *lifetime = lifetime_;
    return 0;
  }

  //! Create a new iterator
  IndexProvider::Iterator::Pointer create_iterator() override {
    return IndexProvider::Iterator::Pointer(
        new (std::nothrow) Iterator(entity_, snapshot_vectors_));
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return entity_->doc_cnt();
  }

  //! Retrieve dimension of vector
  size_t dimension() const override {
    return dimension_;
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return data_type_;
  }

  //! Retrieve vector size in bytes
  size_t element_size() const override {
    return element_size_;
  }

 public:  // provider's unique interface
  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override {
    IndexStorage::MemoryBlock block;
    if (get_vector(key, block) != 0) return nullptr;
    static thread_local LegacyReadBuffer buffer;
    return buffer.assign(block, entity_->vector_size());
  }

  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    block.reset();
    const HnswGraphLock guard(entity_->buffered_graph_mutex(),
                              HnswGraphLock::Mode::kPointRead);
    if (guard.blocked()) return IndexError_NoReady;
    if (const int error = entity_->provider_read_error(); error != 0)
      return error;
    if (!snapshot_vectors_) return entity_->get_vector_by_key(key, block);
    const auto id = entity_->get_id(key);
    if (id >= entity_->committed_doc_count()) return IndexError_NoExist;
    return entity_->get_vector_snapshot(id, block);
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexProvider::Iterator {
   public:
    Iterator(const std::shared_ptr<HnswStreamerEntity> &entity,
             bool snapshot_vectors)
        : entity_(entity), snapshot_vectors_(snapshot_vectors) {}

    //! Retrieve pointer of data
    //! NOTICE: the vec feature will be changed after iterating to next, so
    //! the caller need to keep a copy of it before iterator to next vector
    const void *data() const override {
      const HnswGraphLock guard(entity_->buffered_graph_mutex(),
                                HnswGraphLock::Mode::kPointRead);
      if (!check_graph(guard) || !is_valid()) return nullptr;
      status_ = snapshot_vectors_
                    ? entity_->get_vector_snapshot(cur_id_, block_)
                    : entity_->get_vector(cur_id_, block_);
      return status_ == 0 ? block_.data() : nullptr;
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      return status_ == 0 && cur_id_ < entity_->doc_cnt();
    }

    int status() const override {
      return status_;
    }

    //! Retrieve primary key
    uint64_t key() const override {
      const HnswGraphLock guard(entity_->buffered_graph_mutex(),
                                HnswGraphLock::Mode::kPointRead);
      if (!check_graph(guard) || !is_valid()) return kInvalidKey;
      key_t key;
      status_ = entity_->read_key_snapshot(cur_id_, key);
      return status_ == 0 ? key : kInvalidKey;
    }

    //! Next iterator
    void next() override {
      const HnswGraphLock guard(entity_->buffered_graph_mutex(),
                                HnswGraphLock::Mode::kPointRead);
      if (check_graph(guard) && is_valid()) {
        cur_id_ = get_next_valid_id(cur_id_ + 1);
      }
    }

   private:
    bool check_graph(const HnswGraphLock &guard) const {
      if (status_ == 0) {
        status_ = guard.blocked() ? IndexError_NoReady
                                  : entity_->provider_read_error();
      }
      return status_ == 0;
    }

    node_id_t get_next_valid_id(node_id_t start_id) {
      for (node_id_t i = start_id; i < entity_->doc_cnt(); i++) {
        key_t key;
        status_ = entity_->read_key_snapshot(i, key);
        if (status_ != 0) break;
        if (key != kInvalidKey) return i;
      }
      return kInvalidNodeId;
    }

   private:
    const std::shared_ptr<HnswStreamerEntity> entity_;
    const bool snapshot_vectors_;
    node_id_t cur_id_{0};
    mutable IndexStorage::MemoryBlock block_;
    mutable int status_{0};
  };

 private:
  const size_t dimension_;
  const IndexMeta::DataType data_type_;
  const size_t element_size_;
  const std::shared_ptr<HnswStreamerEntity> entity_;
  const std::string owner_class_;
  std::shared_ptr<const void> lifetime_;
  const bool snapshot_vectors_;
};

}  // namespace core
}  // namespace zvec

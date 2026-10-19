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

#include "flat_distance_matrix.h"
#include "flat_searcher.h"
#include "flat_streamer.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

/*! Brute Force Streamer Provider
 */

template <size_t BATCH_SIZE>
class FlatStreamerProvider : public IndexProvider {
 public:
  //! Constructor
  FlatStreamerProvider(const FlatStreamer<BATCH_SIZE> *owner) {
    total_vector_count_ = owner->entity().vector_count();
    owner_ = owner;
  }

  //! Create a new iterator
  IndexProvider::Iterator::Pointer create_iterator() override {
    return owner_->entity().creater_iterator();
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
  const void *get_vector(uint64_t key) const override {
    return this->get_vector_by_key(key);
  }

  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    return this->get_vector_by_key(key, block);
  }

  int get_vectors(
      const uint64_t *keys, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &blocks) const override {
    return get_vectors_impl(keys, count, blocks);
  }
  int get_vectors(const uint64_t *keys, uint32_t count,
                  IndexProvider::VectorBlocks &blocks) const override {
    return get_vectors_impl(keys, count, blocks);
  }

 private:
  template <typename Blocks>
  int get_vectors_impl(const uint64_t *keys, uint32_t count,
                       Blocks &blocks) const try {
    int ret = owner_->entity().get_vectors_by_key(keys, count, blocks);
    if (ret != 0) {
      blocks.clear();
      return ret;
    }
    for (auto &block : blocks) {
      if (block.type_ == IndexStorage::MemoryBlock::MBT_MMAP) {
        auto owned = IndexStorage::MemoryBlock::AllocateOwned(element_size());
        std::memcpy(const_cast<void *>(owned.data()), block.data(),
                    element_size());
        block = std::move(owned);
      }
    }
    return 0;
  } catch (const std::bad_alloc &) {
    blocks.clear();
    return IndexError_NoMemory;
  }

 public:
  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_->name();
  }

 protected:
  //! Retrieve a vector via primary key
  const void *get_vector_by_key(uint64_t key) const {
    return owner_->get_vector_by_key(key);
  }

  int get_vector_by_key(const uint64_t key,
                        IndexStorage::MemoryBlock &block) const {
    return owner_->get_vector_by_key(key, block);
  }

 private:
  //! Members
  const FlatStreamer<BATCH_SIZE> *owner_{nullptr};
  uint32_t total_vector_count_{0};
};

}  // namespace core
}  // namespace zvec

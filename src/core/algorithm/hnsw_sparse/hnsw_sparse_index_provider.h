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

#include "hnsw_sparse_entity.h"

namespace zvec {
namespace core {

class HnswSparseIndexProvider : public IndexSparseProvider {
 public:
  HnswSparseIndexProvider(const IndexMeta &meta,
                          const HnswSparseEntity::Pointer &entity,
                          const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  HnswSparseIndexProvider(const HnswSparseIndexProvider &) = delete;
  HnswSparseIndexProvider &operator=(const HnswSparseIndexProvider &) = delete;

 public:
  //! Create a new iterator
  IndexSparseProvider::Iterator::Pointer create_iterator() override {
    return IndexSparseProvider::Iterator::Pointer(new (std::nothrow)
                                                      Iterator(entity_));
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return entity_->doc_cnt();
  }

  size_t total_sparse_count() const override {
    return entity_->total_sparse_count();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

  //! Retrieve a vector using a primary key
  int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                        std::string *sparse_indices_buffer,
                        std::string *sparse_values_buffer) const override {
    return entity_->get_sparse_vector_by_key(
        key, sparse_count, sparse_indices_buffer, sparse_values_buffer);
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexSparseProvider::Iterator {
   public:
    explicit Iterator(const HnswSparseEntity::Pointer &entity)
        : entity_(entity) {
      read_current();
    }

    uint32_t sparse_count() const override {
      return sparse_count_;
    }

    const uint32_t *sparse_indices() const override {
      return reinterpret_cast<const uint32_t *>(sparse_indices_buffer_.data());
    }

    const void *sparse_data() const override {
      return sparse_data_buffer_.data();
    }

    bool is_valid() const override {
      return valid_;
    }
    int status() const override {
      return status_;
    }
    uint64_t key() const override {
      return key_;
    }

    void next() override {
      if (status_ == 0 && cur_id_ < entity_->doc_cnt()) {
        ++cur_id_;
        read_current();
      }
    }

   private:
    void read_current() try {
      valid_ = false;
      while (cur_id_ < entity_->doc_cnt()) {
        key_ = entity_->get_key(cur_id_);
        if (key_ != kInvalidKey) break;
        ++cur_id_;
      }
      if (cur_id_ >= entity_->doc_cnt()) return;

      IndexStorage::MemoryBlock block;
      status_ = entity_->get_sparse_data(cur_id_, block);
      if (status_ != 0) return;
      if (!block.data()) {
        status_ = IndexError_ReadData;
        return;
      }
      sparse_indices_buffer_.clear();
      sparse_data_buffer_.clear();
      SparseUtility::ReverseSparseFormat(
          block.data(), &sparse_count_, &sparse_indices_buffer_,
          &sparse_data_buffer_, entity_->sparse_unit_size());
      valid_ = true;
    } catch (const GraphStorageError &error) {
      status_ = error.code;
    } catch (const std::bad_alloc &) {
      status_ = IndexError_NoMemory;
    }

    const HnswSparseEntity::Pointer entity_;
    node_id_t cur_id_{0};
    uint32_t sparse_count_{0};
    ailego::BudgetedString sparse_indices_buffer_;
    ailego::BudgetedString sparse_data_buffer_;
    bool valid_{false};
    int status_{0};
    uint64_t key_{kInvalidKey};
  };

 private:
  const IndexMeta &meta_;
  const HnswSparseEntity::Pointer entity_;
  const std::string owner_class_;
};

}  // namespace core
}  // namespace zvec

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

#include <zvec/ailego/buffer/memory_budget.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_holder.h>
#include "diskann_entity.h"
#include "diskann_file_reader.h"
#include "diskann_pq_table.h"

namespace zvec {
namespace core {

class DiskAnnSearcherEntity : public DiskAnnEntity {
 private:
  class ResidentBudgetToken {
   public:
    explicit ResidentBudgetToken(uint64_t bytes) : bytes_(bytes) {}
    ~ResidentBudgetToken() {
      ailego::MemoryBudgetManager::get_instance().release(
          ailego::MemoryBudgetManager::Category::ResidentMetadata, bytes_);
    }

   private:
    uint64_t bytes_;
  };

 public:
  using Pointer = std::shared_ptr<DiskAnnSearcherEntity>;
  using SegmentPointer = IndexStorage::Segment::Pointer;

 public:
  DiskAnnSearcherEntity() = default;
  virtual ~DiskAnnSearcherEntity() = default;

 public:
  virtual const DiskAnnEntity::Pointer clone() const override;

  int load(const IndexMeta &meta, IndexStorage::Pointer storage);
  int load_pq_segment();
  int load_header_segment();
  int load_vector_segment();
  int load_key_segment();
  int load_key_mapping_segment();
  int load_entrypoint_segment();

  PQTable::Pointer get_pq_table() {
    return pq_table_;
  }

  IndexStorage::Pointer get_storage() {
    return storage_;
  }

  SegmentPointer get_vector_segment() {
    return vector_segment_;
  }

  std::vector<diskann_id_t> &entrypoints() {
    return *entrypoints_;
  }

  uint64_t resident_bytes() const {
    return (pq_table_ ? pq_table_->resident_bytes() : 0) +
           static_cast<uint64_t>(key_buffer_->capacity()) +
           static_cast<uint64_t>(key_mapping_buffer_->capacity()) +
           static_cast<uint64_t>(entrypoints_->capacity()) *
               sizeof(diskann_id_t);
  }

  int charge_resident_budget() {
    if (resident_budget_) {
      return 0;
    }
    const uint64_t bytes = resident_bytes();
    if (!ailego::MemoryBudgetManager::get_instance().try_charge(
            ailego::MemoryBudgetManager::Category::ResidentMetadata, bytes)) {
      LOG_ERROR("DiskANN resident-metadata budget exhausted: request=%llu",
                static_cast<unsigned long long>(bytes));
      return IndexError_NoMemory;
    }
    resident_budget_ = std::make_shared<ResidentBudgetToken>(bytes);
    return 0;
  }

  std::pair<uint32_t, const diskann_id_t *> get_neighbors(
      diskann_id_t id) const override;

  diskann_id_t get_id(diskann_key_t key) const override;
  diskann_key_t get_key(diskann_id_t id) const override;
  const void *get_vector(diskann_id_t id) const override;

 private:
  DiskAnnSearcherEntity(
      const DiskAnnMetaHeader &meta_header, const DiskAnnPqMeta &pq_meta,
      const SegmentPointer &meta_segment, const SegmentPointer &pq_meta_segment,
      const SegmentPointer &pq_data_segment,
      const SegmentPointer &vector_segment, const SegmentPointer &key_segment,
      const SegmentPointer &key_mapping_segment,
      const SegmentPointer &entrypoint_segment, uint32_t num_threads,
      uint32_t list_size, uint32_t cache_nodes_num, bool warm_up,
      uint32_t beam_size, const IndexMeta meta, PQTable::Pointer pq_table,
      const std::shared_ptr<std::string> &key_buffer,
      const std::shared_ptr<std::string> &key_mapping_buffer,
      const std::shared_ptr<std::vector<diskann_id_t>> &entrypoints,
      const std::shared_ptr<ResidentBudgetToken> &resident_budget)
      : DiskAnnEntity(meta_header, pq_meta),
        meta_segment_(meta_segment),
        pq_meta_segment_(pq_meta_segment),
        pq_data_segment_(pq_data_segment),
        vector_segment_(vector_segment),
        key_segment_(key_segment),
        key_mapping_segment_(key_mapping_segment),
        entrypoint_segment_{entrypoint_segment},
        num_threads_{num_threads},
        list_size_{list_size},
        cache_nodes_num_{cache_nodes_num},
        warm_up_{warm_up},
        beam_size_{beam_size},
        meta_{meta},
        pq_table_{pq_table},
        key_buffer_{key_buffer},
        key_mapping_buffer_{key_mapping_buffer},
        entrypoints_{entrypoints},
        resident_budget_{resident_budget} {}

  IndexStorage::Pointer storage_{};

  SegmentPointer meta_segment_{nullptr};
  SegmentPointer pq_meta_segment_{nullptr};
  SegmentPointer pq_data_segment_{nullptr};
  SegmentPointer vector_segment_{nullptr};
  SegmentPointer key_segment_{nullptr};
  SegmentPointer key_mapping_segment_{nullptr};
  SegmentPointer entrypoint_segment_{nullptr};

  uint32_t num_threads_{1};
  uint32_t list_size_{200};
  uint32_t cache_nodes_num_{0};

  bool warm_up_{false};
  uint32_t beam_size_{2};

  IndexMeta meta_;

  PQTable::Pointer pq_table_;
  std::shared_ptr<std::string> key_buffer_{std::make_shared<std::string>()};
  std::shared_ptr<std::string> key_mapping_buffer_{
      std::make_shared<std::string>()};
  std::shared_ptr<std::vector<diskann_id_t>> entrypoints_{
      std::make_shared<std::vector<diskann_id_t>>()};
  std::shared_ptr<ResidentBudgetToken> resident_budget_{};
};

}  // namespace core
}  // namespace zvec

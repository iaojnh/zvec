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

#include "diskann_context.h"
#include <chrono>
#include <limits>
#include <zvec/ailego/buffer/memory_budget.h>
#include "diskann_params.h"
#include "diskann_pq_table.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

DiskAnnContext::DiskAnnContext(const IndexMeta &meta,
                               const IndexMetric::Pointer &measure,
                               const DiskAnnEntity::Pointer &entity)
    : dc_(entity.get(), measure, meta.dimension()), entity_{entity} {}

int DiskAnnContext::init(ContextType type, uint32_t graph_degree,
                         uint32_t pq_chunk_num, uint32_t element_size,
                         uint32_t list_size) {
  type_ = type;
  element_size_ = element_size;
  pq_chunk_num_ = pq_chunk_num;
  list_size_ = list_size;

  if (type == kSearcherContext) {
    // Account the stable per-context working set before allocating it. This
    // includes the visit map, aligned query/PQ/I/O buffers and the two
    // list-sized candidate containers used by graph search.
    const uint64_t fixed_buffers =
        static_cast<uint64_t>(element_size_) * 3 +
        static_cast<uint64_t>(PQTable::kPQCentroidNum) * pq_chunk_num_ *
            sizeof(float) +
        static_cast<uint64_t>(graph_degree) * pq_chunk_num_ * sizeof(uint8_t) +
        static_cast<uint64_t>(DiskAnnUtil::kMaxSectorReadNum) *
            DiskAnnUtil::kSectorSize;
    const uint64_t search_containers =
        (static_cast<uint64_t>(list_size_) + 1) * sizeof(Neighbor) +
        static_cast<uint64_t>(list_size_) * sizeof(Neighbor);
    const uint64_t visit_map = entity_->doc_cnt();
    if (fixed_buffers > std::numeric_limits<uint64_t>::max() -
                            search_containers ||
        fixed_buffers + search_containers >
            std::numeric_limits<uint64_t>::max() - visit_map) {
      return IndexError_NoMemory;
    }
    query_budget_bytes_ = fixed_buffers + search_containers + visit_map;
    if (!ailego::MemoryBudgetManager::get_instance().try_charge(
            ailego::MemoryBudgetManager::Category::QueryWorking,
            query_budget_bytes_)) {
      LOG_ERROR("DiskANN query working-memory budget exhausted: request=%llu",
                static_cast<unsigned long long>(query_budget_bytes_));
      query_budget_bytes_ = 0;
      return IndexError_NoMemory;
    }
  }

  DiskAnnUtil::alloc_aligned((void **)&query_, element_size_, 32);
  DiskAnnUtil::alloc_aligned((void **)&query_rotated_, element_size_, 32);

  int ret;
  switch (type) {
    case kBuilderContext:
      ret = visit_filter_.init(VisitFilter::ByteMap, entity_->doc_cnt(),
                               entity_->doc_cnt(), negative_probility_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }
      break;

    case kSearcherContext:
      ret = visit_filter_.init(filter_mode_, entity_->doc_cnt(),
                               entity_->doc_cnt(), negative_probility_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }

      DiskAnnUtil::alloc_aligned(
          (void **)&pq_table_dist_buffer_,
          PQTable::kPQCentroidNum * pq_chunk_num_ * sizeof(float), 256);
      DiskAnnUtil::alloc_aligned((void **)&pq_coord_buffer_,
                                 graph_degree * pq_chunk_num_ * sizeof(uint8_t),
                                 256);
      DiskAnnUtil::alloc_aligned((void **)&coord_buffer_, element_size_, 256);
      DiskAnnUtil::alloc_aligned(
          (void **)&sector_buffer_,
          DiskAnnUtil::kMaxSectorReadNum * DiskAnnUtil::kSectorSize,
          DiskAnnUtil::kSectorSize);

      ret = setup_io_ctx(io_ctx_);
      if (ret != 0) {
        LOG_ERROR("setup io ctx error, ret=%d", ret);
        return ret;
      }
      break;

    default:
      LOG_ERROR("Init context failed");
      return IndexError_Runtime;
  }

  return 0;
}

DiskAnnContext::~DiskAnnContext() {
  free(query_);
  free(query_rotated_);
  free(pq_table_dist_buffer_);
  free(pq_coord_buffer_);
  free(coord_buffer_);
  free(sector_buffer_);
  visit_filter_.destroy();

  if (type_ == kSearcherContext) {
    destroy_io_ctx(io_ctx_);
  }
  ailego::MemoryBudgetManager::get_instance().release(
      ailego::MemoryBudgetManager::Category::QueryWorking,
      query_budget_bytes_);
}

int DiskAnnContext::update(const ailego::Params &params) {
  uint32_t list_size = list_size_;
  params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size);
  if (type_ == kSearcherContext && list_size != list_size_) {
    const uint64_t old_container_bytes =
        (static_cast<uint64_t>(list_size_) + 1) * sizeof(Neighbor) +
        static_cast<uint64_t>(list_size_) * sizeof(Neighbor);
    const uint64_t new_container_bytes =
        (static_cast<uint64_t>(list_size) + 1) * sizeof(Neighbor) +
        static_cast<uint64_t>(list_size) * sizeof(Neighbor);
    auto &budget = ailego::MemoryBudgetManager::get_instance();
    if (new_container_bytes > old_container_bytes) {
      const uint64_t additional =
          new_container_bytes - old_container_bytes;
      if (!budget.try_charge(
              ailego::MemoryBudgetManager::Category::QueryWorking,
              additional)) {
        LOG_ERROR(
            "DiskANN query working-memory budget exhausted while updating "
            "list_size: old=%u new=%u additional=%llu",
            list_size_, list_size,
            static_cast<unsigned long long>(additional));
        return IndexError_NoMemory;
      }
      query_budget_bytes_ += additional;
    } else {
      const uint64_t released = old_container_bytes - new_container_bytes;
      budget.release(ailego::MemoryBudgetManager::Category::QueryWorking,
                     released);
      query_budget_bytes_ -= released;
    }
  }
  list_size_ = list_size;
  return 0;
}

int DiskAnnContext::update_context(ContextType type, const IndexMeta &meta,
                                   const IndexMetric::Pointer &measure,
                                   const DiskAnnEntity::Pointer &entity,
                                   uint32_t magic_num) {
  if (ailego_unlikely(type != type_)) {
    LOG_ERROR(
        "DiskAnnContext does not support shared by different type, "
        "src=%u dst=%u",
        type_, type);
    return IndexError_Unsupported;
  }

  magic_ = kInvalidMgic;

  switch (type) {
    case kBuilderContext:
      LOG_ERROR("BuildContext does not support update");
      return IndexError_NotImplemented;

    case kSearcherContext:
      break;

    case kReducerContext:
      break;

    default:
      LOG_ERROR("update context failed");
      return IndexError_Runtime;
  }

  entity_ = entity;
  dc_.update(measure, meta.dimension());
  magic_ = magic_num;

  return 0;
}

}  // namespace core
}  // namespace zvec

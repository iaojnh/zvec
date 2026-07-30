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
#include <algorithm>
#include <chrono>
#include <limits>
#include <new>
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
    // Account the stable per-context working set before allocating it.
    const uint64_t pq_dist_bytes =
        static_cast<uint64_t>(PQTable::kPQCentroidNum) * pq_chunk_num_ *
        sizeof(float);
    const uint64_t pq_coord_bytes =
        static_cast<uint64_t>(graph_degree) * pq_chunk_num_ * sizeof(uint8_t);
    const uint64_t sector_bytes =
        static_cast<uint64_t>(DiskAnnUtil::kMaxSectorReadNum) *
        DiskAnnUtil::kSectorSize;
    uint64_t fixed_buffers = static_cast<uint64_t>(element_size_) * 3;
    const uint64_t fixed_parts[] = {pq_dist_bytes, pq_coord_bytes,
                                    sector_bytes};
    for (uint64_t part : fixed_parts) {
      if (fixed_buffers > std::numeric_limits<uint64_t>::max() - part) {
        return IndexError_NoMemory;
      }
      fixed_buffers += part;
    }
    const uint64_t visit_map = entity_->doc_cnt();
    if (fixed_buffers > std::numeric_limits<uint64_t>::max() - visit_map) {
      return IndexError_NoMemory;
    }
    query_fixed_budget_bytes_ = fixed_buffers + visit_map;

    // Heap::limit(0) still reserves one slot. Initialize that invariant here
    // so calculate_query_budget() always matches retained allocations.
    try {
      topk_heap_.limit(0);
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
    if (!calculate_query_budget(topk_, list_size_, &query_budget_bytes_)) {
      query_budget_bytes_ = 0;
      return IndexError_NoMemory;
    }
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
  if (element_size_ == 0 || query_ == nullptr || query_rotated_ == nullptr) {
    return IndexError_NoMemory;
  }

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

      DiskAnnUtil::alloc_aligned((void **)&pq_table_dist_buffer_,
                                 static_cast<size_t>(PQTable::kPQCentroidNum) *
                                     pq_chunk_num_ * sizeof(float),
                                 256);
      DiskAnnUtil::alloc_aligned(
          (void **)&pq_coord_buffer_,
          static_cast<size_t>(graph_degree) * pq_chunk_num_ * sizeof(uint8_t),
          256);
      DiskAnnUtil::alloc_aligned((void **)&coord_buffer_, element_size_, 256);
      DiskAnnUtil::alloc_aligned(
          (void **)&sector_buffer_,
          DiskAnnUtil::kMaxSectorReadNum * DiskAnnUtil::kSectorSize,
          DiskAnnUtil::kSectorSize);
      if (pq_table_dist_buffer_ == nullptr || pq_coord_buffer_ == nullptr ||
          coord_buffer_ == nullptr || sector_buffer_ == nullptr) {
        return IndexError_NoMemory;
      }

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
      ailego::MemoryBudgetManager::Category::QueryWorking, query_budget_bytes_);
}

int DiskAnnContext::update(const ailego::Params &params) {
  uint32_t list_size = list_size_;
  params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size);
  return set_search_params(topk_, list_size);
}

bool DiskAnnContext::calculate_query_budget(uint32_t topk, uint32_t list_size,
                                            uint64_t *bytes) const {
  if (bytes == nullptr) {
    return false;
  }

  const auto checked_add = [](uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
      return false;
    }
    *out = lhs + rhs;
    return true;
  };
  const auto checked_mul = [](uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
      return false;
    }
    *out = lhs * rhs;
    return true;
  };

  uint64_t candidate_slots = 0;
  if (!checked_add(static_cast<uint64_t>(list_size), 1, &candidate_slots) ||
      !checked_add(candidate_slots, static_cast<uint64_t>(list_size),
                   &candidate_slots)) {
    return false;
  }

  uint64_t candidate_bytes = 0;
  if (!checked_mul(candidate_slots, sizeof(Neighbor), &candidate_bytes)) {
    return false;
  }

  const uint64_t requested_heap_slots =
      std::max<uint64_t>(static_cast<uint64_t>(topk), 1);
  const uint64_t retained_heap_slots = std::max<uint64_t>(
      requested_heap_slots, static_cast<uint64_t>(topk_heap_.capacity()));
  uint64_t heap_bytes = 0;
  if (!checked_mul(retained_heap_slots, sizeof(TopkHeap::value_type),
                   &heap_bytes)) {
    return false;
  }

  uint64_t vector_payload_bytes = 0;
  if (!checked_mul(static_cast<uint64_t>(topk),
                   static_cast<uint64_t>(element_size_),
                   &vector_payload_bytes)) {
    return false;
  }
  // The heap always owns one full-vector copy per hit. When fetch_vector is
  // enabled, result materialization owns a second copy until the result is
  // moved out of the context. Charge both conservatively because the fetch
  // flag is set independently from topk/list_size.
  if (!checked_mul(vector_payload_bytes, 2, &vector_payload_bytes)) {
    return false;
  }

  uint64_t total = query_fixed_budget_bytes_;
  return checked_add(total, candidate_bytes, &total) &&
         checked_add(total, heap_bytes, &total) &&
         checked_add(total, vector_payload_bytes, bytes);
}

int DiskAnnContext::set_search_params(uint32_t topk, uint32_t list_size) {
  if (type_ != kSearcherContext) {
    try {
      topk_heap_.limit(topk);
    } catch (const std::bad_alloc &) {
      parameter_error_ = IndexError_NoMemory;
      return parameter_error_;
    }
    topk_ = topk;
    list_size_ = list_size;
    parameter_error_ = 0;
    return 0;
  }

  uint64_t new_budget_bytes = 0;
  if (!calculate_query_budget(topk, list_size, &new_budget_bytes)) {
    parameter_error_ = IndexError_NoMemory;
    return parameter_error_;
  }

  auto &budget = ailego::MemoryBudgetManager::get_instance();
  const uint64_t additional = new_budget_bytes > query_budget_bytes_
                                  ? new_budget_bytes - query_budget_bytes_
                                  : 0;
  if (additional != 0 &&
      !budget.try_charge(ailego::MemoryBudgetManager::Category::QueryWorking,
                         additional)) {
    LOG_ERROR(
        "DiskANN query working-memory budget exhausted: topk=%u list_size=%u "
        "additional=%llu",
        topk, list_size, static_cast<unsigned long long>(additional));
    parameter_error_ = IndexError_NoMemory;
    return parameter_error_;
  }

  try {
    // reserve() can throw. Do it only after the budget reservation, and keep
    // the old parameter values intact if allocation fails.
    topk_heap_.limit(topk);
  } catch (const std::bad_alloc &) {
    budget.release(ailego::MemoryBudgetManager::Category::QueryWorking,
                   additional);
    parameter_error_ = IndexError_NoMemory;
    return parameter_error_;
  }

  // Destroy old vector strings before releasing any payload budget.
  topk_heap_.clear();
  if (new_budget_bytes < query_budget_bytes_) {
    budget.release(ailego::MemoryBudgetManager::Category::QueryWorking,
                   query_budget_bytes_ - new_budget_bytes);
  }
  query_budget_bytes_ = new_budget_bytes;
  topk_ = topk;
  list_size_ = list_size;
  parameter_error_ = 0;
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

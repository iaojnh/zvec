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
#include "diskann_params.h"
#include "diskann_pq_table.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

namespace {

double average(double total, uint64_t count) {
  return count == 0 ? 0.0 : total / static_cast<double>(count);
}

void log_search_diagnostics(const SearchStats &stats, IOContext io_ctx) {
  if (stats.query_count == 0) {
    return;
  }

  LOG_INFO(
      "DiskAnn search diagnostics: queries=%llu, reads/query=%.2f, "
      "cache_hits/query=%.2f, hops/query=%.2f, batches/query=%.2f, "
      "reads/batch=%.2f, max_batch=%u, beam_limit/query=%.2f, "
      "max_beam_limit=%u, io_us/query=%.2f, cpu_us/query=%.2f, "
      "batch_histogram=[1:%llu,2-4:%llu,5-8:%llu,9-16:%llu,17-32:%llu]",
      static_cast<unsigned long long>(stats.query_count),
      average(stats.disk_page_reads, stats.query_count),
      average(stats.cache_hits, stats.query_count),
      average(stats.hop_num, stats.query_count),
      average(stats.io_batch_count, stats.query_count),
      average(stats.io_batch_size_sum, stats.io_batch_count),
      stats.io_batch_size_max,
      average(stats.effective_beam_width_sum, stats.query_count),
      stats.effective_beam_width_max, average(stats.io_us, stats.query_count),
      average(stats.cpu_us, stats.query_count),
      static_cast<unsigned long long>(stats.io_batch_size_histogram[0]),
      static_cast<unsigned long long>(stats.io_batch_size_histogram[1]),
      static_cast<unsigned long long>(stats.io_batch_size_histogram[2]),
      static_cast<unsigned long long>(stats.io_batch_size_histogram[3]),
      static_cast<unsigned long long>(stats.io_batch_size_histogram[4]));

#if defined(_WIN32) || defined(_WIN64)
  if (io_ctx == nullptr || io_ctx == reinterpret_cast<IOContext>(-1)) {
    return;
  }

  const auto &io = io_ctx->diagnostics;
  LOG_INFO(
      "DiskAnn IOCP diagnostics: submit_calls=%llu, submitted_reads=%llu, "
      "immediate_reads=%llu, pending_reads=%llu, pending_ratio=%.2f%%, "
      "max_outstanding=%u, dequeue_calls=%llu, completions/dequeue=%.2f, "
      "max_dequeued_once=%u, iocp_wait_us/query=%.2f",
      static_cast<unsigned long long>(io.submit_calls),
      static_cast<unsigned long long>(io.submitted_reads),
      static_cast<unsigned long long>(io.immediate_reads),
      static_cast<unsigned long long>(io.pending_reads),
      100.0 * average(io.pending_reads, io.submitted_reads), io.max_outstanding,
      static_cast<unsigned long long>(io.dequeue_calls),
      average(io.dequeued_reads, io.dequeue_calls), io.max_dequeued_once,
      average(io.wait_us, stats.query_count));
#else
  (void)io_ctx;
#endif
}

}  // namespace

DiskAnnContext::DiskAnnContext(const IndexMeta &meta,
                               const IndexMetric::Pointer &measure,
                               const DiskAnnEntity::Pointer &entity)
    : IndexContext(measure),
      dc_(entity.get(), measure, meta.dimension()),
      entity_{entity} {}

int DiskAnnContext::init(ContextType type, uint32_t graph_degree,
                         uint32_t pq_chunk_num, uint32_t element_size,
                         uint32_t disk_element_size) {
  if (!entity_ || element_size == 0) {
    LOG_ERROR("Invalid DiskAnn context parameters");
    return IndexError_InvalidArgument;
  }
  type_ = type;
  io_diagnostics_enabled_ = diskann_io_diagnostics_enabled();
  element_size_ = element_size;
  pq_chunk_num_ = pq_chunk_num;

  DiskAnnUtil::alloc_aligned((void **)&query_, element_size_, 32);
  DiskAnnUtil::alloc_aligned((void **)&query_rotated_, element_size_, 32);
  if (!query_ || !query_rotated_) {
    LOG_ERROR("Failed to allocate DiskAnn query buffers");
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
      if (graph_degree == 0 || pq_chunk_num_ == 0) {
        LOG_ERROR("Invalid DiskAnn search context dimensions");
        return IndexError_InvalidArgument;
      }

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
      DiskAnnUtil::alloc_aligned(
          (void **)&coord_buffer_,
          disk_element_size > 0 ? disk_element_size : element_size_, 256);
      DiskAnnUtil::alloc_aligned(
          (void **)&sector_buffer_,
          DiskAnnUtil::kMaxSectorReadNum * DiskAnnUtil::kSectorSize,
          DiskAnnUtil::kSectorSize);
      if (!pq_table_dist_buffer_ || !pq_coord_buffer_ || !coord_buffer_ ||
          !sector_buffer_) {
        LOG_ERROR("Failed to allocate DiskAnn search buffers");
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
  if (io_diagnostics_enabled_) {
    log_search_diagnostics(query_stats_, io_ctx_);
  }

  DiskAnnUtil::free_aligned(query_);
  DiskAnnUtil::free_aligned(query_rotated_);
  DiskAnnUtil::free_aligned(pq_table_dist_buffer_);
  DiskAnnUtil::free_aligned(pq_coord_buffer_);
  DiskAnnUtil::free_aligned(coord_buffer_);
  DiskAnnUtil::free_aligned(sector_buffer_);

  if (type_ == kSearcherContext) {
    destroy_io_ctx(io_ctx_);
  }
}

int DiskAnnContext::update(const ailego::Params &params) {
  uint32_t list_size = list_size_;
  params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size);
  list_size_ = list_size;
  return 0;
}

int DiskAnnContext::update_context(ContextType type, const IndexMeta &meta,
                                   const IndexMetric::Pointer &measure,
                                   const DiskAnnEntity::Pointer &entity,
                                   uint32_t magic_num) {
  if (ailego_unlikely(type != static_cast<ContextType>(type_))) {
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
  update_index_metric(measure);
  dc_.update(entity_.get(), measure, meta.dimension());
  magic_ = magic_num;

  return 0;
}

}  // namespace core
}  // namespace zvec

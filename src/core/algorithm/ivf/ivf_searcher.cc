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
#include "ivf_searcher.h"
#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "ailego/pattern/defer.h"
#include "ivf_centroid_index.h"
#include "ivf_index_provider.h"
#include "ivf_params.h"

namespace zvec {
namespace core {

int IVFSearcher::init(const ailego::Params &parameters) {
  params_ = parameters;
  quantizer_.reset();

  params_.get(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, &bruteforce_threshold_);

  searcher_state_ = STATE_INITED;

  return 0;
}

int IVFSearcher::init(const ailego::Params &parameters,
                      const turbo::Quantizer::Pointer &quantizer) {
  if (!quantizer) {
    return init(parameters);
  }
  int ret = init(parameters);
  if (ret == 0) {
    quantizer_ = quantizer;
  }
  return ret;
}

int IVFSearcher::cleanup() {
  this->unload();

  params_.clear();
  bruteforce_threshold_ = kDefaultBfThreshold;

  searcher_state_ = STATE_INIT;
  return 0;
}

int IVFSearcher::load(IndexStorage::Pointer container,
                      IndexMetric::Pointer /*metric*/) try {
  if (!container) {
    LOG_ERROR("Invalid container");
    return IndexError_InvalidArgument;
  }
  if (searcher_state_ != STATE_INITED) {
    LOG_ERROR("Initalize the searcher first before load index");
    return IndexError_Runtime;
  }

  ailego::ElapsedTime timer;
  bool loaded = false;
  AILEGO_DEFER([&] {
    if (!loaded) unload();
  });
  int ret = IndexHelper::DeserializeFromStorage(container.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from container");
    return ret;
  }

  //! Load centroid index
  centroid_index_ = std::make_shared<IVFCentroidIndex>();
  if (!centroid_index_) {
    return IndexError_NoMemory;
  }
  auto seg = container->get(IVF_CENTROID_SEG_ID, 0);
  if (!seg) {
    LOG_ERROR("Failed to get segment %s", IVF_CENTROID_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  IndexStorage::Pointer seg_container =
      std::make_shared<IndexSegmentStorage>(seg);
  if (!seg_container) {
    return IndexError_NoMemory;
  }
  ret = seg_container->open(std::string(), false);
  if (ret != 0) {
    LOG_ERROR("IndexSegmentStorage load failed for %s", IndexError::What(ret));
    return ret;
  }
  ret = centroid_index_->load(seg_container, params_);
  if (ret != 0) {
    LOG_ERROR("Failed to load index for %s", IndexError::What(ret));
    return ret;
  }

  auto reformer = centroid_index_->reformer();
  if (reformer) {
    //! The centroid index is loaded from the centroid sub-segment which does
    //! not contain the rotator segment. Load the reformer state (e.g. rotation
    //! matrix) from the main container instead.
    ret = reformer->load(container);
    ivf_check_error_code(ret);
  }
  params_.set(PARAM_IVF_SEARCHER_CONVERTER_REFORMER, reformer);

  //! load iverted index
  entity_ = std::make_shared<IVFEntity>();
  if (!entity_) {
    return IndexError_NoMemory;
  }
  ret = entity_->load(container);
  ivf_check_error_code(ret);

  // Restore from the persisted descriptor, including when an injected
  // quantizer was supplied. Legacy postings retain their original metric.
  quantizer_ = entity_->quantizer();

  magic_ = IndexContext::GenerateMagic();

  stats_.set_loaded_count(entity_->vector_count());
  stats_.set_loaded_costtime(timer.milli_seconds());

  searcher_state_ = STATE_LOADED;
  loaded = true;
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int IVFSearcher::unload() {
  magic_ = 0;
  params_.erase(PARAM_IVF_SEARCHER_CONVERTER_REFORMER);
  centroid_index_.reset();
  entity_.reset();
  quantizer_.reset();
  stats_.set_loaded_count(0UL);
  stats_.set_loaded_costtime(0UL);
  stats_.clear_attributes();
  searcher_state_ = STATE_INITED;

  return 0;
}

int IVFSearcher::search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                                Context::Pointer &context) const {
  return search_bf_impl(query, qmeta, 1, context);
}

int IVFSearcher::search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                                uint32_t count,
                                Context::Pointer &context) const {
  return search_batch(query, qmeta, count, context, true);
}

int IVFSearcher::search_bf_by_keys_impl(const void *query,
                                        const std::vector<uint64_t> &keys,
                                        const IndexQueryMeta &qmeta,
                                        Context::Pointer &context) const {
  return search_batch(query, qmeta, 1, context, true, nullptr, &keys);
}

int IVFSearcher::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (count == 0 || count != p_keys.size()) {
    if (context) context->reset();
    return IndexError_InvalidArgument;
  }
  return search_batch(query, qmeta, count, context, true, &p_keys);
}

int IVFSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                             Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

int IVFSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count, Context::Pointer &context) const {
  return search_batch(query, qmeta, count, context,
                      entity_->vector_count() <= bruteforce_threshold_);
}

int IVFSearcher::search_batch(const void *query, const IndexQueryMeta &qmeta,
                              uint32_t count, Context::Pointer &context,
                              bool brute,
                              const std::vector<std::vector<uint64_t>> *p_keys,
                              const std::vector<uint64_t> *single_keys) const
    try {
  bool completed = false;
  AILEGO_DEFER([&] {
    if (!completed && context) context->reset();
  });
  if (searcher_state_ != STATE_LOADED) return IndexError_Runtime;
  if (!query || qmeta.element_size() == 0 ||
      qmeta.element_size() != meta_.element_size() ||
      count > std::numeric_limits<size_t>::max() / qmeta.element_size())
    return IndexError_InvalidArgument;
  auto *ctx = dynamic_cast<IVFSearcherContext *>(context.get());
  if (!ctx || ctx->topk() == 0) return IndexError_InvalidArgument;
  if (ctx->magic() != magic_) {
    const int ret = update_context(ctx);
    ivf_check_error_code(ret);
  }
  auto release_candidates =
      ailego::ScopeGuard::Make([&] { ctx->mutable_result_heap().release(); });
  ctx->reset_results(count);
  auto &entity = ctx->entity();
  auto &filter = ctx->filter();
  auto &centroid_ctx = ctx->centroid_searcher_ctx();
  // Result lists belong to the caller, but transformed queries and centroid
  // candidates are working scratch. Bound those independently of batch size.
  // Account for nprobe as well as vector width when choosing the tile size.
  const size_t row_bytes =
      std::max<size_t>(qmeta.element_size(), entity->meta().element_size());
  const size_t candidate_bytes =
      brute ? 0
            : static_cast<size_t>(centroid_ctx->topk()) * sizeof(IndexDocument);
  const size_t tile_size = std::max<size_t>(
      1, std::min<size_t>(
             32, 64 * 1024 / std::max<size_t>(1, row_bytes + candidate_bytes)));
  for (size_t base = 0; base < count; base += tile_size) {
    const uint32_t n =
        static_cast<uint32_t>(std::min<size_t>(tile_size, count - base));
    const void *input =
        static_cast<const char *>(query) + base * qmeta.element_size();
    if (!brute) {
      centroid_ctx->reset();
      const int ret = centroid_index_->search(input, qmeta, n, centroid_ctx);
      ivf_check_error_code(ret);
    }
    IndexQueryMeta iv_qmeta;
    const void *converted = nullptr;
    int ret = entity->transform(input, qmeta, n, &converted, &iv_qmeta);
    ivf_check_error_code(ret);
    if (n > 1 && !p_keys && !single_keys && !filter.is_valid() &&
        !entity->buffered_storage()) {
      bool handled = false;
      ret = search_many(converted, iv_qmeta, n, ctx, brute, base, &handled);
      ivf_check_error_code(ret);
      if (handled) continue;
    }
    for (uint32_t q = 0; q < n; ++q) {
      const void *one = static_cast<const char *>(converted) +
                        static_cast<size_t>(q) * iv_qmeta.element_size();
      ret = entity->bind_query(one, iv_qmeta);
      ivf_check_error_code(ret);
      auto &stats = ctx->mutable_stats(base + q);
      auto &heap = ctx->mutable_result_heap();
      heap.clear();
      const auto *keys = single_keys ? single_keys
                         : p_keys    ? &(*p_keys)[base + q]
                                     : nullptr;
      if (keys) {
        ret = entity->search_by_p_keys(one, *keys, filter, &heap, &stats);
        ivf_check_error_code(ret);
      } else if (brute) {
        ret = !filter.is_valid() ? entity->search(one, &heap, &stats)
                                 : entity->search(one, filter, &heap, &stats);
        ivf_check_error_code(ret);
      } else {
        auto release_prefetch =
            ailego::ScopeGuard::Make([&] { entity->release_prefetch(); });
        const auto &centroids = centroid_ctx->result(q);
        size_t hinted_scans = 0, next_hint = 0, scans = 0;
        for (size_t i = 0;
             i < centroids.size() && scans < ctx->max_scan_count(); ++i) {
          // An oversized or unadmitted list can execute synchronously. Once
          // consumed, skip it so it cannot block the remaining lookahead.
          if (next_hint < i) {
            next_hint = i;
            hinted_scans = std::max(hinted_scans, scans);
          }
          while (next_hint < centroids.size() &&
                 hinted_scans < ctx->max_scan_count()) {
            bool accepted = false;
            const size_t hinted_count =
                entity->prefetch_list(centroids[next_hint].key(), &accepted);
            if (!accepted) break;
            hinted_scans += hinted_count;
            ++next_hint;
          }
          const auto &centroid = centroids[i];
          uint32_t scanned = 0;
          ret = !filter.is_valid() ? entity->search(centroid.key(), one,
                                                    &scanned, &heap, &stats)
                                   : entity->search(centroid.key(), one, filter,
                                                    &scanned, &heap, &stats);
          ivf_check_error_code(ret);
          scans += scanned;
        }
      }
      heap.sort();
      if (!keys && !filter.is_valid()) {
        ret = entity->retrieve_keys(&heap);
        ivf_check_error_code(ret);
      }
      entity->normalize(q, &heap);
      ret = ctx->topk_to_result(static_cast<uint32_t>(base + q));
      ivf_check_error_code(ret);
    }
  }
  completed = true;
  return 0;
} catch (const std::bad_alloc &) {
  if (context) context->reset();
  return IndexError_NoMemory;
}

int IVFSearcher::search_many(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count, IVFSearcherContext *ctx,
                             bool brute_force, size_t result_offset,
                             bool *handled) const {
  auto &entity = ctx->entity();
  const auto &quantizer = entity->quantizer();
  if (quantizer && quantizer->type() != turbo::QuantizeType::kFp16 &&
      quantizer->type() != turbo::QuantizeType::kFp32) {
    return 0;
  }
  int ret = entity->bind_query(query, qmeta);
  ivf_check_error_code(ret);
  const bool column_major = entity->column_many_supported();
  auto distance = entity->many_distance();
  if (!distance && !column_major) return 0;
  *handled = true;

  // Bound scratch memory independently of the caller's query count.
  constexpr size_t kQueryBatch = 32;
  for (size_t base = 0; base < count; base += kQueryBatch) {
    const size_t n = std::min(kQueryBatch, count - base);
    std::vector<turbo::DistanceImpl> queries;
    std::vector<IndexCandidateHeap> heaps;
    queries.reserve(n);
    heaps.reserve(n);
    for (size_t q = 0; q < n; ++q) {
      const void *data =
          static_cast<const char *>(query) + (base + q) * qmeta.element_size();
      if (column_major) {
        queries.push_back(entity->column_query(data, qmeta));
      } else {
        if (base + q != 0) {
          ret = entity->bind_query(static_cast<const char *>(query) +
                                       (base + q) * qmeta.element_size(),
                                   qmeta);
          ivf_check_error_code(ret);
        }
        queries.push_back(entity->query_distance());
      }
      auto &heap = heaps.emplace_back();
      heap.limit(ctx->topk());
      heap.set_threshold(ctx->threshold());
    }

    std::vector<size_t> next(n, 0), scanned(n, 0);
    const auto pending = [&](size_t q) {
      return brute_force ? next[q] < entity->inverted_list_count()
                         : next[q] < ctx->centroid_searcher_ctx()
                                         ->result(base + q)
                                         .size() &&
                               scanned[q] < ctx->max_scan_count();
    };
    const auto centroid = [&](size_t q) -> size_t {
      return brute_force ? next[q]
                         : ctx->centroid_searcher_ctx()
                               ->result(base + q)[next[q]]
                               .key();
    };
    std::vector<const turbo::DistanceImpl *> group;
    std::vector<IndexCandidateHeap *> group_heaps;
    std::vector<IndexContext::Stats *> group_stats;
    std::vector<size_t> members;
    group.reserve(n);
    group_heaps.reserve(n);
    group_stats.reserve(n);
    members.reserve(n);
    for (;;) {
      size_t first = 0;
      while (first < n && !pending(first)) ++first;
      if (first == n) break;
      const size_t cid = centroid(first);
      group.clear();
      group_heaps.clear();
      group_stats.clear();
      members.clear();
      // Only join queries at the same frontier. This preserves each query's
      // list order, scan limit, and heap tie-breaking order.
      for (size_t q = first; q < n; ++q) {
        if (pending(q) && centroid(q) == cid) {
          group.push_back(&queries[q]);
          group_heaps.push_back(&heaps[q]);
          group_stats.push_back(&ctx->mutable_stats(result_offset + base + q));
          members.push_back(q);
        }
      }
      uint32_t scan_count = 0;
      if (column_major && group.size() == 1) {
        // No shared scan is possible; retain the original single-query path.
        ret = entity->search(cid, group[0]->query_storage().data(), &scan_count,
                             group_heaps[0], group_stats[0]);
      } else {
        ret = entity->search_many(cid, group, distance, group_heaps,
                                  group_stats, &scan_count);
      }
      ivf_check_error_code(ret);
      for (size_t q : members) {
        ++next[q];
        scanned[q] += scan_count;
      }
    }
    for (size_t q = 0; q < n; ++q) {
      heaps[q].sort();
      ret = entity->retrieve_keys(&heaps[q]);
      ivf_check_error_code(ret);
      entity->normalize(result_offset + base + q, &heaps[q]);
      ret = ctx->topk_to_result(
          static_cast<uint32_t>(result_offset + base + q), heaps[q]);
      ivf_check_error_code(ret);
    }
  }
  return 0;
}

const IndexSearcher::Stats &IVFSearcher::stats() const {
  return stats_;
}

IndexSearcher::Context::Pointer IVFSearcher::create_context() const {
  if (searcher_state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before create context");
    return nullptr;
  }

  auto entity = entity_->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone IVFEntity");
    return nullptr;
  }

  auto centroid_index_ctx = centroid_index_->create_context();
  if (!centroid_index_ctx) {
    LOG_ERROR("Failed to create centroid index context");
    return nullptr;
  }

  auto context =
      new (std::nothrow) IVFSearcherContext(entity, centroid_index_ctx);
  if (!context) {
    LOG_ERROR("Failed to alloc IVFSearcherContext");
    return nullptr;
  }
  int ret = context->init(params_);
  if (ret != 0) {
    delete context;
    return nullptr;
  }

  context->set_magic(magic_);

  return Context::Pointer(context);
}

IndexProvider::Pointer IVFSearcher::create_provider() const {
  if (searcher_state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before create provider");
    return nullptr;
  }

  auto entity = entity_->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone IVFEntity");
    return Provider::Pointer();
  }

  auto *provider = new (std::nothrow) IVFIndexProvider(
      (entity->has_orignal_feature() || entity->quantizer()) ? meta_
                                                             : entity->meta(),
      entity, "IVFSearcher");
  if (!provider) {
    LOG_ERROR("Failed to alloc IVFIndexProvider");
    return Provider::Pointer();
  }

  return Provider::Pointer(provider);
}

INDEX_FACTORY_REGISTER_SEARCHER(IVFSearcher);

}  // namespace core
}  // namespace zvec

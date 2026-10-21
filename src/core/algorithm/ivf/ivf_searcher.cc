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

int IVFSearcher::search_batch(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context, bool brute,
    const std::vector<std::vector<uint64_t>> *p_keys) const try {
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
    for (uint32_t q = 0; q < n; ++q) {
      const void *one = static_cast<const char *>(converted) +
                        static_cast<size_t>(q) * iv_qmeta.element_size();
      ret = entity->bind_query(one, iv_qmeta);
      ivf_check_error_code(ret);
      auto &stats = ctx->mutable_stats(base + q);
      auto &heap = ctx->mutable_result_heap();
      heap.clear();
      if (p_keys) {
        ret = entity->search_by_p_keys(one, (*p_keys)[base + q], filter, &heap,
                                       &stats);
        ivf_check_error_code(ret);
      } else if (brute) {
        ret = !filter.is_valid() ? entity->search(one, &heap, &stats)
                                 : entity->search(one, filter, &heap, &stats);
        ivf_check_error_code(ret);
      } else {
        size_t scans = 0;
        for (const auto &centroid : centroid_ctx->result(q)) {
          if (scans >= ctx->max_scan_count()) break;
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
      if (!p_keys && !filter.is_valid()) {
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

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

#include "diskann_searcher.h"
#include "diskann_context.h"
#include "diskann_indexer.h"
#include "diskann_params.h"

namespace zvec {
namespace core {

DiskAnnSearcher::DiskAnnSearcher() {}

DiskAnnSearcher::~DiskAnnSearcher() {}

int DiskAnnSearcher::init(const ailego::Params &search_params) {
  log_diskann_io_backend();

  search_params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size_);
  search_params.get(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, &cache_nodes_num_);
  search_params.get(PARAM_DISKANN_SEARCHER_CACHE_NODE_BUDGET_BYTES,
                    &cache_node_budget_bytes_);
  search_params.get(PARAM_DISKANN_SEARCHER_CACHE_NODE_PAGE_POLICY,
                    &cache_node_page_policy_);
  search_params.get(PARAM_DISKANN_SEARCHER_CACHE_NODE_LIST_PATH,
                    &cache_node_list_path_);
  search_params.get(PARAM_DISKANN_SEARCHER_WARMUP_MODE, &warmup_mode_);
  search_params.get(PARAM_DISKANN_SEARCHER_WARMUP_NODE_NUM, &warmup_node_num_);
  if (cache_nodes_num_ != 0 && cache_node_budget_bytes_ != 0) {
    LOG_ERROR("cache_node_num and cache_node_budget_bytes cannot both be set");
    return IndexError_InvalidArgument;
  }
  if (cache_node_page_policy_ != DISKANN_CACHE_NODE_PAGE_POLICY_KEEP &&
      cache_node_page_policy_ != DISKANN_CACHE_NODE_PAGE_POLICY_EVICT) {
    LOG_ERROR("Unknown cache_node_page_policy: %s",
              cache_node_page_policy_.c_str());
    return IndexError_InvalidArgument;
  }
  if (warmup_mode_ != DISKANN_WARMUP_MODE_NONE &&
      warmup_mode_ != DISKANN_WARMUP_MODE_BFS &&
      warmup_mode_ != DISKANN_WARMUP_MODE_QUERY_SAMPLE &&
      warmup_mode_ != DISKANN_WARMUP_MODE_CACHE_LIST) {
    LOG_ERROR("Unknown DiskANN warmup_mode: %s", warmup_mode_.c_str());
    return IndexError_InvalidArgument;
  }
  if (warmup_mode_ != DISKANN_WARMUP_MODE_NONE && warmup_node_num_ == 0) {
    LOG_ERROR(
        "DiskANN warmup_node_num must be positive when warmup is enabled");
    return IndexError_InvalidArgument;
  }
  if ((warmup_mode_ == DISKANN_WARMUP_MODE_QUERY_SAMPLE ||
       warmup_mode_ == DISKANN_WARMUP_MODE_CACHE_LIST) &&
      cache_node_list_path_.empty()) {
    LOG_ERROR("DiskANN query-sample warmup requires cache_node_list_path");
    return IndexError_InvalidArgument;
  }
  return 0;
}

void DiskAnnSearcher::print_debug_info() {}

int DiskAnnSearcher::cleanup() {
  LOG_INFO("Begin DiskAnnSearcher:cleanup");

  LOG_INFO("End DiskAnnSearcher:cleanup");

  return 0;
}

int DiskAnnSearcher::load(IndexStorage::Pointer storage,
                          IndexMetric::Pointer measure) {
  LOG_INFO("DiskAnnSearcher::load Begin");

  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from storage");
    return ret;
  }

  ret = entity_.load(meta_, storage);
  if (ret != 0) {
    LOG_INFO("Searcher Entity Load Failed");
    entity_ = DiskAnnSearcherEntity();
    return ret;
  }

  try {
    diskann_indexer_ = std::make_shared<DiskAnnIndexer>(meta_);
  } catch (const std::bad_alloc &) {
    entity_ = DiskAnnSearcherEntity();
    return IndexError_NoMemory;
  }

  int res = diskann_indexer_->init(entity_);
  if (res != 0) {
    diskann_indexer_.reset();
    entity_ = DiskAnnSearcherEntity();
    return res;
  }

  ret = diskann_indexer_->configure_cache(
      cache_nodes_num_, cache_node_budget_bytes_, cache_node_list_path_,
      cache_node_page_policy_, warmup_mode_, warmup_node_num_);
  if (ret != 0) {
    diskann_indexer_.reset();
    entity_ = DiskAnnSearcherEntity();
    return ret;
  }

  if (measure) {
    measure_ = measure;
  } else {
    measure_ = IndexFactory::CreateMetric(meta_.metric_name());
    if (!measure_) {
      LOG_ERROR("CreateMetric failed, name: %s", meta_.metric_name().c_str());
      diskann_indexer_.reset();
      entity_ = DiskAnnSearcherEntity();
      return IndexError_NoExist;
    }
    ret = measure_->init(meta_, meta_.metric_params());
    if (ret != 0) {
      LOG_ERROR("IndexMetric init failed, ret=%d", ret);
      diskann_indexer_.reset();
      entity_ = DiskAnnSearcherEntity();
      return ret;
    }
    if (measure_->query_metric()) {
      measure_ = measure_->query_metric();
    }
  }

  stats_.set_loaded_costtime(ailego::Monotime::MilliSeconds() - start_time);
  state_ = STATE_LOADED;

  magic_ = IndexContext::GenerateMagic();

  LOG_INFO("DiskAnnSearcher::load Done");

  return 0;
}

int DiskAnnSearcher::unload() {
  LOG_INFO("DiskAnnSearcher unload index");

  diskann_indexer_.reset();
  entity_ = DiskAnnSearcherEntity();
  measure_.reset();
  state_ = STATE_INITED;

  return 0;
}

int DiskAnnSearcher::update_context(DiskAnnContext *ctx) const {
  const DiskAnnEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }

  return ctx->update_context(DiskAnnContext::kSearcherContext, meta_, measure_,
                             entity, magic_);
}

int DiskAnnSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                                 uint32_t count,
                                 Context::Pointer &context) const {
  // do search
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }
  if (ctx->parameter_error() != 0) {
    return ctx->parameter_error();
  }

  // Context is pooled per index type. When switching between DiskAnn indexes
  // with different element sizes (e.g., fp16 vs fp32), the cached context has
  // undersized buffers. Recreate it to ensure correct buffer allocations.
  if (ctx->magic() != magic_) {
    uint32_t saved_topk = ctx->topk();
    uint32_t saved_list_size = ctx->list_size();
    bool saved_fetch_vector = ctx->fetch_vector();
    context.reset();
    context = create_context();
    if (!context) {
      LOG_ERROR("Failed to recreate context for current streamer");
      return IndexError_Runtime;
    }
    ctx = dynamic_cast<DiskAnnContext *>(context.get());
    int ret = ctx->set_search_params(saved_topk, saved_list_size);
    if (ret != 0) {
      return ret;
    }
    ctx->set_fetch_vector(saved_fetch_vector);
  }

  ctx->clear();
  ctx->resize_results(count);

  for (uint32_t i = 0; i < count; i++) {
    ctx->reset_query(query);

    int ret = diskann_indexer_->knn_search(ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  return 0;
}

int DiskAnnSearcher::search_bf_impl(const void *query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }
  if (ctx->parameter_error() != 0) {
    return ctx->parameter_error();
  }

  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer, recreate it
    //! to ensure buffers are correctly sized for this index's parameters.
    uint32_t saved_topk = ctx->topk();
    uint32_t saved_list_size = ctx->list_size();
    bool saved_fetch_vector = ctx->fetch_vector();
    context.reset();
    context = create_context();
    if (!context) {
      LOG_ERROR("Failed to recreate context for current streamer");
      return IndexError_Runtime;
    }
    ctx = dynamic_cast<DiskAnnContext *>(context.get());
    int ret = ctx->set_search_params(saved_topk, saved_list_size);
    if (ret != 0) {
      return ret;
    }
    ctx->set_fetch_vector(saved_fetch_vector);
  }

  ctx->clear();
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    int ret = diskann_indexer_->linear_search(ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int DiskAnnSearcher::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }
  if (ctx->parameter_error() != 0) {
    return ctx->parameter_error();
  }

  if (ailego_unlikely(p_keys.size() != count)) {
    LOG_ERROR("The size of p_keys is not equal to count");
    return IndexError_InvalidArgument;
  }

  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer, recreate it
    //! to ensure buffers are correctly sized for this index's parameters.
    uint32_t saved_topk = ctx->topk();
    uint32_t saved_list_size = ctx->list_size();
    bool saved_fetch_vector = ctx->fetch_vector();
    context.reset();
    context = create_context();
    if (!context) {
      LOG_ERROR("Failed to recreate context for current streamer");
      return IndexError_Runtime;
    }
    ctx = dynamic_cast<DiskAnnContext *>(context.get());
    int ret = ctx->set_search_params(saved_topk, saved_list_size);
    if (ret != 0) {
      return ret;
    }
    ctx->set_fetch_vector(saved_fetch_vector);
  }

  ctx->clear();
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    int ret = diskann_indexer_->keys_search(p_keys[i], ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int DiskAnnSearcher::get_vector(uint64_t key, Context::Pointer &context,
                                std::string &vector) const {
  auto *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  if (ctx == nullptr || ctx->magic() != magic_) {
    context.reset();
    context = create_context();
    if (!context) {
      return IndexError_Runtime;
    }
  }
  const diskann_id_t id = diskann_indexer_->get_id(key);
  if (id == kInvalidId) {
    return IndexError_NoExist;
  }
  return diskann_indexer_->get_vector(id, context, vector);
}

IndexSearcher::Context::Pointer DiskAnnSearcher::create_context() const {
  const DiskAnnEntity::Pointer search_ctx_entity = entity_.clone();
  if (!search_ctx_entity) {
    LOG_ERROR("Failed to create search context entity");
    return Context::Pointer();
  }

  DiskAnnContext *ctx =
      new (std::nothrow) DiskAnnContext(meta_, measure_, search_ctx_entity);
  if (ctx == nullptr) {
    LOG_ERROR("Failed to allocate DiskAnn Context");
    return Context::Pointer();
  }
  if (ailego_unlikely(ctx->init(DiskAnnContext::kSearcherContext,
                                search_ctx_entity->max_degree(),
                                search_ctx_entity->pq_chunk_num(),
                                meta_.element_size(), list_size_)) != 0) {
    LOG_ERROR("Init DiskAnn Context failed");
    delete ctx;

    return Context::Pointer();
  }

  ctx->set_list_size(list_size_);
  ctx->set_magic(magic_);

  return Context::Pointer(ctx);
}

INDEX_FACTORY_REGISTER_SEARCHER(DiskAnnSearcher);

}  // namespace core
}  // namespace zvec

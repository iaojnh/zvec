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

#include <ailego/math/inner_product_matrix.h>
#include <ailego/pattern/scope_guard.h>
#include "flat_sparse_context.h"

namespace zvec {
namespace core {

using FlatSparseGroupMap = std::map<
    GroupKey, IndexCandidateHeap, std::less<>,
    ailego::BudgetedAllocator<std::pair<const GroupKey, IndexCandidateHeap>>>;

static inline IndexGroupDocumentList ConvertGroupMapToResult(
    const FlatSparseGroupMap &group_map, uint32_t group_num) {
  IndexGroupDocumentList result;
  using Entry = FlatSparseGroupMap::value_type;
  ailego::BudgetedVector<const Entry *> ranked;
  for (const auto &entry : group_map)
    if (!entry.second.empty()) ranked.push_back(&entry);
  std::sort(ranked.begin(), ranked.end(), [](const Entry *a, const Entry *b) {
    return a->second[0].score() < b->second[0].score();
  });
  for (size_t i = 0; i < group_num && i < ranked.size(); ++i) {
    const auto &items = ranked[i]->second.container();
    result.emplace_back(ranked[i]->first,
                        IndexDocumentList(items.begin(), items.end()));
  }
  return result;
}

static inline int FlatSearch(const uint32_t *sparse_count,
                             const uint32_t *sparse_indices,
                             const void *sparse_query, bool with_p_keys,
                             IndexKeyBatchView p_keys,
                             const IndexQueryMeta &qmeta, uint32_t count,
                             const IndexMeta, IndexContext::Pointer &context,
                             FlatSparseEntity *entity) try {
  int ret;

  FlatSparseContext *ctx = dynamic_cast<FlatSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to FlatSparseContext failed");
    return IndexError_Cast;
  }

  auto release_candidates =
      ailego::ScopeGuard::Make([&] { ctx->result_heap()->release(); });
  // reset context results
  ctx->reset_results(count);

  const uint32_t *sparse_indices_tmp = sparse_indices;
  const void *sparse_query_tmp = sparse_query;

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
      return ctx->group_by()(key);
    };

    for (size_t q = 0; q < count; ++q) {
      auto &sparse_query_buffer = ctx->query_buffer();
      sparse_query_buffer.clear();
      SparseUtility::TransSparseFormat(sparse_count[q], sparse_indices_tmp,
                                       sparse_query_tmp, qmeta.unit_size(),
                                       sparse_query_buffer);

      FlatSparseGroupMap group_heap;
      GroupCandidateCollector groups(ctx->group_num(), ctx->group_topk());

      if (with_p_keys) {
        ret = entity->search_group_p_keys(sparse_query_buffer, p_keys[q],
                                          ctx->filter(), group_by,
                                          ctx->group_topk(), &groups);
      } else {
        ret = entity->search_group(sparse_query_buffer, ctx->filter(), group_by,
                                   ctx->group_topk(), &groups);
      }

      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Failed to search group, ret=%s", IndexError::What(ret));
        return ret;
      }

      // sort group heap
      ret = groups.finish_heaps(group_heap, ctx->group_topk());
      if (ret != 0) return ret;
      for (auto &group : group_heap) {
        group.second.sort();
      }

      auto group_result = ConvertGroupMapToResult(group_heap, ctx->group_num());

      // Populate sparse vector data when fetch_vector is enabled
      if (ctx->fetch_vector()) {
        for (auto &group_doc : group_result) {
          for (auto &doc : *group_doc.mutable_docs()) {
            node_id_t id = entity->get_id(doc.key());
            if (id == kInvalidNodeId)
              throw FlatSparseEntity::ReadError{IndexError_NoExist};
            {
              IndexSparseDocument sparse_doc;
              IndexStorage::MemoryBlock vec_block;
              int read_ret = entity->get_sparse_vector(id, vec_block);
              if (read_ret != 0 || !vec_block.data())
                throw FlatSparseEntity::ReadError{
                    read_ret != 0 ? read_ret : IndexError_ReadData};
              const void *sparse_data = vec_block.data();
              if (sparse_data != nullptr) {
                SparseUtility::ReverseSparseFormat(sparse_data, sparse_doc,
                                                   entity->sparse_unit_size());
              }
              // Reconstruct doc with sparse vector data
              doc = IndexDocument(doc.key(), doc.score(), id, nullptr,
                                  std::move(sparse_doc));
            }
          }
        }
      }

      ctx->mutable_group_result(q)->swap(group_result);
      sparse_indices_tmp += sparse_count[q];
      sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                         sparse_count[q] * qmeta.unit_size();
    }
  } else {
    for (size_t q = 0; q < count; ++q) {
      auto &sparse_query_buffer = ctx->query_buffer();
      sparse_query_buffer.clear();
      SparseUtility::TransSparseFormat(sparse_count[q], sparse_indices_tmp,
                                       sparse_query_tmp, qmeta.unit_size(),
                                       sparse_query_buffer);

      auto heap = ctx->result_heap();
      heap->clear();

      if (with_p_keys) {
        ret = entity->search_p_keys(sparse_query_buffer, p_keys[q],
                                    ctx->filter(), heap);
      } else {
        ret = entity->search(sparse_query_buffer, ctx->filter(), heap);
      }

      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Failed to search, ret=%s", IndexError::What(ret));
        return ret;
      }

      ctx->topk_to_result(q);

      sparse_indices_tmp += sparse_count[q];
      sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                         sparse_count[q] * qmeta.unit_size();
    }
  }

  return 0;
} catch (const FlatSparseEntity::ReadError &error) {
  if (context) context->reset();
  return error.status;
} catch (const std::bad_alloc &) {
  if (context) context->reset();
  return IndexError_NoMemory;
}

}  // namespace core
}  // namespace zvec

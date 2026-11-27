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

#include "diskann_algorithm.h"
#include <algorithm>
#include <set>
#include <ailego/pattern/defer.h>
#include <zvec/core/framework/index_holder.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

DiskAnnAlgorithm::DiskAnnAlgorithm(DiskAnnEntity &entity, uint32_t max_degree,
                                   size_t thread_count)
    : entity_(entity), max_degree_(max_degree) {
  const size_t concurrency = std::max<size_t>(1, thread_count);
  const size_t target = std::max<size_t>(
      1, std::min<size_t>(entity.doc_cnt(), concurrency >= kLockCnt / 64
                                                ? kLockCnt
                                                : concurrency * 64));
  size_t count = 1;
  while (count * 2 <= target) count *= 2;
  for (;;) {
    try {
      decltype(lock_pool_) replacement(count);
      lock_pool_.swap(replacement);
      break;
    } catch (const std::bad_alloc &) {
      if (count == 1) throw;
      count /= 2;
    }
  }
}

ailego::BudgetedVector<diskann_id_t> DiskAnnAlgorithm::get_init_ids(
    DiskAnnContext *ctx) {
  const auto &entity = ctx->get_entity();

  ailego::BudgetedVector<diskann_id_t> init_ids;

  init_ids.emplace_back(entity.medoid());

  return init_ids;
}

int DiskAnnAlgorithm::add_node(diskann_id_t id, DiskAnnContext *ctx) try {
  // reset_query copies the vector into context-owned storage, so the page can
  // be released before traversal needs to read any other vector.
  {
    IndexStorage::MemoryBlock block;
    const int ret = entity_.read_vector(id, block);
    if (ret != 0) {
      return ret;
    }
    if (block.data() == nullptr) {
      return IndexError_ReadData;
    }
    ctx->reset_query(block.data());
  }

  ailego::BudgetedVector<diskann_id_t> pruned_list;

  int ret = search_neighbor_and_prune(id, pruned_list, ctx);
  if (ret != 0) {
    return ret;
  }

  {
    auto lock = lock_for(id);
    ret = entity_.set_neighbors(id, pruned_list);
    if (ret != 0) {
      return ret;
    }
  }

  return inter_insert(id, pruned_list, ctx);
} catch (const std::bad_alloc &) {
  ctx->best_list_nodes().release();
  ctx->clear();
  return IndexError_NoMemory;
}

int DiskAnnAlgorithm::prune_node(diskann_id_t id, DiskAnnContext *ctx) try {
  DistCalculator &dc = ctx->dist_calculator();
  dc.clear();

  ailego::BudgetedVector<diskann_id_t> neighbors;
  int ret;
  {
    auto lock = lock_for(id);
    ret = entity_.read_neighbors(id, &neighbors);
  }
  if (ret != 0) {
    return ret;
  }

  if (neighbors.size() > max_degree_) {
    PruneCandidates dummy_pool;
    ailego::BudgetedVector<diskann_id_t> new_out_neighbors;
    PruneVectorCache cache;
    ret = collect_prune_candidates(id, neighbors, dummy_pool, ctx, &cache);
    if (ret != 0) return ret;

    ret = prune_neighbors(id, dummy_pool, new_out_neighbors, ctx, &cache);
    if (ret != 0) {
      return ret;
    }

    cache.clear();
    {
      auto lock = lock_for(id);
      ret = entity_.set_neighbors(id, new_out_neighbors);
      if (ret != 0) {
        return ret;
      }
    }
  }

  return 0;
} catch (const std::bad_alloc &) {
  ctx->clear();
  return IndexError_NoMemory;
}

int DiskAnnAlgorithm::inter_insert(
    diskann_id_t id, ailego::BudgetedVector<diskann_id_t> &pruned_list,
    DiskAnnContext *ctx) {
  for (auto &des : pruned_list) {
    ailego::BudgetedVector<diskann_id_t> new_neighbors;
    bool need_prune = false;

    {
      auto lock = lock_for(des);

      ailego::BudgetedVector<diskann_id_t> neighbors;
      const int ret = entity_.read_neighbors(des, &neighbors);
      if (ret != 0) {
        return ret;
      }

      bool found = false;
      for (diskann_id_t neighbor : neighbors) {
        if (neighbor == id) {
          found = true;
          break;
        }
      }

      if (!found) {
        if (neighbors.size() <
            static_cast<uint64_t>(DiskAnnEntity::kDefaultGraphSlackFactor *
                                  max_degree_)) {
          const int add_ret = entity_.add_neighbor(des, id);
          if (add_ret != 0) {
            return add_ret;
          }
          need_prune = false;
        } else {
          new_neighbors = std::move(neighbors);
          new_neighbors.push_back(id);

          need_prune = true;
        }
      }
    }

    if (need_prune) {
      PruneCandidates new_pool;

      size_t reserve_size = static_cast<size_t>(std::ceil(
          1.05 * DiskAnnEntity::kDefaultGraphSlackFactor * max_degree_));

      new_pool.reserve(std::min<size_t>(reserve_size, max_candidate_size_));

      PruneVectorCache cache;
      int ret =
          collect_prune_candidates(des, new_neighbors, new_pool, ctx, &cache);
      if (ret != 0) return ret;

      ailego::BudgetedVector<diskann_id_t> new_pruned_neighbors;
      ret = prune_neighbors(des, new_pool, new_pruned_neighbors, ctx, &cache);
      if (ret != 0) {
        return ret;
      }

      cache.clear();
      {
        auto lock = lock_for(des);
        ret = entity_.set_neighbors(des, new_pruned_neighbors);
        if (ret != 0) {
          return ret;
        }
      }
    }
  }

  return 0;
}

int DiskAnnAlgorithm::iterate_to_fixed_point(
    const ailego::BudgetedVector<diskann_id_t> &init_ids, diskann_id_t excluded,
    DiskAnnContext *ctx) {
  DistCalculator &dc = ctx->dist_calculator();
  auto &expanded_nodes = ctx->expanded_nodes();
  expanded_nodes.clear();
  NeighborPriorityQueue &best_list_nodes = ctx->best_list_nodes();
  VisitFilter &visit = ctx->visit_filter();

  best_list_nodes.reserve(ctx->list_size());

  for (auto id : init_ids) {
    const float distance = dc.dist(id);
    if (dc.error()) {
      return dc.error_code();
    }

    Neighbor nn = Neighbor(id, distance);
    best_list_nodes.insert(nn);
  }

  while (best_list_nodes.has_unexpanded_node()) {
    auto neighbor = best_list_nodes.closest_unexpanded();
    auto node_id = neighbor.id;

    CollectPruneCandidate(expanded_nodes, neighbor, excluded,
                          max_candidate_size_);

    ailego::BudgetedVector<diskann_id_t> id_scratch;
    {
      auto lock = lock_for(node_id);
      ailego::BudgetedVector<diskann_id_t> neighbors;
      const int ret = entity_.read_neighbors(node_id, &neighbors);
      if (ret != 0) {
        return ret;
      }

      for (diskann_id_t neighbor_id : neighbors) {
        if (!visit.visited(neighbor_id)) {
          id_scratch.push_back(neighbor_id);

          visit.set_visited(neighbor_id);
        }
      }
    }

    if (!entity_.prefer_vector_batch()) {
      // Keep the in-memory builder's scalar fast path unchanged.
      for (diskann_id_t id : id_scratch) {
        const float distance = dc.dist(id);
        if (dc.error()) return dc.error_code();
        best_list_nodes.insert(Neighbor(id, distance));
      }
      continue;
    }
    const int ret = dc.visit_vector_batches(
        id_scratch.data(), id_scratch.size(),
        [&](const diskann_id_t *ids, IndexStorage::MemoryBlock *blocks,
            size_t count) {
          for (size_t i = 0; i < count; ++i) {
            const float distance = dc.dist(blocks[i].data());
            if (dc.error()) return dc.error_code();
            best_list_nodes.insert(Neighbor(ids[i], distance));
          }
          return 0;
        });
    if (ret) return ret;
  }

  return 0;
}

int DiskAnnAlgorithm::collect_prune_candidates(
    diskann_id_t center, const ailego::BudgetedVector<diskann_id_t> &neighbors,
    PruneCandidates &pool, DiskAnnContext *ctx, PruneVectorCache *cache) {
  auto &dc = ctx->dist_calculator();
  if (std::none_of(neighbors.begin(), neighbors.end(),
                   [center](diskann_id_t id) { return id != center; }))
    return 0;

  // Allocate the bounded candidate workspace before optional vector retention.
  // Incremental growth temporarily charges both old and new allocations and
  // can reject a working set that fits, depending on the STL growth policy.
  // Duplicates may make the actual candidate set smaller than this bound, so
  // retain incremental growth when the upper bound cannot be reserved.
  try {
    pool.reserve(std::min(neighbors.size(), size_t(max_candidate_size_)));
  } catch (const std::bad_alloc &) {
  }
  std::set<diskann_id_t> visited;
  const bool batch_reads = entity_.prefer_vector_batch();
  if (!batch_reads) {
    for (const auto id : neighbors) {
      if (id == center || !visited.insert(id).second) continue;
      const float distance = dc.dist(center, id);
      if (dc.error()) return dc.error_code();
      CollectPruneCandidate(pool, Neighbor(id, distance), center,
                            max_candidate_size_);
    }
    return 0;
  }

  IndexStorage::MemoryBlock center_block;
  const int center_ret = entity_.read_vector(center, center_block);
  if (center_ret != 0) return center_ret;
  if (!center_block.data()) return IndexError_ReadData;

  // Charge descriptors before retaining candidates. Optional retention yields
  // to the original small working set when it cannot fit.
  if (neighbors.size() <= max_candidate_size_) {
    try {
      cache->ids.resize(neighbors.size());
      cache->blocks.resize(neighbors.size());
    } catch (const std::bad_alloc &) {
      cache->clear();
    }
  }

  std::array<diskann_id_t, DiskAnnEntity::kVectorReadBatchSize> ids;
  std::array<dist_t, DiskAnnEntity::kVectorReadBatchSize> distances;
  size_t count = 0;
  size_t completed = 0;
  const auto collect = [&]() {
    auto *retained =
        cache->blocks.empty() ? nullptr : cache->blocks.data() + completed;
    dc.dist_batch(center, ids.data(), count, distances.data(), retained,
                  &center_block);
    if (retained && dc.error_code() == IndexError_NoMemory) {
      cache->clear();
      dc.dist_batch(center, ids.data(), count, distances.data(), nullptr,
                    &center_block);
    }
    if (dc.error()) return dc.error_code();
    // Preserve first-visit order, distance orientation and candidate pruning.
    for (size_t i = 0; i < count; ++i) {
      try {
        CollectPruneCandidate(pool, Neighbor(ids[i], distances[i]), center,
                              max_candidate_size_);
      } catch (const std::bad_alloc &) {
        if (cache->blocks.empty()) throw;
        cache->clear();
        CollectPruneCandidate(pool, Neighbor(ids[i], distances[i]), center,
                              max_candidate_size_);
      }
    }
    if (!cache->ids.empty())
      std::copy_n(ids.data(), count, cache->ids.data() + completed);
    completed += count;
    count = 0;
    return 0;
  };
  for (const auto id : neighbors) {
    if (id == center || !visited.insert(id).second) continue;
    ids[count++] = id;
    if (count == ids.size()) {
      const int ret = collect();
      if (ret != 0) return ret;
    }
  }
  const int ret = count ? collect() : 0;
  if (ret == 0 && !cache->ids.empty()) {
    cache->ids.resize(completed);
    cache->blocks.resize(completed);
    cache->complete = true;
  }
  return ret;
}

int DiskAnnAlgorithm::occlude_list(diskann_id_t id, PruneCandidates &pool,
                                   ailego::BudgetedVector<diskann_id_t> &result,
                                   DiskAnnContext *ctx,
                                   PruneVectorCache *cache) {
  if (pool.size() == 0) return 0;

  DistCalculator &dc = ctx->dist_calculator();

  ailego_assert(std::is_sorted(pool.begin(), pool.end()));
  ailego_assert(result.size() == 0);

  if (pool.size() > max_candidate_size_) {
    pool.resize(max_candidate_size_);
  }

  auto &occlude_factor = ctx->occlude_factor();

  occlude_factor.clear();
  occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

  std::array<size_t, DiskAnnEntity::kDefaultMaxOcclusionSize> cached_indices;
  bool retained = cache && cache->complete;
  if (retained) {
    for (size_t i = 0; i < pool.size(); ++i) {
      const auto found =
          std::find(cache->ids.begin(), cache->ids.end(), pool[i].id);
      if (found == cache->ids.end()) {
        retained = false;
        cache->clear();
        break;
      }
      cached_indices[i] = found - cache->ids.begin();
    }
  }

  // Robust pruning compares the same candidate vectors repeatedly. Retain
  // just this bounded candidate set while pruning, instead of re-reading
  // pageable vectors for every pair. Both descriptors and borrowed/owned
  // blocks remain charged to the pool. Under pressure the scalar path needs
  // only two vectors, so this optional working set must not reject a build.
  ailego::BudgetedVector<IndexStorage::MemoryBlock> vectors;
  if (!retained && entity_.prefer_vector_batch()) {
    try {
      vectors.resize(pool.size());
      for (size_t begin = 0; begin < pool.size();
           begin += DiskAnnEntity::kVectorReadBatchSize) {
        const size_t count =
            std::min(DiskAnnEntity::kVectorReadBatchSize, pool.size() - begin);
        std::array<diskann_id_t, DiskAnnEntity::kVectorReadBatchSize> ids;
        for (size_t i = 0; i < count; ++i) ids[i] = pool[begin + i].id;
        const int ret =
            entity_.read_vectors(ids.data(), vectors.data() + begin, count);
        if (ret == IndexError_NoMemory) {
          decltype(vectors)().swap(vectors);
          break;
        }
        if (ret != 0) return ret;
      }
    } catch (const std::bad_alloc &) {
      decltype(vectors)().swap(vectors);
    }
  }

  float cur_alpha = 1;
  while (cur_alpha <= alpha_ && result.size() < max_degree_) {
    for (auto iter = pool.begin();
         result.size() < max_degree_ && iter != pool.end(); ++iter) {
      if (occlude_factor[iter - pool.begin()] > cur_alpha) {
        continue;
      }

      occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();

      if (iter->id != id) {
        result.push_back(iter->id);
      }

      for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
        auto t = iter2 - pool.begin();
        if (occlude_factor[t] > alpha_) {
          continue;
        }

        float djk =
            retained
                ? dc.dist(cache->blocks[cached_indices[t]],
                          cache->blocks[cached_indices[iter - pool.begin()]])
            : vectors.empty()
                ? dc.dist(iter2->id, iter->id)
                : dc.dist(vectors[t], vectors[iter - pool.begin()]);
        if (dc.error()) {
          return dc.error_code();
        }

        occlude_factor[t] =
            (djk == 0) ? std::numeric_limits<float>::max()
                       : std::max(occlude_factor[t], iter2->distance / djk);
      }
    }
    cur_alpha *= 1.2f;
  }

  return 0;
}

int DiskAnnAlgorithm::prune_neighbors(
    diskann_id_t id, PruneCandidates &pool,
    ailego::BudgetedVector<diskann_id_t> &pruned_list, DiskAnnContext *ctx,
    PruneVectorCache *cache) {
  if (pool.size() == 0) {
    pruned_list.clear();
    return 0;
  }

  std::sort(pool.begin(), pool.end());

  int ret;
  for (;;) {
    try {
      pruned_list.clear();
      pruned_list.reserve(max_degree_);
      ret = occlude_list(id, pool, pruned_list, ctx, cache);
      break;
    } catch (const std::bad_alloc &) {
      // Mandatory metadata may grow after candidate collection. Give back
      // optional pins before retrying; preserve the already sorted order.
      if (!cache || !cache->complete) throw;
      cache->clear();
    }
  }
  if (ret != 0) {
    return ret;
  }

  ailego_assert(pruned_list.size() <= max_degree_);

  if (saturate_graph_ && alpha_ > 1) {
    for (const auto &node : pool) {
      if (pruned_list.size() >= max_degree_) {
        break;
      }

      if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) ==
           pruned_list.end()) &&
          node.id != id) {
        pruned_list.push_back(node.id);
      }
    }
  }

  return 0;
}

int DiskAnnAlgorithm::search_neighbor_and_prune(
    diskann_id_t id, ailego::BudgetedVector<diskann_id_t> &pruned_list,
    DiskAnnContext *ctx) {
  const ailego::BudgetedVector<diskann_id_t> init_ids = get_init_ids(ctx);

  int ret = iterate_to_fixed_point(init_ids, id, ctx);
  if (ret != 0) {
    return ret;
  }

  auto &pool = ctx->expanded_nodes();
  pool.erase(std::remove_if(pool.begin(), pool.end(),
                            [id](const auto &node) { return node.id == id; }),
             pool.end());

  ret = prune_neighbors(id, pool, pruned_list, ctx);
  if (ret != 0) {
    return ret;
  }

  return 0;
}

}  // namespace core
}  // namespace zvec

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
#include "diskann_graph_connectivity.h"
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
    std::set<diskann_id_t> dummy_visited;
    PruneCandidates dummy_pool;
    ailego::BudgetedVector<diskann_id_t> new_out_neighbors;

    for (diskann_id_t node_id : neighbors) {
      auto itr = dummy_visited.find(node_id);
      if (itr == dummy_visited.end() && node_id != id) {
        float dist = dc.dist(id, node_id);
        if (dc.error()) {
          return dc.error_code();
        }

        CollectPruneCandidate(dummy_pool, Neighbor(node_id, dist), id,
                              max_candidate_size_);
        dummy_visited.insert(node_id);
      }
    }

    ret = prune_neighbors(id, dummy_pool, new_out_neighbors, ctx);
    if (ret != 0) {
      return ret;
    }

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

int DiskAnnAlgorithm::repair_connectivity(DiskAnnContext *ctx,
                                          size_t *added_edges) try {
  if (added_edges) *added_edges = 0;
  const uint64_t count = entity_.doc_cnt();
  if (count == 0) return 0;
  if (!ctx || count > kInvalidId || entity_.medoid() >= count ||
      max_degree_ == 0 || ctx->list_size() == 0)
    return IndexError_InvalidArgument;
  AILEGO_DEFER([&] { ctx->clear(); });
  DiskAnnGraphConnectivity reachable(entity_, max_degree_);
  const auto medoid = static_cast<diskann_id_t>(entity_.medoid());
  int ret = reachable.visit(medoid, medoid);
  if (ret != 0) return ret;
  if (reachable.reached() == count) return 0;

  size_t fallback = 0;
  size_t repaired = 0;
  ailego::BudgetedVector<diskann_id_t> neighbors;
  for (uint64_t next = 0; next < count; ++next) {
    const auto target = static_cast<diskann_id_t>(next);
    if (reachable.contains(target)) continue;
    ctx->clear();
    {
      IndexStorage::MemoryBlock vector;
      ret = entity_.read_vector(target, vector);
      if (ret != 0) return ret;
      if (!vector.data()) return IndexError_ReadData;
      ctx->reset_query(vector.data());
    }
    ret = iterate_to_fixed_point(get_init_ids(ctx), target, ctx);
    if (ret != 0) return ret;
    auto &candidates = ctx->expanded_nodes();
    std::sort(candidates.begin(), candidates.end());
    diskann_id_t anchor = kInvalidId;
    size_t slot = std::numeric_limits<size_t>::max();
    for (const auto &candidate : candidates) {
      ret = entity_.read_neighbors(candidate.id, &neighbors);
      if (ret != 0) return ret;
      slot = reachable.link_slot(candidate.id, neighbors);
      if (slot != std::numeric_limits<size_t>::max()) {
        anchor = candidate.id;
        break;
      }
    }
    // A finite beam may find only full tree rows. A tree on R reached nodes
    // has R-1 protected edges, so some reached row has room when degree >= 1.
    // Protected counts only grow; permanently skip exhausted fallback rows.
    while (anchor == kInvalidId && fallback < reachable.reached()) {
      const diskann_id_t candidate = reachable.at(fallback);
      ret = entity_.read_neighbors(candidate, &neighbors);
      if (ret != 0) return ret;
      slot = reachable.link_slot(candidate, neighbors);
      if (slot != std::numeric_limits<size_t>::max()) {
        anchor = candidate;
      } else {
        ++fallback;
      }
    }
    if (anchor == kInvalidId) return IndexError_Runtime;
    if (slot == neighbors.size()) {
      neighbors.push_back(target);
    } else {
      neighbors[slot] = target;
    }
    ret = entity_.set_neighbors(anchor, neighbors);
    if (ret != 0) return ret;
    ret = reachable.visit(target, anchor);
    if (ret != 0) return ret;
    ++repaired;
  }

  // Reuse the same admitted workspace for an independent final traversal.
  reachable.reset();
  ret = reachable.visit(medoid, medoid);
  if (ret != 0) return ret;
  if (reachable.reached() != count) return IndexError_Runtime;
  if (added_edges) *added_edges = repaired;
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int DiskAnnAlgorithm::inter_insert(
    diskann_id_t id, ailego::BudgetedVector<diskann_id_t> &pruned_list,
    DiskAnnContext *ctx) {
  DistCalculator &dc = ctx->dist_calculator();

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
      std::set<diskann_id_t> new_visited;
      PruneCandidates new_pool;

      size_t reserve_size = static_cast<size_t>(std::ceil(
          1.05 * DiskAnnEntity::kDefaultGraphSlackFactor * max_degree_));

      new_pool.reserve(std::min<size_t>(reserve_size, max_candidate_size_));

      for (auto node_id : new_neighbors) {
        if (new_visited.find(node_id) == new_visited.end() && node_id != des) {
          float dist = dc.dist(des, node_id);
          if (dc.error()) {
            return dc.error_code();
          }
          CollectPruneCandidate(new_pool, Neighbor(node_id, dist), des,
                                max_candidate_size_);
          new_visited.insert(node_id);
        }
      }

      ailego::BudgetedVector<diskann_id_t> new_pruned_neighbors;
      int ret = prune_neighbors(des, new_pool, new_pruned_neighbors, ctx);
      if (ret != 0) {
        return ret;
      }

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
    float distance = dc.dist(id);
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
    for (size_t begin = 0; begin < id_scratch.size();
         begin += DiskAnnEntity::kVectorReadBatchSize) {
      const size_t count = std::min(DiskAnnEntity::kVectorReadBatchSize,
                                    id_scratch.size() - begin);
      std::array<dist_t, DiskAnnEntity::kVectorReadBatchSize> distances;
      dc.dist_batch(id_scratch.data() + begin, count, distances.data());
      if (dc.error()) {
        return dc.error_code();
      }
      // Preserve the scalar path's insertion order and distance function.
      for (size_t i = 0; i < count; ++i) {
        best_list_nodes.insert(Neighbor(id_scratch[begin + i], distances[i]));
      }
    }
  }

  return 0;
}

int DiskAnnAlgorithm::occlude_list(diskann_id_t id, PruneCandidates &pool,
                                   ailego::BudgetedVector<diskann_id_t> &result,
                                   DiskAnnContext *ctx) {
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

        float djk = dc.dist(iter2->id, iter->id);
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
    ailego::BudgetedVector<diskann_id_t> &pruned_list, DiskAnnContext *ctx) {
  if (pool.size() == 0) {
    pruned_list.clear();
    return 0;
  }

  std::sort(pool.begin(), pool.end());

  pruned_list.clear();
  pruned_list.reserve(max_degree_);

  const int ret = occlude_list(id, pool, pruned_list, ctx);
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

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
#include <random>
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

int DiskAnnAlgorithm::initialize_graph() try {
  const uint64_t count = entity_.doc_cnt();
  if (count == 0) return 0;
  if (count > kInvalidId || max_degree_ == 0) return IndexError_InvalidArgument;
  const size_t degree = std::min<uint64_t>(max_degree_, count - 1);
  ailego::BudgetedVector<diskann_id_t> neighbors;
  neighbors.reserve(degree);
  std::mt19937_64 random(0);
  std::uniform_int_distribution<diskann_id_t> pick(
      0, static_cast<diskann_id_t>(count - 1));
  // Only one row is retained in workspace; buffered entities keep the graph
  // itself pageable. Seed edges span the full corpus before insertion begins.
  for (uint64_t id = 0; id < count; ++id) {
    neighbors.clear();
    while (neighbors.size() < degree) {
      const auto neighbor = pick(random);
      if (neighbor != id && std::find(neighbors.begin(), neighbors.end(),
                                      neighbor) == neighbors.end())
        neighbors.push_back(neighbor);
    }
    const int ret = entity_.set_neighbors(id, neighbors);
    if (ret != 0) return ret;
  }
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
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

  const auto guidance = entity_.graph_quantization();
  bool quantized = guidance.quantizer != nullptr;
  if (quantized) {
    if ((!guidance.codes && !guidance.paged_codes) || guidance.code_size == 0)
      return IndexError_NoReady;
    dc.bind_quantizer(guidance.quantizer, guidance.codes, guidance.code_size,
                      guidance.paged_codes);
    const int ret = dc.quantize_query(ctx->query());
    if (ret == IndexError_NoMemory) {
      dc.clear();
      quantized = false;
    } else if (ret != 0) {
      return ret;
    }
  }

  for (auto id : init_ids) {
    float distance;
    if (quantized) {
      dc.quantized_dist(&id, 1, &distance);
    } else {
      distance = dc.dist(id);
    }
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

    if (!quantized && !entity_.prefer_vector_batch()) {
      // Keep the in-memory builder's scalar fast path unchanged.
      for (diskann_id_t id : id_scratch) {
        const float distance = dc.dist(id);
        if (dc.error()) return dc.error_code();
        best_list_nodes.insert(Neighbor(id, distance));
      }
      continue;
    }
    if (quantized) {
      for (size_t begin = 0; begin < id_scratch.size();
           begin += DiskAnnEntity::kVectorReadBatchSize) {
        const size_t count = std::min(DiskAnnEntity::kVectorReadBatchSize,
                                      id_scratch.size() - begin);
        std::array<dist_t, DiskAnnEntity::kVectorReadBatchSize> distances;
        dc.quantized_dist(id_scratch.data() + begin,
                          static_cast<uint32_t>(count), distances.data());
        if (dc.error()) return dc.error_code();
        for (size_t i = 0; i < count; ++i)
          best_list_nodes.insert(Neighbor(id_scratch[begin + i], distances[i]));
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

  // Approximate scores guide traversal only. Restore exact query distances
  // and heap order before existing-neighbor union, robust pruning or repair.
  if (quantized) {
    for (size_t begin = 0; begin < expanded_nodes.size();
         begin += DiskAnnEntity::kVectorReadBatchSize) {
      const size_t count = std::min(DiskAnnEntity::kVectorReadBatchSize,
                                    expanded_nodes.size() - begin);
      std::array<diskann_id_t, DiskAnnEntity::kVectorReadBatchSize> ids;
      std::array<dist_t, DiskAnnEntity::kVectorReadBatchSize> distances;
      for (size_t i = 0; i < count; ++i) ids[i] = expanded_nodes[begin + i].id;
      dc.dist_batch(ids.data(), count, distances.data());
      if (dc.error()) return dc.error_code();
      for (size_t i = 0; i < count; ++i)
        expanded_nodes[begin + i].distance = distances[i];
    }
    std::make_heap(expanded_nodes.begin(), expanded_nodes.end());
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
  ailego::BudgetedVector<diskann_id_t> existing;
  {
    auto lock = lock_for(id);
    ret = entity_.read_neighbors(id, &existing);
  }
  if (ret != 0) return ret;
  auto &dc = ctx->dist_calculator();
  const auto missing = [&](diskann_id_t neighbor) {
    return neighbor != id && std::none_of(pool.begin(), pool.end(),
                                          [neighbor](const auto &candidate) {
                                            return candidate.id == neighbor;
                                          });
  };
  if (!entity_.prefer_vector_batch()) {
    for (const auto neighbor : existing) {
      if (!missing(neighbor)) continue;
      const float distance = dc.dist(neighbor);
      if (dc.error()) return dc.error_code();
      CollectPruneCandidate(pool, Neighbor(neighbor, distance), id,
                            max_candidate_size_);
    }
  } else {
    for (size_t begin = 0; begin < existing.size();
         begin += DiskAnnEntity::kVectorReadBatchSize) {
      const size_t size = std::min(DiskAnnEntity::kVectorReadBatchSize,
                                   existing.size() - begin);
      std::array<diskann_id_t, DiskAnnEntity::kVectorReadBatchSize> ids;
      std::array<size_t, DiskAnnEntity::kVectorReadBatchSize> slots;
      std::array<IndexStorage::MemoryBlock, DiskAnnEntity::kVectorReadBatchSize>
          blocks;
      slots.fill(slots.size());
      size_t count = 0;
      if (entity_.prefer_vector_batch()) {
        for (size_t i = 0; i < size; ++i)
          if (missing(existing[begin + i])) {
            slots[i] = count;
            ids[count++] = existing[begin + i];
          }
        int status = 0;
        try {
          status = entity_.read_vectors(ids.data(), blocks.data(), count);
        } catch (const std::bad_alloc &) {
          status = IndexError_NoMemory;
        }
        if (status == IndexError_NoMemory) {
          for (auto &block : blocks) block.reset();
        } else if (status)
          return status;
      }
      for (size_t i = 0; i < size; ++i) {
        const auto neighbor = existing[begin + i];
        // Recheck against the evolving bounded pool. An earlier insertion may
        // have removed a candidate, so the scalar read remains a rare fallback.
        if (!missing(neighbor)) continue;
        const float distance = slots[i] < count && blocks[slots[i]].data()
                                   ? dc.dist(blocks[slots[i]].data())
                                   : dc.dist(neighbor);
        if (dc.error()) return dc.error_code();
        try {
          CollectPruneCandidate(pool, Neighbor(neighbor, distance), id,
                                max_candidate_size_);
        } catch (const std::bad_alloc &) {
          for (auto &block : blocks) block.reset();
          CollectPruneCandidate(pool, Neighbor(neighbor, distance), id,
                                max_candidate_size_);
        }
      }
    }
  }
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

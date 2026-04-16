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

#include <variant>
#include <zvec/core/framework/index_streamer.h>
#include "hnsw_streamer_bench_entity.h"
#include "hnsw_streamer_entity.h"

namespace zvec {
namespace core {

class HnswStreamerEntitySet {
 public:
  enum Options {
    kUnknown = 0,
    kMMap = 1,
    kMMapBench = 2,
    kBufferPool = 3,
  };
  typedef std::shared_ptr<HnswStreamerEntitySet> Pointer;
  using EntityVariant = std::variant<std::unique_ptr<HnswStreamerEntity>,
                                     std::unique_ptr<HnswStreamerBenchEntity>>;

 public:
  HnswStreamerEntitySet(Options opt, IndexStreamer::Stats &stats)
      : options_(opt) {
    switch (opt) {
      case kMMap:
        entity_ = std::make_unique<HnswStreamerEntity>(stats);
        break;
      case kMMapBench:
        entity_ = std::make_unique<HnswStreamerBenchEntity>(stats);
        break;
      default:
        break;
    }
  }

  HnswStreamerEntitySet(HnswStreamerEntity::UPointer normal_entity)
      : options_(kMMap) {
    entity_ = std::move(normal_entity);
  }

  HnswStreamerEntitySet(HnswStreamerBenchEntity::UPointer bench_entity)
      : options_(kMMapBench) {
    entity_ = std::move(bench_entity);
  }

 public:
  int cleanup() {
    return std::visit([&](const auto &e) { return e->cleanup(); }, entity_);
  }

  key_t get_key(node_id_t id) const {
    return std::visit([&](const auto &e) { return e->get_key(id); }, entity_);
  }

  const HnswStreamerEntitySet::Pointer clone() const {
    return std::visit([&](const auto &e) {
      return Pointer(new HnswStreamerEntitySet(e->clone_uptr()));
    }, entity_);
  }

  const void *get_vector_by_key(key_t key) const {
    return std::visit(
        [&](const auto &e) -> const void * { return e->get_vector_by_key(key); },
        entity_);
  }

  const void *get_vector(node_id_t id) const {
    return std::visit(
        [&](const auto &e) -> const void * {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<T, std::unique_ptr<HnswStreamerEntity>>) {
            return e->get_vector(id);
          } else {
            return e->get_vector_new(id);
          }
        },
        entity_);
  }

  int get_vector(const node_id_t *ids, uint32_t count,
                 const void **vecs) const {
    return std::visit(
        [&](const auto &e) { return e->get_vector(ids, count, vecs); }, entity_);
  }

  int get_vector(const node_id_t id, IndexStorage::MemoryBlock &block) const {
    return std::visit(
        [&](const auto &e) -> int {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<T, std::unique_ptr<HnswStreamerEntity>>) {
            return e->get_vector(id, block);
          } else {
            return e->get_vector_new(id, block);
          }
        },
        entity_);
  }

  int get_vector(const node_id_t *ids, uint32_t count,
                 std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
    return std::visit(
        [&](const auto &e) -> int {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<T, std::unique_ptr<HnswStreamerEntity>>) {
            return e->get_vector(ids, count, vec_blocks);
          } else {
            return e->get_vector_new(ids, count, vec_blocks);
          }
        },
        entity_);
  }

  const Neighbors get_neighbors(level_t level, node_id_t id) const {
    return std::visit(
        [&](const auto &e) -> Neighbors {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<T, std::unique_ptr<HnswStreamerEntity>>) {
            return e->get_neighbors(level, id);
          } else {
            return e->get_neighbors_new(level, id);
          }
        },
        entity_);
  }

  int add_vector(level_t level, key_t key, const void *vec, node_id_t *id) {
    return std::visit(
        [&](auto &e) { return e->add_vector(level, key, vec, id); }, entity_);
  }

  int add_vector_with_id(level_t level, node_id_t id, const void *vec) {
    return std::visit(
        [&](auto &e) { return e->add_vector_with_id(level, id, vec); }, entity_);
  }

  int update_neighbors(
      level_t level, node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
    return std::visit(
        [&](auto &e) { return e->update_neighbors(level, id, neighbors); },
        entity_);
  }

  void add_neighbor(level_t level, node_id_t id, uint32_t size,
                    node_id_t neighbor_id) {
    std::visit(
        [&](auto &e) { e->add_neighbor(level, id, size, neighbor_id); }, entity_);
  }

  int dump(const IndexDumper::Pointer &dumper) {
    return std::visit([&](auto &e) { return e->dump(dumper); }, entity_);
  }

  void update_ep_and_level(node_id_t ep, level_t level) {
    std::visit([&](auto &e) { e->update_ep_and_level(ep, level); }, entity_);
  }

  int get_vector_by_key(const key_t key,
                        IndexStorage::MemoryBlock &block) const {
    return std::visit(
        [&](const auto &e) { return e->get_vector_by_key(key, block); },
        entity_);
  }

  inline size_t neighbor_cnt(level_t level) const {
    return std::visit(
        [&](const auto &e) { return e->neighbor_cnt(level); }, entity_);
  }

  inline size_t l0_neighbor_cnt() const {
    return std::visit(
        [&](const auto &e) { return e->l0_neighbor_cnt(); }, entity_);
  }

  inline size_t min_neighbor_cnt() const {
    return std::visit(
        [&](const auto &e) { return e->min_neighbor_cnt(); }, entity_);
  }

  inline size_t upper_neighbor_cnt() const {
    return std::visit(
        [&](const auto &e) { return e->upper_neighbor_cnt(); }, entity_);
  }

  inline node_id_t *mutable_doc_cnt() {
    return std::visit([&](auto &e) { return e->mutable_doc_cnt(); }, entity_);
  }

  inline node_id_t doc_cnt() const {
    return std::visit([&](const auto &e) { return e->doc_cnt(); }, entity_);
  }

  inline size_t scaling_factor() const {
    return std::visit(
        [&](const auto &e) { return e->scaling_factor(); }, entity_);
  }

  inline size_t prune_cnt() const {
    return std::visit([&](const auto &e) { return e->prune_cnt(); }, entity_);
  }

  inline node_id_t entry_point() const {
    return std::visit(
        [&](const auto &e) { return e->entry_point(); }, entity_);
  }

  inline level_t cur_max_level() const {
    return std::visit(
        [&](const auto &e) { return e->cur_max_level(); }, entity_);
  }

  size_t vector_size() const {
    return std::visit([&](const auto &e) { return e->vector_size(); }, entity_);
  }

  //! Retrieve node size
  size_t node_size() const {
    return std::visit([&](const auto &e) { return e->node_size(); }, entity_);
  }

  //! Retrieve ef constuction
  size_t ef_construction() const {
    return std::visit(
        [&](const auto &e) { return e->ef_construction(); }, entity_);
  }

  void set_vector_size(size_t size) {
    std::visit([&](auto &e) { e->set_vector_size(size); }, entity_);
  }

  void set_prune_cnt(size_t v) {
    std::visit([&](auto &e) { e->set_prune_cnt(v); }, entity_);
  }

  void set_scaling_factor(size_t val) {
    std::visit([&](auto &e) { e->set_scaling_factor(val); }, entity_);
  }

  void set_l0_neighbor_cnt(size_t cnt) {
    std::visit([&](auto &e) { e->set_l0_neighbor_cnt(cnt); }, entity_);
  }

  void set_min_neighbor_cnt(size_t cnt) {
    std::visit([&](auto &e) { e->set_min_neighbor_cnt(cnt); }, entity_);
  }

  void set_upper_neighbor_cnt(size_t cnt) {
    std::visit([&](auto &e) { e->set_upper_neighbor_cnt(cnt); }, entity_);
  }

  void set_ef_construction(size_t ef) {
    std::visit([&](auto &e) { e->set_ef_construction(ef); }, entity_);
  }

  int init(size_t max_doc_cnt) {
    return std::visit(
        [&](auto &e) { return e->init(max_doc_cnt); }, entity_);
  }

  int flush(uint64_t checkpoint) {
    return std::visit(
        [&](auto &e) { return e->flush(checkpoint); }, entity_);
  }

  int open(IndexStorage::Pointer stg, uint64_t max_index_size, bool check_crc) {
    return std::visit(
        [&](auto &e) { return e->open(stg, max_index_size, check_crc); },
        entity_);
  }

  int close() {
    return std::visit([&](auto &e) { return e->close(); }, entity_);
  }

  void set_use_key_info_map(bool use_id_map) {
    std::visit([&](auto &e) { e->set_use_key_info_map(use_id_map); }, entity_);
  }

  //! Set meta information from entity
  int set_index_meta(const IndexMeta &meta) const {
    return std::visit(
        [&](const auto &e) { return e->set_index_meta(meta); }, entity_);
  }

  //! Get meta information from entity
  int get_index_meta(IndexMeta *meta) const {
    return std::visit(
        [&](const auto &e) { return e->get_index_meta(meta); }, entity_);
  }

  //! Set params: chunk size
  inline void set_chunk_size(size_t val) {
    std::visit([&](auto &e) { e->set_chunk_size(val); }, entity_);
  }

  //! Set params
  inline void set_filter_same_key(bool val) {
    std::visit([&](auto &e) { e->set_filter_same_key(val); }, entity_);
  }

  //! Set params
  inline void set_get_vector(bool val) {
    std::visit([&](auto &e) { e->set_get_vector(val); }, entity_);
  }

  //! Get vector local id by key
  inline node_id_t get_id(key_t key) const {
    return std::visit([&](const auto &e) { return e->get_id(key); }, entity_);
  }

  void print_key_map() const {
    std::visit([&](const auto &e) { e->print_key_map(); }, entity_);
  }

  //! Get l0 neighbors size
  inline size_t neighbors_size() const {
    return std::visit(
        [&](const auto &e) { return e->neighbors_size(); }, entity_);
  }

  //! Get neighbors size for level > 0
  inline size_t upper_neighbors_size() const {
    return std::visit(
        [&](const auto &e) { return e->upper_neighbors_size(); }, entity_);
  }

 private:
  Options options_{kUnknown};
  EntityVariant entity_;
};

}  // namespace core
}  // namespace zvec
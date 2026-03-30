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

 public:
  HnswStreamerEntitySet(Options opt, IndexStreamer::Stats &stats)
      : options_(opt) {
    switch (opt) {
      case kMMap:
        normal_entity_ = std::make_unique<HnswStreamerEntity>(stats);
        break;
      case kMMapBench:
        bench_entity_ = std::make_unique<HnswStreamerBenchEntity>(stats);
        break;
    }
  }

  HnswStreamerEntitySet(HnswStreamerEntity::UPointer normal_entity) : options_(kMMap) {
    normal_entity_ = std::move(normal_entity);
  }

  HnswStreamerEntitySet(HnswStreamerBenchEntity::UPointer bench_entity) : options_(kMMapBench) {
    bench_entity_ = std::move(bench_entity);
  }

 public:
  int cleanup() {
    switch (options_) {
      case kMMap:
        return normal_entity_->cleanup();
      case kMMapBench:
        return bench_entity_->cleanup();
    }
  }

  key_t get_key(node_id_t id) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_key(id);
      case kMMapBench:
        return bench_entity_->get_key(id);
    }
  }

  const HnswStreamerEntitySet::Pointer clone() const {
    switch (options_) {
      case kMMap:
        return Pointer(new HnswStreamerEntitySet(normal_entity_->clone_uptr()));
      case kMMapBench:
        return Pointer(new HnswStreamerEntitySet(bench_entity_->clone_uptr()));
    }
  }

  const void *get_vector_by_key(key_t key) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_vector_by_key(key);
      case kMMapBench:
        return bench_entity_->get_vector_by_key(key);
    }
  }

  const void *get_vector(node_id_t id) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_vector(id);
      case kMMapBench:
        return bench_entity_->get_vector_new(id);
    }
  }

  int get_vector(const node_id_t *ids, uint32_t count,
                 const void **vecs) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_vector(ids, count, vecs);
      case kMMapBench:
        return bench_entity_->get_vector(ids, count, vecs);
    }
  }

  int get_vector(const node_id_t id, IndexStorage::MemoryBlock &block) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_vector(id, block);
      case kMMapBench:
        return bench_entity_->get_vector_new(id, block);
    }
  }

  int get_vector(const node_id_t *ids, uint32_t count,
                 std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_vector(ids, count, vec_blocks);
      case kMMapBench:
        return bench_entity_->get_vector_new(ids, count, vec_blocks);
    }
  }

  const Neighbors get_neighbors(level_t level, node_id_t id) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_neighbors(level, id);
      case kMMapBench:
        return bench_entity_->get_neighbors_new(level, id);
    }
  }

  int add_vector(level_t level, key_t key, const void *vec, node_id_t *id) {
    switch (options_) {
      case kMMap:
        return normal_entity_->add_vector(level, key, vec, id);
      case kMMapBench:
        return bench_entity_->add_vector(level, key, vec, id);
    }
  }

  int add_vector_with_id(level_t level, node_id_t id, const void *vec) {
    switch (options_) {
      case kMMap:
        return normal_entity_->add_vector_with_id(level, id, vec);
      case kMMapBench:
        return bench_entity_->add_vector_with_id(level, id, vec);
    }
  }

  int update_neighbors(
      level_t level, node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
    switch (options_) {
      case kMMap:
        return normal_entity_->update_neighbors(level, id, neighbors);
      case kMMapBench:
        return bench_entity_->update_neighbors(level, id, neighbors);
    }
  }

  void add_neighbor(level_t level, node_id_t id, uint32_t size,
                    node_id_t neighbor_id) {
    switch (options_) {
      case kMMap:
        normal_entity_->add_neighbor(level, id, size, neighbor_id);
        break;
      case kMMapBench:
        bench_entity_->add_neighbor(level, id, size, neighbor_id);
        break;
    }
  }

  int dump(const IndexDumper::Pointer &dumper) {
    switch (options_) {
      case kMMap:
        return normal_entity_->dump(dumper);
      case kMMapBench:
        return bench_entity_->dump(dumper);
    }
  }

  void update_ep_and_level(node_id_t ep, level_t level) {
    switch (options_) {
      case kMMap:
        normal_entity_->update_ep_and_level(ep, level);
        break;
      case kMMapBench:
        bench_entity_->update_ep_and_level(ep, level);
        break;
    }
  }

  int get_vector_by_key(const key_t key,
                        IndexStorage::MemoryBlock &block) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_vector_by_key(key, block);
      case kMMapBench:
        return bench_entity_->get_vector_by_key(key, block);
    }
  }

  inline size_t neighbor_cnt(level_t level) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->neighbor_cnt(level);
      case kMMapBench:
        return bench_entity_->neighbor_cnt(level);
    }
  }

  inline size_t l0_neighbor_cnt() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->l0_neighbor_cnt();
      case kMMapBench:
        return bench_entity_->l0_neighbor_cnt();
    }
  }

  inline size_t min_neighbor_cnt() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->min_neighbor_cnt();
      case kMMapBench:
        return bench_entity_->min_neighbor_cnt();
    }
  }

  inline size_t upper_neighbor_cnt() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->upper_neighbor_cnt();
      case kMMapBench:
        return bench_entity_->upper_neighbor_cnt();
    }
  }

  inline node_id_t *mutable_doc_cnt() {
    switch (options_) {
      case kMMap:
        return normal_entity_->mutable_doc_cnt();
      case kMMapBench:
        return bench_entity_->mutable_doc_cnt();
    }
  }

  inline node_id_t doc_cnt() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->doc_cnt();
      case kMMapBench:
        return bench_entity_->doc_cnt();
    }
  }

  inline size_t scaling_factor() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->scaling_factor();
      case kMMapBench:
        return bench_entity_->scaling_factor();
    }
  }
  inline size_t prune_cnt() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->prune_cnt();
      case kMMapBench:
        return bench_entity_->prune_cnt();
    }
  }

  inline node_id_t entry_point() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->entry_point();
      case kMMapBench:
        return bench_entity_->entry_point();
    }
  }

  inline level_t cur_max_level() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->cur_max_level();
      case kMMapBench:
        return bench_entity_->cur_max_level();
    }
  }

  size_t vector_size() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->vector_size();
      case kMMapBench:
        return bench_entity_->vector_size();
    }
  }

  //! Retrieve node size
  size_t node_size() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->node_size();
      case kMMapBench:
        return bench_entity_->node_size();
    }
  }

  //! Retrieve ef constuction
  size_t ef_construction() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->ef_construction();
      case kMMapBench:
        return bench_entity_->ef_construction();
    }
  }

  void set_vector_size(size_t size) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_vector_size(size);
        break;
      case kMMapBench:
        bench_entity_->set_vector_size(size);
        break;
    }
  }

  void set_prune_cnt(size_t v) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_prune_cnt(v);
        break;
      case kMMapBench:
        bench_entity_->set_prune_cnt(v);
        break;
    }
  }

  void set_scaling_factor(size_t val) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_scaling_factor(val);
        break;
      case kMMapBench:
        bench_entity_->set_scaling_factor(val);
        break;
    }
  }

  void set_l0_neighbor_cnt(size_t cnt) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_l0_neighbor_cnt(cnt);
        break;
      case kMMapBench:
        bench_entity_->set_l0_neighbor_cnt(cnt);
        break;
    }
  }

  void set_min_neighbor_cnt(size_t cnt) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_min_neighbor_cnt(cnt);
        break;
      case kMMapBench:
        bench_entity_->set_min_neighbor_cnt(cnt);
        break;
    }
  }

  void set_upper_neighbor_cnt(size_t cnt) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_upper_neighbor_cnt(cnt);
        break;
      case kMMapBench:
        bench_entity_->set_upper_neighbor_cnt(cnt);
        break;
    }
  }

  void set_ef_construction(size_t ef) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_ef_construction(ef);
        break;
      case kMMapBench:
        bench_entity_->set_ef_construction(ef);
        break;
    }
  }

  int init(size_t max_doc_cnt) {
    switch (options_) {
      case kMMap:
        return normal_entity_->init(max_doc_cnt);
      case kMMapBench:
        return bench_entity_->init(max_doc_cnt);
    }
  }

  int flush(uint64_t checkpoint) {
    switch (options_) {
      case kMMap:
        return normal_entity_->flush(checkpoint);
      case kMMapBench:
        return bench_entity_->flush(checkpoint);
    }
  }

  int open(IndexStorage::Pointer stg, uint64_t max_index_size, bool check_crc) {
    switch (options_) {
      case kMMap:
        return normal_entity_->open(stg, max_index_size, check_crc);
      case kMMapBench:
        return bench_entity_->open(stg, max_index_size, check_crc);
    }
  }

  int close() {
    switch (options_) {
      case kMMap:
        return normal_entity_->close();
      case kMMapBench:
        return bench_entity_->close();
    }
  }

  void set_use_key_info_map(bool use_id_map) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_use_key_info_map(use_id_map);
        break;
      case kMMapBench:
        bench_entity_->set_use_key_info_map(use_id_map);
        break;
    }
  }

  //! Set meta information from entity
  int set_index_meta(const IndexMeta &meta) const {
    switch (options_) {
      case kMMap:
        normal_entity_->set_index_meta(meta);
        break;
      case kMMapBench:
        bench_entity_->set_index_meta(meta);
        break;
    }
  }

  //! Get meta information from entity
  int get_index_meta(IndexMeta *meta) const {
    switch (options_) {
      case kMMap:
        normal_entity_->get_index_meta(meta);
        break;
      case kMMapBench:
        bench_entity_->get_index_meta(meta);
        break;
    }
  }

  //! Set params: chunk size
  inline void set_chunk_size(size_t val) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_chunk_size(val);
        break;
      case kMMapBench:
        bench_entity_->set_chunk_size(val);
        break;
    }
  }

  //! Set params
  inline void set_filter_same_key(bool val) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_filter_same_key(val);
        break;
      case kMMapBench:
        bench_entity_->set_filter_same_key(val);
        break;
    }
  }

  //! Set params
  inline void set_get_vector(bool val) {
    switch (options_) {
      case kMMap:
        normal_entity_->set_get_vector(val);
        break;
      case kMMapBench:
        bench_entity_->set_get_vector(val);
        break;
    }
  }

  //! Get vector local id by key
  inline node_id_t get_id(key_t key) const {
    switch (options_) {
      case kMMap:
        return normal_entity_->get_id(key);
      case kMMapBench:
        return bench_entity_->get_id(key);
    }
  }

  void print_key_map() const {
    switch (options_) {
      case kMMap:
        normal_entity_->print_key_map();
        break;
      case kMMapBench:
        bench_entity_->print_key_map();
        break;
    }
  }

  //! Get l0 neighbors size
  inline size_t neighbors_size() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->neighbors_size();
      case kMMapBench:
        return bench_entity_->neighbors_size();
    }
  }

  //! Get neighbors size for level > 0
  inline size_t upper_neighbors_size() const {
    switch (options_) {
      case kMMap:
        return normal_entity_->upper_neighbors_size();
      case kMMapBench:
        return bench_entity_->upper_neighbors_size();
    }
  }

 private:
  Options options_{kUnknown};
  std::unique_ptr<HnswStreamerEntity> normal_entity_{nullptr};
  std::unique_ptr<HnswStreamerBenchEntity> bench_entity_{nullptr};
};

}  // namespace core
}  // namespace zvec
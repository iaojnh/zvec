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

 public:
  HnswStreamerEntitySet(Options opt, IndexStreamer::Stats &stats)
      : options_(opt) {
    switch (opt) {
      case kMMap:
        normal_entity_ = new HnswStreamerEntity(stats);
        break;
      case kMMapBench:
        bench_entity_ = new HnswStreamerBenchEntity(stats);
        break;
    }
  }

  ~HnswStreamerEntitySet() {
    delete normal_entity_;
    delete bench_entity_;
  }

 public:
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


 private:
  Options options_{kUnknown};
  HnswStreamerEntity *normal_entity_{nullptr};
  HnswStreamerBenchEntity *bench_entity_{nullptr};
};

}  // namespace core
}  // namespace zvec
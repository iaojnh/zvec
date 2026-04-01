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

#include "hnsw_streamer_bench_entity.h"
#include <stdlib.h>
#include <numeric> 
#include <ailego/utility/memory_helper.h>

// #define DEBUG_PRINT

namespace zvec {
namespace core {

const std::string HnswStreamerBenchEntity::kGraphHeaderSegmentId =
    "graph.header";
const std::string HnswStreamerBenchEntity::kGraphFeaturesSegmentId =
    "graph.features";
const std::string HnswStreamerBenchEntity::kGraphKeysSegmentId = "graph.keys";
const std::string HnswStreamerBenchEntity::kGraphNeighborsSegmentId =
    "graph.neighbors";
const std::string HnswStreamerBenchEntity::kGraphOffsetsSegmentId =
    "graph.offsets";
const std::string HnswStreamerBenchEntity::kGraphMappingSegmentId =
    "graph.mapping";
const std::string HnswStreamerBenchEntity::kHnswHeaderSegmentId = "hnsw.header";
const std::string HnswStreamerBenchEntity::kHnswNeighborsSegmentId =
    "hnsw.neighbors";
const std::string HnswStreamerBenchEntity::kHnswOffsetsSegmentId =
    "hnsw.offsets";

int64_t HnswStreamerBenchEntity::dump_segment(
    const IndexDumper::Pointer &dumper, const std::string &segment_id,
    const void *data, size_t size) const {
  size_t len = dumper->write(data, size);
  if (len != size) {
    LOG_ERROR("Dump segment %s data failed, expect: %lu, actual: %lu",
              segment_id.c_str(), size, len);
    return IndexError_WriteData;
  }

  size_t padding_size = AlignSize(size) - size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  uint32_t crc = ailego::Crc32c::Hash(data, size);
  int ret = dumper->append(segment_id, size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d", segment_id.c_str(), ret);
    return ret;
  }

  return len + padding_size;
}

int64_t HnswStreamerBenchEntity::dump_header(const IndexDumper::Pointer &dumper,
                                             const HNSWHeader &hd) const {
  //! dump basic graph header. header is aligned and does not need padding
  int64_t graph_hd_size =
      dump_segment(dumper, kGraphHeaderSegmentId, &hd.graph, hd.graph.size);
  if (graph_hd_size < 0) {
    return graph_hd_size;
  }

  //! dump basic graph header. header is aligned and does not need padding
  int64_t hnsw_hd_size =
      dump_segment(dumper, kHnswHeaderSegmentId, &hd.hnsw, hd.hnsw.size);
  if (hnsw_hd_size < 0) {
    return hnsw_hd_size;
  }

  return graph_hd_size + hnsw_hd_size;
}


HnswStreamerBenchEntity::HnswStreamerBenchEntity(IndexStreamer::Stats &stats)
    : stats_(stats) {}

HnswStreamerBenchEntity::~HnswStreamerBenchEntity() {}

int HnswStreamerBenchEntity::init(size_t max_doc_cnt) {
  if (std::pow(scaling_factor(), kMaxGraphLayers) < max_doc_cnt) {
    LOG_ERROR("scalingFactor=%zu is too small", scaling_factor());
    return IndexError_InvalidArgument;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  broker_ = std::make_shared<ChunkBroker>(stats_);
  upper_neighbor_index_ = std::make_shared<NIHashMap>();
  keys_map_lock_ = std::make_shared<ailego::SharedMutex>();
  keys_map_ = std::make_shared<HashMap<key_t, node_id_t>>();
  if (!keys_map_ || !upper_neighbor_index_ || !broker_ || !keys_map_lock_) {
    LOG_ERROR("HnswStreamerBenchEntity new object failed");
    return IndexError_NoMemory;
  }
  keys_map_->set_empty_key(kInvalidKey);

  neighbor_size_ = neighbors_size();
  upper_neighbor_size_ = upper_neighbors_size();

  //! vector + key + level 0 neighbors
  size_t size = vector_size() + sizeof(key_t) + neighbor_size_;

  size = AlignSize(size);
  set_node_size(size);
  return 0;
}

int HnswStreamerBenchEntity::cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);
  mutable_header()->clear();
  chunk_size_ = kDefaultChunkSize;
  node_index_mask_bits_ = 0U;
  node_index_mask_ = 0U;
  node_cnt_per_chunk_ = 0U;
  neighbor_size_ = 0U;
  upper_neighbor_size_ = 0U;
  if (upper_neighbor_index_) {
    upper_neighbor_index_->cleanup();
  }
  if (keys_map_) {
    keys_map_->clear();
  }
  node_chunks_.clear();
  upper_neighbor_chunks_.clear();
  filter_same_key_ = false;
  get_vector_enabled_ = false;
  broker_.reset();

  return 0;
}

int HnswStreamerBenchEntity::update_neighbors(
    level_t level, node_id_t id,
    const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
  std::vector<char> buffer(level == 0 ? neighbor_size_ : upper_neighbor_size_);
  NeighborsHeader *hd = reinterpret_cast<NeighborsHeader *>(buffer.data());
  hd->neighbor_cnt = neighbors.size();
  size_t i = 0;
  for (; i < neighbors.size(); ++i) {
    hd->neighbors[i] = neighbors[i].first;
  }

  auto loc = get_neighbor_chunk_loc(level, id);
  size_t size = reinterpret_cast<char *>(&hd->neighbors[i]) - &buffer[0];
  size_t ret = loc.first->write(loc.second, hd, size);
  if (ailego_unlikely(ret != size)) {
    LOG_ERROR("Write neighbor header failed, ret=%zu", ret);

    return IndexError_Runtime;
  }

  return 0;
}

const Neighbors HnswStreamerBenchEntity::get_neighbors(level_t level,
                                                       node_id_t id) const {
  Chunk *chunk = nullptr;
  size_t offset = 0UL;
  size_t neighbor_size = neighbor_size_;
  if (level == 0UL) {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    offset =
        (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);

    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    ailego_assert_with(chunk_idx < node_chunks_.size(), "invalid chunk idx");
    chunk = node_chunks_[chunk_idx].get();
  } else {
    auto p = get_upper_neighbor_chunk_loc(level, id);
    chunk = upper_neighbor_chunks_[p.first].get();
    offset = p.second;
    neighbor_size = upper_neighbor_size_;
  }

  ailego_assert_with(offset < chunk->data_size(), "invalid chunk offset");
  IndexStorage::MemoryBlock neighbor_block;
  size_t size = chunk->read(offset, neighbor_block, neighbor_size);
  if (ailego_unlikely(size != neighbor_size)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", size);
    return Neighbors();
  }
  return Neighbors(neighbor_block);
}

const Neighbors HnswStreamerBenchEntity::get_neighbors_new(level_t level,
                                                           node_id_t id) const {
  if (id) {
    return get_neighbors(level, id);
  } else {
    const void *src = neighbors_value_ptr_->data() + id * neighbor_size_;
    const NeighborsHeader *header =
        reinterpret_cast<const NeighborsHeader *>(src);
    return Neighbors(header->neighbor_cnt, header->neighbors);
  }
}

//! Get vector data by key
const void *HnswStreamerBenchEntity::get_vector(node_id_t id) const {
  auto loc = get_vector_chunk_loc(id);
  const void *vec = nullptr;
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
  ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                     "invalid chunk offset");

  size_t read_size = vector_size();

  size_t ret = node_chunks_[loc.first]->read(loc.second, &vec, read_size);
  if (ailego_unlikely(ret != read_size)) {
    LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
              loc.second, read_size, ret);
  }

  return vec;
}

const void *HnswStreamerBenchEntity::get_vector_new(node_id_t id) const {
  return vector_value_ptr_->data() + vector_size() * id;
  // return get_vector(id);
}

int HnswStreamerBenchEntity::get_vector(const node_id_t *ids, uint32_t count,
                                        const void **vecs) const {
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");

    size_t read_size = vector_size();

    size_t ret = node_chunks_[loc.first]->read(loc.second, &vecs[i], read_size);
    if (ailego_unlikely(ret != read_size)) {
      LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
                loc.second, read_size, ret);
      return IndexError_ReadData;
    }
  }
  return 0;
}

int HnswStreamerBenchEntity::get_vector(
    const node_id_t id, IndexStorage::MemoryBlock &block) const {
  auto loc = get_vector_chunk_loc(id);
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
  ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                     "invalid chunk offset");

  size_t read_size = vector_size();

  size_t ret = node_chunks_[loc.first]->read(loc.second, block, read_size);
  if (ailego_unlikely(ret != read_size)) {
    LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
              loc.second, read_size, ret);
    return IndexError_ReadData;
  }
  return 0;
}

int HnswStreamerBenchEntity::get_vector_new(
    const node_id_t id, IndexStorage::MemoryBlock &block) const {
  const void *data = vector_value_ptr_->data() + vector_size() * id;
  block.reset((void *)data);
  return 0;
  // return get_vector(id, block);
}

int HnswStreamerBenchEntity::get_vector_new(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
  vec_blocks.reserve(count);
  for (int i = 0; i < count; i++) {
    const void *data = vector_value_ptr_->data() + vector_size() * ids[i];
    vec_blocks.emplace_back((void *)data);
  }
  return 0;
  // return get_vector(ids, count, vec_blocks);
}

int HnswStreamerBenchEntity::get_vector(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");

    size_t read_size = vector_size();

    size_t ret =
        node_chunks_[loc.first]->read(loc.second, vec_blocks[i], read_size);
    if (ailego_unlikely(ret != read_size)) {
      LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
                loc.second, read_size, ret);
      return IndexError_ReadData;
    }
  }
  return 0;
}

key_t HnswStreamerBenchEntity::get_key(node_id_t id) const {
  if (use_key_info_map_) {
    auto loc = get_key_chunk_loc(id);
    IndexStorage::MemoryBlock key_block;
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");
    size_t ret =
        node_chunks_[loc.first]->read(loc.second, key_block, sizeof(key_t));
    if (ailego_unlikely(ret != sizeof(key_t))) {
      LOG_ERROR("Read vector failed, ret=%zu", ret);
      return kInvalidKey;
    }

    return *reinterpret_cast<const key_t *>(key_block.data());
  } else {
    return id;
  }
}

void HnswStreamerBenchEntity::add_neighbor(level_t level, node_id_t id,
                                           uint32_t size,
                                           node_id_t neighbor_id) {
  auto loc = get_neighbor_chunk_loc(level, id);
  size_t offset =
      loc.second + sizeof(NeighborsHeader) + size * sizeof(node_id_t);
  ailego_assert_with(size < neighbor_cnt(level), "invalid neighbor size");
  ailego_assert_with(offset < loc.first->data_size(), "invalid chunk offset");
  size_t ret = loc.first->write(offset, &neighbor_id, sizeof(node_id_t));
  if (ailego_unlikely(ret != sizeof(node_id_t))) {
    LOG_ERROR("Write neighbor id failed, ret=%zu", ret);
    return;
  }

  uint32_t neighbors = size + 1;
  ret = loc.first->write(loc.second, &neighbors, sizeof(uint32_t));
  if (ailego_unlikely(ret != sizeof(uint32_t))) {
    LOG_ERROR("Write neighbor cnt failed, ret=%zu", ret);
  }

  return;
}

int HnswStreamerBenchEntity::init_chunks(const Chunk::Pointer &header_chunk) {
  if (header_chunk->data_size() < header_size()) {
    LOG_ERROR("Invalid header chunk size");
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock header_block;
  size_t size = header_chunk->read(0UL, header_block, header_size());
  if (ailego_unlikely(size != header_size())) {
    LOG_ERROR("Read header chunk failed");
    return IndexError_ReadData;
  }
  *mutable_header() =
      *reinterpret_cast<const HNSWHeader *>(header_block.data());

  int ret = check_hnsw_index(&header());
  if (ret != 0) {
    broker_->close();
    return ret;
  }

  node_chunks_.resize(broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_NODE));
  for (auto seq = 0UL; seq < node_chunks_.size(); ++seq) {
    node_chunks_[seq] = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_NODE, seq);
    if (!node_chunks_[seq]) {
      LOG_ERROR("Missing hnsw streamer data chunk %zu th of %zu", seq,
                node_chunks_.size());
      return IndexError_InvalidFormat;
    }
  }

  upper_neighbor_chunks_.resize(
      broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR));
  for (auto seq = 0UL; seq < upper_neighbor_chunks_.size(); ++seq) {
    upper_neighbor_chunks_[seq] =
        broker_->get_chunk(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR, seq);
    if (!upper_neighbor_chunks_[seq]) {
      LOG_ERROR("Missing hnsw streamer index chunk %zu th of %zu", seq,
                upper_neighbor_chunks_.size());
      return IndexError_InvalidFormat;
    }
  }

  return 0;
}

int HnswStreamerBenchEntity::open(IndexStorage::Pointer stg,
                                  uint64_t max_index_size, bool check_crc) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool huge_page = stg->isHugePage();
  LOG_DEBUG("huge_page: %d", (int)huge_page);
  int ret = init_chunk_params(max_index_size, huge_page);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("init_chunk_params failed for %s", IndexError::What(ret));
    return ret;
  }
  ret = broker_->open(std::move(stg), max_index_size_, chunk_size_, check_crc);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Open index failed for %s", IndexError::What(ret));
    return ret;
  }
  ret = upper_neighbor_index_->init(broker_, upper_neighbor_chunk_size_,
                                    scaling_factor(), estimate_doc_capacity(),
                                    kUpperHashMemoryInflateRatio);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Init neighbor hash map failed");
    return ret;
  }

  //! init header
  auto header_chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                                         ChunkBroker::kDefaultChunkSeqId);
  if (!header_chunk) {  // open empty index, create one
    auto p =
        broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                             ChunkBroker::kDefaultChunkSeqId, header_size());
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc header chunk failed");
      return p.first;
    }
    size_t size = p.second->write(0UL, &header(), header_size());
    if (ailego_unlikely(size != header_size())) {
      LOG_ERROR("Write header chunk failed");
      return IndexError_WriteData;
    }
    return 0;
  }

  //! Open an exist hnsw index
  ret = init_chunks(header_chunk);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  //! total docs including features wrote in index but neighbors may not ready
  node_id_t total_vecs = 0;
  if (node_chunks_.size() > 0) {
    size_t last_idx = node_chunks_.size() - 1;
    auto last_chunk = node_chunks_[last_idx];
    if (last_chunk->data_size() % node_size()) {
      LOG_WARN("The index may broken");
      return IndexError_InvalidFormat;
    }
    total_vecs = last_idx * node_cnt_per_chunk_ +
                 node_chunks_[last_idx]->data_size() / node_size();
  }

  LOG_INFO(
      "Open index, l0NeighborCnt=%zu upperNeighborCnt=%zu "
      "efConstruction=%zu curDocCnt=%u totalVecs=%u maxLevel=%u",
      l0_neighbor_cnt(), upper_neighbor_cnt(), ef_construction(), doc_cnt(),
      total_vecs, cur_max_level());
  //! try to correct the docCnt if index not fully flushed
  if (doc_cnt() != total_vecs) {
    LOG_WARN("Index closed abnormally, using totalVecs as curDocCnt");
    *mutable_doc_cnt() = total_vecs;
  }
  if (filter_same_key_ || get_vector_enabled_) {
    if (use_key_info_map_) {
      for (node_id_t id = 0U; id < doc_cnt(); ++id) {
        if (get_key(id) == kInvalidKey) {
          continue;
        }
        (*keys_map_)[get_key(id)] = id;
      }
    }
  }
  vector_value_ptr_ = std::make_shared<std::string>();
  vector_value_ptr_->reserve(vector_size() * doc_cnt());
  for (int i = 0; i < doc_cnt(); i++) {
    vector_value_ptr_->append((const char *)get_vector(i), vector_size());
  }

  neighbors_value_ptr_ = std::make_shared<std::string>();
  neighbors_value_ptr_->reserve(neighbor_size_ * doc_cnt());
  for (int i = 0; i < doc_cnt(); i++) {
    Neighbors neighbor = get_neighbors(0, i);
    neighbors_value_ptr_->append((const char *)neighbor.neighbor_block.data(),
                                 neighbor_size_);
  }

  stats_.set_loaded_count(doc_cnt());

  return 0;
}

int HnswStreamerBenchEntity::close() {
  LOG_DEBUG("close index");

  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  mutable_header()->reset();
  upper_neighbor_index_->cleanup();
  keys_map_->clear();
  header_.clear();
  node_chunks_.clear();
  upper_neighbor_chunks_.clear();

  return broker_->close();
}

int HnswStreamerBenchEntity::flush(uint64_t checkpoint) {
  LOG_INFO("Flush index, curDocs=%u", doc_cnt());

  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  int ret = broker_->flush(checkpoint);
  if (ret != 0) {
    return ret;
  }

  return 0;
}

int HnswStreamerBenchEntity::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Dump index, curDocs=%u", doc_cnt());

  //! sort by keys, to support get_vector by key in searcher
  std::vector<key_t> keys(doc_cnt());
  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    keys[i] = get_key(i);
  }

  //! dump neighbors
  auto get_level = [&](node_id_t id) {
    auto it = upper_neighbor_index_->find(id);
    if (it == upper_neighbor_index_->end()) {
      return 0U;
    };
    auto meta = reinterpret_cast<const UpperNeighborIndexMeta *>(&it->second);
    return meta->level;
  };
  auto ret = dump_segments(dumper, keys.data(), get_level);
  if (ailego_unlikely(ret < 0)) {
    return ret;
  }
  *stats_.mutable_dumped_size() += ret;

  return 0;
}

int HnswStreamerBenchEntity::check_hnsw_index(const HNSWHeader *hd) const {
  if (l0_neighbor_cnt() != hd->l0_neighbor_cnt() ||
      upper_neighbor_cnt() != hd->upper_neighbor_cnt()) {
    LOG_ERROR("Param neighbor cnt: %zu:%zu mismatch index previous %zu:%zu",
              l0_neighbor_cnt(), upper_neighbor_cnt(), hd->l0_neighbor_cnt(),
              hd->upper_neighbor_cnt());
    return IndexError_Mismatch;
  }
  if (vector_size() != hd->vector_size()) {
    LOG_ERROR("vector size %zu mismatch index previous %zu", vector_size(),
              hd->vector_size());
    return IndexError_Mismatch;
  }
  if (ef_construction() != hd->ef_construction()) {
    LOG_WARN("Param efConstruction %zu mismatch index previous %zu",
             ef_construction(), hd->ef_construction());
  }
  if (scaling_factor() != hd->scaling_factor()) {
    LOG_WARN("Param scalingFactor %zu mismatch index previous %zu",
             scaling_factor(), hd->scaling_factor());
    return IndexError_Mismatch;
  }
  if (prune_cnt() != hd->neighbor_prune_cnt()) {
    LOG_WARN("Param pruneCnt %zu mismatch index previous %zu", prune_cnt(),
             hd->neighbor_prune_cnt());
    return IndexError_Mismatch;
  }
  if ((hd->entry_point() != kInvalidNodeId &&
       hd->entry_point() >= hd->doc_cnt()) ||
      (hd->entry_point() == kInvalidNodeId && hd->doc_cnt() > 0U)) {
    LOG_WARN("Invalid entryPoint %u, docCnt %u", hd->entry_point(),
             hd->doc_cnt());
    return IndexError_InvalidFormat;
  }
  if (hd->entry_point() == kInvalidNodeId &&
      broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_NODE) > 0) {
    LOG_WARN("The index is broken, maybe it haven't flush");
    return IndexError_InvalidFormat;
  }

  return 0;
}

int HnswStreamerBenchEntity::add_vector(level_t level, key_t key,
                                        const void *vec, node_id_t *id) {
  Chunk::Pointer node_chunk;
  size_t chunk_offset = -1UL;

  std::lock_guard<std::mutex> lock(mutex_);
  // duplicate check
  if (ailego_unlikely(filter_same_key_ && get_id(key) != kInvalidNodeId)) {
    LOG_WARN("Try to add duplicate key, ignore it");
    return IndexError_Duplicate;
  }

  node_id_t local_id = static_cast<node_id_t>(doc_cnt());
  uint32_t chunk_index = node_chunks_.size() - 1U;
  if (chunk_index == -1U ||
      (node_chunks_[chunk_index]->data_size() >=
       node_cnt_per_chunk_ * node_size())) {  // no space left and need to alloc
    if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      return IndexError_IndexFull;
    }
    chunk_index++;
    auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_index,
                                  chunk_size_);
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc data chunk failed");
      return p.first;
    }
    node_chunk = p.second;
    chunk_offset = 0UL;
    node_chunks_.emplace_back(node_chunk);
  } else {
    node_chunk = node_chunks_[chunk_index];
    chunk_offset = node_chunk->data_size();
  }

  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  //! level 0 neighbors is inited to zero by default

  int ret = add_upper_neighbor(level, local_id);
  if (ret != 0) {
    return ret;
  }

  chunk_offset += node_size();
  if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
    LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
    return IndexError_Runtime;
  }
  if (filter_same_key_ || get_vector_enabled_) {
    if (use_key_info_map_) {
      keys_map_lock_->lock();
      (*keys_map_)[key] = local_id;
      keys_map_lock_->unlock();
    }
  }

  *mutable_doc_cnt() += 1;
  broker_->mark_dirty();
  *id = local_id;

  return 0;
}

int HnswStreamerBenchEntity::add_vector_with_id(level_t level, node_id_t id,
                                                const void *vec) {
  Chunk::Pointer node_chunk;
  size_t chunk_offset = -1UL;
  key_t key = id;

  std::lock_guard<std::mutex> lock(mutex_);

  // duplicate check
  if (ailego_unlikely(filter_same_key_ && get_id(key) != kInvalidNodeId)) {
    LOG_WARN("Try to add duplicate key, ignore it");
    return IndexError_Duplicate;
  }

  // set node_chunk & chunk_offset if succeed
  auto func_get_node_chunk_and_offset = [&](node_id_t node_id) -> int {
    uint32_t chunk_index = node_id >> node_index_mask_bits_;
    ailego_assert_with(chunk_index <= node_chunks_.size(), "invalid chunk idx");
    // belongs to next chunk
    if (chunk_index == node_chunks_.size()) {
      if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
        LOG_ERROR("add vector failed for no memory quota");
        return IndexError_IndexFull;
      }
      auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_index,
                                    chunk_size_);
      if (ailego_unlikely(p.first != 0)) {
        LOG_ERROR("Alloc data chunk failed");
        return p.first;
      }
      node_chunk = p.second;
      node_chunks_.emplace_back(node_chunk);
    }

    node_chunk = node_chunks_[chunk_index];
    chunk_offset = (node_id & node_index_mask_) * node_size();
    return 0;
  };

  for (size_t start_id = doc_cnt(); start_id < id; ++start_id) {
    if (auto ret = func_get_node_chunk_and_offset(start_id); ret != 0) {
      LOG_ERROR("func_get_node_chunk_and_offset failed");
      return ret;
    }
    size_t size = node_chunk->write(chunk_offset + vector_size(), &kInvalidKey,
                                    sizeof(key_t));
    if (ailego_unlikely(size != sizeof(key_t))) {
      LOG_ERROR("Chunk write key failed, ret=%zu", size);
      return IndexError_WriteData;
    }

    chunk_offset += node_size();
    if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
      return IndexError_Runtime;
    }
  }

  if (auto ret = func_get_node_chunk_and_offset(id); ret != 0) {
    LOG_ERROR("func_get_node_chunk_and_offset failed");
    return ret;
  }

  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  //! level 0 neighbors is inited to zero by default

  int ret = add_upper_neighbor(level, id);
  if (ret != 0) {
    return ret;
  }

  if (*mutable_doc_cnt() <= id) {
    *mutable_doc_cnt() = id + 1;
    chunk_offset += node_size();
    if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
      return IndexError_Runtime;
    }
  }

  if (filter_same_key_ || get_vector_enabled_) {
    if (use_key_info_map_) {
      keys_map_lock_->lock();
      (*keys_map_)[key] = id;
      keys_map_lock_->unlock();
    }
  }

  broker_->mark_dirty();

  return 0;
}

void HnswStreamerBenchEntity::update_ep_and_level(node_id_t ep, level_t level) {
  base_header_.hnsw.entry_point = ep;
  base_header_.hnsw.max_level = level;
  flush_header();

  return;
}

const HnswStreamerBenchEntity::Pointer HnswStreamerBenchEntity::clone() const {
  std::vector<Chunk::Pointer> node_chunks;
  node_chunks.reserve(node_chunks_.size());
  for (size_t i = 0UL; i < node_chunks_.size(); ++i) {
    node_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!node_chunks[i])) {
      LOG_ERROR("HnswStreamerBenchEntity get chunk failed in clone");
      return HnswStreamerBenchEntity::Pointer();
    }
  }

  std::vector<Chunk::Pointer> upper_neighbor_chunks;
  upper_neighbor_chunks.reserve(upper_neighbor_chunks_.size());
  for (size_t i = 0UL; i < upper_neighbor_chunks_.size(); ++i) {
    upper_neighbor_chunks.emplace_back(upper_neighbor_chunks_[i]->clone());
    if (ailego_unlikely(!upper_neighbor_chunks[i])) {
      LOG_ERROR("HnswStreamerBenchEntity get chunk failed in clone");
      return HnswStreamerBenchEntity::Pointer();
    }
  }

  HnswStreamerBenchEntity *entity = new (std::nothrow) HnswStreamerBenchEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_,
      upper_neighbor_mask_bits_, filter_same_key_, get_vector_enabled_,
      upper_neighbor_index_, keys_map_lock_, keys_map_, use_key_info_map_,
      std::move(node_chunks), std::move(upper_neighbor_chunks), broker_,
      vector_value_ptr_, neighbors_value_ptr_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("HnswStreamerBenchEntity new failed");
  }
  return HnswStreamerBenchEntity::Pointer(entity);
}

HnswStreamerBenchEntity::UPointer HnswStreamerBenchEntity::clone_uptr() const {
  std::vector<Chunk::Pointer> node_chunks;
  node_chunks.reserve(node_chunks_.size());
  for (size_t i = 0UL; i < node_chunks_.size(); ++i) {
    node_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!node_chunks[i])) {
      LOG_ERROR("HnswStreamerBenchEntity get chunk failed in clone");
      return HnswStreamerBenchEntity::UPointer();
    }
  }

  std::vector<Chunk::Pointer> upper_neighbor_chunks;
  upper_neighbor_chunks.reserve(upper_neighbor_chunks_.size());
  for (size_t i = 0UL; i < upper_neighbor_chunks_.size(); ++i) {
    upper_neighbor_chunks.emplace_back(upper_neighbor_chunks_[i]->clone());
    if (ailego_unlikely(!upper_neighbor_chunks[i])) {
      LOG_ERROR("HnswStreamerBenchEntity get chunk failed in clone");
      return HnswStreamerBenchEntity::UPointer();
    }
  }

  HnswStreamerBenchEntity *entity = new (std::nothrow) HnswStreamerBenchEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_,
      upper_neighbor_mask_bits_, filter_same_key_, get_vector_enabled_,
      upper_neighbor_index_, keys_map_lock_, keys_map_, use_key_info_map_,
      std::move(node_chunks), std::move(upper_neighbor_chunks), broker_,
      vector_value_ptr_, neighbors_value_ptr_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("HnswStreamerBenchEntity new failed");
  }
  return HnswStreamerBenchEntity::UPointer(entity);
}


int64_t HnswStreamerBenchEntity::dump_mapping_segment(
    const IndexDumper::Pointer &dumper, const key_t *keys) const {
  std::vector<node_id_t> mapping(doc_cnt());

  std::iota(mapping.begin(), mapping.end(), 0U);
  std::sort(mapping.begin(), mapping.end(),
            [&](node_id_t i, node_id_t j) { return keys[i] < keys[j]; });

  size_t size = mapping.size() * sizeof(node_id_t);

  return dump_segment(dumper, kGraphMappingSegmentId, mapping.data(), size);
}

int64_t HnswStreamerBenchEntity::dump_segments(
    const IndexDumper::Pointer &dumper, key_t *keys,
    const std::function<level_t(node_id_t)> &get_level) const {
  HNSWHeader dump_hd(header());

  dump_hd.graph.node_size = AlignSize(vector_size());

  std::vector<node_id_t> n2o_mapping;  // map new id to origin id
  std::vector<node_id_t> o2n_mapping;  // map origin id to new id
  if (!o2n_mapping.empty()) {
    dump_hd.hnsw.entry_point = o2n_mapping[entry_point()];
  }

  //! Dump header
  int64_t hd_size = dump_header(dumper, dump_hd);
  if (hd_size < 0) {
    return hd_size;
  }

  //! Dump vectors
  int64_t vecs_size = dump_vectors(dumper, n2o_mapping);
  if (vecs_size < 0) {
    return vecs_size;
  }

  //! Dump neighbors
  auto neighbors_size =
      dump_neighbors(dumper, get_level, n2o_mapping, o2n_mapping);
  if (neighbors_size < 0) {
    return neighbors_size;
  }
  //! free memory
  n2o_mapping = std::vector<node_id_t>();
  o2n_mapping = std::vector<node_id_t>();

  //! Dump keys
  size_t key_segment_size = doc_cnt() * sizeof(key_t);
  int64_t keys_size =
      dump_segment(dumper, kGraphKeysSegmentId, keys, key_segment_size);
  if (keys_size < 0) {
    return keys_size;
  }

  //! Dump mapping
  int64_t mapping_size = dump_mapping_segment(dumper, keys);
  if (mapping_size < 0) {
    return mapping_size;
  }

  return hd_size + keys_size + vecs_size + neighbors_size + mapping_size;
}

int64_t HnswStreamerBenchEntity::dump_vectors(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping) const {
  size_t vector_dump_size = vector_size();

  size_t padding_size = AlignSize(vector_dump_size) - vector_dump_size;

  std::vector<char> padding(padding_size);
  memset(padding.data(), 0, sizeof(char) * padding_size);
  const void *data = nullptr;
  uint32_t crc = 0U;
  size_t vecs_size = 0UL;

  //! dump vectors
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    data = get_vector(reorder_mapping.empty() ? id : reorder_mapping[id]);
    if (ailego_unlikely(!data)) {
      return IndexError_ReadData;
    }
    size_t len = dumper->write(data, vector_size());
    if (len != vector_size()) {
      LOG_ERROR("Dump vectors failed, write=%zu expect=%zu", len,
                vector_size());
      return IndexError_WriteData;
    }

    crc = ailego::Crc32c::Hash(data, vector_size(), crc);
    vecs_size += vector_size();

    if (padding_size == 0) {
      continue;
    }

    len = dumper->write(padding.data(), padding_size);
    if (len != padding_size) {
      LOG_ERROR("Dump vectors failed, write=%zu expect=%zu", len, padding_size);
      return IndexError_WriteData;
    }
    crc = ailego::Crc32c::Hash(padding.data(), padding_size, crc);
    vecs_size += padding_size;
  }

  int ret = dumper->append(kGraphFeaturesSegmentId, vecs_size, 0UL, crc);
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment meta failed, ret %d", ret);
    return ret;
  }

  return vecs_size;
}

int64_t HnswStreamerBenchEntity::dump_graph_neighbors(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping,
    const std::vector<node_id_t> &neighbor_mapping) const {
  std::vector<GraphNeighborMeta> graph_meta;
  graph_meta.reserve(doc_cnt());
  size_t offset = 0;
  uint32_t crc = 0;
  std::vector<node_id_t> mapping(l0_neighbor_cnt());

  uint32_t min_neighbor_count = 10000;
  uint32_t max_neighbor_count = 0;
  size_t sum_neighbor_count = 0;

  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    const Neighbors neighbors =
        get_neighbors(0, reorder_mapping.empty() ? id : reorder_mapping[id]);
    ailego_assert_with(!!neighbors.data, "invalid neighbors");
    ailego_assert_with(neighbors.size() <= l0_neighbor_cnt(),
                       "invalid neighbors");

    uint32_t neighbor_count = neighbors.size();
    if (neighbor_count < min_neighbor_count) {
      min_neighbor_count = neighbor_count;
    }
    if (neighbor_count > max_neighbor_count) {
      max_neighbor_count = neighbor_count;
    }
    sum_neighbor_count += neighbor_count;

    graph_meta.emplace_back(offset, neighbor_count);
    size_t size = neighbors.size() * sizeof(node_id_t);
    const node_id_t *data = &neighbors[0];
    if (!neighbor_mapping.empty()) {
      for (node_id_t i = 0; i < neighbors.size(); ++i) {
        mapping[i] = neighbor_mapping[neighbors[i]];
      }
      data = mapping.data();
    }
    if (dumper->write(data, size) != size) {
      LOG_ERROR("Dump graph neighbor id=%u failed, size %lu", id, size);
      return IndexError_WriteData;
    }
    crc = ailego::Crc32c::Hash(data, size, crc);
    offset += size;
  }

  uint32_t average_neighbor_count = 0;
  if (doc_cnt() > 0) {
    average_neighbor_count = sum_neighbor_count / doc_cnt();
  }
  LOG_INFO(
      "Dump hnsw graph: min_neighbor_count[%u] max_neighbor_count[%u] "
      "average_neighbor_count[%u]",
      min_neighbor_count, max_neighbor_count, average_neighbor_count);

  size_t padding_size = 0;
  int ret = CalcAndAddPadding(dumper, offset, &padding_size);
  if (ret != 0) {
    return ret;
  }
  ret = dumper->append(kGraphNeighborsSegmentId, offset, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s failed, ret %d",
              kGraphNeighborsSegmentId.c_str(), ret);
    return ret;
  }

  //! dump level 0 neighbors meta
  auto len = dump_segment(dumper, kGraphOffsetsSegmentId, graph_meta.data(),
                          graph_meta.size() * sizeof(GraphNeighborMeta));
  if (len < 0) {
    return len;
  }

  return len + offset + padding_size;
}

int64_t HnswStreamerBenchEntity::dump_upper_neighbors(
    const IndexDumper::Pointer &dumper,
    const std::function<level_t(node_id_t)> &get_level,
    const std::vector<node_id_t> &reorder_mapping,
    const std::vector<node_id_t> &neighbor_mapping) const {
  std::vector<HnswNeighborMeta> hnsw_meta;
  hnsw_meta.reserve(doc_cnt());
  size_t offset = 0;
  uint32_t crc = 0;
  std::vector<node_id_t> buffer(upper_neighbor_cnt() + 1);
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    node_id_t new_id = reorder_mapping.empty() ? id : reorder_mapping[id];
    auto level = get_level(new_id);
    if (level == 0) {
      hnsw_meta.emplace_back(0U, 0U);
      continue;
    }
    hnsw_meta.emplace_back(offset, level);
    ailego_assert_with((size_t)level < kMaxGraphLayers, "invalid level");
    for (level_t cur_level = 1; cur_level <= level; ++cur_level) {
      const Neighbors neighbors = get_neighbors(cur_level, new_id);
      ailego_assert_with(!!neighbors.data, "invalid neighbors");
      ailego_assert_with(neighbors.size() <= neighbor_cnt(cur_level),
                         "invalid neighbors");
      memset(buffer.data(), 0, sizeof(node_id_t) * buffer.size());
      buffer[0] = neighbors.size();
      if (neighbor_mapping.empty()) {
        memcpy(&buffer[1], &neighbors[0], neighbors.size() * sizeof(node_id_t));
      } else {
        for (node_id_t i = 0; i < neighbors.size(); ++i) {
          buffer[i + 1] = neighbor_mapping[neighbors[i]];
        }
      }
      if (dumper->write(buffer.data(), sizeof(node_id_t) * buffer.size()) !=
          sizeof(node_id_t) * buffer.size()) {
        LOG_ERROR("Dump graph neighbor id=%u failed, size %lu", id,
                  sizeof(node_id_t) * buffer.size());
        return IndexError_WriteData;
      }
      crc = ailego::Crc32c::Hash(buffer.data(),
                                 sizeof(node_id_t) * buffer.size(), crc);
      offset += sizeof(node_id_t) * buffer.size();
    }
  }
  size_t padding_size = 0;
  int ret = CalcAndAddPadding(dumper, offset, &padding_size);
  if (ret != 0) {
    return ret;
  }

  ret = dumper->append(kHnswNeighborsSegmentId, offset, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s failed, ret %d", kHnswNeighborsSegmentId.c_str(),
              ret);
    return ret;
  }

  //! dump level 0 neighbors meta
  auto len = dump_segment(dumper, kHnswOffsetsSegmentId, hnsw_meta.data(),
                          hnsw_meta.size() * sizeof(HnswNeighborMeta));
  if (len < 0) {
    return len;
  }

  return len + offset + padding_size;
}

int HnswStreamerBenchEntity::CalcAndAddPadding(
    const IndexDumper::Pointer &dumper, size_t data_size,
    size_t *padding_size) {
  *padding_size = AlignSize(data_size) - data_size;
  if (*padding_size == 0) {
    return 0;
  }

  std::string padding(*padding_size, '\0');
  if (dumper->write(padding.data(), *padding_size) != *padding_size) {
    LOG_ERROR("Append padding failed, size %lu", *padding_size);
    return IndexError_WriteData;
  }
  return 0;
}


}  // namespace core
}  // namespace zvec

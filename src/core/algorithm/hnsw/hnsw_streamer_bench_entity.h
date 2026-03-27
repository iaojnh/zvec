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

#include <ailego/parallel/lock.h>
#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>
#include <zvec/ailego/container/heap.h>
#include <zvec/core/framework/index_framework.h>
#include "hnsw_chunk.h"
#include "hnsw_entity.h"
#include "hnsw_index_hash.h"
#include "hnsw_params.h"

namespace zvec {
namespace core {

//! HnswStreamerBenchEntity manage vector data, pkey, and node's neighbors
class HnswStreamerBenchEntity {
 public:  // override
  typedef std::shared_ptr<HnswStreamerBenchEntity> Pointer;

  //! Cleanup
  //! return 0 on success, or errCode in failure
  int cleanup();

  //! Make a copy of streamer entity, to support thread-safe operation.
  //! The segment in container cannot be read concurrenly
  const HnswStreamerBenchEntity::Pointer clone() const;

  //! Get primary key of the node id
  key_t get_key(node_id_t id) const;

  //! Get vector feature data by key
  const void *get_vector(node_id_t id) const;

  const void *get_vector_new(node_id_t id) const;

  //! Get vectors feature data by local ids
  int get_vector(const node_id_t *ids, uint32_t count, const void **vecs) const;

  int get_vector(const node_id_t id, IndexStorage::MemoryBlock &block) const;

  int get_vector(const node_id_t *ids, uint32_t count,
                 std::vector<IndexStorage::MemoryBlock> &vec_blocks) const;

  int get_vector_new(const node_id_t id,
                     IndexStorage::MemoryBlock &block) const;

  int get_vector_new(const node_id_t *ids, uint32_t count,
                     std::vector<IndexStorage::MemoryBlock> &vec_blocks) const;

  //! Get the node id's neighbors on graph level
  //! Note: the neighbors cannot be modified, using the following
  //! method to get WritableNeighbors if want to
  const Neighbors get_neighbors(level_t level, node_id_t id) const;
  const Neighbors get_neighbors_new(level_t level, node_id_t id) const;

  //! Add vector and key to hnsw entity, and local id will be saved in id
  int add_vector(level_t level, key_t key, const void *vec, node_id_t *id);

  //! Add vector and id to hnsw entity
  int add_vector_with_id(level_t level, node_id_t id, const void *vec);

  int update_neighbors(
      level_t level, node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors);

  //! Append neighbor_id to node id neighbors on level
  //! Notice: the caller must be ensure the neighbors not full
  void add_neighbor(level_t level, node_id_t id, uint32_t size,
                    node_id_t neighbor_id);

  //! Dump index by dumper
  int dump(const IndexDumper::Pointer &dumper);

  void update_ep_and_level(node_id_t ep, level_t level);

  const void *get_vector_by_key(key_t key) const {
    auto id = get_id(key);
    return id == kInvalidNodeId ? nullptr : get_vector(id);
  }

  int get_vector_by_key(const key_t key,
                        IndexStorage::MemoryBlock &block) const {
    auto id = get_id(key);
    if (id != kInvalidNodeId) {
      return get_vector(id, block);
    } else {
      return IndexError_InvalidArgument;
    }
  }

 public:  // hnsw entity public
  //! Get max neighbor size of graph level
  inline size_t neighbor_cnt(level_t level) const {
    return level == 0 ? base_header_.graph.l0_neighbor_count
                      : base_header_.hnsw.upper_neighbor_count;
  }

  //! get max neighbor size of graph level 0
  inline size_t l0_neighbor_cnt() const {
    return base_header_.graph.l0_neighbor_count;
  }

  //! get min neighbor size of graph
  inline size_t min_neighbor_cnt() const {
    return base_header_.graph.min_neighbor_count;
  }

  //! get upper neighbor size of graph level other than 0
  inline size_t upper_neighbor_cnt() const {
    return base_header_.hnsw.upper_neighbor_count;
  }

  //! Get current total doc of the hnsw graph
  inline node_id_t *mutable_doc_cnt() {
    return &base_header_.graph.doc_count;
  }

  inline node_id_t doc_cnt() const {
    return base_header_.graph.doc_count;
  }

  //! Get hnsw graph scaling params
  inline size_t scaling_factor() const {
    return base_header_.hnsw.scaling_factor;
  }

  //! Get prune_size
  inline size_t prune_cnt() const {
    return base_header_.graph.prune_neighbor_count;
  }

  //! Current entity of top level graph
  inline node_id_t entry_point() const {
    return base_header_.hnsw.entry_point;
  }

  //! Current max graph level
  inline level_t cur_max_level() const {
    return base_header_.hnsw.max_level;
  }

  //! Retrieve index vector size
  size_t vector_size() const {
    return base_header_.graph.vector_size;
  }

  //! Retrieve node size
  size_t node_size() const {
    return base_header_.graph.node_size;
  }

  //! Retrieve ef constuction
  size_t ef_construction() const {
    return base_header_.graph.ef_construction;
  }

  void set_vector_size(size_t size) {
    base_header_.graph.vector_size = size;
  }

  void set_prune_cnt(size_t v) {
    base_header_.graph.prune_neighbor_count = v;
  }

  void set_scaling_factor(size_t val) {
    base_header_.hnsw.scaling_factor = val;
  }

  void set_l0_neighbor_cnt(size_t cnt) {
    base_header_.graph.l0_neighbor_count = cnt;
  }

  void set_min_neighbor_cnt(size_t cnt) {
    base_header_.graph.min_neighbor_count = cnt;
  }

  void set_upper_neighbor_cnt(size_t cnt) {
    base_header_.hnsw.upper_neighbor_count = cnt;
  }

  void set_ef_construction(size_t ef) {
    base_header_.graph.ef_construction = ef;
  }

  static int CalcAndAddPadding(const IndexDumper::Pointer &dumper,
                               size_t data_size, size_t *padding_size);

 protected:
  inline const HNSWHeader &header() const {
    return base_header_;
  }

  inline HNSWHeader *mutable_header() {
    return &base_header_;
  }

  inline size_t header_size() const {
    return sizeof(base_header_);
  }

  void set_node_size(size_t size) {
    base_header_.graph.node_size = size;
  }

  //! Dump all segment by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_segments(
      const IndexDumper::Pointer &dumper, key_t *keys,
      const std::function<level_t(node_id_t)> &get_level) const;

  static inline size_t AlignSize(size_t size) {
    return (size + 0x1F) & (~0x1F);
  }

  static inline size_t AlignPageSize(size_t size) {
    size_t page_mask = ailego::MemoryHelper::PageSize() - 1;
    return (size + page_mask) & (~page_mask);
  }

  static inline size_t AlignHugePageSize(size_t size) {
    size_t page_mask = ailego::MemoryHelper::HugePageSize() - 1;
    return (size + page_mask) & (~page_mask);
  }

 private:
  //! dump mapping segment, for get_vector_by_key in provider
  int64_t dump_mapping_segment(const IndexDumper::Pointer &dumper,
                               const key_t *keys) const;

  //! dump hnsw head by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_header(const IndexDumper::Pointer &dumper,
                      const HNSWHeader &hd) const;

  //! dump vectors by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_vectors(const IndexDumper::Pointer &dumper,
                       const std::vector<node_id_t> &reorder_mapping) const;

  //! dump hnsw neighbors by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_neighbors(const IndexDumper::Pointer &dumper,
                         const std::function<level_t(node_id_t)> &get_level,
                         const std::vector<node_id_t> &reorder_mapping,
                         const std::vector<node_id_t> &neighbor_mapping) const {
    auto len1 = dump_graph_neighbors(dumper, reorder_mapping, neighbor_mapping);
    if (len1 < 0) {
      return len1;
    }
    auto len2 = dump_upper_neighbors(dumper, get_level, reorder_mapping,
                                     neighbor_mapping);
    if (len2 < 0) {
      return len2;
    }

    return len1 + len2;
  }

  //! dump segment by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_segment(const IndexDumper::Pointer &dumper,
                       const std::string &segment_id, const void *data,
                       size_t size) const;

  //! Dump level 0 neighbors
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_graph_neighbors(
      const IndexDumper::Pointer &dumper,
      const std::vector<node_id_t> &reorder_mapping,
      const std::vector<node_id_t> &neighbor_mapping) const;

  //! Dump upper level neighbors
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_upper_neighbors(
      const IndexDumper::Pointer &dumper,
      const std::function<level_t(node_id_t)> &get_level,
      const std::vector<node_id_t> &reorder_mapping,
      const std::vector<node_id_t> &neighbor_mapping) const;

 public:
  const static std::string kGraphHeaderSegmentId;
  const static std::string kGraphFeaturesSegmentId;
  const static std::string kGraphKeysSegmentId;
  const static std::string kGraphNeighborsSegmentId;
  const static std::string kGraphOffsetsSegmentId;
  const static std::string kGraphMappingSegmentId;
  const static std::string kHnswHeaderSegmentId;
  const static std::string kHnswNeighborsSegmentId;
  const static std::string kHnswOffsetsSegmentId;

  constexpr static uint32_t kRevision = 0U;
  constexpr static size_t kMaxGraphLayers = 15;
  constexpr static uint32_t kDefaultEfConstruction = 500;
  constexpr static uint32_t kDefaultEf = 500;
  constexpr static uint32_t kDefaultUpperMaxNeighborCnt = 50;  // M of HNSW
  constexpr static uint32_t kDefaultL0MaxNeighborCnt = 100;
  constexpr static uint32_t kMaxNeighborCnt = 65535;
  constexpr static float kDefaultScanRatio = 0.1f;
  constexpr static uint32_t kDefaultMinScanLimit = 10000;
  constexpr static uint32_t kDefaultMaxScanLimit =
      std::numeric_limits<uint32_t>::max();
  constexpr static float kDefaultBFNegativeProbability = 0.001f;
  constexpr static uint32_t kDefaultScalingFactor = 50U;
  constexpr static uint32_t kDefaultBruteForceThreshold = 1000U;
  constexpr static uint32_t kDefaultDocsHardLimit = 1 << 30U;  // 1 billion
  constexpr static float kDefaultDocsSoftLimitRatio = 0.9f;
  constexpr static size_t kMaxChunkSize = 0xFFFFFFFF;
  constexpr static size_t kDefaultChunkSize = 2UL * 1024UL * 1024UL;
  constexpr static size_t kDefaultMaxChunkCnt = 50000UL;
  constexpr static float kDefaultNeighborPruneMultiplier =
      1.0f;  // prune_cnt = upper_max_neighbor_cnt * multiplier
  constexpr static float kDefaultL0MaxNeighborCntMultiplier =
      2.0f;  // l0_max_neighbor_cnt = upper_max_neighbor_cnt * multiplier

 public:
  //! Constructor
  HnswStreamerBenchEntity(IndexStreamer::Stats &stats);

  //! Destructor
  ~HnswStreamerBenchEntity();

  //! Get vector feature data by key


  //! Init entity
  int init(size_t max_doc_cnt);

  //! Flush graph entity to disk
  //! return 0 on success, or errCode in failure
  int flush(uint64_t checkpoint);

  //! Open entity from storage
  //! return 0 on success, or errCode in failure
  int open(IndexStorage::Pointer stg, uint64_t max_index_size, bool check_crc);

  //! Close entity
  //! return 0 on success, or errCode in failure
  int close();

  void set_use_key_info_map(bool use_id_map) {
    use_key_info_map_ = use_id_map;
    LOG_DEBUG("use_key_info_map_: %d", (int)use_key_info_map_);
  }

  //! Set meta information from entity
  int set_index_meta(const IndexMeta &meta) const {
    return IndexHelper::SerializeToStorage(meta, broker_->storage().get());
  }

  //! Get meta information from entity
  int get_index_meta(IndexMeta *meta) const {
    return IndexHelper::DeserializeFromStorage(broker_->storage().get(), meta);
  }

  //! Set params: chunk size
  inline void set_chunk_size(size_t val) {
    chunk_size_ = val;
  }

  //! Set params
  inline void set_filter_same_key(bool val) {
    filter_same_key_ = val;
  }

  //! Set params
  inline void set_get_vector(bool val) {
    get_vector_enabled_ = val;
  }

  //! Get vector local id by key
  inline node_id_t get_id(key_t key) const {
    if (use_key_info_map_) {
      keys_map_lock_->lock_shared();
      auto it = keys_map_->find(key);
      keys_map_lock_->unlock_shared();
      return it == keys_map_->end() ? kInvalidNodeId : it->second;
    } else {
      return key;
    }
  }

  void print_key_map() const {
    LOG_DEBUG("key map begins");
    auto iter = keys_map_->begin();
    while (iter != keys_map_->end()) {
      LOG_DEBUG("key: %lld, id: %u", (long long)iter->first, iter->second);
      iter++;
    }
    LOG_DEBUG("key map ends");
  }

  //! Get l0 neighbors size
  inline size_t neighbors_size() const {
    return sizeof(NeighborsHeader) + l0_neighbor_cnt() * sizeof(node_id_t);
  }

  //! Get neighbors size for level > 0
  inline size_t upper_neighbors_size() const {
    return sizeof(NeighborsHeader) + upper_neighbor_cnt() * sizeof(node_id_t);
  }


 private:
  union UpperNeighborIndexMeta {
    struct {
      uint32_t level : 4;
      uint32_t index : 28;  // index is composite type: chunk idx, and the
                            // N th neighbors in chunk, they two composite
                            // the 28 bits location
    };
    uint32_t data;
  };

  template <class Key, class T>
  using HashMap = google::dense_hash_map<Key, T, std::hash<Key>>;
  template <class Key, class T>
  using HashMapPointer = std::shared_ptr<HashMap<Key, T>>;

  template <class Key>
  using HashSet = google::dense_hash_set<Key, std::hash<Key>>;
  template <class Key>
  using HashSetPointer = std::shared_ptr<HashSet<Key>>;

  //! upper neighbor index hashmap
  using NIHashMap = HnswIndexHashMap<node_id_t, uint32_t>;
  using NIHashMapPointer = std::shared_ptr<NIHashMap>;

  //! Private construct, only be called by clone method
  HnswStreamerBenchEntity(IndexStreamer::Stats &stats, const HNSWHeader &hd,
                        size_t chunk_size, uint32_t node_index_mask_bits,
                        uint32_t upper_neighbor_mask_bits, bool filter_same_key,
                        bool get_vector_enabled,
                        const NIHashMapPointer &upper_neighbor_index,
                        std::shared_ptr<ailego::SharedMutex> &keys_map_lock,
                        const HashMapPointer<key_t, node_id_t> &keys_map,
                        bool use_key_info_map,
                        std::vector<Chunk::Pointer> &&node_chunks,
                        std::vector<Chunk::Pointer> &&upper_neighbor_chunks,
                        const ChunkBroker::Pointer &broker,
                        std::shared_ptr<std::string> vector_value_ptr,
                        std::shared_ptr<std::string> neighbors_value_ptr)
      : stats_(stats),
        chunk_size_(chunk_size),
        node_index_mask_bits_(node_index_mask_bits),
        node_cnt_per_chunk_(1UL << node_index_mask_bits_),
        node_index_mask_(node_cnt_per_chunk_ - 1),
        upper_neighbor_mask_bits_(upper_neighbor_mask_bits),
        upper_neighbor_mask_((1U << upper_neighbor_mask_bits_) - 1),
        filter_same_key_(filter_same_key),
        get_vector_enabled_(get_vector_enabled),
        use_key_info_map_(use_key_info_map),
        upper_neighbor_index_(upper_neighbor_index),
        keys_map_lock_(keys_map_lock),
        keys_map_(keys_map),
        node_chunks_(std::move(node_chunks)),
        upper_neighbor_chunks_(std::move(upper_neighbor_chunks)),
        broker_(broker),
        vector_value_ptr_(vector_value_ptr),
        neighbors_value_ptr_(neighbors_value_ptr) {
    *mutable_header() = hd;

    neighbor_size_ = neighbors_size();
    upper_neighbor_size_ = upper_neighbors_size();
  }

  //! Called only in searching procedure per context, so no need to lock
  void sync_chunks(ChunkBroker::CHUNK_TYPE type, size_t idx,
                   std::vector<Chunk::Pointer> *chunks) const {
    if (ailego_likely(idx < chunks->size())) {
      return;
    }
    for (size_t i = chunks->size(); i <= idx; ++i) {
      auto chunk = broker_->get_chunk(type, i);
      // the storage can ensure get chunk will success after the first get
      ailego_assert_with(!!chunk, "get chunk failed");
      chunks->emplace_back(std::move(chunk));
    }
  }

  //! return pair: chunk index + chunk offset
  inline std::pair<uint32_t, uint32_t> get_vector_chunk_loc(
      node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size();

    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    return std::make_pair(chunk_idx, offset);
  }

  //! return pair: chunk index + chunk offset
  inline std::pair<uint32_t, uint32_t> get_key_chunk_loc(node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size() + vector_size();

    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    return std::make_pair(chunk_idx, offset);
  }

  inline std::pair<uint32_t, uint32_t> get_upper_neighbor_chunk_loc(
      level_t level, node_id_t id) const {
    auto it = upper_neighbor_index_->find(id);
    ailego_assert_abort(it != upper_neighbor_index_->end(),
                        "Get upper neighbor header failed");
    auto meta = reinterpret_cast<const UpperNeighborIndexMeta *>(&it->second);
    uint32_t chunk_idx = (meta->index) >> upper_neighbor_mask_bits_;
    uint32_t offset = (((meta->index) & upper_neighbor_mask_) + level - 1) *
                      upper_neighbor_size_;
    sync_chunks(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR, chunk_idx,
                &upper_neighbor_chunks_);
    ailego_assert_abort(chunk_idx < upper_neighbor_chunks_.size(),
                        "invalid chunk idx");
    ailego_assert_abort(offset < upper_neighbor_chunks_[chunk_idx]->data_size(),
                        "invalid chunk offset");
    return std::make_pair(chunk_idx, offset);
  }

  //! return pair: chunk + chunk offset
  inline std::pair<Chunk *, size_t> get_neighbor_chunk_loc(level_t level,
                                                           node_id_t id) const {
    if (level == 0UL) {
      uint32_t chunk_idx = id >> node_index_mask_bits_;
      uint32_t offset =
          (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);

      sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
      ailego_assert_abort(chunk_idx < node_chunks_.size(), "invalid chunk idx");
      ailego_assert_abort(offset < node_chunks_[chunk_idx]->data_size(),
                          "invalid chunk offset");
      return std::make_pair(node_chunks_[chunk_idx].get(), offset);
    } else {
      auto p = get_upper_neighbor_chunk_loc(level, id);
      return std::make_pair(upper_neighbor_chunks_[p.first].get(), p.second);
    }
  }

  //! Chunk hnsw index valid
  int check_hnsw_index(const HNSWHeader *hd) const;

  size_t get_total_upper_neighbors_size(level_t level) const {
    return level * upper_neighbor_size_;
  }

  //! Add upper neighbor header and reserve space for upper neighbor
  int add_upper_neighbor(level_t level, node_id_t id) {
    if (level == 0) {
      return 0;
    }
    Chunk::Pointer chunk;
    uint64_t chunk_offset = -1UL;
    size_t neighbors_size = get_total_upper_neighbors_size(level);
    uint64_t chunk_index = upper_neighbor_chunks_.size() - 1UL;
    if (chunk_index == -1UL ||
        (upper_neighbor_chunks_[chunk_index]->padding_size() <
         neighbors_size)) {  // no space left and need to alloc
      chunk_index++;
      if (ailego_unlikely(upper_neighbor_chunks_.capacity() ==
                          upper_neighbor_chunks_.size())) {
        LOG_ERROR("add upper neighbor failed for no memory quota");
        return IndexError_IndexFull;
      }
      auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR,
                                    chunk_index, upper_neighbor_chunk_size_);
      if (ailego_unlikely(p.first != 0)) {
        LOG_ERROR("Alloc data chunk failed");
        return p.first;
      }
      chunk = p.second;
      chunk_offset = 0UL;
      upper_neighbor_chunks_.emplace_back(chunk);
    } else {
      chunk = upper_neighbor_chunks_[chunk_index];
      chunk_offset = chunk->data_size();
    }
    ailego_assert_with((size_t)level < kMaxGraphLayers, "invalid level");
    ailego_assert_with(chunk_offset % upper_neighbor_size_ == 0,
                       "invalid offset");
    ailego_assert_with((chunk_offset / upper_neighbor_size_) <
                           (1U << upper_neighbor_mask_bits_),
                       "invalid offset");
    ailego_assert_with(chunk_index < (1U << (28 - upper_neighbor_mask_bits_)),
                       "invalid chunk index");
    UpperNeighborIndexMeta meta;
    meta.level = level;
    meta.index = (chunk_index << upper_neighbor_mask_bits_) |
                 (chunk_offset / upper_neighbor_size_);
    chunk_offset += upper_neighbor_size_ * level;
    if (ailego_unlikely(!upper_neighbor_index_->insert(id, meta.data))) {
      LOG_ERROR("HashMap insert value failed");
      return IndexError_Runtime;
    }

    if (ailego_unlikely(chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", (size_t)chunk_offset);
      return IndexError_Runtime;
    }

    return 0;
  }

  size_t estimate_doc_capacity() const {
    return node_chunks_.capacity() * node_cnt_per_chunk_;
  }

  int init_chunk_params(size_t max_index_size, bool huge_page) {
    node_cnt_per_chunk_ = std::max<uint32_t>(1, chunk_size_ / node_size());
    //! align node cnt per chunk to pow of 2
    node_index_mask_bits_ = std::ceil(std::log2(node_cnt_per_chunk_));
    node_cnt_per_chunk_ = 1UL << node_index_mask_bits_;
    if (huge_page) {
      chunk_size_ = AlignHugePageSize(node_cnt_per_chunk_ * node_size());
    } else {
      chunk_size_ = AlignPageSize(node_cnt_per_chunk_ * node_size());
    }
    node_index_mask_ = node_cnt_per_chunk_ - 1;

    if (max_index_size == 0UL) {
      max_index_size_ = chunk_size_ * kDefaultMaxChunkCnt;
    } else {
      max_index_size_ = max_index_size;
    }

    //! To get a balanced upper neighbor chunk size.
    //! If the upper chunk size is equal to node chunk size, it may waste
    //! upper neighbor chunk space; if the upper neighbor chunk size is too
    //! small, the will need large upper neighbor chunks index space. So to
    //! get a balanced ratio be sqrt of the node/neighbor size ratio
    float ratio =
        std::sqrt(node_size() * scaling_factor() * 1.0f / upper_neighbor_size_);
    if (huge_page) {
      upper_neighbor_chunk_size_ = AlignHugePageSize(
          std::max(get_total_upper_neighbors_size(kMaxGraphLayers),
                   static_cast<size_t>(chunk_size_ / ratio)));
    } else {
      upper_neighbor_chunk_size_ = AlignPageSize(
          std::max(get_total_upper_neighbors_size(kMaxGraphLayers),
                   static_cast<size_t>(chunk_size_ / ratio)));
    }
    upper_neighbor_mask_bits_ =
        std::ceil(std::log2(upper_neighbor_chunk_size_ / upper_neighbor_size_));
    upper_neighbor_mask_ = (1 << upper_neighbor_mask_bits_) - 1;

    size_t max_node_chunk_cnt = std::ceil(max_index_size_ / chunk_size_);
    size_t max_upper_chunk_cnt = std::ceil(
        (max_node_chunk_cnt * node_cnt_per_chunk_ * 1.0f / scaling_factor()) /
        (upper_neighbor_chunk_size_ / upper_neighbor_size_));
    max_upper_chunk_cnt =
        max_upper_chunk_cnt + std::ceil(max_upper_chunk_cnt / scaling_factor());

    //! reserve space to avoid memmove in chunks vector emplace chunk, so
    //! as to lock-free in reading chunk
    node_chunks_.reserve(max_node_chunk_cnt);
    upper_neighbor_chunks_.reserve(max_upper_chunk_cnt);

    LOG_DEBUG(
        "Settings: nodeSize=%zu chunkSize=%u upperNeighborSize=%u "
        "upperNeighborChunkSize=%u "
        "nodeCntPerChunk=%u maxChunkCnt=%zu maxNeighborChunkCnt=%zu "
        "maxIndexSize=%zu ratio=%.3f",
        node_size(), chunk_size_, upper_neighbor_size_,
        upper_neighbor_chunk_size_, node_cnt_per_chunk_, max_node_chunk_cnt,
        max_upper_chunk_cnt, max_index_size_, ratio);

    return 0;
  }

  //! Init node chunk and neighbor chunks
  int init_chunks(const Chunk::Pointer &header_chunk);

  int flush_header(void) {
    if (!broker_->dirty()) {
      // do not need to flush
      return 0;
    }
    auto header_chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                                           ChunkBroker::kDefaultChunkSeqId);
    if (ailego_unlikely(!header_chunk)) {
      LOG_ERROR("get header chunk failed");
      return IndexError_Runtime;
    }
    size_t size = header_chunk->write(0UL, &header(), header_size());
    if (ailego_unlikely(size != header_size())) {
      LOG_ERROR("Write header chunk failed");
      return IndexError_WriteData;
    }

    return 0;
  }

 private:
  HnswStreamerBenchEntity(const HnswStreamerBenchEntity &) = delete;
  HnswStreamerBenchEntity &operator=(const HnswStreamerBenchEntity &) = delete;
  static constexpr float kUpperHashMemoryInflateRatio = 2.0f;

 private:
  IndexStreamer::Stats &stats_;
  HNSWHeader base_header_{};
  HNSWHeader header_{};
  std::mutex mutex_{};
  size_t max_index_size_{0UL};
  uint32_t chunk_size_{kDefaultChunkSize};
  uint32_t upper_neighbor_chunk_size_{kDefaultChunkSize};
  uint32_t node_index_mask_bits_{0U};
  uint32_t node_cnt_per_chunk_{0U};
  uint32_t node_index_mask_{0U};
  uint32_t neighbor_size_{0U};
  uint32_t upper_neighbor_size_{0U};
  //! UpperNeighborIndex.index composite chunkIdx and offset in chunk by the
  //! following mask
  uint32_t upper_neighbor_mask_bits_{0U};
  uint32_t upper_neighbor_mask_{0U};
  bool filter_same_key_{false};
  bool get_vector_enabled_{false};
  bool use_key_info_map_{true};

  NIHashMapPointer upper_neighbor_index_{};

  mutable std::shared_ptr<ailego::SharedMutex> keys_map_lock_{};
  HashMapPointer<key_t, node_id_t> keys_map_{};

  //! the chunks will be changed in searcher, so need mutable
  //! data chunk include: vector, key, level 0 neighbors
  mutable std::vector<Chunk::Pointer> node_chunks_{};

  //! upper neighbor chunk inlude: UpperNeighborHeader + (1~level) neighbors
  mutable std::vector<Chunk::Pointer> upper_neighbor_chunks_{};

  ChunkBroker::Pointer broker_{};  // chunk broker

  std::shared_ptr<std::string> vector_value_ptr_{};
  std::shared_ptr<std::string> neighbors_value_ptr_{};
};

}  // namespace core
}  // namespace zvec
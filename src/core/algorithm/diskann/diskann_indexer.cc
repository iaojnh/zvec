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

#include "diskann_indexer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "diskann_buffer_pool_shim.h"
#include "diskann_params.h"

namespace zvec {
namespace core {

DiskAnnIndexer::DiskAnnIndexer(const IndexMeta &meta) {
  meta_ = meta;
}

DiskAnnIndexer::~DiskAnnIndexer() {
  destroy_io_ctx(init_ctx_);
  if (centroid_data_) {
    free(centroid_data_);
  }
  coord_cache_.clear();
  neighbor_cache_.clear();
  std::vector<diskann_id_t>().swap(neighbor_cache_buffer_);
  DiskAnnUtil::free_aligned(coord_cache_buf_);
  coord_cache_buf_ = nullptr;
  if (cache_memory_charge_bytes_ != 0) {
    ailego::MemoryLimitPool::get_instance().release_external(
        cache_memory_charge_bytes_);
    cache_memory_charge_bytes_ = 0;
  }
}

int DiskAnnIndexer::init(DiskAnnSearcherEntity &entity) {
  entity_ = &entity;

  storage_ = entity.get_storage();
  auto vector_segment = entity.get_vector_segment();

  pq_table_ = entity.get_pq_table();

  index_segment_offset_ = vector_segment->data_offset();

  if (auto *pool = storage_->vec_buffer_pool()) {
    // BufferPool backend: route every sector read through the VecBufferPool
    // paged cache.  The reader holds a raw pool pointer, so the storage (and
    // thus the pool) must outlive this indexer -- keep storage_ alive and do
    // NOT cleanup here.
    buffer_pool_ = pool;
    reader_ = std::make_shared<BufferPoolAlignedFileReader>(pool);
    reader_->open(storage_->file_path());
  } else {
    // Direct backend (FileReadStorage): open our own O_DIRECT + libaio reader
    // and release the storage as before.
    reader_ = std::make_shared<LinuxAlignedFileReader>();
    reader_->open(storage_->file_path());
    storage_->cleanup();
    storage_.reset();
  }

  int ret = setup_io_ctx(init_ctx_);
  if (ret != 0) {
    LOG_ERROR("setup io ctx error");
    return ret;
  }

  max_node_size_ = entity.max_node_size();
  disk_bytes_per_point_ = meta_.element_size();

  node_per_sector_ = entity.node_per_sector();
  aligned_dim_ = meta_.dimension();

  pq_chunk_num_ = entity.pq_chunk_num();

  medoid_ = entity.medoid();

  entrypoints_.push_back(medoid_);
  auto &entrypoints = entity.entrypoints();
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    entrypoints_.push_back(entrypoints[i]);
  }

  doc_cnt_ = entity.doc_cnt();

  max_degree_ = entity.max_degree();

  sector_num_per_node_ =
      DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);
  if (beam_width_ > sector_num_per_node_ * DiskAnnUtil::kMaxSectorReadNum) {
    LOG_ERROR("Beamwidth can not be higher than kMaxSectorReadNum");

    return IndexError_InvalidArgument;
  }

  DiskAnnUtil::alloc_aligned((void **)(&centroid_data_),
                             entrypoints_.size() * aligned_dim_ * sizeof(float),
                             32);

  use_medroids_data_as_centroids();

  return 0;
}

int DiskAnnIndexer::use_medroids_data_as_centroids() {
  LOG_INFO("Loading centroid data from medoid vector data");

  std::vector<diskann_id_t> nodes_to_read;
  std::vector<void *> medoid_bufs;
  std::vector<std::pair<uint32_t, diskann_id_t *>> neighbor_bufs;

  std::vector<float> centroid_buffer;

  size_t dim = meta_.dimension();
  centroid_buffer.resize(dim);

  nodes_to_read.push_back(medoid_);
  medoid_bufs.push_back(&(centroid_buffer[0]));
  neighbor_bufs.emplace_back(0, nullptr);

  auto read_status = read_nodes(nodes_to_read, medoid_bufs, neighbor_bufs);

  if (read_status[0] == true) {
    for (uint32_t i = 0; i < dim; i++) centroid_data_[i] = centroid_buffer[i];
  } else {
    LOG_ERROR("Failed to read medoid");
    return IndexError_Runtime;
  }

  return 0;
}

diskann_key_t DiskAnnIndexer::get_key(diskann_id_t id) const {
  return entity_->get_key(id);
}

diskann_id_t DiskAnnIndexer::get_id(diskann_key_t key) const {
  return entity_->get_id(key);
}

std::vector<bool> DiskAnnIndexer::read_nodes(
    const std::vector<diskann_id_t> &node_ids,
    std::vector<void *> &coord_buffers,
    std::vector<std::pair<uint32_t, diskann_id_t *>> &neighbor_buffers) {
  std::vector<AlignedRead> read_reqs;
  std::vector<bool> retval(node_ids.size(), true);

  uint8_t *buf = nullptr;
  auto sector_num =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);
  DiskAnnUtil::alloc_aligned(
      (void **)&buf, node_ids.size() * sector_num * DiskAnnUtil::kSectorSize,
      DiskAnnUtil::kSectorSize);

  for (size_t i = 0; i < node_ids.size(); ++i) {
    auto node_id = node_ids[i];

    AlignedRead read;
    read.len = sector_num * DiskAnnUtil::kSectorSize;
    read.buf = buf + i * sector_num * DiskAnnUtil::kSectorSize;
    read.offset =
        index_segment_offset_ +
        DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                     DiskAnnUtil::kSectorSize, node_id) *
            DiskAnnUtil::kSectorSize;
    read_reqs.push_back(read);
  }

  int read_ret = reader_->read(read_reqs, init_ctx_);
  if (read_ret != 0) {
    LOG_ERROR("read_nodes: reader_->read failed, ret=%d", read_ret);
    for (size_t i = 0; i < retval.size(); i++) {
      retval[i] = false;
    }
    DiskAnnUtil::free_aligned(buf);
    return retval;
  }

  for (uint32_t i = 0; i < read_reqs.size(); i++) {
    uint8_t *node_buf =
        DiskAnnUtil::offset_to_node(node_per_sector_, max_node_size_,
                                    (uint8_t *)read_reqs[i].buf, node_ids[i]);

    if (coord_buffers[i] != nullptr) {
      void *node_coords = node_buf;
      memcpy(coord_buffers[i], node_coords, disk_bytes_per_point_);
    }

    if (neighbor_buffers[i].second != nullptr) {
      uint32_t *node_neighbor =
          DiskAnnUtil::offset_to_node_neighbor(node_buf, meta_.element_size());
      uint32_t neighbor_num = *node_neighbor;

      neighbor_buffers[i].first = neighbor_num;
      memcpy(neighbor_buffers[i].second, node_neighbor + 1,
             neighbor_num * sizeof(diskann_id_t));
    }
  }

  DiskAnnUtil::free_aligned(buf);

  return retval;
}

int DiskAnnIndexer::load_cache_list(
    const std::vector<diskann_id_t> &node_list) {
  LOG_INFO("Loading the cache list into memory");

  size_t num_cached_nodes = node_list.size();
  int reserve_ret = reserve_cache_memory(num_cached_nodes);
  if (reserve_ret != 0) {
    return reserve_ret;
  }

  neighbor_cache_buffer_.resize(num_cached_nodes * (max_degree_ + 1), 0);

  size_t coord_cache_buf_len = num_cached_nodes * aligned_dim_;
  DiskAnnUtil::alloc_aligned((void **)&coord_cache_buf_,
                             coord_cache_buf_len * meta_.unit_size(),
                             8 * meta_.unit_size());

  memset(coord_cache_buf_, 0, coord_cache_buf_len * meta_.unit_size());
  allocated_cache_payload_bytes_ =
      static_cast<uint64_t>(num_cached_nodes) *
      cache_payload_bytes_per_node();

  constexpr size_t BLOCK_SIZE = 8;
  size_t num_blocks = DiskAnnUtil::div_round_up(num_cached_nodes, BLOCK_SIZE);
  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_idx = block * BLOCK_SIZE;
    size_t end_idx = std::min(num_cached_nodes, (block + 1) * BLOCK_SIZE);

    std::vector<diskann_id_t> nodes_to_read;
    std::vector<void *> coord_buffers;
    std::vector<std::pair<uint32_t, diskann_id_t *>> neighbor_buffers;
    for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++) {
      nodes_to_read.push_back(node_list[node_idx]);
      coord_buffers.push_back(reinterpret_cast<uint8_t *>(coord_cache_buf_) +
                              node_idx * meta_.element_size());
      neighbor_buffers.emplace_back(
          0, neighbor_cache_buffer_.data() + node_idx * (max_degree_ + 1));
    }

    auto read_status =
        read_nodes(nodes_to_read, coord_buffers, neighbor_buffers);

    for (size_t i = 0; i < read_status.size(); i++) {
      if (read_status[i] == true) {
        coord_cache_.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
        neighbor_cache_.insert(
            std::make_pair(nodes_to_read[i], neighbor_buffers[i]));
      }
    }
  }

  LOG_INFO(
      "Load Cache List Done: nodes=%zu payload_bytes=%llu "
      "estimated_bytes=%llu",
      num_cached_nodes,
      static_cast<unsigned long long>(allocated_cache_payload_bytes_),
      static_cast<unsigned long long>(
          static_cast<uint64_t>(num_cached_nodes) *
          cache_estimated_bytes_per_node()));

  return 0;
}

int DiskAnnIndexer::reserve_cache_memory(uint64_t node_count) {
  const uint64_t estimated_bytes_per_node =
      cache_estimated_bytes_per_node();
  if (estimated_bytes_per_node != 0 &&
      node_count > std::numeric_limits<uint64_t>::max() /
                       estimated_bytes_per_node) {
    LOG_ERROR("DiskANN node cache size overflow: nodes=%llu bytes_per_node=%llu",
              static_cast<unsigned long long>(node_count),
              static_cast<unsigned long long>(estimated_bytes_per_node));
    return IndexError_NoMemory;
  }
  const uint64_t estimated_cache_bytes =
      node_count * estimated_bytes_per_node;
  if (estimated_cache_bytes > std::numeric_limits<size_t>::max()) {
    LOG_ERROR("DiskANN node cache does not fit the platform address space");
    return IndexError_NoMemory;
  }
  auto &memory_pool = ailego::MemoryLimitPool::get_instance();
  if (memory_pool.capacity() == 0) {
    return 0;
  }
  if (estimated_cache_bytes == cache_memory_charge_bytes_) {
    return 0;
  }

  if (estimated_cache_bytes > cache_memory_charge_bytes_) {
    const uint64_t additional_charge =
        estimated_cache_bytes - cache_memory_charge_bytes_;
    if (!memory_pool.try_charge_external(additional_charge)) {
      LOG_ERROR(
          "DiskANN node cache exceeds the shared memory budget: "
          "requested=%llu available=%llu capacity=%llu",
          static_cast<unsigned long long>(estimated_cache_bytes),
          static_cast<unsigned long long>(memory_pool.available()),
          static_cast<unsigned long long>(memory_pool.capacity()));
      return IndexError_NoMemory;
    }
  } else if (estimated_cache_bytes < cache_memory_charge_bytes_) {
    memory_pool.release_external(cache_memory_charge_bytes_ -
                                 estimated_cache_bytes);
  }
  cache_memory_charge_bytes_ = estimated_cache_bytes;

  LOG_INFO(
      "DiskANN shared cache budget: node_cache_charge_bytes=%llu "
      "memory_limit_used=%llu memory_limit_available=%llu "
      "memory_limit_capacity=%llu",
      static_cast<unsigned long long>(cache_memory_charge_bytes_),
      static_cast<unsigned long long>(memory_pool.used()),
      static_cast<unsigned long long>(memory_pool.available()),
      static_cast<unsigned long long>(memory_pool.capacity()));
  return 0;
}

void DiskAnnIndexer::manage_cache_page_overlap(
    const std::vector<diskann_id_t> &node_list, bool evict) {
  if (buffer_pool_ == nullptr || node_list.empty()) {
    return;
  }

  const uint64_t sector_num =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_,
                                      DiskAnnUtil::kSectorSize);
  std::vector<uint64_t> page_ids;
  page_ids.reserve(node_list.size() * sector_num);
  for (diskann_id_t node_id : node_list) {
    const uint64_t first_page =
        index_segment_offset_ / DiskAnnUtil::kSectorSize +
        DiskAnnUtil::get_node_sector(
            node_per_sector_, max_node_size_, DiskAnnUtil::kSectorSize,
            node_id);
    for (uint64_t i = 0; i < sector_num; ++i) {
      page_ids.push_back(first_page + i);
    }
  }

  const auto stats = diskann_buffer_pool_manage_overlap(
      buffer_pool_, page_ids.data(), page_ids.size(), evict);
  LOG_INFO(
      "DiskANN L1/L2 overlap: policy=%s unique_source_pages=%llu "
      "resident_before=%llu duplicate_bytes_before=%llu evicted=%llu "
      "resident_after=%llu",
      evict ? "evict" : "keep",
      static_cast<unsigned long long>(stats.unique_pages),
      static_cast<unsigned long long>(stats.resident_pages_before),
      static_cast<unsigned long long>(
          stats.resident_pages_before * DiskAnnUtil::kSectorSize),
      static_cast<unsigned long long>(stats.evicted_pages),
      static_cast<unsigned long long>(stats.resident_pages_after));
}

int DiskAnnIndexer::load_node_list(
    const std::string &path, uint32_t node_limit,
    std::vector<diskann_id_t> &node_list) const {
  std::ifstream input(path);
  if (!input.is_open()) {
    LOG_ERROR("Failed to open DiskANN cache node list: %s", path.c_str());
    return IndexError_InvalidArgument;
  }

  node_list.clear();
  std::unordered_set<diskann_id_t> seen;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line) &&
         (node_limit == 0 || node_list.size() < node_limit)) {
    ++line_number;
    const size_t comment = line.find('#');
    if (comment != std::string::npos) {
      line.resize(comment);
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream fields(line);
    std::string token;
    while (fields >> token &&
           (node_limit == 0 || node_list.size() < node_limit)) {
      size_t parsed = 0;
      uint64_t value = 0;
      try {
        value = std::stoull(token, &parsed);
      } catch (const std::exception &) {
        LOG_ERROR("Invalid node id in %s at line %zu: %s", path.c_str(),
                  line_number, token.c_str());
        return IndexError_InvalidArgument;
      }
      if (parsed != token.size() || value >= doc_cnt_ ||
          value > std::numeric_limits<diskann_id_t>::max()) {
        LOG_ERROR("Out-of-range node id in %s at line %zu: %s", path.c_str(),
                  line_number, token.c_str());
        return IndexError_InvalidArgument;
      }
      const auto node_id = static_cast<diskann_id_t>(value);
      if (seen.insert(node_id).second) {
        node_list.push_back(node_id);
      }
    }
  }

  if (node_list.empty()) {
    LOG_ERROR("DiskANN cache node list is empty: %s", path.c_str());
    return IndexError_InvalidArgument;
  }
  LOG_INFO("Loaded DiskANN cache node list: path=%s nodes=%zu", path.c_str(),
           node_list.size());
  return 0;
}

void DiskAnnIndexer::warmup_node_pages(
    const std::vector<diskann_id_t> &node_list,
    const std::unordered_set<diskann_id_t> &excluded_nodes) {
  if (buffer_pool_ == nullptr || node_list.empty()) {
    return;
  }

  const uint64_t sector_num =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_,
                                      DiskAnnUtil::kSectorSize);
  std::vector<uint64_t> page_ids;
  page_ids.reserve(node_list.size() * sector_num);
  for (diskann_id_t node_id : node_list) {
    if (excluded_nodes.find(node_id) != excluded_nodes.end()) {
      continue;
    }
    const uint64_t first_page =
        index_segment_offset_ / DiskAnnUtil::kSectorSize +
        DiskAnnUtil::get_node_sector(
            node_per_sector_, max_node_size_, DiskAnnUtil::kSectorSize,
            node_id);
    for (uint64_t page = 0; page < sector_num; ++page) {
      page_ids.push_back(first_page + page);
    }
  }
  if (page_ids.empty()) {
    return;
  }

  std::sort(page_ids.begin(), page_ids.end());
  page_ids.erase(std::unique(page_ids.begin(), page_ids.end()),
                 page_ids.end());
  size_t resident_before = 0;
  for (uint64_t page_id : page_ids) {
    if (buffer_pool_->is_page_resident(page_id)) {
      ++resident_before;
    }
    buffer_pool_->set_page_priority(
        page_id, ailego::VecBufferPool::kHighPriority);
  }

  size_t cursor = 0;
  while (cursor < page_ids.size()) {
    size_t end = cursor + 1;
    while (end < page_ids.size() &&
           page_ids[end] == page_ids[end - 1] + 1) {
      ++end;
    }
    buffer_pool_->prefetch_pages(
        page_ids[cursor], end - cursor,
        ailego::VecBufferPool::kHighPriority);
    cursor = end;
  }

  size_t resident_after = 0;
  for (uint64_t page_id : page_ids) {
    if (buffer_pool_->is_page_resident(page_id)) {
      ++resident_after;
    }
  }
  LOG_INFO(
      "DiskANN semantic warmup: requested_nodes=%zu excluded_nodes=%zu "
      "unique_pages=%zu resident_before=%zu resident_after=%zu priority=high",
      node_list.size(), excluded_nodes.size(), page_ids.size(),
      resident_before, resident_after);
}

int DiskAnnIndexer::configure_cache(
    uint32_t cache_node_num, uint64_t cache_node_budget_bytes,
    const std::string &cache_node_list_path,
    const std::string &cache_node_page_policy,
    const std::string &warmup_mode, uint32_t warmup_node_num) {
  if (cache_node_budget_bytes != 0) {
    cache_node_num = cache_node_count_for_budget(cache_node_budget_bytes);
    LOG_INFO(
        "DiskANN node-cache budget: budget_bytes=%llu "
        "estimated_bytes_per_node=%llu payload_bytes_per_node=%llu "
        "resolved_nodes=%u",
        static_cast<unsigned long long>(cache_node_budget_bytes),
        static_cast<unsigned long long>(cache_estimated_bytes_per_node()),
        static_cast<unsigned long long>(cache_payload_bytes_per_node()),
        cache_node_num);
  }

  std::vector<diskann_id_t> cached_nodes;
  if (cache_node_num != 0) {
    if (cache_node_budget_bytes != 0) {
      int ret = reserve_cache_memory(cache_node_num);
      if (ret != 0) {
        return ret;
      }
    }
    if (!cache_node_list_path.empty()) {
      int ret =
          load_node_list(cache_node_list_path, cache_node_num, cached_nodes);
      if (ret != 0) {
        return ret;
      }
    } else {
      cache_bfs_levels(cache_node_num, cached_nodes,
                       cache_node_budget_bytes == 0);
    }

    int ret = load_cache_list(cached_nodes);
    if (ret != 0) {
      return ret;
    }
    manage_cache_page_overlap(
        cached_nodes,
        cache_node_page_policy == DISKANN_CACHE_NODE_PAGE_POLICY_EVICT);
  }

  if (warmup_mode == DISKANN_WARMUP_MODE_NONE) {
    return 0;
  }
  if (buffer_pool_ == nullptr) {
    LOG_WARN("DiskANN semantic warmup skipped without BufferReadStorage");
    return 0;
  }

  std::vector<diskann_id_t> warmup_nodes;
  if (warmup_mode == DISKANN_WARMUP_MODE_BFS) {
    cache_bfs_levels(warmup_node_num, warmup_nodes,
                     /*enforce_legacy_ten_percent_cap=*/false);
  } else {
    int ret =
        load_node_list(cache_node_list_path, warmup_node_num, warmup_nodes);
    if (ret != 0) {
      return ret;
    }
  }

  std::unordered_set<diskann_id_t> excluded_nodes;
  if (cache_node_page_policy == DISKANN_CACHE_NODE_PAGE_POLICY_EVICT) {
    excluded_nodes.insert(cached_nodes.begin(), cached_nodes.end());
  }
  warmup_node_pages(warmup_nodes, excluded_nodes);
  return 0;
}

void DiskAnnIndexer::cache_bfs_levels(uint64_t num_nodes_to_cache,
                                      std::vector<diskann_id_t> &node_list,
                                      bool enforce_legacy_ten_percent_cap) {
  std::set<diskann_id_t> node_set;

  size_t tenp_cnt = static_cast<uint64_t>(std::round(doc_cnt_ * 0.1));
  if (enforce_legacy_ten_percent_cap && num_nodes_to_cache > tenp_cnt) {
    LOG_WARN(
        "Reducing nodes to cache from: %zu, to: (10 percent of total nodes: "
        "%zu)",
        (size_t)num_nodes_to_cache, (size_t)tenp_cnt);

    num_nodes_to_cache = tenp_cnt == 0 ? 1 : tenp_cnt;
  }

  LOG_INFO("Begin to cache %zu Nodes", (size_t)num_nodes_to_cache);

  std::unordered_set<diskann_id_t> cur_level;
  std::unordered_set<diskann_id_t> prev_level;

  for (uint64_t iter = 0;
       iter < entrypoints_.size() && cur_level.size() < num_nodes_to_cache;
       iter++) {
    cur_level.insert(entrypoints_[iter]);
  }

  uint64_t level = 1;
  uint64_t prev_node_set_size = 0;
  while ((node_set.size() + cur_level.size() < num_nodes_to_cache) &&
         cur_level.size() != 0) {
    prev_level.swap(cur_level);

    cur_level.clear();

    std::vector<diskann_id_t> nodes_to_expand;
    nodes_to_expand.reserve(prev_level.size());

    for (const diskann_id_t &id : prev_level) {
      if (node_set.find(id) != node_set.end()) {
        continue;
      }

      node_set.insert(id);
      nodes_to_expand.push_back(id);
    }

    std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

    bool finish_flag = false;

    constexpr uint64_t BLOCK_SIZE = 1024;
    uint64_t nblocks =
        DiskAnnUtil::div_round_up(nodes_to_expand.size(), BLOCK_SIZE);
    for (size_t block = 0; block < nblocks && !finish_flag; block++) {
      size_t start = block * BLOCK_SIZE;
      size_t end = std::min((uint64_t)((block + 1) * BLOCK_SIZE),
                            (uint64_t)(nodes_to_expand.size()));
      const size_t block_size = end - start;

      std::vector<diskann_id_t> nodes_to_read(nodes_to_expand.begin() + start,
                                              nodes_to_expand.begin() + end);
      std::vector<void *> coord_buffers(block_size, nullptr);

      std::vector<std::pair<uint32_t, std::vector<diskann_id_t>>>
          neighbor_buffers;
      neighbor_buffers.reserve(block_size);

      for (size_t i = 0; i < block_size; i++) {
        neighbor_buffers.emplace_back(
            0, std::vector<diskann_id_t>(max_degree_ + 1));
      }

      std::vector<std::pair<uint32_t, diskann_id_t *>> neighbor_buffers_ptr;
      neighbor_buffers_ptr.reserve(block_size);
      for (size_t i = 0; i < block_size; i++) {
        neighbor_buffers_ptr.emplace_back(neighbor_buffers[i].first,
                                          neighbor_buffers[i].second.data());
      }

      auto read_status =
          read_nodes(nodes_to_read, coord_buffers, neighbor_buffers_ptr);

      for (uint32_t i = 0; i < read_status.size(); i++) {
        if (read_status[i] == false) {
          continue;
        } else {
          neighbor_buffers[i].first = neighbor_buffers_ptr[i].first;
          uint32_t neighbor_num = neighbor_buffers[i].first;
          diskann_id_t *neighbors = neighbor_buffers[i].second.data();

          for (uint32_t j = 0; j < neighbor_num && !finish_flag; j++) {
            if (node_set.find(neighbors[j]) == node_set.end()) {
              cur_level.insert(neighbors[j]);
            }
            if (cur_level.size() + node_set.size() >= num_nodes_to_cache) {
              finish_flag = true;
            }
          }
        }
      }
    }

    size_t total_size = node_set.size();

    LOG_INFO("Level: %zu, Cached Size: %zu, Total Cached Size: %zu",
             (size_t)level, (size_t)(total_size - prev_node_set_size),
             total_size);

    prev_node_set_size = total_size;
    level++;
  }

  ailego_assert(node_set.size() + cur_level.size() == num_nodes_to_cache ||
                cur_level.size() == 0);

  node_list.clear();
  node_list.reserve(node_set.size() + cur_level.size());

  for (auto node : node_set) {
    node_list.push_back(node);
  }

  for (auto node : cur_level) {
    node_list.push_back(node);
  }

  size_t total_size = node_list.size();
  LOG_INFO("Level: %zu, Cached Size: %zu, Total Cached Size: %zu",
           (size_t)level, (size_t)(total_size - prev_node_set_size),
           (size_t)total_size);

  return;
}

int DiskAnnIndexer::linear_search(DiskAnnContext *ctx) {
  auto &stats = ctx->query_stats();
  auto &dc = ctx->dist_calculator();
  auto &topk_heap = ctx->topk_heap();

  topk_heap.clear();

  IOContext &io_ctx = ctx->io_ctx();
  void *aligned_query_raw = ctx->query();

  void *data_buf = reinterpret_cast<void *>(ctx->coord_buffer());

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);

  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime cpu_timer;

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * beam_width_);

  std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * beam_width_);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * beam_width_);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * beam_width_);

  uint64_t sector_buffer_idx = 0;

  diskann_id_t id = 0;
  while (id < doc_cnt_) {
    while (frontier.size() < beam_width_) {
      if (!ctx->filter().is_valid() || !ctx->filter()(get_key(id))) {
        auto iter = neighbor_cache_.find(id);
        if (iter != neighbor_cache_.end()) {
          cached_neighbors.push_back(
              std::make_tuple(id, iter->second.first, iter->second.second));
          stats.cache_hits++;
        } else {
          frontier.push_back(id);
        }
      }

      id++;
      if (id >= doc_cnt_) {
        break;
      }
    }

    if (!frontier.empty()) {
      for (uint64_t i = 0; i < frontier.size(); i++) {
        diskann_id_t cur_id = frontier[i];

        std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
        frontier_neighbor.first = cur_id;
        frontier_neighbor.second = sector_buffer + sector_num_per_node *
                                                       sector_buffer_idx *
                                                       DiskAnnUtil::kSectorSize;
        frontier_neighbors.push_back(frontier_neighbor);

        sector_buffer_idx++;

        frontier_read_reqs.emplace_back(
            index_segment_offset_ +
                DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                             DiskAnnUtil::kSectorSize, cur_id) *
                    DiskAnnUtil::kSectorSize,
            sector_num_per_node * DiskAnnUtil::kSectorSize,
            frontier_neighbor.second);

        stats.disk_page_reads++;
        stats.io_num++;
      }

      io_timer.reset();

      int read_ret = reader_->read(frontier_read_reqs, io_ctx);
      stats.io_us += io_timer.micro_seconds();
      if (read_ret != 0) {
        LOG_ERROR("linear_search: reader_->read failed, ret=%d", read_ret);
        ctx->set_error(true);
        return IndexError_Runtime;
      }
    }

    for (auto &cached_neighbor : cached_neighbors) {
      auto global_cache_iter = coord_cache_.find(std::get<0>(cached_neighbor));
      void *node_fp_coords_copy = global_cache_iter->second;

      float cur_expanded_dist = dc.dist(aligned_query_raw, node_fp_coords_copy);

      topk_heap.emplace(
          std::get<0>(cached_neighbor),
          VectorInfo(cur_expanded_dist, make_vector_copy(node_fp_coords_copy)));
    }

    for (auto &frontier_neighbor : frontier_neighbors) {
      uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
          node_per_sector_, max_node_size_, frontier_neighbor.second,
          frontier_neighbor.first);

      void *node_fp_coords = node_disk_buf;
      memcpy(data_buf, node_fp_coords, disk_bytes_per_point_);

      float cur_expanded_dist = dc.dist(aligned_query_raw, data_buf);

      topk_heap.emplace(
          frontier_neighbor.first,
          VectorInfo(cur_expanded_dist, make_vector_copy(data_buf)));

      stats.cpu_us += cpu_timer.micro_seconds();
    }

    frontier.clear();
    frontier_neighbors.clear();
    frontier_read_reqs.clear();
    cached_neighbors.clear();
    sector_buffer_idx = 0;
  }

  stats.total_us += query_timer.micro_seconds();

  return 0;
}

int DiskAnnIndexer::keys_search(const std::vector<uint64_t> &keys,
                                DiskAnnContext *ctx) {
  auto &stats = ctx->query_stats();
  auto &dc = ctx->dist_calculator();
  auto &topk_heap = ctx->topk_heap();

  topk_heap.clear();

  IOContext &io_ctx = ctx->io_ctx();
  void *aligned_query_raw = ctx->query();

  void *data_buf = reinterpret_cast<void *>(ctx->coord_buffer());

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);

  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime cpu_timer;

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * beam_width_);

  std::vector<std::pair<uint32_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * beam_width_);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * beam_width_);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * beam_width_);

  uint64_t sector_buffer_idx = 0;

  size_t idx = 0;
  while (idx < keys.size()) {
    while (frontier.size() < beam_width_) {
      if (!ctx->filter().is_valid() || !ctx->filter()(keys[idx])) {
        diskann_id_t id = get_id(keys[idx]);

        auto iter = neighbor_cache_.find(id);
        if (iter != neighbor_cache_.end()) {
          cached_neighbors.push_back(
              std::make_tuple(id, iter->second.first, iter->second.second));
          stats.cache_hits++;
        } else {
          frontier.push_back(id);
        }
      }

      idx++;
      if (idx >= keys.size()) {
        break;
      }
    }

    if (!frontier.empty()) {
      for (uint64_t i = 0; i < frontier.size(); i++) {
        diskann_id_t cur_id = frontier[i];

        std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
        frontier_neighbor.first = cur_id;
        frontier_neighbor.second = sector_buffer + sector_num_per_node *
                                                       sector_buffer_idx *
                                                       DiskAnnUtil::kSectorSize;
        frontier_neighbors.push_back(frontier_neighbor);

        sector_buffer_idx++;

        frontier_read_reqs.emplace_back(
            index_segment_offset_ +
                DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                             DiskAnnUtil::kSectorSize, cur_id) *
                    DiskAnnUtil::kSectorSize,
            sector_num_per_node * DiskAnnUtil::kSectorSize,
            frontier_neighbor.second);

        stats.disk_page_reads++;
        stats.io_num++;
      }

      io_timer.reset();

      int read_ret = reader_->read(frontier_read_reqs, io_ctx);
      stats.io_us += io_timer.micro_seconds();
      if (read_ret != 0) {
        LOG_ERROR("keys_search: reader_->read failed, ret=%d", read_ret);
        ctx->set_error(true);
        return IndexError_Runtime;
      }
    }

    for (auto &cached_neighbor : cached_neighbors) {
      auto global_cache_iter = coord_cache_.find(std::get<0>(cached_neighbor));
      void *node_fp_coords_copy = global_cache_iter->second;

      float cur_expanded_dist = dc.dist(aligned_query_raw, node_fp_coords_copy);

      topk_heap.emplace(
          std::get<0>(cached_neighbor),
          VectorInfo(cur_expanded_dist, make_vector_copy(node_fp_coords_copy)));
    }

    for (auto &frontier_neighbor : frontier_neighbors) {
      uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
          node_per_sector_, max_node_size_, frontier_neighbor.second,
          frontier_neighbor.first);

      void *node_fp_coords = node_disk_buf;
      memcpy(data_buf, node_fp_coords, disk_bytes_per_point_);

      float cur_expanded_dist = dc.dist(aligned_query_raw, data_buf);

      topk_heap.emplace(
          frontier_neighbor.first,
          VectorInfo(cur_expanded_dist, make_vector_copy(data_buf)));

      stats.cpu_us += cpu_timer.micro_seconds();
    }

    frontier.clear();
    frontier_neighbors.clear();
    frontier_read_reqs.clear();
    cached_neighbors.clear();
    sector_buffer_idx = 0;
  }

  stats.total_us += query_timer.micro_seconds();

  return 0;
}

int DiskAnnIndexer::get_vector(diskann_id_t id, IndexContext::Pointer &context,
                               std::string &vector) {
  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());

  auto &stats = ctx->query_stats();

  IOContext &io_ctx = ctx->io_ctx();

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);

  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime cpu_timer;

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * beam_width_);

  std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * beam_width_);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * beam_width_);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * beam_width_);

  auto iter = neighbor_cache_.find(id);
  if (iter != neighbor_cache_.end()) {
    void *node_fp_coords_copy = iter->second.second;

    vector.resize(meta_.element_size());
    ::memcpy(&(vector[0]), node_fp_coords_copy, meta_.element_size());

    return 0;
  } else {
    std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
    frontier_neighbor.first = id;
    frontier_neighbor.second = sector_buffer;
    frontier_neighbors.push_back(frontier_neighbor);

    frontier_read_reqs.emplace_back(
        index_segment_offset_ +
            DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                         DiskAnnUtil::kSectorSize, id) *
                DiskAnnUtil::kSectorSize,
        sector_num_per_node * DiskAnnUtil::kSectorSize,
        frontier_neighbor.second);

    stats.disk_page_reads++;
    stats.io_num++;

    io_timer.reset();

    reader_->read(frontier_read_reqs, io_ctx);
    stats.io_us += io_timer.micro_seconds();

    uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
        node_per_sector_, max_node_size_, frontier_neighbor.second, id);

    void *node_fp_coords = node_disk_buf;

    vector.resize(meta_.element_size());
    ::memcpy(&(vector[0]), node_fp_coords, meta_.element_size());

    stats.cpu_us += cpu_timer.micro_seconds();
  }

  return 0;
}

int DiskAnnIndexer::knn_search(DiskAnnContext *ctx) {
  int ret = cached_beam_search(ctx);
  if (ret != 0) {
    return ret;
  }

  if (ctx->group_by_search()) {
    ret = cached_beam_search_by_group(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

int DiskAnnIndexer::cached_beam_search(DiskAnnContext *ctx) {
  auto &stats = ctx->query_stats();
  auto &dc = ctx->dist_calculator();
  auto &topk_heap = ctx->topk_heap();
  auto &visit_filter = ctx->visit_filter();

  topk_heap.clear();

  IOContext &io_ctx = ctx->io_ctx();

  uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

  const uint64_t sector_num_per_node =
      node_per_sector_ > 0
          ? 1
          : DiskAnnUtil::div_round_up(max_node_size_, DiskAnnUtil::kSectorSize);

  pq_table_->preprocess_pq_dist_table(ctx->query_rotated(),
                                      ctx->pq_table_dist_buffer());

  ailego::ElapsedTime query_timer;
  ailego::ElapsedTime io_timer;
  ailego::ElapsedTime cpu_timer;

  NeighborPriorityQueue candidates;

  candidates.reserve(ctx->list_size());

  diskann_id_t best_medoid = 0;
  float best_dist = (std::numeric_limits<float>::max)();
  for (uint64_t cur_m = 0; cur_m < entrypoints_.size(); cur_m++) {
    float cur_expanded_dist =
        dc.dist(ctx->query(), centroid_data_ + aligned_dim_ * cur_m);

    if (cur_expanded_dist < best_dist) {
      best_medoid = entrypoints_[cur_m];
      best_dist = cur_expanded_dist;
    }
  }

  float dist;
  pq_table_->compute_dists(1, &best_medoid, pq_chunk_num_,
                           ctx->pq_table_dist_buffer(), ctx->pq_coord_buffer(),
                           &dist);
  candidates.insert(Neighbor(best_medoid, dist));
  visit_filter.set_visited(best_medoid);

  uint32_t num_ios = 0;

  std::vector<diskann_id_t> frontier;
  frontier.reserve(2 * beam_width_);

  std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
  frontier_neighbors.reserve(2 * beam_width_);

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * beam_width_);

  std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
      cached_neighbors;
  cached_neighbors.reserve(2 * beam_width_);

  while (candidates.has_unexpanded_node() && num_ios < io_limit_) {
    frontier.clear();
    frontier_neighbors.clear();
    frontier_read_reqs.clear();
    cached_neighbors.clear();

    uint64_t sector_buffer_idx = 0;

    uint32_t num_seen = 0;
    while (candidates.has_unexpanded_node() && frontier.size() < beam_width_ &&
           num_seen < beam_width_) {
      auto neighbor = candidates.closest_unexpanded();
      num_seen++;

      auto iter = neighbor_cache_.find(neighbor.id);
      if (iter != neighbor_cache_.end()) {
        cached_neighbors.push_back(std::make_tuple(
            neighbor.id, iter->second.first, iter->second.second));
        stats.cache_hits++;
      } else {
        frontier.push_back(neighbor.id);
      }
    }

    if (!frontier.empty()) {
      stats.hop_num++;

      for (uint64_t i = 0; i < frontier.size(); i++) {
        diskann_id_t cur_id = frontier[i];

        std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
        frontier_neighbor.first = cur_id;
        frontier_neighbor.second = sector_buffer + sector_num_per_node *
                                                       sector_buffer_idx *
                                                       DiskAnnUtil::kSectorSize;
        frontier_neighbors.push_back(frontier_neighbor);

        sector_buffer_idx++;

        frontier_read_reqs.emplace_back(
            index_segment_offset_ +
                DiskAnnUtil::get_node_sector(node_per_sector_, max_node_size_,
                                             DiskAnnUtil::kSectorSize, cur_id) *
                    DiskAnnUtil::kSectorSize,
            sector_num_per_node * DiskAnnUtil::kSectorSize,
            frontier_neighbor.second);

        stats.disk_page_reads++;
        stats.io_num++;
        num_ios++;
      }

      io_timer.reset();

      int read_ret = reader_->read(frontier_read_reqs, io_ctx);
      stats.io_us += io_timer.micro_seconds();
      if (read_ret != 0) {
        LOG_ERROR("cached_beam_search: reader_->read failed, ret=%d", read_ret);
        ctx->set_error(true);
        return IndexError_Runtime;
      }
    }

    for (auto &cached_neighbor : cached_neighbors) {
      auto global_cache_iter = coord_cache_.find(std::get<0>(cached_neighbor));
      void *node_fp_coords_copy = global_cache_iter->second;

      float cur_expanded_dist = dc.dist(ctx->query(), node_fp_coords_copy);

      if (!ctx->filter().is_valid() ||
          !ctx->filter()(get_key(std::get<0>(cached_neighbor)))) {
        topk_heap.emplace(std::get<0>(cached_neighbor),
                          VectorInfo(cur_expanded_dist,
                                     make_vector_copy(node_fp_coords_copy)));
      }

      uint32_t neighbor_num = std::get<1>(cached_neighbor);
      diskann_id_t *node_neighbors = std::get<2>(cached_neighbor);

      cpu_timer.reset();

      std::vector<float> distances(neighbor_num);
      pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                               ctx->pq_table_dist_buffer(),
                               ctx->pq_coord_buffer(), distances.data());

      stats.dist_num += neighbor_num;
      stats.cpu_us += cpu_timer.micro_seconds();

      for (uint64_t m = 0; m < neighbor_num; ++m) {
        diskann_id_t id = node_neighbors[m];
        visit_filter.set_visited(id);

        Neighbor nn(id, distances[m]);
        candidates.insert(nn);
      }
    }

    for (auto &frontier_neighbor : frontier_neighbors) {
      uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
          node_per_sector_, max_node_size_, frontier_neighbor.second,
          frontier_neighbor.first);
      uint32_t *node_buf = DiskAnnUtil::offset_to_node_neighbor(
          node_disk_buf, meta_.element_size());
      uint32_t neighbor_num = *node_buf;

      void *node_fp_coords = node_disk_buf;

      float cur_expanded_dist = dc.dist(ctx->query(), node_fp_coords);

      if (!ctx->filter().is_valid() ||
          !ctx->filter()(get_key(frontier_neighbor.first))) {
        topk_heap.emplace(
            frontier_neighbor.first,
            VectorInfo(cur_expanded_dist, make_vector_copy(node_fp_coords)));
      }

      diskann_id_t *node_neighbors =
          reinterpret_cast<diskann_id_t *>(node_buf + 1);

      cpu_timer.reset();
      std::vector<float> distances(neighbor_num);
      pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                               ctx->pq_table_dist_buffer(),
                               ctx->pq_coord_buffer(), distances.data());

      stats.dist_num += neighbor_num;
      stats.cpu_us += cpu_timer.micro_seconds();

      cpu_timer.reset();
      for (uint64_t m = 0; m < neighbor_num; ++m) {
        diskann_id_t id = node_neighbors[m];
        visit_filter.set_visited(id);
        stats.dist_num++;

        Neighbor nn(id, distances[m]);
        candidates.insert(nn);
      }

      stats.cpu_us += cpu_timer.micro_seconds();
    }
  }

  stats.total_us += query_timer.micro_seconds();

  return 0;
}

int DiskAnnIndexer::cached_beam_search_in_mem(DiskAnnContext * /*ctx*/) {
  return IndexError_NotImplemented;
}

int DiskAnnIndexer::cached_beam_search_by_group(DiskAnnContext *ctx) {
  if (!ctx->group_by().is_valid()) {
    return 0;
  }

  std::function<std::string(diskann_id_t)> group_by = [&](diskann_id_t id) {
    return ctx->group_by()(get_key(id));
  };

  // devide into groups
  auto &topk_heap = ctx->topk_heap();
  auto &visit_filter = ctx->visit_filter();

  std::map<std::string, TopkHeap> &group_topk_heaps = ctx->group_topk_heaps();

  for (uint32_t i = 0; i < topk_heap.size(); ++i) {
    diskann_id_t id = topk_heap[i].first;
    auto info = topk_heap[i].second;

    std::string group_id = group_by(id);

    auto &group_topk_heap = group_topk_heaps[group_id];
    if (group_topk_heap.empty()) {
      group_topk_heap.limit(ctx->group_topk());
    }

    topk_heap.emplace(id, info);
  }

  // stage 2, expand to reach group num as possible
  if (group_topk_heaps.size() < ctx->group_num()) {
    NeighborPriorityQueue candidates;

    candidates.reserve(ctx->list_size());

    for (uint32_t i = 0; i < topk_heap.size(); ++i) {
      diskann_id_t id = topk_heap[i].first;
      float score = topk_heap[i].second.dist_;

      visit_filter.set_visited(id);
      candidates.insert(Neighbor(id, score));
    }

    ailego::ElapsedTime io_timer;
    ailego::ElapsedTime query_timer;
    ailego::ElapsedTime cpu_timer;

    auto &stats = ctx->query_stats();
    auto &dc = ctx->dist_calculator();

    IOContext &io_ctx = ctx->io_ctx();

    void *data_buf = reinterpret_cast<void *>(ctx->coord_buffer());
    uint8_t *sector_buffer = reinterpret_cast<uint8_t *>(ctx->sector_buffer());

    const uint64_t sector_num_per_node =
        node_per_sector_ > 0 ? 1
                             : DiskAnnUtil::div_round_up(
                                   max_node_size_, DiskAnnUtil::kSectorSize);

    pq_table_->preprocess_pq_dist_table(ctx->query_rotated(),
                                        ctx->pq_table_dist_buffer());

    uint32_t num_ios = 0;

    std::vector<diskann_id_t> frontier;
    frontier.reserve(2 * beam_width_);
    std::vector<std::pair<diskann_id_t, uint8_t *>> frontier_neighbors;
    frontier_neighbors.reserve(2 * beam_width_);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width_);
    std::vector<std::tuple<diskann_id_t, uint32_t, diskann_id_t *>>
        cached_neighbors;
    cached_neighbors.reserve(2 * beam_width_);

    uint64_t sector_buffer_idx;

    while (candidates.has_unexpanded_node() && num_ios < io_limit_) {
      frontier.clear();
      frontier_neighbors.clear();
      frontier_read_reqs.clear();
      cached_neighbors.clear();
      sector_buffer_idx = 0;

      uint32_t num_seen = 0;
      while (candidates.has_unexpanded_node() &&
             frontier.size() < beam_width_ && num_seen < beam_width_) {
        auto neighbor = candidates.closest_unexpanded();
        num_seen++;

        auto iter = neighbor_cache_.find(neighbor.id);
        if (iter != neighbor_cache_.end()) {
          cached_neighbors.push_back(std::make_tuple(
              neighbor.id, iter->second.first, iter->second.second));
          stats.cache_hits++;
        } else {
          frontier.push_back(neighbor.id);
        }
      }

      if (!frontier.empty()) {
        stats.hop_num++;

        for (uint64_t i = 0; i < frontier.size(); i++) {
          diskann_id_t cur_id = frontier[i];

          std::pair<diskann_id_t, uint8_t *> frontier_neighbor;
          frontier_neighbor.first = cur_id;
          frontier_neighbor.second =
              sector_buffer + sector_num_per_node * sector_buffer_idx *
                                  DiskAnnUtil::kSectorSize;
          frontier_neighbors.push_back(frontier_neighbor);

          sector_buffer_idx++;

          frontier_read_reqs.emplace_back(
              index_segment_offset_ + DiskAnnUtil::get_node_sector(
                                          node_per_sector_, max_node_size_,
                                          DiskAnnUtil::kSectorSize, cur_id) *
                                          DiskAnnUtil::kSectorSize,
              sector_num_per_node * DiskAnnUtil::kSectorSize,
              frontier_neighbor.second);

          stats.disk_page_reads++;
          stats.io_num++;
          num_ios++;
        }

        io_timer.reset();

        reader_->read(frontier_read_reqs, io_ctx);  // synchronous IO linux
        stats.io_us += io_timer.micro_seconds();
      }

      for (auto &cached_neighbor : cached_neighbors) {
        auto global_cache_iter =
            coord_cache_.find(std::get<0>(cached_neighbor));
        void *node_fp_coords_copy = global_cache_iter->second;

        float cur_expanded_dist = dc.dist(ctx->query(), node_fp_coords_copy);

        if (!ctx->filter().is_valid() ||
            !ctx->filter()(get_key(std::get<0>(cached_neighbor)))) {
          std::string group_id = group_by(std::get<0>(cached_neighbor));

          auto &group_topk_heap = group_topk_heaps[group_id];
          if (group_topk_heap.empty()) {
            group_topk_heap.limit(ctx->group_topk());
          }

          group_topk_heap.emplace_back(
              std::get<0>(cached_neighbor),
              VectorInfo(cur_expanded_dist,
                         make_vector_copy(node_fp_coords_copy)));

          if (group_topk_heaps.size() >= ctx->group_num()) {
            break;
          }
        }

        uint64_t neighbor_num = std::get<1>(cached_neighbor);
        diskann_id_t *node_neighbors = std::get<2>(cached_neighbor);

        cpu_timer.reset();

        std::vector<float> distances(neighbor_num);
        pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                                 ctx->pq_table_dist_buffer(),
                                 ctx->pq_coord_buffer(), distances.data());

        stats.dist_num += neighbor_num;
        stats.cpu_us += cpu_timer.micro_seconds();

        for (uint64_t m = 0; m < neighbor_num; ++m) {
          diskann_id_t id = node_neighbors[m];
          visit_filter.set_visited(id);

          Neighbor nn(id, distances[m]);
          candidates.insert(nn);
        }
      }

      for (auto &frontier_neighbor : frontier_neighbors) {
        uint8_t *node_disk_buf = DiskAnnUtil::offset_to_node(
            node_per_sector_, max_node_size_, frontier_neighbor.second,
            frontier_neighbor.first);
        uint32_t *node_buf = DiskAnnUtil::offset_to_node_neighbor(
            node_disk_buf, meta_.element_size());
        uint32_t neighbor_num = *node_buf;

        void *node_fp_coords = node_disk_buf;
        memcpy(data_buf, node_fp_coords, disk_bytes_per_point_);

        float cur_expanded_dist = dc.dist(ctx->query(), data_buf);

        if (!ctx->filter().is_valid() ||
            !ctx->filter()(get_key(frontier_neighbor.first))) {
          std::string group_id = group_by(frontier_neighbor.first);

          auto &group_topk_heap = group_topk_heaps[group_id];
          if (group_topk_heap.empty()) {
            group_topk_heap.limit(ctx->group_topk());
          }

          group_topk_heap.emplace_back(
              frontier_neighbor.first,
              VectorInfo(cur_expanded_dist, make_vector_copy(data_buf)));

          if (group_topk_heaps.size() >= ctx->group_num()) {
            break;
          }
        }

        cpu_timer.reset();

        std::vector<float> distances(neighbor_num);
        diskann_id_t *node_neighbors =
            reinterpret_cast<diskann_id_t *>(node_buf + 1);
        pq_table_->compute_dists(neighbor_num, node_neighbors, pq_chunk_num_,
                                 ctx->pq_table_dist_buffer(),
                                 ctx->pq_coord_buffer(), distances.data());

        stats.dist_num += neighbor_num;
        stats.cpu_us += cpu_timer.micro_seconds();

        cpu_timer.reset();
        for (uint64_t m = 0; m < neighbor_num; ++m) {
          diskann_id_t id = node_neighbors[m];
          visit_filter.set_visited(id);
          stats.dist_num++;

          Neighbor nn(id, distances[m]);
          candidates.insert(nn);
        }

        stats.cpu_us += cpu_timer.micro_seconds();
      }
    }

    stats.total_us += query_timer.micro_seconds();
  }

  return 0;
}

}  // namespace core
}  // namespace zvec

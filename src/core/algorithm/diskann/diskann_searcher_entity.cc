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

#include "diskann_searcher_entity.h"
#include <limits>
#include <new>

namespace zvec {
namespace core {

const DiskAnnEntity::Pointer DiskAnnSearcherEntity::clone() const {
  auto meta_segment = meta_segment_->clone();
  if (ailego_unlikely(!meta_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto pq_meta_segment = pq_meta_segment_->clone();
  if (ailego_unlikely(!pq_meta_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnPqMetaSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto pq_data_segment = pq_data_segment_->clone();
  if (ailego_unlikely(!pq_data_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnPqDataSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto vector_segment = vector_segment_->clone();
  if (ailego_unlikely(!vector_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnVectorSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto key_segment = key_segment_->clone();
  if (ailego_unlikely(!key_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnKeySegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto key_mapping_segment = key_mapping_segment_->clone();
  if (ailego_unlikely(!key_mapping_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnKeyMappingSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto entrypoint_segment = entrypoint_segment_->clone();
  if (ailego_unlikely(!entrypoint_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  DiskAnnSearcherEntity *entity = new (std::nothrow) DiskAnnSearcherEntity(
      meta_header_, pq_meta_, meta_segment, pq_meta_segment, pq_data_segment,
      vector_segment, key_segment, key_mapping_segment, entrypoint_segment,
      num_threads_, list_size_, cache_nodes_num_, warm_up_, beam_size_, meta_,
      pq_table_, key_buffer_, key_mapping_buffer_, entrypoints_,
      resident_budget_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("DiskAnnSearcherEntity new failed");
  }

  return DiskAnnEntity::Pointer(entity);
}

int DiskAnnSearcherEntity::load(const IndexMeta &meta,
                                IndexStorage::Pointer storage) {
  meta_ = meta;

  storage_ = storage;

  int ret;
  ret = load_header_segment();
  if (ret != 0) {
    LOG_ERROR("Load Header Segment Failed, ret = %d", ret);

    return ret;
  }

  // Reserve the complete resident working set before any PQ/key vectors are
  // allocated. Segment sizes are a conservative upper bound for the copied
  // payloads; the transposed PQ table is the only additional allocation.
  ret = reserve_resident_budget();
  if (ret != 0) {
    return ret;
  }

  try {
    ret = load_pq_segment();
    if (ret != 0) {
      LOG_ERROR("Load PQ Meta Segment Failed, ret = %d", ret);
      clear_resident_data();
      return ret;
    }

    ret = load_key_segment();
    if (ret != 0) {
      LOG_ERROR("Load Key Segment Failed, ret = %d", ret);
      clear_resident_data();
      return ret;
    }

    ret = load_key_mapping_segment();
    if (ret != 0) {
      LOG_ERROR("Load Key Mapping Segment Failed, ret = %d", ret);
      clear_resident_data();
      return ret;
    }

    ret = load_entrypoint_segment();
    if (ret != 0) {
      LOG_WARN("Load EntryPoint Segment Failed, ret = %d", ret);
      clear_resident_data();
      return ret;
    }

    ret = load_vector_segment();
    if (ret != 0) {
      LOG_ERROR("Load Vector Segment Failed, ret = %d", ret);
      clear_resident_data();
      return ret;
    }

    return 0;
  } catch (const std::bad_alloc &) {
    LOG_ERROR("DiskANN resident allocation failed");
    clear_resident_data();
    return IndexError_NoMemory;
  }
}

int DiskAnnSearcherEntity::reserve_resident_budget() {
  pq_meta_segment_ = storage_->get(kDiskAnnPqMetaSegmentId);
  pq_data_segment_ = storage_->get(kDiskAnnPqDataSegmentId);
  key_segment_ = storage_->get(kDiskAnnKeySegmentId);
  key_mapping_segment_ = storage_->get(kDiskAnnKeyMappingSegmentId);
  entrypoint_segment_ = storage_->get(kDiskAnnEntryPointSegmentId);
  if (!pq_meta_segment_ || !pq_data_segment_ || !key_segment_ ||
      !key_mapping_segment_ || !entrypoint_segment_ ||
      pq_meta_segment_->data_size() < sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("DiskANN resident segment is missing or invalid");
    return IndexError_InvalidFormat;
  }

  const auto checked_add = [](uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
      return false;
    }
    *out = lhs + rhs;
    return true;
  };
  const auto checked_mul = [](uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
      return false;
    }
    *out = lhs * rhs;
    return true;
  };

  uint64_t transposed_table_bytes = 0;
  if (!checked_mul(static_cast<uint64_t>(PQTable::kPQCentroidNum),
                   static_cast<uint64_t>(meta_.element_size()),
                   &transposed_table_bytes)) {
    return IndexError_NoMemory;
  }

  uint64_t bytes = transposed_table_bytes;
  const uint64_t segment_bytes[] = {
      static_cast<uint64_t>(pq_meta_segment_->data_size()),
      static_cast<uint64_t>(pq_data_segment_->data_size()),
      static_cast<uint64_t>(key_segment_->data_size()),
      static_cast<uint64_t>(key_mapping_segment_->data_size()),
      static_cast<uint64_t>(entrypoint_segment_->data_size())};
  for (uint64_t segment_size : segment_bytes) {
    if (!checked_add(bytes, segment_size, &bytes)) {
      return IndexError_NoMemory;
    }
  }

  auto &budget = ailego::MemoryBudgetManager::get_instance();
  if (!budget.try_charge(
          ailego::MemoryBudgetManager::Category::ResidentMetadata, bytes)) {
    LOG_ERROR("DiskANN resident-metadata budget exhausted: request=%llu",
              static_cast<unsigned long long>(bytes));
    return IndexError_NoMemory;
  }
  try {
    resident_budget_ = std::make_shared<ResidentBudgetToken>(bytes);
  } catch (const std::bad_alloc &) {
    budget.release(ailego::MemoryBudgetManager::Category::ResidentMetadata,
                   bytes);
    return IndexError_NoMemory;
  }
  return 0;
}

void DiskAnnSearcherEntity::clear_resident_data() {
  pq_table_.reset();
  std::string().swap(*key_buffer_);
  std::string().swap(*key_mapping_buffer_);
  std::vector<diskann_id_t>().swap(*entrypoints_);
  resident_budget_.reset();
}

int DiskAnnSearcherEntity::load_pq_segment() {
  // load pq meta
  pq_meta_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqMetaSegmentId);
  if (!pq_meta_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  size_t read_size;
  size_t offset = 0;

  // 1. read pq meta
  read_size = pq_meta_segment_->fetch(offset, &pq_meta_, sizeof(DiskAnnPqMeta));
  if (read_size != sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              sizeof(DiskAnnPqMeta), read_size);

    return IndexError_ReadData;
  }
  offset += read_size;

  uint32_t pq_dimension = meta_.dimension();
  if (meta_.metric_name() == "Cosine") {
    const uint32_t removed_dimensions =
        meta_.data_type() == IndexMeta::DataType::DT_FP32 ? 1 : 2;
    if (pq_dimension <= removed_dimensions) {
      LOG_ERROR("Invalid DiskANN cosine dimension");
      return IndexError_InvalidFormat;
    }
    pq_dimension -= removed_dimensions;
  }
  const uint64_t pq_element_size =
      static_cast<uint64_t>(pq_dimension) * meta_.unit_size();
  const uint64_t expected_pivot_bytes =
      static_cast<uint64_t>(PQTable::kPQCentroidNum) * pq_element_size;
  if (pq_meta_.full_pivot_data_size != expected_pivot_bytes ||
      pq_meta_.centroid_data_size != pq_element_size ||
      pq_meta_.chunk_num == 0 || pq_meta_.chunk_num > pq_dimension ||
      pq_meta_.chunk_num > std::numeric_limits<uint32_t>::max()) {
    LOG_ERROR("Invalid DiskANN PQ metadata");
    return IndexError_InvalidFormat;
  }
  const uint64_t chunk_offset_bytes =
      (pq_meta_.chunk_num + 1) * sizeof(uint32_t);
  uint64_t remaining = pq_meta_segment_->data_size() - sizeof(DiskAnnPqMeta);
  const uint64_t pq_parts[] = {pq_meta_.full_pivot_data_size,
                               pq_meta_.centroid_data_size, chunk_offset_bytes};
  for (uint64_t part : pq_parts) {
    if (part > remaining || part > std::numeric_limits<size_t>::max()) {
      LOG_ERROR("DiskANN PQ metadata exceeds its segment");
      return IndexError_InvalidFormat;
    }
    remaining -= part;
  }

  if (meta_header_.doc_cnt != 0 &&
      pq_meta_.chunk_num >
          std::numeric_limits<uint64_t>::max() / meta_header_.doc_cnt) {
    return IndexError_InvalidFormat;
  }
  const uint64_t pq_data_bytes = meta_header_.doc_cnt * pq_meta_.chunk_num;
  pq_data_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqDataSegmentId);
  if (!pq_data_segment_ || pq_data_bytes > pq_data_segment_->data_size() ||
      pq_data_bytes > std::numeric_limits<size_t>::max()) {
    LOG_ERROR("Miss, invalid, or undersized segment %s",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  // 2. read full pivot data
  std::vector<uint8_t> full_pivot_data(pq_meta_.full_pivot_data_size);
  read_size = pq_meta_segment_->fetch(offset, full_pivot_data.data(),
                                      pq_meta_.full_pivot_data_size);
  if (read_size != pq_meta_.full_pivot_data_size) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)(pq_meta_.full_pivot_data_size), (size_t)read_size);
    return IndexError_ReadData;
  }
  offset += read_size;

  // 3. read centroid
  std::vector<uint8_t> centroid(pq_meta_.centroid_data_size);
  read_size = pq_meta_segment_->fetch(offset, centroid.data(),
                                      pq_meta_.centroid_data_size);
  if (read_size != pq_meta_.centroid_data_size) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)(pq_meta_.centroid_data_size), (size_t)read_size);
    return IndexError_ReadData;
  }
  offset += read_size;

  // 4. chunk offset
  std::vector<uint32_t> chunk_offsets(pq_meta_.chunk_num + 1);

  read_size =
      pq_meta_segment_->fetch(offset, chunk_offsets.data(), chunk_offset_bytes);
  if (read_size != chunk_offset_bytes) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              static_cast<size_t>(chunk_offset_bytes), read_size);
    return IndexError_ReadData;
  }
  if (chunk_offsets.front() != 0 || chunk_offsets.back() != pq_dimension) {
    LOG_ERROR("Invalid DiskANN PQ chunk boundaries");
    return IndexError_InvalidFormat;
  }
  for (size_t i = 1; i < chunk_offsets.size(); ++i) {
    if (chunk_offsets[i] < chunk_offsets[i - 1] ||
        chunk_offsets[i] > pq_dimension) {
      LOG_ERROR("Invalid DiskANN PQ chunk boundary at %zu", i);
      return IndexError_InvalidFormat;
    }
  }

  // load pq data
  std::vector<uint8_t> pq_data(static_cast<size_t>(pq_data_bytes));
  read_size = pq_data_segment_->fetch(0, pq_data.data(), pq_data.size());
  if (read_size != pq_data.size()) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str(), pq_data.size(),
              read_size);
    return IndexError_ReadData;
  }

  pq_table_ = std::make_shared<PQTable>(meta_, pq_meta_.chunk_num);

  return pq_table_->init(full_pivot_data, centroid, chunk_offsets, pq_data);
}

int DiskAnnSearcherEntity::load_header_segment() {
  meta_segment_ = storage_->get(kDiskAnnMetaSegmentId);
  if (!meta_segment_ ||
      meta_segment_->data_size() < sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Miss or invalid segment %s", kDiskAnnMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (meta_segment_->fetch(0, &meta_header_, sizeof(DiskAnnMetaHeader)) !=
      sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return IndexError_ReadData;
  }
  if (meta_header_.doc_cnt > std::numeric_limits<diskann_id_t>::max() ||
      meta_header_.doc_cnt == 0 || meta_header_.ndims != meta_.dimension() ||
      meta_header_.max_node_size > std::numeric_limits<uint32_t>::max() ||
      meta_header_.max_degree == 0 ||
      meta_header_.max_degree > std::numeric_limits<uint32_t>::max() ||
      meta_header_.node_per_sector > std::numeric_limits<uint32_t>::max() ||
      meta_header_.medoid >= meta_header_.doc_cnt) {
    LOG_ERROR("Invalid DiskANN header fields");
    return IndexError_InvalidFormat;
  }

  const uint64_t expected_node_size =
      static_cast<uint64_t>(meta_.element_size()) + sizeof(uint32_t) +
      meta_header_.max_degree * sizeof(diskann_id_t);
  if (meta_header_.max_node_size != expected_node_size ||
      meta_header_.node_per_sector !=
          DiskAnnUtil::kSectorSize / expected_node_size) {
    LOG_ERROR("Invalid DiskANN node layout");
    return IndexError_InvalidFormat;
  }

  const uint64_t sectors_per_node =
      meta_header_.node_per_sector == 0
          ? DiskAnnUtil::div_round_up(meta_header_.max_node_size,
                                      DiskAnnUtil::kSectorSize)
          : 1;
  const uint64_t sector_count =
      meta_header_.node_per_sector == 0
          ? meta_header_.doc_cnt * sectors_per_node
          : DiskAnnUtil::div_round_up(meta_header_.doc_cnt,
                                      meta_header_.node_per_sector);
  if (sector_count >
          std::numeric_limits<uint64_t>::max() / DiskAnnUtil::kSectorSize ||
      meta_header_.index_size != sector_count * DiskAnnUtil::kSectorSize) {
    LOG_ERROR("Invalid DiskANN index size");
    return IndexError_InvalidFormat;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_vector_segment() {
  vector_segment_ = storage_->get(kDiskAnnVectorSegmentId);
  if (!vector_segment_ ||
      meta_header_.index_size > vector_segment_->data_size()) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnVectorSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_key_segment() {
  // load key
  key_segment_ = storage_->get(kDiskAnnKeySegmentId);
  if (!key_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnKeySegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  if (doc_cnt() > std::numeric_limits<size_t>::max() / sizeof(diskann_key_t)) {
    return IndexError_InvalidFormat;
  }
  size_t key_data_len = doc_cnt() * sizeof(diskann_key_t);
  if (key_data_len > key_segment_->data_size()) {
    LOG_ERROR("DiskANN key segment is undersized");
    return IndexError_InvalidFormat;
  }

  key_buffer_->resize(key_data_len);
  if (key_segment_->fetch(0, key_buffer_->data(), key_data_len) !=
      key_data_len) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeySegmentId.c_str());
    return IndexError_ReadData;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_entrypoint_segment() {
  entrypoint_segment_ = storage_->get(kDiskAnnEntryPointSegmentId);
  if (!entrypoint_segment_ ||
      entrypoint_segment_->data_size() < sizeof(uint32_t)) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  uint32_t entrypoint_cnt = 0;
  if (entrypoint_segment_->fetch(0, &entrypoint_cnt, sizeof(uint32_t)) !=
      sizeof(uint32_t)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_ReadData;
  }

  if (entrypoint_cnt != 0) {
    size_t entrypoint_data_len = entrypoint_cnt * sizeof(diskann_id_t);
    if (entrypoint_data_len >
        entrypoint_segment_->data_size() - sizeof(uint32_t)) {
      LOG_ERROR("Invalid entrypoint count in segment %s",
                kDiskAnnEntryPointSegmentId.c_str());
      return IndexError_InvalidFormat;
    }

    entrypoints_->resize(entrypoint_cnt);
    if (entrypoint_segment_->fetch(sizeof(uint32_t), entrypoints_->data(),
                                   entrypoint_data_len) !=
        entrypoint_data_len) {
      LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
      return IndexError_ReadData;
    }
    for (diskann_id_t id : *entrypoints_) {
      if (id >= doc_cnt()) {
        LOG_ERROR("Invalid DiskANN entrypoint id: %u", id);
        return IndexError_InvalidFormat;
      }
    }
  }

  return 0;
}


int DiskAnnSearcherEntity::load_key_mapping_segment() {
  key_mapping_segment_ = storage_->get(kDiskAnnKeyMappingSegmentId);
  if (!key_mapping_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  if (doc_cnt() > std::numeric_limits<size_t>::max() / sizeof(diskann_id_t)) {
    return IndexError_InvalidFormat;
  }
  size_t key_mapping_data_len = doc_cnt() * sizeof(diskann_id_t);
  if (key_mapping_data_len > key_mapping_segment_->data_size()) {
    LOG_ERROR("DiskANN key-mapping segment is undersized");
    return IndexError_InvalidFormat;
  }

  key_mapping_buffer_->resize(key_mapping_data_len);
  if (key_mapping_segment_->fetch(0, key_mapping_buffer_->data(),
                                  key_mapping_data_len) !=
      key_mapping_data_len) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_ReadData;
  }
  const auto *mapping =
      reinterpret_cast<const diskann_id_t *>(key_mapping_buffer_->data());
  for (size_t i = 0; i < doc_cnt(); ++i) {
    if (mapping[i] >= doc_cnt()) {
      LOG_ERROR("Invalid DiskANN key-mapping id: %u", mapping[i]);
      return IndexError_InvalidFormat;
    }
  }

  return 0;
}

//! Get vector local id by key
diskann_id_t DiskAnnSearcherEntity::get_id(diskann_key_t key) const {
  const diskann_id_t *key_mapping_data_ptr =
      reinterpret_cast<const diskann_id_t *>(key_mapping_buffer_->data());

  const diskann_key_t *key_data_ptr =
      reinterpret_cast<const diskann_key_t *>(key_buffer_->data());

  //! Do binary search
  diskann_id_t start = 0UL;
  diskann_id_t end = doc_cnt();
  diskann_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    diskann_id_t local_id = key_mapping_data_ptr[idx];

    const diskann_key_t local_key = key_data_ptr[local_id];

    if (local_key < key) {
      start = idx + 1;
    } else if (local_key > key) {
      end = idx;
    } else {
      return local_id;
    }
  }

  return kInvalidId;
}

diskann_key_t DiskAnnSearcherEntity::get_key(diskann_id_t id) const {
  if (id >= doc_cnt()) {
    return kInvalidKey;
  }
  const diskann_key_t *key_data_ptr =
      reinterpret_cast<const diskann_key_t *>(key_buffer_->data());

  return key_data_ptr[id];
}

const void *DiskAnnSearcherEntity::get_vector(diskann_id_t id) const {
  if (!vector_segment_) {
    LOG_ERROR("Vector segment is null");
    return nullptr;
  }
  if (id >= doc_cnt()) {
    LOG_ERROR("Invalid vector id: %u", id);
    return nullptr;
  }

  uint64_t sector_offset =
      DiskAnnUtil::get_node_sector(node_per_sector(), max_node_size(),
                                   DiskAnnUtil::kSectorSize, id) *
      DiskAnnUtil::kSectorSize;
  uint64_t within_sector_offset =
      (node_per_sector() == 0 ? 0 : (id % node_per_sector()) * max_node_size());
  uint64_t total_offset = sector_offset + within_sector_offset;

  size_t read_size = meta_.element_size();
  const void *vec;
  if (ailego_unlikely(vector_segment_->read(total_offset, &vec, read_size) !=
                      read_size)) {
    LOG_ERROR("Read vector from segment failed, id: %u, offset: %llu", id,
              static_cast<unsigned long long>(total_offset));
    return nullptr;
  }

  return vec;
}

std::pair<uint32_t, const diskann_id_t *> DiskAnnSearcherEntity::get_neighbors(
    diskann_id_t id) const {
  if (!vector_segment_ || id >= doc_cnt()) {
    return std::make_pair(0, nullptr);
  }

  uint64_t read_sector_offset =
      DiskAnnUtil::get_node_sector(node_per_sector(), max_node_size(),
                                   DiskAnnUtil::kSectorSize, id) *
      DiskAnnUtil::kSectorSize;
  uint64_t node_vec_offset =
      read_sector_offset +
      (node_per_sector() == 0 ? 0 : (id % node_per_sector()) * max_node_size());

  const void *data;
  if (ailego_unlikely(
          vector_segment_->read(node_vec_offset, &data, max_node_size()) !=
          max_node_size())) {
    LOG_ERROR("Read neighbors from segment failed");
    return {0, nullptr};
  }

  const uint8_t *data_ptr = reinterpret_cast<const uint8_t *>(data);
  const diskann_id_t *node_neighbor =
      reinterpret_cast<const diskann_id_t *>(data_ptr + meta_.element_size());

  auto neighbor_num = *node_neighbor;
  if (neighbor_num > meta_header_.max_degree) {
    LOG_ERROR("Invalid neighbor count for node %u", id);
    return {0, nullptr};
  }
  for (uint32_t i = 0; i < neighbor_num; ++i) {
    if (node_neighbor[i + 1] >= doc_cnt()) {
      LOG_ERROR("Invalid neighbor id for node %u", id);
      return {0, nullptr};
    }
  }

  return std::make_pair(neighbor_num, node_neighbor + 1);
}

}  // namespace core
}  // namespace zvec

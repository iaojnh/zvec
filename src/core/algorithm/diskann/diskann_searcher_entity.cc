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
#include <cinttypes>
#include <limits>
#include <stdexcept>

namespace zvec {
namespace core {

namespace {

bool checked_multiply(size_t lhs, size_t rhs, size_t *result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_add(size_t lhs, size_t rhs, size_t *result) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

int get_pq_layout(const IndexMeta &meta, uint32_t *dimension,
                  size_t *full_pivot_size, size_t *centroid_size) {
  uint32_t pq_dimension = meta.dimension();
  size_t unit_size = 0;
  switch (meta.data_type()) {
    case IndexMeta::DataType::DT_FP32:
      unit_size = sizeof(float);
      if (meta.metric_name() == "Cosine") {
        if (pq_dimension <= 1) {
          return IndexError_InvalidFormat;
        }
        --pq_dimension;
      }
      break;
    case IndexMeta::DataType::DT_FP16:
      unit_size = sizeof(ailego::Float16);
      if (meta.metric_name() == "Cosine") {
        if (pq_dimension <= 2) {
          return IndexError_InvalidFormat;
        }
        pq_dimension -= 2;
      }
      break;
    default:
      return IndexError_Unsupported;
  }

  if (pq_dimension == 0 ||
      !checked_multiply(pq_dimension, unit_size, centroid_size) ||
      !checked_multiply(*centroid_size, PQTable::kPQCentroidNum,
                        full_pivot_size)) {
    return IndexError_InvalidFormat;
  }
  *dimension = pq_dimension;
  return 0;
}

}  // namespace

void DiskAnnSearcherEntity::clear() {
  release_storage();
  pq_table_.reset();
  key_buffer_.reset();
  key_mapping_buffer_.reset();
  entrypoints_.clear();
  meta_.clear();
  meta_header_ = {};
  pq_meta_ = {};
}

void DiskAnnSearcherEntity::release_storage() {
  storage_.reset();
  meta_segment_.reset();
  pq_meta_segment_.reset();
  pq_data_segment_.reset();
  vector_segment_.reset();
  key_segment_.reset();
  key_mapping_segment_.reset();
  entrypoint_segment_.reset();
}

const DiskAnnEntity::Pointer DiskAnnSearcherEntity::clone() const {
  std::unique_ptr<DiskAnnSearcherEntity> entity(new (std::nothrow)
                                                    DiskAnnSearcherEntity());
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("DiskAnnSearcherEntity new failed");
    return DiskAnnEntity::Pointer();
  }

  try {
    entity->meta_header_ = meta_header_;
    entity->pq_meta_ = pq_meta_;
    entity->meta_ = meta_;
    entity->pq_table_ = pq_table_;
    entity->key_buffer_ = key_buffer_;
    entity->key_mapping_buffer_ = key_mapping_buffer_;
    entity->entrypoints_ = entrypoints_;
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to clone in-memory DiskAnn entity");
    return DiskAnnEntity::Pointer();
  }

  return DiskAnnEntity::Pointer(entity.release());
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

  ret = load_pq_segment();
  if (ret != 0) {
    LOG_ERROR("Load PQ Meta Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_key_segment();
  if (ret != 0) {
    LOG_ERROR("Load Key Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_key_mapping_segment();
  if (ret != 0) {
    LOG_ERROR("Load Key Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_entrypoint_segment();
  if (ret != 0) {
    LOG_WARN("Load EntryPoint Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_vector_segment();
  if (ret != 0) {
    LOG_ERROR("Load Vector Segment Failed, ret = %d", ret);

    return ret;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_pq_segment() {
  const void *data = nullptr;

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
  read_size = pq_meta_segment_->read(offset, &data, sizeof(DiskAnnPqMeta));
  if (read_size != sizeof(DiskAnnPqMeta) || data == nullptr) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              sizeof(DiskAnnPqMeta), read_size);

    return IndexError_ReadData;
  }

  memcpy(reinterpret_cast<uint8_t *>(&pq_meta_), data, sizeof(DiskAnnPqMeta));
  offset += read_size;
  uint32_t pq_dimension = 0;
  size_t expected_full_pivot_size = 0;
  size_t expected_centroid_size = 0;
  int layout_ret = get_pq_layout(
      meta_, &pq_dimension, &expected_full_pivot_size, &expected_centroid_size);
  if (layout_ret != 0) {
    LOG_ERROR("Invalid DiskAnn PQ layout for the configured index metadata");
    return layout_ret;
  }

  size_t chunk_offsets_count = 0;
  size_t expected_chunk_offsets_size = 0;
  if (pq_meta_.chunk_num == 0 || meta_header_.doc_cnt == 0 ||
      pq_meta_.chunk_num > pq_dimension ||
      pq_meta_.full_pivot_data_size != expected_full_pivot_size ||
      pq_meta_.centroid_data_size != expected_centroid_size ||
      pq_meta_.chunk_num > std::numeric_limits<uint32_t>::max() ||
      !checked_add(static_cast<size_t>(pq_meta_.chunk_num), 1,
                   &chunk_offsets_count) ||
      !checked_multiply(chunk_offsets_count, sizeof(uint32_t),
                        &expected_chunk_offsets_size) ||
      (pq_meta_.chunk_offsets_size != 0 &&
       pq_meta_.chunk_offsets_size != expected_chunk_offsets_size)) {
    LOG_ERROR("Invalid DiskAnn PQ metadata sizes");
    return IndexError_InvalidFormat;
  }

  size_t pq_meta_data_size = sizeof(DiskAnnPqMeta);
  if (!checked_add(pq_meta_data_size, expected_full_pivot_size,
                   &pq_meta_data_size) ||
      !checked_add(pq_meta_data_size, expected_centroid_size,
                   &pq_meta_data_size) ||
      !checked_add(pq_meta_data_size, expected_chunk_offsets_size,
                   &pq_meta_data_size) ||
      pq_meta_segment_->data_size() < pq_meta_data_size) {
    LOG_ERROR("DiskAnn PQ metadata segment is shorter than its layout");
    return IndexError_InvalidFormat;
  }

  // 2. read full pivot data
  std::vector<uint8_t> full_pivot_data;
  std::vector<uint8_t> centroid;
  std::vector<uint32_t> chunk_offsets;
  try {
    full_pivot_data.resize(expected_full_pivot_size);
    centroid.resize(expected_centroid_size);
    chunk_offsets.resize(chunk_offsets_count);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn PQ metadata buffers");
    return IndexError_NoMemory;
  }

  read_size = pq_meta_segment_->read(offset, &data, expected_full_pivot_size);
  if (read_size != expected_full_pivot_size || data == nullptr) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              expected_full_pivot_size, read_size);
    return IndexError_ReadData;
  }
  memcpy(full_pivot_data.data(), data, read_size);
  offset += read_size;

  // 3. read centroid
  read_size = pq_meta_segment_->read(offset, &data, expected_centroid_size);
  if (read_size != expected_centroid_size || data == nullptr) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              expected_centroid_size, read_size);
    return IndexError_ReadData;
  }
  memcpy(centroid.data(), data, read_size);
  offset += read_size;

  // 4. chunk offset
  read_size =
      pq_meta_segment_->read(offset, &data, expected_chunk_offsets_size);
  if (read_size != expected_chunk_offsets_size || data == nullptr) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              expected_chunk_offsets_size, read_size);
    return IndexError_ReadData;
  }
  memcpy(chunk_offsets.data(), data, read_size);

  if (chunk_offsets.front() != 0 || chunk_offsets.back() != pq_dimension) {
    LOG_ERROR("Invalid DiskAnn PQ chunk offset boundaries");
    return IndexError_InvalidFormat;
  }
  for (size_t i = 1; i < chunk_offsets.size(); ++i) {
    if (chunk_offsets[i - 1] >= chunk_offsets[i] ||
        chunk_offsets[i] > pq_dimension) {
      LOG_ERROR("Invalid DiskAnn PQ chunk offsets");
      return IndexError_InvalidFormat;
    }
  }

  // load pq data
  std::vector<uint8_t> pq_data;
  pq_data_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqDataSegmentId);
  if (!pq_data_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  size_t pq_data_size = 0;
  if (meta_header_.doc_cnt > std::numeric_limits<size_t>::max() ||
      !checked_multiply(static_cast<size_t>(meta_header_.doc_cnt),
                        static_cast<size_t>(pq_meta_.chunk_num),
                        &pq_data_size) ||
      pq_data_segment_->data_size() < pq_data_size) {
    LOG_ERROR("Invalid DiskAnn PQ data size");
    return IndexError_InvalidFormat;
  }

  try {
    pq_data.resize(pq_data_size);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn PQ data buffer");
    return IndexError_NoMemory;
  }

  read_size = pq_data_segment_->fetch(0, pq_data.data(), pq_data_size);

  if (read_size != pq_data_size) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str(), pq_data_size,
              read_size);

    return IndexError_ReadData;
  }

  try {
    pq_table_ = std::make_shared<PQTable>(
        meta_, static_cast<uint32_t>(pq_meta_.chunk_num));
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn PQ table");
    return IndexError_NoMemory;
  }

  return pq_table_->init(full_pivot_data, centroid, chunk_offsets, pq_data);
}

int DiskAnnSearcherEntity::load_header_segment() {
  const void *data = nullptr;
  meta_segment_ = storage_->get(kDiskAnnMetaSegmentId);
  if (!meta_segment_ ||
      meta_segment_->data_size() < sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Miss or invalid segment %s", kDiskAnnMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (meta_segment_->read(0, reinterpret_cast<const void **>(&data),
                          sizeof(DiskAnnMetaHeader)) !=
          sizeof(DiskAnnMetaHeader) ||
      data == nullptr) {
    LOG_ERROR("Read segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(reinterpret_cast<uint8_t *>(&meta_header_), data,
         sizeof(DiskAnnMetaHeader));

  if (meta_header_.doc_cnt == 0 || meta_header_.doc_cnt > kInvalidId) {
    LOG_ERROR("Invalid DiskAnn document count: %" PRIu64, meta_header_.doc_cnt);
    return IndexError_InvalidFormat;
  }
  if (meta_header_.ndims != meta_.dimension()) {
    LOG_ERROR("Invalid DiskAnn dimension: stored=%" PRIu64 " expected=%u",
              meta_header_.ndims, meta_.dimension());
    return IndexError_InvalidFormat;
  }
  if (meta_header_.medoid >= meta_header_.doc_cnt) {
    LOG_ERROR("Invalid DiskAnn medoid: %" PRIu64, meta_header_.medoid);
    return IndexError_InvalidFormat;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_vector_segment() {
  vector_segment_ = storage_->get(kDiskAnnVectorSegmentId);
  if (!vector_segment_) {
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

  size_t key_data_len = 0;
  if (doc_cnt() > std::numeric_limits<size_t>::max() ||
      !checked_multiply(static_cast<size_t>(doc_cnt()), sizeof(diskann_key_t),
                        &key_data_len) ||
      key_segment_->data_size() < key_data_len) {
    LOG_ERROR("Invalid DiskAnn key segment size");
    return IndexError_InvalidFormat;
  }

  const void *data = nullptr;
  if (key_segment_->read(0, reinterpret_cast<const void **>(&data),
                         key_data_len) != key_data_len ||
      data == nullptr) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeySegmentId.c_str());
    return IndexError_ReadData;
  }

  try {
    auto key_buffer = std::make_shared<std::vector<diskann_key_t>>(
        static_cast<size_t>(doc_cnt()));
    memcpy(key_buffer->data(), data, key_data_len);
    key_buffer_ = std::move(key_buffer);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn key buffer");
    return IndexError_NoMemory;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_entrypoint_segment() {
  entrypoint_segment_ = storage_->get(kDiskAnnEntryPointSegmentId);
  if (!entrypoint_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  if (entrypoint_segment_->data_size() < sizeof(uint32_t)) {
    LOG_ERROR("Invalid segment %s size", kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  const void *data = nullptr;
  if (entrypoint_segment_->read(0, reinterpret_cast<const void **>(&data),
                                sizeof(uint32_t)) != sizeof(uint32_t) ||
      data == nullptr) {
    LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_ReadData;
  }

  uint32_t entrypoint_cnt = 0;
  memcpy(&entrypoint_cnt, data, sizeof(uint32_t));

  size_t entrypoint_data_len = 0;
  size_t expected_segment_size = 0;
  if (entrypoint_cnt > meta_header_.doc_cnt ||
      !checked_multiply(entrypoint_cnt, sizeof(diskann_id_t),
                        &entrypoint_data_len) ||
      !checked_add(sizeof(uint32_t), entrypoint_data_len,
                   &expected_segment_size) ||
      entrypoint_segment_->data_size() != expected_segment_size) {
    LOG_ERROR("Invalid DiskAnn entrypoint count or segment size: count=%u",
              entrypoint_cnt);
    return IndexError_InvalidFormat;
  }

  std::vector<diskann_id_t> entrypoints;
  try {
    entrypoints.resize(entrypoint_cnt);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn entrypoints");
    return IndexError_NoMemory;
  } catch (const std::length_error &) {
    LOG_ERROR("Invalid DiskAnn entrypoint count: %u", entrypoint_cnt);
    return IndexError_InvalidFormat;
  }

  if (entrypoint_data_len != 0 &&
      (entrypoint_segment_->read(sizeof(uint32_t),
                                 reinterpret_cast<const void **>(&data),
                                 entrypoint_data_len) != entrypoint_data_len ||
       data == nullptr)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_ReadData;
  }
  if (entrypoint_data_len != 0) {
    memcpy(entrypoints.data(), data, entrypoint_data_len);
  }
  for (diskann_id_t id : entrypoints) {
    if (id >= meta_header_.doc_cnt) {
      LOG_ERROR("Invalid DiskAnn entrypoint id: %u", id);
      return IndexError_InvalidFormat;
    }
  }
  entrypoints_ = std::move(entrypoints);

  return 0;
}


int DiskAnnSearcherEntity::load_key_mapping_segment() {
  key_mapping_segment_ = storage_->get(kDiskAnnKeyMappingSegmentId);
  if (!key_mapping_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  if (!key_buffer_ || key_buffer_->size() != doc_cnt()) {
    LOG_ERROR("DiskAnn keys must be loaded before the key mapping");
    return IndexError_InvalidFormat;
  }

  size_t key_mapping_data_len = 0;
  if (doc_cnt() > std::numeric_limits<size_t>::max() ||
      !checked_multiply(static_cast<size_t>(doc_cnt()), sizeof(diskann_id_t),
                        &key_mapping_data_len) ||
      key_mapping_segment_->data_size() < key_mapping_data_len) {
    LOG_ERROR("Invalid DiskAnn key mapping segment size");
    return IndexError_InvalidFormat;
  }

  const void *data = nullptr;
  if (key_mapping_segment_->read(0, reinterpret_cast<const void **>(&data),
                                 key_mapping_data_len) !=
          key_mapping_data_len ||
      data == nullptr) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_ReadData;
  }

  try {
    auto key_mapping_buffer = std::make_shared<std::vector<diskann_id_t>>(
        static_cast<size_t>(doc_cnt()));
    memcpy(key_mapping_buffer->data(), data, key_mapping_data_len);

    std::vector<uint8_t> seen_mapping_ids(key_mapping_buffer->size(), 0);
    diskann_key_t previous_key = 0;
    bool have_previous_key = false;
    bool reached_invalid_keys = false;
    for (size_t i = 0; i < key_mapping_buffer->size(); ++i) {
      const diskann_id_t local_id = (*key_mapping_buffer)[i];
      if (local_id >= key_buffer_->size()) {
        LOG_ERROR("Invalid DiskAnn key mapping id: %u", local_id);
        return IndexError_InvalidFormat;
      }
      if (seen_mapping_ids[local_id] != 0) {
        LOG_ERROR("Duplicate DiskAnn key mapping id: %u", local_id);
        return IndexError_InvalidFormat;
      }
      seen_mapping_ids[local_id] = 1;

      const diskann_key_t local_key = (*key_buffer_)[local_id];
      if (local_key == kInvalidKey) {
        reached_invalid_keys = true;
        continue;
      }
      if (reached_invalid_keys ||
          (have_previous_key && local_key <= previous_key)) {
        LOG_ERROR("DiskAnn key mapping is not strictly ordered");
        return IndexError_InvalidFormat;
      }
      previous_key = local_key;
      have_previous_key = true;
    }
    key_mapping_buffer_ = std::move(key_mapping_buffer);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn key mapping buffer");
    return IndexError_NoMemory;
  }

  return 0;
}

//! Get vector local id by key
diskann_id_t DiskAnnSearcherEntity::get_id(diskann_key_t key) const {
  if (key == kInvalidKey || !key_mapping_buffer_ || !key_buffer_ ||
      key_mapping_buffer_->size() != key_buffer_->size()) {
    return kInvalidId;
  }

  //! Do binary search
  size_t start = 0;
  size_t end = key_mapping_buffer_->size();
  while (start < end) {
    const size_t idx = start + (end - start) / 2;
    const diskann_id_t local_id = (*key_mapping_buffer_)[idx];
    if (local_id >= key_buffer_->size()) {
      return kInvalidId;
    }

    const diskann_key_t local_key = (*key_buffer_)[local_id];

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
  if (!key_buffer_ || id >= key_buffer_->size()) {
    return kInvalidKey;
  }
  return (*key_buffer_)[id];
}

}  // namespace core
}  // namespace zvec

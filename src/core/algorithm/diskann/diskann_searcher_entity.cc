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
#include "diskann_util.h"

namespace zvec {
namespace core {

void DiskAnnSearcherEntity::clear() {
  release_storage();
  pq_codes_.reset();
  key_buffer_.reset();
  key_mapping_buffer_.reset();
  entrypoints_.clear();
  meta_.clear();
  meta_header_ = {};
  pq_meta_ = {};
  legacy_pq_layout_ = false;
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

  entity->meta_header_ = meta_header_;
  entity->pq_meta_ = pq_meta_;
  entity->meta_ = meta_;
  entity->legacy_pq_layout_ = legacy_pq_layout_;
  entity->pq_codes_ = pq_codes_;
  if (!pq_codes_) entity->pq_data_segment_ = pq_data_segment_;
  entity->key_buffer_ = key_buffer_;
  entity->key_mapping_buffer_ = key_mapping_buffer_;
  if (!key_buffer_) entity->key_segment_ = key_segment_;
  if (!key_mapping_buffer_) entity->key_mapping_segment_ = key_mapping_segment_;
  entity->entrypoints_ = entrypoints_;

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
  if (read_size != sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              sizeof(DiskAnnPqMeta), read_size);

    return IndexError_ReadData;
  }

  memcpy(reinterpret_cast<uint8_t *>(&pq_meta_), data, sizeof(DiskAnnPqMeta));
  offset += read_size;

  int ret =
      DiskAnnUtil::normalize_pq_meta(meta_, &pq_meta_, &legacy_pq_layout_);
  if (ret != 0) {
    return ret;
  }

  if (pq_meta_.chunk_num == 0 || meta_header_.doc_cnt == 0 ||
      pq_meta_.quantizer_meta_buffer_size == 0 ||
      pq_meta_.chunk_num > meta_.dimension()) {
    LOG_ERROR("Invalid empty DiskAnn PQ metadata");
    return IndexError_InvalidFormat;
  }

  // The serialized quantizer meta buffer follows the PQ meta in this segment;
  // it is NOT parsed here.  The searcher/streamer reads it on demand via
  // read_pq_quantizer_meta_buffer() and constructs the quantizer itself.

  // 2. load pq codes (uint8[chunk_num] per vector)
  pq_data_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqDataSegmentId);
  if (!pq_data_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  size_t code_bytes = meta_header_.doc_cnt * pq_meta_.chunk_num;
  if (pq_data_segment_->data_size() < code_bytes)
    return IndexError_InvalidFormat;
  if (storage_->vec_buffer_pool()) {
    pq_codes_.reset();
    return 0;
  }

  try {
    auto pq_codes = std::make_shared<std::string>(code_bytes, '\0');
    read_size = pq_data_segment_->fetch(0, &(*pq_codes)[0], code_bytes);
    if (read_size != code_bytes) {
      LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
                DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(), code_bytes,
                (size_t)read_size);

      return IndexError_ReadData;
    }
    pq_codes_ = std::move(pq_codes);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn PQ code buffer");
    return IndexError_NoMemory;
  }

  return 0;
}

int DiskAnnSearcherEntity::read_pq_quantizer_meta_buffer(
    std::string *meta_buffer) const {
  if (!meta_buffer || !pq_meta_segment_) {
    return IndexError_InvalidArgument;
  }

  // The meta buffer is stored right after the DiskAnnPqMeta header in the
  // segment.
  const void *data = nullptr;
  size_t read_size = pq_meta_segment_->read(
      sizeof(DiskAnnPqMeta), &data, pq_meta_.quantizer_meta_buffer_size);
  if (read_size != pq_meta_.quantizer_meta_buffer_size) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)(pq_meta_.quantizer_meta_buffer_size), (size_t)read_size);
    return IndexError_ReadData;
  }

  meta_buffer->assign(reinterpret_cast<const char *>(data), read_size);
  return 0;
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
      sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(reinterpret_cast<uint8_t *>(&meta_header_), data,
         sizeof(DiskAnnMetaHeader));

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

  size_t key_data_len = doc_cnt() * sizeof(diskann_key_t);
  if (key_segment_->data_size() < key_data_len) return IndexError_InvalidFormat;
  if (storage_->vec_buffer_pool()) {
    key_buffer_.reset();
    return 0;
  }

  const void *data = nullptr;
  if (key_segment_->read(0, reinterpret_cast<const void **>(&data),
                         key_data_len) != key_data_len) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeySegmentId.c_str());
    return IndexError_ReadData;
  }

  try {
    auto key_buffer = std::make_shared<std::string>(key_data_len, '\0');
    memcpy(&(*key_buffer)[0], data, key_data_len);
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

  const void *data = nullptr;

  if (entrypoint_segment_->read(0, reinterpret_cast<const void **>(&data),
                                sizeof(uint32_t)) != sizeof(uint32_t)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_ReadData;
  }

  uint32_t entrypoint_cnt = 0;
  memcpy(&entrypoint_cnt, data, sizeof(uint32_t));

  if (entrypoint_cnt != 0) {
    size_t entrypoint_data_len = entrypoint_cnt * sizeof(diskann_id_t);

    if (entrypoint_segment_->read(sizeof(uint32_t),
                                  reinterpret_cast<const void **>(&data),
                                  entrypoint_data_len) != entrypoint_data_len) {
      LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
      return IndexError_ReadData;
    }

    entrypoints_.resize(entrypoint_cnt);
    memcpy(&(entrypoints_[0]), data, entrypoint_data_len);
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

  size_t key_mapping_data_len = doc_cnt() * sizeof(diskann_id_t);
  if (key_mapping_segment_->data_size() < key_mapping_data_len)
    return IndexError_InvalidFormat;
  if (storage_->vec_buffer_pool()) {
    key_mapping_buffer_.reset();
    return 0;
  }

  const void *data = nullptr;
  if (key_mapping_segment_->read(0, reinterpret_cast<const void **>(&data),
                                 key_mapping_data_len) !=
      key_mapping_data_len) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_ReadData;
  }

  try {
    auto key_mapping_buffer =
        std::make_shared<std::string>(key_mapping_data_len, '\0');
    memcpy(&(*key_mapping_buffer)[0], data, key_mapping_data_len);
    key_mapping_buffer_ = std::move(key_mapping_buffer);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate DiskAnn key mapping buffer");
    return IndexError_NoMemory;
  }

  return 0;
}

//! Get vector local id by key
diskann_id_t DiskAnnSearcherEntity::get_id(diskann_key_t key) const {
  diskann_id_t id = kInvalidId;
  return find_id(key, &id) == 0 ? id : kInvalidId;
}

int DiskAnnSearcherEntity::find_id(diskann_key_t key, diskann_id_t *id) const
    try {
  if (!id) return IndexError_InvalidArgument;
  *id = kInvalidId;

  //! Do binary search
  diskann_id_t start = 0UL;
  diskann_id_t end = doc_cnt();
  diskann_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    diskann_id_t local_id;
    if (key_mapping_buffer_) {
      std::memcpy(&local_id,
                  key_mapping_buffer_->data() + idx * sizeof(local_id),
                  sizeof(local_id));
    } else {
      IndexStorage::MemoryBlock block;
      if (!key_mapping_segment_ ||
          key_mapping_segment_->read(idx * sizeof(local_id), block,
                                     sizeof(local_id)) != sizeof(local_id) ||
          !block.data())
        return IndexError_ReadData;
      std::memcpy(&local_id, block.data(), sizeof(local_id));
    }
    diskann_key_t local_key;
    int ret = read_key(local_id, &local_key);
    if (ret != 0) return ret;

    if (local_key < key) {
      start = idx + 1;
    } else if (local_key > key) {
      end = idx;
    } else {
      *id = local_id;
      return 0;
    }
  }

  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_ReadData;
}

diskann_key_t DiskAnnSearcherEntity::get_key(diskann_id_t id) const {
  diskann_key_t key = kInvalidKey;
  return read_key(id, &key) == 0 ? key : kInvalidKey;
}

int DiskAnnSearcherEntity::read_key(diskann_id_t id, diskann_key_t *key) const
    try {
  if (!key || id >= doc_cnt()) return IndexError_InvalidArgument;
  const size_t offset = static_cast<size_t>(id) * sizeof(*key);
  if (key_buffer_) {
    std::memcpy(key, key_buffer_->data() + offset, sizeof(*key));
  } else {
    IndexStorage::MemoryBlock block;
    if (!key_segment_ ||
        key_segment_->read(offset, block, sizeof(*key)) != sizeof(*key) ||
        !block.data())
      return IndexError_ReadData;
    std::memcpy(key, block.data(), sizeof(*key));
  }
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_ReadData;
}

}  // namespace core
}  // namespace zvec

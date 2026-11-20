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

namespace {

// Read directly into the retained table. A storage-owned view of the whole
// table can exceed the page-cache budget even when bypass reads are possible.
int LoadTable(const IndexStorage::Segment &segment, uint64_t count,
              size_t width, std::shared_ptr<const std::string> *out) {
  if (count > segment.data_size() / width) return IndexError_ReadData;
  const size_t bytes = static_cast<size_t>(count) * width;
  if (bytes > std::string().max_size()) return IndexError_InvalidLength;
  try {
    auto table = std::make_shared<std::string>(bytes, '\0');
    if (segment.fetch(0, table->data(), bytes) != bytes)
      return IndexError_ReadData;
    *out = std::move(table);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
  return 0;
}

}  // namespace

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
  entity->key_buffer_ = key_buffer_;
  entity->key_mapping_buffer_ = key_mapping_buffer_;
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
  IndexStorage::MemoryBlock data;

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
  read_size = pq_meta_segment_->read(offset, data, sizeof(DiskAnnPqMeta));
  if (read_size != sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              sizeof(DiskAnnPqMeta), read_size);

    return IndexError_ReadData;
  }

  memcpy(reinterpret_cast<uint8_t *>(&pq_meta_), data.data(),
         sizeof(DiskAnnPqMeta));
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

  return LoadTable(*pq_data_segment_, doc_cnt(), pq_meta_.chunk_num,
                   &pq_codes_);
}

int DiskAnnSearcherEntity::read_pq_quantizer_meta_buffer(
    std::string *meta_buffer) const {
  if (!meta_buffer || !pq_meta_segment_) {
    return IndexError_InvalidArgument;
  }

  // The meta buffer is stored right after the DiskAnnPqMeta header in the
  // segment.
  const auto size = pq_meta_.quantizer_meta_buffer_size;
  const size_t segment_size = pq_meta_segment_->data_size();
  if (segment_size < sizeof(DiskAnnPqMeta) ||
      size > segment_size - sizeof(DiskAnnPqMeta) ||
      size > meta_buffer->max_size()) {
    return IndexError_ReadData;
  }
  try {
    std::string buffer(size, '\0');
    if (pq_meta_segment_->fetch(sizeof(DiskAnnPqMeta), buffer.data(), size) !=
        size) {
      LOG_ERROR("Failed to fetch DiskAnn PQ quantizer metadata");
      return IndexError_ReadData;
    }
    meta_buffer->swap(buffer);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
  return 0;
}

int DiskAnnSearcherEntity::load_header_segment() {
  IndexStorage::MemoryBlock data;
  meta_segment_ = storage_->get(kDiskAnnMetaSegmentId);
  if (!meta_segment_ ||
      meta_segment_->data_size() < sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Miss or invalid segment %s", kDiskAnnMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (meta_segment_->read(0, data, sizeof(DiskAnnMetaHeader)) !=
      sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(reinterpret_cast<uint8_t *>(&meta_header_), data.data(),
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

  return LoadTable(*key_segment_, doc_cnt(), sizeof(diskann_key_t),
                   &key_buffer_);
}

int DiskAnnSearcherEntity::load_entrypoint_segment() {
  entrypoint_segment_ = storage_->get(kDiskAnnEntryPointSegmentId);
  if (!entrypoint_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  uint32_t count = 0;
  if (entrypoint_segment_->fetch(0, &count, sizeof(count)) != sizeof(count))
    return IndexError_ReadData;
  const size_t size = entrypoint_segment_->data_size();
  if (size < sizeof(count) ||
      count > (size - sizeof(count)) / sizeof(diskann_id_t))
    return IndexError_ReadData;
  try {
    std::vector<diskann_id_t> entries(count);
    const size_t bytes = entries.size() * sizeof(diskann_id_t);
    if (bytes && entrypoint_segment_->fetch(sizeof(count), entries.data(),
                                            bytes) != bytes)
      return IndexError_ReadData;
    entrypoints_.swap(entries);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
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

  return LoadTable(*key_mapping_segment_, doc_cnt(), sizeof(diskann_id_t),
                   &key_mapping_buffer_);
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
  const diskann_key_t *key_data_ptr =
      reinterpret_cast<const diskann_key_t *>(key_buffer_->data());

  return key_data_ptr[id];
}

}  // namespace core
}  // namespace zvec

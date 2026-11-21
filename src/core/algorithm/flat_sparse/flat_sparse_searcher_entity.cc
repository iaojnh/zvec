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

#include "flat_sparse_searcher_entity.h"
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_helper.h>
#include "utility/legacy_read_buffer.h"
#include "flat_sparse_utility.h"

namespace zvec {
namespace core {

FlatSparseSearcherEntity::FlatSparseSearcherEntity() = default;

int FlatSparseSearcherEntity::load(const IndexStorage::Pointer &container,
                                   const IndexMeta &index_meta) {
  if (container_) {
    LOG_ERROR("An storage instance is already opened");
    return IndexError_Duplicate;
  }

  int ret = this->load_container(container);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to load storage index");
    return ret;
  }

  if (init_measure(index_meta) != 0) {
    LOG_ERROR("Failed to init measure");
    return IndexError_InvalidFormat;
  }

  container_ = container;
  return 0;
}

int FlatSparseSearcherEntity::init_measure(const IndexMeta &meta) {
  measure_ = IndexFactory::CreateMetric(meta.metric_name());
  if (!measure_) {
    LOG_ERROR("Failed to create measure %s", meta.metric_name().c_str());
    return IndexError_NoExist;
  }
  int ret = measure_->init(meta, meta.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failled to init measure, ret=%d", ret);
    return ret;
  }

  if (!measure_->sparse_distance()) {
    LOG_ERROR("Invalid measure distance");
    return IndexError_InvalidArgument;
  }

  search_sparse_distance_ = measure_->sparse_distance();

  if (measure_->query_metric() && measure_->query_metric()->distance()) {
    search_sparse_distance_ = measure_->query_metric()->sparse_distance();
  }
  sparse_unit_size_ = meta.unit_size();

  return 0;
}

int FlatSparseSearcherEntity::load_container(
    const IndexStorage::Pointer &container) {
  // meta
  auto segment = container->get(PARAM_FLAT_SPARSE_META_SEG_ID);
  if (!segment || segment->data_size() < sizeof(meta_)) {
    LOG_ERROR("Missing segment %s, or invalid segment size",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  if (ailego_unlikely(segment->fetch(0, &meta_, sizeof(meta_)) !=
                      sizeof(meta_))) {
    LOG_ERROR("Failed to read meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_ReadData;
  }

  // keys segment
  keys_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID);
  if (!keys_chunk_) {
    LOG_ERROR("Missing segment %s", PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  // mapping segment
  mapping_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID);
  if (!mapping_chunk_) {
    LOG_ERROR("Missing segment %s",
              PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  // offset segment
  sparse_offset_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID);
  if (!sparse_offset_chunk_) {
    LOG_ERROR("Missing segment %s",
              PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  // data segment
  sparse_data_chunk_ = container->get(PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID);
  if (!sparse_data_chunk_) {
    LOG_ERROR("Missing segment %s", PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  return 0;
}

int FlatSparseSearcherEntity::unload() {
  container_.reset();
  sparse_data_chunk_.reset();
  sparse_offset_chunk_.reset();
  keys_chunk_.reset();
  mapping_chunk_.reset();

  return 0;
}

FlatSparseSearcherEntity::Pointer FlatSparseSearcherEntity::clone() const {
  auto entity = new (std::nothrow)
      FlatSparseSearcherEntity(meta_, sparse_data_chunk_, sparse_offset_chunk_,
                               keys_chunk_, mapping_chunk_);
  return FlatSparseSearcherEntity::Pointer(entity);
}

int FlatSparseSearcherEntity::get_sparse_vector_ptr_by_id(
    node_id_t id, const void **sparse_vector_ptr,
    uint32_t *sparse_vector_len_ptr) const {
  *sparse_vector_ptr = nullptr;
  *sparse_vector_len_ptr = 0;
  if (id >= doc_cnt()) return IndexError_OutOfRange;
  const size_t offset_chunk_offset =
      static_cast<size_t>(id) * offset_size_per_node();
  alignas(uint64_t) char offset_info[sizeof(uint64_t) + sizeof(uint32_t)];
  if (sparse_offset_chunk_->fetch(offset_chunk_offset, offset_info,
                                  sizeof(offset_info)) != sizeof(offset_info))
    return IndexError_ReadData;

  // sparse offset
  uint64_t sparse_offset = *(uint64_t *)offset_info;
  uint32_t sparse_vector_len =
      *(uint32_t *)((uint8_t *)offset_info + sizeof(uint64_t));

  if (sparse_vector_len > 0) {
    const void *sparse_data =
        get_sparse_vector_data(sparse_offset, sparse_vector_len);
    if (ailego_unlikely(sparse_data == nullptr)) {
      LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)sparse_offset,
                sparse_vector_len);

      return IndexError_ReadData;
    }
    *sparse_vector_ptr = sparse_data;
    *sparse_vector_len_ptr = sparse_vector_len;
  }

  return 0;
}

int FlatSparseSearcherEntity::get_sparse_vector_ptr_by_id(
    node_id_t id, IndexStorage::MemoryBlock &block, uint32_t *length) const {
  block.reset();
  *length = 0;
  if (id >= doc_cnt()) return IndexError_OutOfRange;
  // Copy the small offset record so only the vector data needs a live pin.
  char record[sizeof(uint64_t) + sizeof(uint32_t)];
  if (sparse_offset_chunk_->fetch(static_cast<size_t>(id) * sizeof(record),
                                  record, sizeof(record)) != sizeof(record))
    return IndexError_ReadData;
  uint64_t offset;
  std::memcpy(&offset, record, sizeof(offset));
  std::memcpy(length, record + sizeof(offset), sizeof(*length));
  if (!*length) return 0;
  if (sparse_data_chunk_->read(offset, block, *length) != *length)
    return IndexError_ReadData;
  return 0;
}

const void *FlatSparseSearcherEntity::get_sparse_vector_data(
    uint64_t offset, uint32_t length) const {
  static thread_local LegacyReadBuffer buffer;
  IndexStorage::MemoryBlock block;
  auto size = sparse_data_chunk_->read(offset, block, length);
  if (size != length) {
    LOG_ERROR(
        "read sparse vector data failed: offset=%zu, "
        "length=%u, size=%zu",
        (size_t)offset, length, size);
    return nullptr;
  }
  return buffer.assign(block, length);
}


node_id_t FlatSparseSearcherEntity::get_id(uint64_t key) const {
  if (ailego_unlikely(!mapping_chunk_)) {
    LOG_ERROR("Index missing mapping segment");
    return kInvalidNodeId;
  }

  //! Do binary search
  node_id_t start = 0UL;
  node_id_t end = doc_cnt();
  node_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    node_id_t local_id;
    if (ailego_unlikely(
            mapping_chunk_->fetch(static_cast<size_t>(idx) * sizeof(node_id_t),
                                  &local_id,
                                  sizeof(local_id)) != sizeof(node_id_t))) {
      LOG_ERROR("Read key from segment failed");
      throw ReadError{IndexError_ReadData};
    }
    uint64_t mkey;
    if (ailego_unlikely(
            keys_chunk_->fetch(static_cast<size_t>(local_id) * sizeof(uint64_t),
                               &mkey, sizeof(mkey)) != sizeof(uint64_t))) {
      LOG_ERROR("Read key from segment failed");
      throw ReadError{IndexError_ReadData};
    }
    if (mkey < key) {
      start = idx + 1;
    } else if (mkey > key) {
      end = idx;
    } else {
      return local_id;
    }
  }
  return kInvalidNodeId;
}

uint64_t FlatSparseSearcherEntity::get_key(node_id_t id) const {
  if (id >= doc_cnt()) return kInvalidKey;
  uint64_t key;
  if (ailego_unlikely(
          keys_chunk_->fetch(static_cast<size_t>(id) * sizeof(uint64_t), &key,
                             sizeof(key)) != sizeof(uint64_t))) {
    LOG_ERROR("Read key from segment failed");
    throw ReadError{IndexError_ReadData};
  }
  return key;
}

}  // namespace core
}  // namespace zvec

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
#include <ailego/pattern/scope_guard.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

namespace {

// These resident buffers share the process RSS limit with the page cache.
// Their lifetime is independent of the capacity configured for native I/O.
class ProcessHeapReservation {
 public:
  ProcessHeapReservation() = default;
  ProcessHeapReservation(const ProcessHeapReservation &) = delete;
  ProcessHeapReservation &operator=(const ProcessHeapReservation &) = delete;
  ~ProcessHeapReservation() {
    if (bytes_) {
      ailego::MemoryLimitPool::get_instance().release_heap_process_backing(
          bytes_, stamp_);
    }
  }

  template <typename Allocate>
  void allocate(size_t pending, Allocate &&allocate) {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (!pool.try_reserve_process_backing(pending)) throw std::bad_alloc();
    const auto refund = ailego::ScopeGuard::Make([&] {
      if (pending) pool.finish_process_backing(pending, false);
    });
    const size_t bytes = allocate();
    if (bytes > pending) throw std::bad_alloc();
    stamp_ = pool.finish_process_backing(bytes, true);
    bytes_ = bytes;
    pending -= bytes;
  }

 private:
  size_t bytes_{0};
  ailego::ProcessRssBudget::MaterializationStamp stamp_{};
};

template <typename T>
int LoadTable(const IndexStorage::Segment &segment, size_t offset,
              uint64_t count, std::shared_ptr<const std::vector<T>> *out) {
  const size_t size = segment.data_size();
  if (offset > size || count > (size - offset) / sizeof(T))
    return IndexError_ReadData;
  if (count > std::vector<T>().max_size()) return IndexError_InvalidLength;
  const size_t bytes = static_cast<size_t>(count) * sizeof(T);
  struct Table {
    ProcessHeapReservation memory;
    std::vector<T> values;
    ailego::MemoryLimitPool::HeapResidentRange resident;
  };
  try {
    auto table = std::make_shared<Table>();
    if (bytes) {
      table->memory.allocate(bytes, [&] {
        table->values.resize(count);
        return table->values.capacity() * sizeof(T);
      });
      table->resident.observe(table->values.data(), bytes);
      if (segment.fetch(offset, table->values.data(), bytes) != bytes)
        return segment.read_error();
    }
    *out = std::shared_ptr<const std::vector<T>>(table, &table->values);
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
  entrypoints_.reset();
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
  if (!storage) return IndexError_InvalidArgument;
  clear();
  bool loaded = false;
  const auto rollback = ailego::ScopeGuard::Make([&] {
    if (!loaded) clear();
  });
  meta_ = meta;
  storage_ = std::move(storage);

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

  loaded = true;
  return 0;
}

int DiskAnnSearcherEntity::load_pq_segment() {
  // load pq meta
  pq_meta_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqMetaSegmentId);
  if (!pq_meta_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  DiskAnnPqMeta pq_meta;
  const size_t read_size =
      pq_meta_segment_->fetch(0, &pq_meta, sizeof(pq_meta));
  if (read_size != sizeof(pq_meta)) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(), sizeof(pq_meta),
              read_size);
    return pq_meta_segment_->read_error();
  }
  pq_meta_ = pq_meta;

  int ret =
      DiskAnnUtil::normalize_pq_meta(meta_, &pq_meta_, &legacy_pq_layout_);
  if (ret != 0) {
    return ret;
  }

  if (pq_meta_.chunk_num == 0 || pq_meta_.quantizer_meta_buffer_size == 0 ||
      pq_meta_.chunk_num > meta_.dimension()) {
    LOG_ERROR("Invalid empty DiskAnn PQ metadata");
    return IndexError_InvalidFormat;
  }

  // The serialized quantizer meta buffer follows the PQ meta in this segment;
  // The searcher and streamer call load_quantizer() after loading the tables.

  // 2. load pq codes (uint8[chunk_num] per vector)
  pq_data_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqDataSegmentId);
  if (!pq_data_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  return LoadTable(*pq_data_segment_, 0, doc_cnt() * pq_meta_.chunk_num,
                   &pq_codes_);
}

int DiskAnnSearcherEntity::load_quantizer(
    turbo::Quantizer::Pointer &quantizer) const {
  if (!pq_meta_segment_) return IndexError_InvalidArgument;

  const uint64_t declared_size = pq_meta_.quantizer_meta_buffer_size;
  const size_t segment_size = pq_meta_segment_->data_size();
  if (segment_size < sizeof(DiskAnnPqMeta) ||
      declared_size > segment_size - sizeof(DiskAnnPqMeta)) {
    return IndexError_ReadData;
  }
  const size_t size = static_cast<size_t>(declared_size);
  ProcessHeapReservation memory;
  std::string buffer;
  ailego::MemoryLimitPool::HeapResidentRange resident;
  if (size > buffer.max_size() ||
      size >= std::numeric_limits<size_t>::max() / 2)
    return IndexError_InvalidLength;
  try {
    // Include the terminator and space for the string's rounded capacity.
    const size_t inline_capacity = buffer.capacity();
    const size_t reservation = size > inline_capacity ? 2 * (size + 1) : 0;
    memory.allocate(reservation, [&] {
      buffer.resize(size);
      return reservation ? buffer.capacity() + 1 : 0;
    });
    resident.observe(buffer.data(), size);
    if (pq_meta_segment_->fetch(sizeof(DiskAnnPqMeta), buffer.data(), size) !=
        size) {
      LOG_ERROR("Failed to fetch DiskAnn PQ quantizer metadata");
      return pq_meta_segment_->read_error();
    }
    quantizer = DiskAnnUtil::create_quantizer_from_meta_buffer(
        buffer, meta_, pq_meta_.chunk_num, legacy_pq_layout_);
    return quantizer ? 0 : IndexError_NoExist;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
}

int DiskAnnSearcherEntity::load_header_segment() {
  meta_segment_ = storage_->get(kDiskAnnMetaSegmentId);
  if (!meta_segment_ ||
      meta_segment_->data_size() < sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Miss or invalid segment %s", kDiskAnnMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  DiskAnnMetaHeader header;
  if (meta_segment_->fetch(0, &header, sizeof(header)) != sizeof(header)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return meta_segment_->read_error();
  }
  if (header.doc_cnt == 0 || header.doc_cnt > kInvalidId ||
      header.medoid >= header.doc_cnt) {
    return IndexError_InvalidFormat;
  }
  meta_header_ = header;
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

  return LoadTable(*key_segment_, 0, doc_cnt(), &key_buffer_);
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
    return entrypoint_segment_->read_error();
  const int ret =
      LoadTable(*entrypoint_segment_, sizeof(count), count, &entrypoints_);
  if (ret != 0) return ret;
  for (diskann_id_t id : *entrypoints_)
    if (id >= doc_cnt()) return IndexError_InvalidFormat;
  return 0;
}

int DiskAnnSearcherEntity::load_key_mapping_segment() {
  key_mapping_segment_ = storage_->get(kDiskAnnKeyMappingSegmentId);
  if (!key_mapping_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  return LoadTable(*key_mapping_segment_, 0, doc_cnt(), &key_mapping_buffer_);
}

//! Get vector local id by key
diskann_id_t DiskAnnSearcherEntity::get_id(diskann_key_t key) const {
  const auto &key_mapping = *key_mapping_buffer_;
  const auto &keys = *key_buffer_;

  //! Do binary search
  diskann_id_t start = 0UL;
  diskann_id_t end = doc_cnt();
  diskann_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    const diskann_id_t local_id = key_mapping[idx];
    if (local_id >= keys.size()) return kInvalidId;

    const diskann_key_t local_key = keys[local_id];

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
  return (*key_buffer_)[id];
}

}  // namespace core
}  // namespace zvec

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

#include "flat_sparse_streamer_entity.h"
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <ailego/pattern/defer.h>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_error.h>
#include "algorithm/flat/flat_dump_keys.h"
#include "algorithm/flat/flat_utility.h"
#include "utility/legacy_read_buffer.h"
#include "flat_sparse_index_format.h"
#include "flat_sparse_utility.h"

namespace zvec {
namespace core {

FlatSparseStreamerEntity::FlatSparseStreamerEntity(IndexStreamer::Stats &stats)
    : stats_(stats) {}

int FlatSparseStreamerEntity::open(IndexStorage::Pointer storage,
                                   const IndexMeta &meta) try {
  if (storage_) {
    LOG_ERROR("An storage instance is already opened");
    return IndexError_Duplicate;
  }

  use_id_map_ = true;
  meta.streamer_params().get(PARAM_FLAT_USE_ID_MAP, &use_id_map_);
  keys_map_lock_ = std::make_shared<ailego::SharedMutex>();
  if (!keys_map_lock_) {
    LOG_ERROR("FlatSparseStreamerEntity new object failed");
    return IndexError_NoMemory;
  }
  keys_map_ = std::make_shared<KeyMap>();

  if (storage->get(PARAM_FLAT_SPARSE_META_SEG_ID) ||
      storage->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID)) {
    int ret = this->load_storage(storage, meta);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to load storage index");
      return ret;
    }
  } else {
    int ret = this->init_storage(storage, meta);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to load storage index");
      return ret;
    }
  }

  if (init_metric(meta) != 0) {
    LOG_ERROR("Failed to init metric");
    return IndexError_InvalidFormat;
  }

  // reserve data chunk
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_MAX_DATA_CHUNK_CNT,
                             &max_data_chunk_cnt_);
  sparse_data_chunks_.reserve(max_data_chunk_cnt_);

  // reserve offset chunk
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_MAX_DOC_CNT,
                             &max_doc_cnt_);
  sparse_offset_chunks_.reserve(max_doc_cnt_ / doc_cnt_per_offset_chunk() + 1);
  sparse_unit_size_ = meta.unit_size();

  LOG_DEBUG(
      "FlatSparseStreamerEntity open success, doc_count[%u], "
      "data_chunk_size[%u], offset_chunk_size[%u], data_chunk_count[%zu], "
      "offset_chunk_count[%zu]",
      meta_.doc_cnt, streamer_meta_.data_chunk_size,
      streamer_meta_.offset_chunk_size, sparse_data_chunks_.size(),
      sparse_offset_chunks_.size());

  storage_ = storage;
  return 0;
} catch (const std::bad_alloc &) {
  sparse_data_chunks_.clear();
  sparse_offset_chunks_.clear();
  keys_map_.reset();
  storage_.reset();
  return IndexError_NoMemory;
}

int FlatSparseStreamerEntity::init_metric(const IndexMeta &meta) {
  metric_ = IndexFactory::CreateMetric(meta.metric_name());
  if (!metric_) {
    LOG_ERROR("Failed to create metric %s", meta.metric_name().c_str());
    return IndexError_NoExist;
  }
  int ret = metric_->init(meta, meta.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failled to init metric, ret=%d", ret);
    return ret;
  }

  if (!metric_->sparse_distance()) {
    LOG_ERROR("Invalid metric distance");
    return IndexError_InvalidArgument;
  }

  search_sparse_distance_ = metric_->sparse_distance();

  if (metric_->query_metric() && metric_->query_metric()->distance()) {
    search_sparse_distance_ = metric_->query_metric()->sparse_distance();
  }

  return 0;
}

int FlatSparseStreamerEntity::load_storage(IndexStorage::Pointer storage,
                                           const IndexMeta &meta) {
  size_t index_size{0};

  // load meta
  auto segment = storage->get(PARAM_FLAT_SPARSE_META_SEG_ID);

  if (!segment || segment->data_size() < sizeof(meta_)) {
    LOG_ERROR("Missing segment %s, or invalid segment size",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock data_block;
  if (ailego_unlikely(segment->read(0, data_block, sizeof(meta_)) !=
                      sizeof(meta_))) {
    LOG_ERROR("Failed to read meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_ReadData;
  }
  meta_ = *(reinterpret_cast<const decltype(meta_) *>(data_block.data()));
  index_size += segment->capacity();

  // load streamer meta
  segment = storage->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID);
  if (!segment || segment->data_size() < sizeof(streamer_meta_)) {
    LOG_ERROR("Missing segment %s, or invalid segment size",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  if (ailego_unlikely(segment->read(0, data_block, sizeof(streamer_meta_)) !=
                      sizeof(streamer_meta_))) {
    LOG_ERROR("Failed to read streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_ReadData;
  }
  streamer_meta_ =
      *(reinterpret_cast<const decltype(streamer_meta_) *>(data_block.data()));
  index_size += segment->capacity();

  uint32_t meta_data_chunk_size{streamer_meta_.data_chunk_size};
  uint32_t meta_offset_chunk_size{streamer_meta_.offset_chunk_size};
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_DATA_CHUNK_SIZE,
                             &meta_data_chunk_size);
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_OFFSET_CHUNK_SIZE,
                             &meta_offset_chunk_size);
  if (streamer_meta_.data_chunk_size != meta_data_chunk_size ||
      streamer_meta_.offset_chunk_size != meta_offset_chunk_size) {
    LOG_ERROR(
        "Invalid streamer meta chunk size data[%u] offset[%u], expect data[%u] "
        "offset[%u]",
        streamer_meta_.data_chunk_size, streamer_meta_.offset_chunk_size,
        meta_data_chunk_size, meta_offset_chunk_size);
    return IndexError_InvalidFormat;
  }

  // check chunk cnt
  if (streamer_meta_.data_chunk_count > max_data_chunk_cnt_ ||
      meta_.doc_cnt > max_doc_cnt_) {
    LOG_ERROR(
        "Invalid data chunk count[%u] doc count[%u], expect less than "
        "chunk count[%u] doc count[%u]",
        streamer_meta_.data_chunk_count, meta_.doc_cnt, max_data_chunk_cnt_,
        max_doc_cnt_);
    return IndexError_InvalidFormat;
  }

  // load offset chunks
  for (size_t i = 0; i < streamer_meta_.offset_chunk_count; ++i) {
    std::string segment_id =
        ailego::StringHelper::Concat(PARAM_FLAT_SPARSE_OFFSET_SEG_ID_PREFIX, i);
    segment = storage->get(segment_id);
    if (!segment) {
      LOG_ERROR("Missing segment %s", segment_id.c_str());
      return IndexError_InvalidFormat;
    }
    sparse_offset_chunks_.emplace_back(segment);
    index_size += segment->capacity();
  }
  // load data chunks
  for (size_t i = 0; i < streamer_meta_.data_chunk_count; ++i) {
    std::string segment_id =
        ailego::StringHelper::Concat(PARAM_FLAT_SPARSE_DATA_SEG_ID_PREFIX, i);
    segment = storage->get(segment_id);
    if (!segment) {
      LOG_ERROR("Missing segment %s", segment_id.c_str());
    }
    sparse_data_chunks_.emplace_back(segment);
    index_size += segment->capacity();
  }

  // load keys
  for (node_id_t i = 0; use_id_map_ && i < meta_.doc_cnt; ++i) {
    uint64_t key;
    int ret = read_key(i, &key);
    if (ret != 0) return ret;
    if (key != kInvalidKey) (*keys_map_)[key] = i;
  }

  stats_.set_index_size(index_size);
  stats_.set_check_point(storage->check_point());
  stats_.set_create_time(meta_.create_time);
  stats_.set_update_time(meta_.update_time);
  stats_.set_loaded_count(meta_.doc_cnt);

  return 0;
}

int FlatSparseStreamerEntity::init_storage(IndexStorage::Pointer storage,
                                           const IndexMeta &meta) {
  meta_.create_time = ailego::Realtime::Seconds();
  stats_.set_create_time(meta_.create_time);
  meta_.update_time = ailego::Realtime::Seconds();
  stats_.set_update_time(meta_.update_time);
  meta_.doc_cnt = 0;

  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_DATA_CHUNK_SIZE,
                             &streamer_meta_.data_chunk_size);
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_OFFSET_CHUNK_SIZE,
                             &streamer_meta_.offset_chunk_size);

  // append meta segment
  size_t size = ailego_align(sizeof(meta_), ailego::MemoryHelper::PageSize());
  int ret = storage->append(PARAM_FLAT_SPARSE_META_SEG_ID, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to append meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return ret;
  }
  auto segment = storage->get(PARAM_FLAT_SPARSE_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("Failed to write meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  *stats_.mutable_index_size() += size;

  // append streamer meta segment
  size = ailego_align(sizeof(streamer_meta_), ailego::MemoryHelper::PageSize());
  ret = storage->append(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to append streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return ret;
  }
  segment = storage->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &streamer_meta_, sizeof(streamer_meta_)) !=
      sizeof(streamer_meta_)) {
    LOG_ERROR("Failed to write streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  *stats_.mutable_index_size() += size;

  return 0;
}

int FlatSparseStreamerEntity::close() {
  storage_.reset();
  sparse_data_chunks_.clear();
  sparse_offset_chunks_.clear();

  keys_map_lock_.reset();
  keys_map_.reset();

  return 0;
}

int FlatSparseStreamerEntity::flush(uint64_t checkpoint) {
  // flush meta
  meta_.update_time = ailego::Realtime::Seconds();
  stats_.set_update_time(meta_.update_time);
  auto segment = storage_->get(PARAM_FLAT_SPARSE_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("Failed to write meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  // flush streamer meta
  streamer_meta_.data_chunk_count = sparse_data_chunks_.size();
  streamer_meta_.offset_chunk_count = sparse_offset_chunks_.size();
  segment = storage_->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &streamer_meta_, sizeof(streamer_meta_)) !=
      sizeof(streamer_meta_)) {
    LOG_ERROR("Failed to write streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  if (checkpoint != 0) {
    storage_->refresh(checkpoint);
  }
  int ret = storage_->flush();
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to flush storage for %s", IndexError::What(ret));
    return ret;
  }
  if (checkpoint != 0) {
    stats_.set_check_point(checkpoint);
  }

  return 0;
}

int FlatSparseStreamerEntity::dump(const IndexDumper::Pointer &dumper) {
  ailego::ElapsedTime stamp;

  int ret;
  // meta
  ret = dump_meta(dumper.get());
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_meta = stamp.milli_seconds();

  // offset & data
  ret = dump_offset_data(dumper.get());
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_offset_data = stamp.milli_seconds() - duration_dump_meta;

  // keys
  FlatDumpKeys keys;
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    uint64_t key = kInvalidKey;
    ret = read_key(id, &key);
    if (ret != 0) return ret;
    ret = keys.append(key);
    if (ret != 0) return ret;
  }
  ret = keys.write_keys(dumper.get(), PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID);
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_keys =
      stamp.milli_seconds() - duration_dump_offset_data - duration_dump_meta;

  // mapping
  ret = keys.write_mapping(dumper.get(), PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID);
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_mapping = stamp.milli_seconds() -
                               duration_dump_offset_data - duration_dump_meta -
                               duration_dump_keys;

  LOG_INFO(
      "Dump index meta: %zu ms, offset & data: %zu ms, keys: %zu ms, "
      "mapping: %zu ms",
      (size_t)duration_dump_meta, (size_t)duration_dump_offset_data,
      (size_t)duration_dump_keys, (size_t)duration_dump_mapping);

  return 0;
}

int FlatSparseStreamerEntity::dump_offset_data(IndexDumper *dumper) {
  ailego::ElapsedTime stamp;

  uint64_t written_length = 0;

  // write data
  int ret;
  node_id_t total_doc_cnt = doc_cnt();
  for (node_id_t node_id = 0; node_id < total_doc_cnt; node_id++) {
    uint32_t target_vector_len;
    IndexStorage::MemoryBlock target_vector_block;
    ret = get_sparse_vector_ptr_by_id(node_id, target_vector_block,
                                      &target_vector_len);
    if (ret != 0) {
      LOG_ERROR("Failed to get vector, node_id=%u, error: %s", node_id,
                IndexError::What(ret));
      return ret;
    }
    const void *target_vector = target_vector_block.data();
    ret = dump_sparse_vector_data(target_vector, target_vector_len, dumper);
    if (ret != 0) {
      LOG_ERROR("Failed to dump sparse vector data, node_id=%u, error: %s",
                node_id, IndexError::What(ret));
      return ret;
    }

    written_length += target_vector_len;
  }

  // append data segment
  if (dumper->append(PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID, written_length, 0,
                     0) != 0) {
    LOG_ERROR("append data segment failed");
    return IndexError_WriteData;
  }

  auto duration_dump_data = stamp.milli_seconds();

  // The streamer already persists each row length. Re-read that small
  // metadata in a second pass instead of retaining or spilling an N-row copy.
  constexpr size_t kRecordBytes = sizeof(uint64_t) + sizeof(uint32_t);
  std::array<char, 1024 * kRecordBytes> batch{};
  uint64_t offset = 0;
  for (size_t first = 0; first < total_doc_cnt;) {
    const size_t count = std::min(size_t{1024}, total_doc_cnt - first);
    for (size_t i = 0; i < count; ++i) {
      const size_t id = first + i;
      const size_t chunk = id / get_offset_info_number_per_chunk();
      const size_t pos =
          id % get_offset_info_number_per_chunk() * offset_size_per_node() +
          sizeof(uint64_t);
      uint32_t length = 0;
      if (sparse_offset_chunks_[chunk]->fetch(pos, &length, sizeof(length)) !=
          sizeof(length))
        return IndexError_ReadData;
      std::memcpy(batch.data() + i * kRecordBytes, &offset, sizeof(offset));
      std::memcpy(batch.data() + i * kRecordBytes + sizeof(offset), &length,
                  sizeof(length));
      offset += length;
    }
    const size_t bytes = count * kRecordBytes;
    if (dumper->write(batch.data(), bytes) != bytes)
      return IndexError_WriteData;
    first += count;
  }
  if (offset != written_length) return IndexError_InvalidFormat;
  ret = dumper->append(PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID,
                       static_cast<size_t>(total_doc_cnt) * kRecordBytes, 0, 0);
  if (ret != 0) return ret;

  auto duration_dump_offset = stamp.milli_seconds() - duration_dump_data;

  LOG_INFO("Dump offset: %zu ms, data: %zu ms", (size_t)duration_dump_offset,
           (size_t)duration_dump_data);

  return 0;
}

int FlatSparseStreamerEntity::dump_sparse_vector_data(const void *data,
                                                      uint32_t length,
                                                      IndexDumper *dumper) {
  if (dumper->write(data, length) != length) {
    return IndexError_WriteData;
  }
  return 0;
}

int FlatSparseStreamerEntity::dump_meta(IndexDumper *dumper) {
  if (dumper->write(&meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("write meta failed");
    return IndexError_WriteData;
  }

  size_t meta_padding_size = ailego_align(sizeof(meta_), 32) - sizeof(meta_);
  if (meta_padding_size) {
    std::string padding(meta_padding_size, '\0');
    if (dumper->write(padding.data(), meta_padding_size) != meta_padding_size) {
      LOG_ERROR("write meta padding failed");
      return IndexError_WriteData;
    }
  }
  return dumper->append(PARAM_FLAT_SPARSE_META_SEG_ID, sizeof(meta_),
                        meta_padding_size, 0);
}

FlatSparseStreamerEntity::Pointer FlatSparseStreamerEntity::clone() const try {
  auto entity = new (std::nothrow) FlatSparseStreamerEntity(
      stats_, meta_, streamer_meta_, keys_map_lock_, keys_map_,
      sparse_data_chunks_, sparse_offset_chunks_, use_id_map_);
  return FlatSparseStreamerEntity::Pointer(entity);
} catch (const std::bad_alloc &) {
  return nullptr;
}

int FlatSparseStreamerEntity::add(uint64_t key, std::string_view sparse_vector,
                                  const uint32_t sparse_count) try {
  if (!use_id_map_) {
    if (key >= kInvalidNodeId) return IndexError_InvalidArgument;
    return add_vector_with_id(static_cast<node_id_t>(key), sparse_vector,
                              sparse_count);
  }
  uint32_t sparse_vector_len = sparse_vector.size();

  sparse_vector_len = AlignSize(sparse_vector_len);

  if (sparse_vector_len > streamer_meta_.data_chunk_size) {
    LOG_ERROR(
        "Sparse Vector Length exceed the chunk size, sparse vec len: %u, chunk "
        "size: %u",
        sparse_vector_len, streamer_meta_.data_chunk_size);
    return IndexError_InvalidArgument;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  node_id_t local_id = doc_cnt();

  if (ailego_unlikely(local_id >= max_doc_cnt_)) {
    LOG_ERROR("Add vector failed for exceed max doc count: %u", max_doc_cnt_);
    return IndexError_IndexFull;
  }

  // duplicate check
  if (ailego_unlikely(get_id(key) != kInvalidNodeId)) {
    LOG_WARN("Try to add duplicate key, ignore it");
    return IndexError_Duplicate;
  }

  auto pending_key = prepare_key(key, local_id);

  uint32_t data_index, data_offset, offset_index, offset_offset;
  int ret = write_sparse_vector_to_chunk(sparse_vector, sparse_vector_len,
                                         data_index, data_offset);
  if (ret != 0) return ret;
  bool committed = false;
  AILEGO_DEFER([&] {
    if (!committed) sparse_data_chunks_[data_index]->resize(data_offset);
  });
  ret = get_new_sparse_offset_chunk(offset_index, offset_offset);
  if (ret != 0) return ret;
  const uint64_t sparse_offset = (uint64_t{data_index} << 32U) + data_offset;
  ret = write_sparse_offset_to_chunk(offset_index, offset_offset, sparse_offset,
                                     sparse_vector_len, key);
  if (ret != 0) return ret;
  committed = true;

  // Direct IDs are already stored in the offset records.
  publish_key(key, local_id, std::move(pending_key));

  inc_doc_count();
  inc_total_sparse_count(sparse_count);

  return 0;
} catch (const ReadError &error) {
  return error.status;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int FlatSparseStreamerEntity::add_vector_with_id(
    uint32_t id, std::string_view sparse_vector,
    const uint32_t sparse_count) try {
  if (id >= max_doc_cnt_) return IndexError_IndexFull;
  uint32_t sparse_vector_len = sparse_vector.size();

  sparse_vector_len = AlignSize(sparse_vector_len);

  if (sparse_vector_len > streamer_meta_.data_chunk_size) {
    LOG_ERROR(
        "Sparse Vector Length exceed the chunk size, sparse vec len: %u, chunk "
        "size: %u",
        sparse_vector_len, streamer_meta_.data_chunk_size);
    return IndexError_InvalidArgument;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  int ret = 0;
  if (id >= doc_cnt()) {
    for (auto i = doc_cnt(); i <= id; i++) {
      node_id_t local_id = doc_cnt();
      if (ailego_unlikely(local_id >= max_doc_cnt_)) {
        LOG_ERROR("Add vector failed for exceed max doc count: %u",
                  max_doc_cnt_);
        return IndexError_IndexFull;
      }
      uint32_t sparse_data_chunk_index, sparse_data_chunk_offset,
          sparse_offset_chunk_index, sparse_offset_chunk_offset;
      auto pending_key = prepare_key(i, i < id ? kInvalidNodeId : i);
      uint32_t stored_length = i < id ? 0 : sparse_vector_len;
      if (i < id) {
        ret = write_sparse_vector_to_chunk("", 0, sparse_data_chunk_index,
                                           sparse_data_chunk_offset);
      } else {
        ret = write_sparse_vector_to_chunk(sparse_vector, sparse_vector_len,
                                           sparse_data_chunk_index,
                                           sparse_data_chunk_offset);
      }
      if (ret != 0) return ret;
      bool committed = false;
      AILEGO_DEFER([&] {
        if (!committed)
          sparse_data_chunks_[sparse_data_chunk_index]->resize(
              sparse_data_chunk_offset);
      });
      uint64_t sparse_offset =
          ((uint64_t)sparse_data_chunk_index << 32U) + sparse_data_chunk_offset;
      ret = get_new_sparse_offset_chunk(sparse_offset_chunk_index,
                                        sparse_offset_chunk_offset);
      if (ret != 0) return ret;
      uint64_t written_key = kInvalidKey;
      if (i == id) {
        written_key = i;
      }
      ret = write_sparse_offset_to_chunk(
          sparse_offset_chunk_index, sparse_offset_chunk_offset, sparse_offset,
          stored_length, written_key);
      if (ret != 0) return ret;
      committed = true;
      publish_key(i, written_key, std::move(pending_key));
      inc_doc_count();
    }
  } else {
    auto pending_key = prepare_key(id, id);
    uint32_t sparse_data_chunk_index, sparse_data_chunk_offset;
    ret = write_sparse_vector_to_chunk(sparse_vector, sparse_vector_len,
                                       sparse_data_chunk_index,
                                       sparse_data_chunk_offset);
    if (ret != 0) return ret;
    bool committed = false;
    AILEGO_DEFER([&] {
      if (!committed)
        sparse_data_chunks_[sparse_data_chunk_index]->resize(
            sparse_data_chunk_offset);
    });
    uint64_t sparse_offset =
        ((uint64_t)sparse_data_chunk_index << 32U) + sparse_data_chunk_offset;
    uint32_t sparse_offset_chunk_index =
        id / get_offset_info_number_per_chunk();
    uint32_t sparse_offset_chunk_offset =
        id % get_offset_info_number_per_chunk() * offset_size_per_node();
    ret = write_sparse_offset_to_chunk(sparse_offset_chunk_index,
                                       sparse_offset_chunk_offset,
                                       sparse_offset, sparse_vector_len, id);
    if (ret != 0) return ret;
    committed = true;
    publish_key(id, id, std::move(pending_key));
  }
  inc_total_sparse_count(sparse_count);
  return 0;
} catch (const ReadError &error) {
  return error.status;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int FlatSparseStreamerEntity::write_sparse_vector_to_chunk(
    std::string_view sparse_vector, const uint32_t sparse_vector_len,
    uint32_t &sparse_data_chunk_index, uint32_t &sparse_data_chunk_offset) {
  // get sparse data chunk and offset for write sparse vector
  Chunk::Pointer sparse_data_chunk;
  sparse_data_chunk_offset = -1U;
  sparse_data_chunk_index = sparse_data_chunks_.size() - 1U;
  if (sparse_data_chunk_index == -1U ||
      sparse_data_chunks_[sparse_data_chunk_index]->data_size() +
              sparse_vector_len >
          streamer_meta_.data_chunk_size) {
    if (ailego_unlikely(sparse_data_chunks_.capacity() ==
                        sparse_data_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      if (sparse_data_chunk_index != -1U) {
        LOG_ERROR(
            "capacity: %zu, chunk used size: %zu, chunk size: %u, "
            "sparse_vector_len: %u",
            sparse_data_chunks_.capacity(),
            sparse_data_chunks_[sparse_data_chunk_index]->data_size(),
            streamer_meta_.data_chunk_size, sparse_vector_len);
      }
      return IndexError_IndexFull;
    }

    sparse_data_chunk = alloc_new_data_chunk(sparse_data_chunks_.size());
    if (ailego_unlikely(!sparse_data_chunk)) {
      LOG_ERROR("allocate data chunk failed");
      return IndexError_NoMemory;
    }
    sparse_data_chunks_.emplace_back(sparse_data_chunk);
    sparse_data_chunk_index = sparse_data_chunks_.size() - 1U;
    sparse_data_chunk_offset = 0UL;
  } else {
    sparse_data_chunk = sparse_data_chunks_[sparse_data_chunk_index];
    sparse_data_chunk_offset = sparse_data_chunk->data_size();
  }

  bool committed = false;
  AILEGO_DEFER([&] {
    if (!committed) sparse_data_chunk->resize(sparse_data_chunk_offset);
  });
  // write sparse vector
  if (sparse_vector.size() > 0) {
    if (ailego_unlikely(write_sparse_vector_data(
                            sparse_data_chunk_index, sparse_data_chunk_offset,
                            sparse_vector.data(), sparse_vector.size()) != 0)) {
      LOG_ERROR("write sparse vector failed");
      return IndexError_NoMemory;
    }
  }

  // resize chunk
  if (sparse_vector_len > 0) {
    uint32_t sparse_data_chunk_size =
        sparse_data_chunk_offset + sparse_vector_len;
    if (ailego_unlikely(sparse_data_chunk->resize(sparse_data_chunk_size) !=
                        sparse_data_chunk_size)) {
      LOG_ERROR("Sparse Chunk resize to %u failed", sparse_data_chunk_size);
      return IndexError_Runtime;
    }
  }
  committed = true;
  return 0;
}

int FlatSparseStreamerEntity::get_new_sparse_offset_chunk(
    uint32_t &sparse_offset_chunk_index, uint32_t &sparse_offset_chunk_offset) {
  // get sparse offset chunk and offset for write new info
  Chunk::Pointer sparse_offset_chunk;
  sparse_offset_chunk_offset = -1U;
  sparse_offset_chunk_index = sparse_offset_chunks_.size() - 1U;
  if (sparse_offset_chunk_index == -1U ||
      sparse_offset_chunks_[sparse_offset_chunk_index]->data_size() +
              offset_size_per_node() >
          streamer_meta_.offset_chunk_size) {
    // no space left and need to allocate new offset chunk
    if (ailego_unlikely(sparse_offset_chunks_.capacity() ==
                        sparse_offset_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      return IndexError_IndexFull;
    }

    sparse_offset_chunk = alloc_new_offset_chunk(sparse_offset_chunks_.size());
    if (ailego_unlikely(!sparse_offset_chunk)) {
      LOG_ERROR("allocate offset chunk failed");
      return IndexError_NoMemory;
    }
    sparse_offset_chunks_.emplace_back(sparse_offset_chunk);
    sparse_offset_chunk_index = sparse_offset_chunks_.size() - 1U;
    sparse_offset_chunk_offset = 0UL;
  } else {
    sparse_offset_chunk = sparse_offset_chunks_[sparse_offset_chunk_index];
    sparse_offset_chunk_offset = sparse_offset_chunk->data_size();
  }
  return 0;
}

int FlatSparseStreamerEntity::write_sparse_offset_to_chunk(
    const uint32_t sparse_offset_chunk_index,
    const uint32_t sparse_offset_chunk_offset, const uint64_t sparse_offset,
    const uint32_t sparse_vector_len, const uint64_t node_id) {
  // One complete record: cross-page admission finishes before any field is
  // changed, so updates cannot mix a new offset with an old length.
  char record[3 * sizeof(uint64_t)]{};
  std::memcpy(record, &sparse_offset, sizeof(sparse_offset));
  std::memcpy(record + sizeof(uint64_t), &sparse_vector_len,
              sizeof(sparse_vector_len));
  std::memcpy(record + 2 * sizeof(uint64_t), &node_id, sizeof(node_id));
  const IndexStorage::SegmentData write{sparse_offset_chunk_offset,
                                        sizeof(record), record};
  return sparse_offset_chunks_[sparse_offset_chunk_index]->write_batch(&write,
                                                                       1)
             ? 0
             : IndexError_WriteData;
}

uint64_t FlatSparseStreamerEntity::get_key(node_id_t node_id) const {
  uint64_t key = kInvalidKey;
  if (node_id >= doc_cnt()) return kInvalidKey;
  int ret = read_key(node_id, &key);
  if (ret != 0) throw ReadError{ret};
  return key;
}

int FlatSparseStreamerEntity::read_key(node_id_t node_id, uint64_t *key) const {
  if (node_id >= doc_cnt()) return IndexError_OutOfRange;
  uint32_t offset_chunk_index = node_id / get_offset_info_number_per_chunk();
  uint32_t offset_chunk_key_offset =
      node_id % get_offset_info_number_per_chunk() * offset_size_per_node() +
      2 * sizeof(uint64_t);

  if (ailego_unlikely(sparse_offset_chunks_[offset_chunk_index]->fetch(
                          offset_chunk_key_offset, key, sizeof(*key)) !=
                      sizeof(uint64_t))) {
    LOG_ERROR("Read key failed, offset=%u, node_id=%u", offset_chunk_key_offset,
              node_id);
    return IndexError_ReadData;
  };

  return 0;
}

int FlatSparseStreamerEntity::get_sparse_vector_ptr_by_id(
    node_id_t node_id, const void **sparse_vector_ptr,
    uint32_t *sparse_vector_len_ptr) const {
  *sparse_vector_ptr = nullptr;
  *sparse_vector_len_ptr = 0;
  if (node_id >= doc_cnt()) return IndexError_OutOfRange;
  uint32_t offset_chunk_index = node_id / get_offset_info_number_per_chunk();
  uint32_t offset_chunk_offset =
      node_id % get_offset_info_number_per_chunk() * offset_size_per_node();

  // LOG_DEBUG("Read sparse vector, offset chunk=%u, offset=%u, len=%u",
  //           offset_chunk_index, offset_chunk_offset, offset_size_per_node());

  auto offset_chunk = sparse_offset_chunks_[offset_chunk_index];

  alignas(uint64_t) char offset_info[3 * sizeof(uint64_t)];
  size_t read_len = offset_chunk->fetch(offset_chunk_offset, offset_info,
                                        offset_size_per_node());
  if (ailego_unlikely(read_len != offset_size_per_node())) {
    LOG_ERROR("Read offset info failed, offset=%u, read_len=%zu, expect=%u",
              offset_chunk_offset, read_len, offset_size_per_node());
    return IndexError_ReadData;
  };

  // sparse offset
  uint64_t sparse_offset = *(uint64_t *)offset_info;
  uint32_t sparse_vector_len =
      *(uint32_t *)((uint8_t *)offset_info + sizeof(uint64_t));

  uint32_t sparse_data_chunk_index =
      static_cast<uint32_t>((sparse_offset >> 32) & 0xFFFFFFFF);
  uint32_t sparse_data_chunk_offset =
      static_cast<uint32_t>(sparse_offset & 0xFFFFFFFF);

  if (sparse_vector_len > 0) {
    const void *sparse_data = get_sparse_vector_data(
        sparse_data_chunk_index, sparse_data_chunk_offset, sparse_vector_len);
    if (ailego_unlikely(sparse_data == nullptr)) {
      LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)sparse_offset,
                sparse_vector_len);

      return IndexError_ReadData;
    }
    *sparse_vector_ptr = sparse_data;
    *sparse_vector_len_ptr = sparse_vector_len;
  }

  // LOG_DEBUG("Read sparse vector, data chunk=%u, offset=%u, len=%u",
  //           sparse_data_chunk_index, sparse_data_chunk_offset,
  //           sparse_vector_len);

  return 0;
}

int FlatSparseStreamerEntity::get_sparse_vector_ptr_by_id(
    node_id_t node_id, IndexStorage::MemoryBlock &sparse_vector_block,
    uint32_t *sparse_vector_len_ptr) const {
  sparse_vector_block = IndexStorage::MemoryBlock{};
  *sparse_vector_len_ptr = 0;
  if (node_id >= doc_cnt()) return IndexError_OutOfRange;
  uint32_t offset_chunk_index = node_id / get_offset_info_number_per_chunk();
  uint32_t offset_chunk_offset =
      node_id % get_offset_info_number_per_chunk() * offset_size_per_node();

  // LOG_DEBUG("Read sparse vector, offset chunk=%u, offset=%u, len=%u",
  //           offset_chunk_index, offset_chunk_offset, offset_size_per_node());

  auto offset_chunk = sparse_offset_chunks_[offset_chunk_index];

  alignas(uint64_t) char offset_info[3 * sizeof(uint64_t)];
  size_t read_len = offset_chunk->fetch(offset_chunk_offset, offset_info,
                                        sizeof(offset_info));
  if (ailego_unlikely(read_len != offset_size_per_node())) {
    LOG_ERROR("Read offset info failed, offset=%u, read_len=%zu, expect=%u",
              offset_chunk_offset, read_len, offset_size_per_node());
    return IndexError_ReadData;
  };

  // sparse offset
  uint64_t sparse_offset = *(uint64_t *)offset_info;
  uint32_t sparse_vector_len =
      *(uint32_t *)((uint8_t *)offset_info + sizeof(uint64_t));

  uint32_t sparse_data_chunk_index =
      static_cast<uint32_t>((sparse_offset >> 32) & 0xFFFFFFFF);
  uint32_t sparse_data_chunk_offset =
      static_cast<uint32_t>(sparse_offset & 0xFFFFFFFF);

  if (sparse_vector_len > 0) {
    get_sparse_vector_data(sparse_data_chunk_index, sparse_data_chunk_offset,
                           sparse_vector_len, sparse_vector_block);
    if (ailego_unlikely(sparse_vector_block.data() == nullptr)) {
      LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)sparse_offset,
                sparse_vector_len);

      return IndexError_ReadData;
    }
    *sparse_vector_len_ptr = sparse_vector_len;
  }

  return 0;
}

int FlatSparseStreamerEntity::write_sparse_vector_data(uint32_t chunk_index,
                                                       uint64_t offset,
                                                       const void *data,
                                                       uint32_t length) {
  auto size = sparse_data_chunks_[chunk_index]->write(offset, data, length);
  if (size != length) {
    LOG_ERROR(
        "write sparse vector data failed: chunk_index=%u, offset=%zu, "
        "length=%u, size=%zu, chunk_data_size=%zu",
        chunk_index, (size_t)offset, length, size,
        sparse_data_chunks_[chunk_index]->data_size());
    return IndexError_WriteData;
  }
  // LOG_DEBUG(
  //     "write_sparse_vector_data: chunk_index=%u, offset=%lu, length=%u, "
  //     "data=%p",
  //     chunk_index, offset, length, data);
  return 0;
}

const void *FlatSparseStreamerEntity::get_sparse_vector_data(
    uint32_t chunk_index, uint64_t offset, uint32_t length) const {
  static thread_local LegacyReadBuffer buffer;
  IndexStorage::MemoryBlock block;
  auto size = sparse_data_chunks_[chunk_index]->read(offset, block, length);
  if (size != length) {
    LOG_ERROR(
        "read sparse vector data failed: chunk_index=%u, offset=%zu, "
        "length=%u, size=%zu",
        chunk_index, (size_t)offset, length, size);
    return nullptr;
  }
  // LOG_DEBUG(
  //     "get_sparse_vector_data: chunk_index=%u, offset=%lu, length=%u, "
  //     "data=%p",
  //     chunk_index, offset, length, data);
  return buffer.assign(block, length);
}

int FlatSparseStreamerEntity::get_sparse_vector_data(
    uint32_t chunk_index, uint64_t offset, uint32_t length,
    IndexStorage::MemoryBlock &block) const {
  auto size = sparse_data_chunks_[chunk_index]->read(offset, block, length);
  if (size != length) {
    LOG_ERROR(
        "read sparse vector data failed: chunk_index=%u, offset=%zu, "
        "length=%u, size=%zu",
        chunk_index, (size_t)offset, length, size);
    return IndexError_ReadData;
  }
  return 0;
}

}  // namespace core
}  // namespace zvec

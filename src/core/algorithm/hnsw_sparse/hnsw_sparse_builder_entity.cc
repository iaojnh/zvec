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
#include "hnsw_sparse_builder_entity.h"
#include <array>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/hash/crc32c.h>
#include "utility/sparse_utility.h"
#include "utility/temporary_buffer_storage.h"

namespace zvec {
namespace core {

// All six growing arrays share one writable page table. Segment directories
// grow per MiB, not per document. Only ingestion appends; graph workers update
// existing rows under the algorithm's node locks.
struct HnswSparseBuilderEntity::PagedState {
  enum Buffer { Meta, Keys, Level0, Upper, Payload, Levels, Count };
  static constexpr size_t kChunkSize = 1024 * 1024;
  struct Stream {
    size_t size{0};
    ailego::BudgetedVector<IndexStorage::Segment::Pointer> chunks;
  };
  TemporaryBufferStorage::Pointer file;
  std::array<Stream, Count> streams;

  int append(Buffer which, const void *data, size_t bytes) {
    auto &stream = streams[which];
    if (bytes > SIZE_MAX - stream.size) return IndexError_InvalidArgument;
    const size_t end = stream.size + bytes;
    try {
      while (stream.chunks.size() <
             end / kChunkSize + (end % kChunkSize != 0)) {
        auto name =
            std::to_string(which) + "." + std::to_string(stream.chunks.size());
        int ret = file->storage()->append(name, kChunkSize);
        if (ret != 0) return ret;
        auto segment = file->storage()->get(name);
        if (!segment) return IndexError_WriteData;
        stream.chunks.push_back(std::move(segment));
      }
      // Publish length only after successful writes. A failed append poisons
      // the entity; it is never silently retried against partial records.
      int ret =
          transfer(which, stream.size, const_cast<void *>(data), bytes, true);
      if (ret == 0) stream.size = end;
      return ret;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_WriteData;
    }
  }

  int transfer(Buffer which, size_t offset, void *data, size_t bytes,
               bool write) const {
    const auto &stream = streams[which];
    if ((!data && bytes) ||
        (!write && (offset > stream.size || bytes > stream.size - offset)))
      return IndexError_InvalidArgument;
    auto *ptr = static_cast<char *>(data);
    try {
      while (bytes) {
        size_t chunk = offset / kChunkSize;
        size_t in_chunk = offset % kChunkSize;
        size_t length = std::min(bytes, kChunkSize - in_chunk);
        if (chunk >= stream.chunks.size()) return IndexError_InvalidArgument;
        const auto &segment = stream.chunks[chunk];
        size_t done = write ? segment->write(in_chunk, ptr, length)
                            : segment->fetch(in_chunk, ptr, length);
        if (done != length)
          return write ? IndexError_WriteData : IndexError_ReadData;
        offset += length;
        ptr += length;
        bytes -= length;
      }
      return 0;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return write ? IndexError_WriteData : IndexError_ReadData;
    }
  }

  int read(Buffer which, size_t offset, size_t bytes,
           IndexStorage::MemoryBlock &block) const {
    block = {};
    const auto &stream = streams[which];
    if (offset > stream.size || bytes > stream.size - offset)
      return IndexError_InvalidArgument;
    try {
      // A row may span pages/chunks. Copy only this row; no escaping pins and
      // no thread-local alias shared by two distance operands.
      // Metadata and neighbor rows are usually much smaller than an OS page.
      // Account their actual capacity instead of allocating a mapped page for
      // each distance operand, which would exhaust small concurrent budgets.
      using Row = ailego::BudgetedVector<uint64_t>;
      auto row = std::allocate_shared<Row>(
          ailego::BudgetedAllocator<Row>{},
          bytes / sizeof(uint64_t) + (bytes % sizeof(uint64_t) != 0));
      int ret = transfer(which, offset, row->data(), bytes, false);
      if (ret == 0)
        block = IndexStorage::MemoryBlock::MakeSharedView(row->data(), row);
      return ret;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
  }
};

HnswSparseBuilderEntity::~HnswSparseBuilderEntity() = default;

int HnswSparseBuilderEntity::record_error(int ret) const {
  if (ret != 0) {
    int expected = 0;
    error_.compare_exchange_strong(expected, ret);
  }
  return ret;
}

HnswSparseBuilderEntity::HnswSparseBuilderEntity() {
  update_ep_and_level(kInvalidNodeId, 0U);
}

int HnswSparseBuilderEntity::cleanup() {
  paged_.reset();
  error_ = 0;
  sparse_data_offset_ = 0;
  std::string().swap(sparse_data_buffer_);
  memory_quota_ = 0UL;
  neighbors_size_ = 0U;
  upper_neighbors_size_ = 0U;
  padding_size_ = 0U;
  vectors_buffer_.clear();
  keys_buffer_.clear();
  neighbors_buffer_.clear();
  upper_neighbors_buffer_.clear();
  neighbors_index_.clear();

  vectors_buffer_.shrink_to_fit();
  keys_buffer_.shrink_to_fit();
  neighbors_buffer_.shrink_to_fit();
  upper_neighbors_buffer_.shrink_to_fit();
  neighbors_index_.shrink_to_fit();

  this->HnswSparseEntity::cleanup();

  return 0;
}

int HnswSparseBuilderEntity::init() {
  size_t size = vector_size();

  size += sparse_meta_size();

  //! aligned size to 32
  set_node_size(AlignSize(size));
  //! if node size is aligned to 1k, the build performance will downgrade
  if (node_size() % 1024 == 0) {
    set_node_size(AlignSize(node_size() + 1));
  }

  padding_size_ = node_size() - size;

  neighbors_size_ = neighbors_size();
  upper_neighbors_size_ = upper_neighbors_size();

  if (ailego::MemoryLimitPool::get_instance().capacity() != 0) {
    auto candidate = std::make_unique<PagedState>();
    std::string prefix;
    int ret =
        TemporaryBufferStorage::DefaultPrefix("zvec-sparse-build", &prefix);
    if (ret == 0)
      ret = TemporaryBufferStorage::CreateEmpty(prefix, &candidate->file);
    if (ret != 0) return record_error(ret);
    paged_ = std::move(candidate);
  }

  return 0;
}

int HnswSparseBuilderEntity::reserve_space(size_t docs,
                                           size_t total_sparse_count) {
  if (status() != 0) return status();
  if (paged_) return 0;  // Physical pages, not corpus size, consume the pool.
  if (memory_quota_ > 0 && (node_size() * docs + neighbors_size_ * docs +
                                sizeof(SparseNeighborIndex) * docs >
                            memory_quota_)) {
    return IndexError_NoMemory;
  }

  vectors_buffer_.reserve(node_size() * docs);
  keys_buffer_.reserve(sizeof(key_t) * docs);
  neighbors_buffer_.reserve(neighbors_size_ * docs);
  neighbors_index_.reserve(docs);

  sparse_data_buffer_.reserve(sizeof(uint32_t) * docs +
                              (sizeof(uint32_t)) * total_sparse_count +
                              sparse_unit_size() * total_sparse_count);

  return 0;
}

int HnswSparseBuilderEntity::add_vector(level_t level, key_t key,
                                        const uint32_t sparse_count,
                                        const uint32_t *sparse_indices,
                                        const void *sparse_vec, node_id_t *id) {
  if (status() != 0) return status();
  if (!id || doc_cnt() == kInvalidNodeId || level < 0 ||
      (sparse_count && (!sparse_indices || !sparse_vec)))
    return IndexError_InvalidArgument;
  if (ailego_unlikely(sparse_count > HnswSparseEntity::kSparseMaxDimSize)) {
    LOG_WARN(
        "Failed to add sparse vector: number of non-zero elements (%u) exceeds "
        "maximum allowed (%u), key=%zu",
        sparse_count, HnswSparseEntity::kSparseMaxDimSize, (size_t)key);
    return IndexError_InvalidValue;
  }

  std::string sparse_buffer;
  SparseUtility::TransSparseFormat(sparse_count, sparse_indices, sparse_vec,
                                   sparse_unit_size(), sparse_buffer);

  uint32_t sparse_len = sparse_buffer.size();

  if (paged_) {
    std::vector<char> meta(node_size(), 0);
    std::memcpy(meta.data() + vector_size(), &sparse_data_offset_,
                sizeof(uint64_t));
    std::memcpy(meta.data() + vector_size() + sizeof(uint64_t), &sparse_len,
                sizeof(sparse_len));
    const SparseNeighborIndex index(paged_->streams[PagedState::Upper].size,
                                    level);
    std::vector<char> zeros(std::max(neighbors_size_, upper_neighbors_size_),
                            0);
    int ret = paged_->append(PagedState::Meta, meta.data(), meta.size());
    if (!ret) ret = paged_->append(PagedState::Keys, &key, sizeof(key));
    if (!ret)
      ret =
          paged_->append(PagedState::Payload, sparse_buffer.data(), sparse_len);
    if (!ret)
      ret = paged_->append(PagedState::Level0, zeros.data(), neighbors_size_);
    if (!ret) ret = paged_->append(PagedState::Levels, &index, sizeof(index));
    for (level_t l = 0; !ret && l < level; ++l)
      ret = paged_->append(PagedState::Upper, zeros.data(),
                           upper_neighbors_size_);
    if (ret) return record_error(ret);
    sparse_data_offset_ += sparse_len;
    *id = (*mutable_doc_cnt())++;
    return 0;
  }

  if (memory_quota_ > 0 &&
      (vectors_buffer_.capacity() + keys_buffer_.capacity() +
           neighbors_buffer_.capacity() + upper_neighbors_buffer_.capacity() +
           neighbors_index_.capacity() * sizeof(SparseNeighborIndex) +
           sparse_len >
       memory_quota_)) {
    LOG_ERROR("Add vector failed, used memory exceed quota, cur_doc=%u",
              doc_cnt());
    return IndexError_NoMemory;
  }

  vectors_buffer_.append(reinterpret_cast<const char *>(&sparse_data_offset_),
                         sizeof(uint64_t));
  vectors_buffer_.append(reinterpret_cast<const char *>(&sparse_len),
                         sizeof(uint32_t));
  vectors_buffer_.append(sizeof(uint32_t),
                         '\0');  // reserve to make it up to meta size
  vectors_buffer_.append(padding_size_, '\0');

  keys_buffer_.append(reinterpret_cast<const char *>(&key), sizeof(key));

  sparse_data_buffer_.append(sparse_buffer.data(), sparse_len);
  sparse_data_offset_ += sparse_len;

  // init level 0 neighbors
  neighbors_buffer_.append(neighbors_size_, '\0');

  neighbors_index_.emplace_back(upper_neighbors_buffer_.size(), level);

  // init upper layer neighbors
  for (level_t cur_level = 1; cur_level <= level; ++cur_level) {
    upper_neighbors_buffer_.append(upper_neighbors_size_, '\0');
  }

  *id = (*mutable_doc_cnt())++;

  return 0;
}

key_t HnswSparseBuilderEntity::get_key(node_id_t id) const {
  if (id >= doc_cnt()) {
    record_error(IndexError_InvalidArgument);
    return kInvalidKey;
  }
  if (!paged_) return reinterpret_cast<const key_t *>(keys_buffer_.data())[id];
  key_t key = kInvalidKey;
  record_error(paged_->transfer(PagedState::Keys, size_t(id) * sizeof(key),
                                &key, sizeof(key), false));
  return key;
}

const void *HnswSparseBuilderEntity::get_vector_meta(node_id_t id) const {
  if (!paged_)
    return id < doc_cnt() ? vectors_buffer_.data() + id * node_size() : nullptr;
  // Legacy pointer APIs last until the next call on this thread. Build and
  // distance paths use owning blocks whenever two records coexist.
  thread_local IndexStorage::MemoryBlock scratch;
  return get_vector_meta(id, scratch) == 0 ? scratch.data() : nullptr;
}

int HnswSparseBuilderEntity::get_vector_meta(
    node_id_t id, IndexStorage::MemoryBlock &block) const {
  block = {};
  if (id >= doc_cnt()) return record_error(IndexError_InvalidArgument);
  if (paged_)
    return record_error(paged_->read(PagedState::Meta, size_t(id) * node_size(),
                                     node_size(), block));
  block.reset(const_cast<char *>(vectors_buffer_.data() + id * node_size()));
  return 0;
}

int HnswSparseBuilderEntity::get_vector_metas(const node_id_t *ids,
                                              uint32_t count,
                                              const void **vecs) const {
  if (paged_) {
    thread_local std::vector<IndexStorage::MemoryBlock> scratch;
    scratch.clear();
    int ret = get_vector_metas(ids, count, scratch);
    if (ret != 0) return ret;
    for (uint32_t i = 0; i < count; ++i) vecs[i] = scratch[i].data();
    return 0;
  }
  for (uint32_t i = 0; i < count; ++i) {
    vecs[i] = get_vector_meta(ids[i]);
    if (!vecs[i]) return record_error(IndexError_ReadData);
  }
  return 0;
}

int HnswSparseBuilderEntity::get_vector_metas(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &blocks) const {
  for (uint32_t i = 0; i < count; ++i) {
    IndexStorage::MemoryBlock block;
    int ret = get_vector_meta(ids[i], block);
    if (ret != 0) return ret;
    blocks.push_back(std::move(block));
  }
  return 0;
}

const void *HnswSparseBuilderEntity::get_sparse_data(uint64_t offset,
                                                     uint32_t len) const {
  if (!paged_)
    return offset <= sparse_data_buffer_.size() &&
                   len <= sparse_data_buffer_.size() - offset
               ? sparse_data_buffer_.data() + offset
               : nullptr;
  thread_local IndexStorage::MemoryBlock scratch;
  return get_sparse_data(offset, len, scratch) == 0 ? scratch.data() : nullptr;
}

int HnswSparseBuilderEntity::get_sparse_data(
    uint64_t offset, uint32_t len, IndexStorage::MemoryBlock &block) const {
  block = {};
  if (paged_)
    return record_error(paged_->read(PagedState::Payload, offset, len, block));
  const void *data = get_sparse_data(offset, len);
  if (!data) return record_error(IndexError_ReadData);
  block.reset(const_cast<void *>(data));
  return 0;
}

const void *HnswSparseBuilderEntity::get_sparse_data(node_id_t id) const {
  if (!paged_) return get_sparse_data_from_vector(get_vector_meta(id)).first;
  thread_local IndexStorage::MemoryBlock scratch;
  return get_sparse_data(id, scratch) == 0 ? scratch.data() : nullptr;
}

int HnswSparseBuilderEntity::get_sparse_data(
    node_id_t id, IndexStorage::MemoryBlock &block) const {
  IndexStorage::MemoryBlock meta;
  int ret = get_vector_meta(id, meta);
  if (ret != 0) {
    block = {};
    return ret;
  }
  int length = 0;
  return get_sparse_data_from_vector(meta.data(), block, length);
}

std::pair<const void *, uint32_t>
HnswSparseBuilderEntity::get_sparse_data_from_vector(const void *vec) const {
  if (!vec) {
    record_error(IndexError_ReadData);
    return {nullptr, 0};
  }
  uint64_t offset;
  uint32_t length;
  const auto *meta = static_cast<const char *>(vec) + vector_size();
  std::memcpy(&offset, meta, sizeof(offset));
  std::memcpy(&length, meta + sizeof(offset), sizeof(length));
  return {get_sparse_data(offset, length), length};
}

int HnswSparseBuilderEntity::get_sparse_data_from_vector(
    const void *vec, IndexStorage::MemoryBlock &block,
    int &sparse_length) const {
  block = {};
  sparse_length = 0;
  if (!vec) return record_error(IndexError_ReadData);
  uint64_t offset;
  uint32_t length;
  const auto *meta = static_cast<const char *>(vec) + vector_size();
  std::memcpy(&offset, meta, sizeof(offset));
  std::memcpy(&length, meta + sizeof(offset), sizeof(length));
  int ret = get_sparse_data(offset, length, block);
  if (!ret) sparse_length = length;
  return ret;
}

HnswSparseBuilderEntity::SparseNeighborIndex
HnswSparseBuilderEntity::neighbor_index(node_id_t id) const {
  SparseNeighborIndex index;
  if (id >= doc_cnt()) {
    record_error(IndexError_InvalidArgument);
  } else if (paged_) {
    record_error(paged_->transfer(PagedState::Levels,
                                  size_t(id) * sizeof(index), &index,
                                  sizeof(index), false));
  } else {
    index = neighbors_index_[id];
  }
  return index;
}

level_t HnswSparseBuilderEntity::get_level(node_id_t id) const {
  return neighbor_index(id).level;
}

int HnswSparseBuilderEntity::neighbor_block(
    level_t level, node_id_t id, IndexStorage::MemoryBlock &block) const {
  block = {};
  if (id >= doc_cnt() || level < 0)
    return record_error(IndexError_InvalidArgument);
  size_t offset = size_t(id) * neighbors_size_;
  if (level) {
    auto index = neighbor_index(id);
    if (level > static_cast<level_t>(index.level))
      return record_error(IndexError_InvalidArgument);
    offset = index.offset + size_t(level - 1) * upper_neighbors_size_;
  }
  if (status()) return status();
  if (paged_)
    return record_error(
        paged_->read(level ? PagedState::Upper : PagedState::Level0, offset,
                     level ? upper_neighbors_size_ : neighbors_size_, block));
  const auto &buffer = level ? upper_neighbors_buffer_ : neighbors_buffer_;
  block.reset(const_cast<char *>(buffer.data() + offset));
  return 0;
}

const Neighbors HnswSparseBuilderEntity::get_neighbors(level_t level,
                                                       node_id_t id) const {
  IndexStorage::MemoryBlock block;
  if (neighbor_block(level, id, block) != 0) return {};
  return Neighbors(std::move(block));
}

int HnswSparseBuilderEntity::write_neighbors(level_t level, node_id_t id,
                                             const void *data) {
  if (!paged_) return 0;  // The block already aliases the in-memory row.
  size_t offset = level ? neighbor_index(id).offset +
                              size_t(level - 1) * upper_neighbors_size_
                        : size_t(id) * neighbors_size_;
  if (status()) return status();
  return record_error(
      paged_->transfer(level ? PagedState::Upper : PagedState::Level0, offset,
                       const_cast<void *>(data),
                       level ? upper_neighbors_size_ : neighbors_size_, true));
}

int HnswSparseBuilderEntity::update_neighbors(
    level_t level, node_id_t id, const TopkHeap::Container &neighbors) {
  if (neighbors.size() > neighbor_cnt(level))
    return record_error(IndexError_InvalidArgument);
  IndexStorage::MemoryBlock block;
  int ret = neighbor_block(level, id, block);
  if (ret != 0) return ret;
  auto *hd = const_cast<NeighborsHeader *>(
      static_cast<const NeighborsHeader *>(block.data()));
  for (size_t i = 0; i < neighbors.size(); ++i)
    hd->neighbors[i] = neighbors[i].first;
  hd->neighbor_cnt = neighbors.size();
  return write_neighbors(level, id, hd);
}

void HnswSparseBuilderEntity::add_neighbor(level_t level, node_id_t id,
                                           uint32_t /*size*/,
                                           node_id_t neighbor_id) {
  IndexStorage::MemoryBlock block;
  if (neighbor_block(level, id, block) != 0) return;
  auto *hd = const_cast<NeighborsHeader *>(
      static_cast<const NeighborsHeader *>(block.data()));
  if (hd->neighbor_cnt >= neighbor_cnt(level)) {
    record_error(IndexError_InvalidArgument);
    return;
  }
  hd->neighbors[hd->neighbor_cnt++] = neighbor_id;
  write_neighbors(level, id, hd);
}

int HnswSparseBuilderEntity::dump(const IndexDumper::Pointer &dumper) {
  if (status()) return status();
  auto ret = dump_segments(dumper, [&](node_id_t id) { return get_level(id); });
  if (status()) return status();
  return ret < 0 ? ret : 0;
}

}  // namespace core
}  // namespace zvec

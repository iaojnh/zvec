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

#include "diskann_builder_entity.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <ailego/pattern/defer.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_factory.h>
#include "utility/immutable_segment_view.h"
#include "utility/ordinal_access_holder.h"
#include "utility/paged_record_sort.h"
#include "diskann_algorithm.h"
#include "diskann_file_reader.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

struct DiskAnnBuilderEntity::BufferedBuildState {
  std::string prefix;
  std::string directory;
  std::string vector_path;
  std::string graph_path;
  std::string code_path;
  IndexStorage::Pointer vector_storage;
  IndexStorage::Pointer graph_storage;
  IndexStorage::Segment::Pointer vectors;
  size_t vector_stride{0};
  size_t vector_prefix{0};
  IndexStorage::Segment::Pointer graph;
  IndexStorage::Segment::Pointer keys;
  mutable ailego::File codes;
  bool vectors_sealed{false};

  int ensure_directory() {
    if (!directory.empty()) return 0;
    static std::atomic<uint64_t> sequence{0};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (size_t attempt = 0; attempt < 64; ++attempt) {
      std::string candidate =
          prefix + ".diskann-build-" + std::to_string(stamp) + "-" +
          std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
      // Prepare all potentially allocating strings before owning a directory.
      std::string vectors_candidate = candidate + "/vectors";
      std::string graph_candidate = candidate + "/graph";
      std::string codes_candidate = candidate + "/codes";
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        directory = std::move(candidate);
        vector_path = std::move(vectors_candidate);
        graph_path = std::move(graph_candidate);
        code_path = std::move(codes_candidate);
        return 0;
      }
      if (error && error != std::errc::file_exists) return IndexError_OpenFile;
    }
    return IndexError_OpenFile;
  }

  void release_vectors() {
    vectors.reset();
    if (vector_storage) vector_storage->close();
    vector_storage.reset();
    vectors_sealed = false;
    vector_stride = 0;
    vector_prefix = 0;
    if (!vector_path.empty()) ailego::File::Delete(vector_path);
  }

  void clear_files() {
    release_vectors();
    graph.reset();
    keys.reset();
    if (graph_storage) graph_storage->close();
    graph_storage.reset();
    codes.close();
    if (!graph_path.empty()) ailego::File::Delete(graph_path);
    if (!code_path.empty()) ailego::File::Delete(code_path);
    // Never recursively remove a caller-supplied directory or unknown files.
    if (!directory.empty()) {
      std::error_code error;
      std::filesystem::remove(directory, error);
    }
    directory.clear();
    vector_path.clear();
    graph_path.clear();
    code_path.clear();
  }

  ~BufferedBuildState() {
    clear_files();
  }
};

DiskAnnBuilderEntity::DiskAnnBuilderEntity() = default;
DiskAnnBuilderEntity::~DiskAnnBuilderEntity() = default;

int DiskAnnBuilderEntity::enable_buffered_build(
    const std::string &scratch_prefix) {
  if (scratch_prefix.empty() || doc_cnt() != 0 || reserved_docs_ != 0 ||
      buffered_state_) {
    return IndexError_InvalidArgument;
  }
  try {
    auto state = std::make_unique<BufferedBuildState>();
    // Configuration and post-dump builder reset must not require writable
    // scratch storage. Create it only when a build actually needs it.
    state->prefix = scratch_prefix;
    buffered_state_ = std::move(state);
    return 0;
  } catch (const std::exception &error) {
    LOG_ERROR("Failed to configure DiskANN build scratch storage: %s",
              error.what());
    return IndexError_Runtime;
  }
}

void DiskAnnBuilderEntity::observe_degree(uint32_t degree) {
  uint32_t previous = max_observed_degree_.load(std::memory_order_relaxed);
  while (previous < degree &&
         !max_observed_degree_.compare_exchange_weak(
             previous, degree, std::memory_order_relaxed)) {
  }
}

void DiskAnnBuilderEntity::clear() {
  buffered_state_.reset();
  reserved_docs_ = 0;
  code_bytes_ = 0;
  code_bytes_written_ = 0;
  max_degree_ = 0;
  list_size_ = 0;
  memory_limit_ = 0;
  num_threads_ = 0;
  max_build_degree_ = 0;
  max_observed_degree_ = 0;
  mem_index_file_.clear();
  index_path_prefix_.clear();
  release_vectors();
  std::string().swap(keys_buffer_);
  neighbor_pool_.clear();
  neighbors_compacted_ = false;
  reserved_docs_ = 0;
  entrypoints_.clear();
  meta_.clear();
  pq_quantizer_.reset();
  std::vector<uint8_t>().swap(block_compressed_data_);
  meta_header_.clear();
  pq_meta_.clear();
}

int DiskAnnBuilderEntity::init(const IndexMeta &meta, uint32_t max_degree,
                               uint32_t list_size, double memory_limit,
                               uint32_t build_threads) {
  clear();
  meta_ = meta;

  max_degree_ = max_degree;
  list_size_ = list_size;

  memory_limit_ = memory_limit;

  num_threads_ = build_threads;

  const double build_degree = static_cast<double>(max_degree_) *
                              static_cast<double>(kDefaultGraphSlackFactor);
  if (meta_.element_size() == 0 || max_degree_ == 0 ||
      build_degree > (std::numeric_limits<uint32_t>::max() - sizeof(uint32_t)) /
                         sizeof(diskann_id_t)) {
    return IndexError_InvalidArgument;
  }
  // Preserve the original float multiplication/rounding for ordinary inputs.
  max_build_degree_ = max_degree_ * kDefaultGraphSlackFactor;
  neighbor_size_ = sizeof(uint32_t) + max_build_degree_ * sizeof(diskann_id_t);

  return 0;
}

void DiskAnnBuilderEntity::release_vectors() {
  ailego::BudgetedVector<const void *>().swap(vector_ptrs_);
  vector_lifetime_.reset();
  if (buffered_state_) {
    buffered_state_->release_vectors();
  }
}

int DiskAnnBuilderEntity::prepare_vectors_for_graph() try {
  if (!buffered_state_) return 0;
  auto &state = *buffered_state_;
  if (state.vectors_sealed) return 0;
  if (doc_cnt() == 0 || doc_cnt() != reserved_docs_ || !state.vectors ||
      !state.vector_storage) {
    return IndexError_NoReady;
  }
  // Graph workers have not started. Flush errors must propagate before
  // releasing the writable pool and its staging/metadata reservations.
  int ret = state.vector_storage->flush();
  if (ret != 0) return ret;
  state.vectors.reset();
  state.vector_storage->close();
  state.vector_storage.reset();
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  if (!storage) return IndexError_NoExist;
  ret = storage->init(ailego::Params{});
  if (ret != 0) return ret;
  ret = storage->open(state.vector_path, false);
  if (ret != 0) return ret;
  auto vectors = storage->get("data");
  if (!vectors) return IndexError_ReadData;
  state.vector_storage = std::move(storage);
  state.vectors = std::move(vectors);
  state.vectors_sealed = true;
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int DiskAnnBuilderEntity::reserve_space(uint32_t docs,
                                        std::shared_ptr<const void> lifetime) {
  const size_t count = docs;
  if (count == 0 || doc_cnt() != 0 || reserved_docs_ != 0 ||
      meta_.element_size() == 0 || neighbor_size_ == 0 ||
      count > std::numeric_limits<size_t>::max() / meta_.element_size() ||
      count > std::numeric_limits<size_t>::max() / neighbor_size_ ||
      count > std::numeric_limits<size_t>::max() / sizeof(diskann_key_t) ||
      (!buffered_state_ && !lifetime)) {
    return IndexError_InvalidArgument;
  }
  try {
    if (buffered_state_) {
      auto open = [](const std::string &path, size_t bytes,
                     IndexStorage::Pointer *storage,
                     IndexStorage::Segment::Pointer *segment) -> int {
        *storage = IndexFactory::CreateStorage("BufferStorage");
        if (!*storage) return IndexError_NoExist;
        int ret = (*storage)->init(ailego::Params{});
        if (ret != 0) return ret;
        ret = (*storage)->open(path, true);
        if (ret != 0) return ret;
        ret = (*storage)->append("data", bytes);
        if (ret != 0) return ret;
        *segment = (*storage)->get("data");
        return *segment ? 0 : IndexError_Runtime;
      };
      auto &state = *buffered_state_;
      const size_t page = ailego::kVectorPageSize;
      const size_t vector_size = meta_.element_size();
      // A near-page-sized vector otherwise straddles two pages on most reads.
      // Pad only these temporary rows: each vector then fits in one cache page,
      // with at most one-third extra scratch space. Small/multi-page vectors
      // and the final index format stay unchanged.
      const bool aligned = vector_size >= page - page / 4 && vector_size < page;
      const size_t stride = aligned ? page : vector_size;
      const size_t prefix_room = aligned ? page - 1 : 0;
      if (count > (std::numeric_limits<size_t>::max() - prefix_room) / stride) {
        return IndexError_InvalidArgument;
      }
      state.vector_stride = stride;
      state.vector_prefix = 0;
      int ret = state.ensure_directory();
      if (ret == 0) {
        ret = open(state.vector_path, stride * count + prefix_room,
                   &state.vector_storage, &state.vectors);
        if (ret == 0 && aligned) {
          state.vector_prefix =
              (page - state.vectors->data_offset() % page) % page;
        }
      }
      if (ret == 0) {
        ret =
            open(state.graph_path, static_cast<size_t>(neighbor_size_) * count,
                 &state.graph_storage, &state.graph);
      }
      if (ret == 0) {
        ret =
            state.graph_storage->append("keys", sizeof(diskann_key_t) * count);
        if (ret == 0) {
          state.keys = state.graph_storage->get("keys");
          if (!state.keys) ret = IndexError_Runtime;
        }
      }
      if (ret != 0) {
        state.clear_files();
        return ret;
      }
    } else {
      keys_buffer_.reserve(sizeof(diskann_key_t) * count);
      vector_ptrs_.reserve(count);
      int ret = neighbor_pool_.reserve(docs, max_build_degree_);
      if (ret != 0) return ret;
      vector_lifetime_ = std::move(lifetime);
    }
    reserved_docs_ = docs;
    return 0;
  } catch (const std::bad_alloc &) {
    if (buffered_state_) buffered_state_->clear_files();
    return IndexError_NoMemory;
  } catch (const std::exception &error) {
    if (buffered_state_) buffered_state_->clear_files();
    LOG_ERROR("Failed to reserve DiskANN build storage: %s", error.what());
    return IndexError_Runtime;
  }
}

int DiskAnnBuilderEntity::add_vector(diskann_key_t key, const void *vec) try {
  if (neighbors_compacted_) return IndexError_NoReady;
  if (!vec) return IndexError_ReadData;
  if (buffered_state_) {
    auto &state = *buffered_state_;
    if (doc_cnt() >= reserved_docs_ || !state.vectors || !state.graph) {
      return IndexError_InvalidArgument;
    }
    const size_t vector_size = meta_.element_size();
    const uint32_t neighbor_count = 0;
    if (state.vectors->write(
            state.vector_prefix + doc_cnt() * state.vector_stride, vec,
            vector_size) != vector_size)
      return state.vectors->write_error();
    if (state.graph->write(doc_cnt() * neighbor_size_, &neighbor_count,
                           sizeof(neighbor_count)) != sizeof(neighbor_count))
      return state.graph->write_error();
    if (!state.keys) return IndexError_WriteData;
    if (state.keys->write(doc_cnt() * sizeof(key), &key, sizeof(key)) !=
        sizeof(key))
      return state.keys->write_error();
    ++*mutable_doc_cnt();
    return 0;
  }
  if (!vector_lifetime_) return IndexError_NoReady;
  if (doc_cnt() >= reserved_docs_) return IndexError_InvalidArgument;
  vector_ptrs_.push_back(vec);
  keys_buffer_.append(reinterpret_cast<const char *>(&key), sizeof(key));

  (*mutable_doc_cnt())++;

  return 0;
} catch (const std::exception &error) {
  LOG_ERROR("Failed to stage DiskANN vector: %s", error.what());
  return IndexError_Runtime;
}

const void *DiskAnnBuilderEntity::get_vector(diskann_id_t id) const {
  if (buffered_state_ || id >= vector_ptrs_.size()) return nullptr;
  return vector_ptrs_[id];
}

int DiskAnnBuilderEntity::read_vector(diskann_id_t id,
                                      IndexStorage::MemoryBlock &block) const
    try {
  block.reset();
  if (id >= doc_cnt()) return IndexError_InvalidArgument;
  if (!buffered_state_) return DiskAnnEntity::read_vector(id, block);
  const auto &segment = buffered_state_->vectors;
  const size_t size = meta_.element_size();
  if (!segment ||
      segment->read_immutable(
          buffered_state_->vector_prefix +
              static_cast<size_t>(id) * buffered_state_->vector_stride,
          block, size) != size ||
      !block.data()) {
    block.reset();
    return IndexError_ReadData;
  }
  return 0;
} catch (const std::exception &error) {
  block.reset();
  LOG_ERROR("Failed to read DiskANN build vector: %s", error.what());
  return IndexError_Runtime;
}

int DiskAnnBuilderEntity::read_vectors(const diskann_id_t *ids,
                                       IndexStorage::MemoryBlock *blocks,
                                       size_t count) const try {
  if (count == 0) return 0;
  if (!ids || !blocks || count > kVectorReadBatchSize)
    return IndexError_InvalidArgument;
  for (size_t i = 0; i < count; ++i) blocks[i].reset();
  for (size_t i = 0; i < count; ++i) {
    if (ids[i] >= doc_cnt()) return IndexError_InvalidArgument;
  }
  if (!buffered_state_) return DiskAnnEntity::read_vectors(ids, blocks, count);
  const auto &segment = buffered_state_->vectors;
  if (!segment) return IndexError_ReadData;
  std::array<IndexStorage::Segment::BorrowedRead, kVectorReadBatchSize> reads;
  const size_t size = meta_.element_size();
  for (size_t i = 0; i < count; ++i) {
    reads[i] = {
        segment.get(),
        buffered_state_->vector_prefix +
            static_cast<size_t>(ids[i]) * buffered_state_->vector_stride,
        size, &blocks[i]};
  }
  if (!segment->read_borrowed_batch_immutable(reads.data(), count)) {
    for (size_t i = 0; i < count; ++i) blocks[i].reset();
    const int ret = segment->read_error();
    return ret != 0 ? ret : IndexError_ReadData;
  }
  return 0;
} catch (const std::bad_alloc &) {
  for (size_t i = 0; i < count; ++i) blocks[i].reset();
  return IndexError_NoMemory;
} catch (const std::exception &) {
  for (size_t i = 0; i < count; ++i) blocks[i].reset();
  return IndexError_ReadData;
}

namespace {
class BufferedVectorReader : public DiskAnnEntity::VectorReader {
 public:
  BufferedVectorReader(std::shared_ptr<ailego::VecBufferPool> pool,
                       const std::string &path, size_t offset, size_t stride,
                       size_t rows, ailego::MemoryReservation memory)
      : memory_(std::move(memory)),
        pool_(pool),
        reader_(std::move(pool)),
        offset_(offset),
        stride_(stride),
        rows_(rows) {
    const int ret = reader_.open_from_pool(path);
    if (ret) throw std::bad_alloc();
  }
  ~BufferedVectorReader() override {
    reset();
  }
  int submit(const diskann_id_t *ids, size_t count) override try {
    if (backing_ || !ids || !count ||
        count > DiskAnnEntity::kVectorReadBatchSize)
      return IndexError_InvalidArgument;
    for (size_t i = 0; i < count; ++i)
      if (ids[i] >= rows_) return IndexError_InvalidArgument;
    bool resident = pool_->cache_enabled();
    for (size_t i = 0; i < count && resident; ++i) {
      const size_t first =
          (offset_ + ids[i] * stride_) / ailego::kVectorPageSize;
      for (size_t j = 0; j < stride_ / ailego::kVectorPageSize; ++j)
        resident = resident && pool_->is_page_resident(first + j);
    }
    // A hot batch can be borrowed directly by read_vectors(). Do not replace
    // that zero-copy path with an owned read-ahead arena and extra memcpy.
    if (resident) return IndexError_NotImplemented;
    ailego::MemoryReservation metadata;
    metadata.add(sizeof(Backing) + 64);
    backing_ = std::make_shared<Backing>();
    backing_->memory.swap(metadata);
    backing_->data.resize(count * stride_);
    ailego::ReserveVectorCapacity(requests_, request_memory_, count);
    ailego::ReserveVectorCapacity(completed_, request_memory_, count);
    requests_.clear();
    for (size_t i = 0; i < count; ++i)
      requests_.emplace_back(offset_ + ids[i] * stride_, stride_,
                             backing_->data.data() + i * stride_);
    count_ = count;
    const int ret = reader_.submit(batch_, requests_, ctx_);
    if (ret) reset();
    return ret;
  } catch (const std::bad_alloc &) {
    reset();
    return IndexError_NoMemory;
  }
  int finish(IndexStorage::MemoryBlock *blocks, size_t count) override {
    if (!backing_ || count != count_ || !blocks)
      return IndexError_InvalidArgument;
    while (batch_.n_reaped < batch_.n_submitted) {
      const int ret = reader_.get_completed(batch_, ctx_, 1, completed_);
      if (ret < 0) {
        reset();
        return ret;
      }
    }
    for (size_t i = 0; i < count; ++i)
      blocks[i] = IndexStorage::MemoryBlock::MakeSharedView(
          backing_->data.data() + i * stride_, backing_);
    backing_.reset();
    batch_ = PendingBatch{};
    return 0;
  }

 private:
  void reset() {
    reader_.release_io_ctx(ctx_);
    destroy_io_ctx(ctx_);
    batch_ = PendingBatch{};
    backing_.reset();
  }
  struct Backing {
    ailego::MemoryReservation memory;
    ailego::BudgetedMappedBuffer data;
  };
  ailego::MemoryReservation memory_, request_memory_;
  std::shared_ptr<ailego::VecBufferPool> pool_;
  BufferPoolAlignedFileReader reader_;
  IOContext ctx_{nullptr};
  PendingBatch batch_;
  std::shared_ptr<Backing> backing_;
  std::vector<AlignedRead> requests_;
  std::vector<uint32_t> completed_;
  size_t offset_, stride_, rows_, count_{0};
};
}  // namespace

std::unique_ptr<DiskAnnEntity::VectorReader>
DiskAnnBuilderEntity::create_vector_reader() const try {
  if (!buffered_state_ || !buffered_state_->vectors_sealed) return nullptr;
  const auto &state = *buffered_state_;
  const size_t page = ailego::kVectorPageSize;
  const size_t offset = state.vectors->data_offset() + state.vector_prefix;
  if (offset % page || state.vector_stride % page ||
      state.vector_stride / page > MAX_IO_DEPTH / kVectorReadBatchSize)
    return nullptr;
  auto pool = state.vector_storage->vec_buffer_pool();
  if (!pool) return nullptr;
#if defined(__linux__) && !defined(__ANDROID__)
  if (pool->io_backend_type() == ailego::IOBackendType::kPread) return nullptr;
#elif !defined(_WIN32) && !defined(_WIN64)
  return nullptr;
#endif
  ailego::MemoryReservation memory;
  memory.add(sizeof(BufferedVectorReader));
  return std::make_unique<BufferedVectorReader>(
      std::move(pool), state.vector_path, offset, state.vector_stride,
      doc_cnt(), std::move(memory));
} catch (const std::bad_alloc &) {
  return nullptr;
}

diskann_key_t DiskAnnBuilderEntity::get_key(diskann_id_t id) const {
  diskann_key_t key = kInvalidKey;
  return read_key(id, &key) == 0 ? key : kInvalidKey;
}

int DiskAnnBuilderEntity::read_key(diskann_id_t id, diskann_key_t *key) const {
  if (!key || id >= doc_cnt()) return IndexError_InvalidArgument;
  size_t offset = (size_t)id * sizeof(diskann_key_t);
  if (buffered_state_) {
    auto segment = buffered_state_->keys;
    return segment && segment->fetch(offset, key, sizeof(*key)) == sizeof(*key)
               ? 0
               : IndexError_ReadData;
  }
  std::memcpy(key, keys_buffer_.data() + offset, sizeof(*key));
  return 0;
}

//! Get vector local id by key
diskann_id_t DiskAnnBuilderEntity::get_id(diskann_key_t /*key*/) const {
  LOG_ERROR("DiskAnnBuilderEntity::get_id not implemented.");
  return kInvalidId;
}

std::pair<uint32_t, const diskann_id_t *> DiskAnnBuilderEntity::get_neighbors(
    diskann_id_t id) const {
  if (buffered_state_ || id >= doc_cnt()) return {0, nullptr};
  return neighbor_pool_.get(id);
}

int DiskAnnBuilderEntity::read_neighbors(
    diskann_id_t id, ailego::BudgetedVector<diskann_id_t> *neighbors) const
    try {
  if (!neighbors || id >= doc_cnt()) return IndexError_InvalidArgument;
  neighbors->clear();
  if (!buffered_state_) return DiskAnnEntity::read_neighbors(id, neighbors);
  const auto &segment = buffered_state_->graph;
  if (!segment) return IndexError_ReadData;
  const size_t offset = static_cast<size_t>(id) * neighbor_size_;
  uint32_t count = 0;
  if (segment->fetch(offset, &count, sizeof(count)) != sizeof(count)) {
    return segment->read_error();
  }
  if (count > max_build_degree_) return IndexError_ReadData;
  neighbors->resize(count);
  const size_t size = count * sizeof(diskann_id_t);
  if (size != 0 &&
      segment->fetch(offset + sizeof(count), neighbors->data(), size) != size) {
    neighbors->clear();
    return segment->read_error();
  }
  if (std::any_of(
          neighbors->begin(), neighbors->end(),
          [&](diskann_id_t neighbor) { return neighbor >= doc_cnt(); })) {
    neighbors->clear();
    return IndexError_ReadData;
  }
  return 0;
} catch (const std::bad_alloc &) {
  if (neighbors) neighbors->clear();
  return IndexError_NoMemory;
} catch (const std::exception &error) {
  if (neighbors) neighbors->clear();
  LOG_ERROR("Failed to read DiskANN build adjacency: %s", error.what());
  return IndexError_Runtime;
}

int DiskAnnBuilderEntity::set_neighbors(
    diskann_id_t id,
    const ailego::BudgetedVector<diskann_id_t> &neighbor_ids) try {
  if (id >= doc_cnt() || neighbor_ids.size() > max_build_degree_ ||
      std::any_of(
          neighbor_ids.begin(), neighbor_ids.end(),
          [&](diskann_id_t neighbor) { return neighbor >= doc_cnt(); })) {
    return IndexError_InvalidArgument;
  }
  if (buffered_state_) {
    const auto &segment = buffered_state_->graph;
    if (!segment) return IndexError_WriteData;
    // Graph callers hold the per-node lock. Publish the count last and let
    // the storage coalesce same-page writes without a per-update heap copy.
    // Any partial I/O failure aborts the build; it must not be dumped.
    const size_t offset = static_cast<size_t>(id) * neighbor_size_;
    const uint32_t count = static_cast<uint32_t>(neighbor_ids.size());
    const std::array<IndexStorage::SegmentData, 2> writes{
        {{offset + sizeof(count), neighbor_ids.size() * sizeof(diskann_id_t),
          neighbor_ids.data()},
         {offset, sizeof(count), &count}}};
    if (!segment->write_batch(writes.data(), writes.size()))
      return segment->write_error();
    observe_degree(count);
    return 0;
  }
  if (neighbors_compacted_) return IndexError_NoReady;
  const int ret =
      neighbor_pool_.set(id, neighbor_ids.data(), neighbor_ids.size());
  if (ret == 0) observe_degree(static_cast<uint32_t>(neighbor_ids.size()));
  return ret;
} catch (const std::exception &error) {
  LOG_ERROR("Failed to write DiskANN build adjacency: %s", error.what());
  return IndexError_Runtime;
}

int DiskAnnBuilderEntity::add_neighbor(diskann_id_t id,
                                       diskann_id_t neighbor_id) try {
  if (id >= doc_cnt() || neighbor_id >= doc_cnt())
    return IndexError_InvalidArgument;
  if (buffered_state_) {
    const auto &segment = buffered_state_->graph;
    if (!segment) return IndexError_WriteData;
    const size_t offset = static_cast<size_t>(id) * neighbor_size_;
    uint32_t count = 0;
    if (segment->fetch(offset, &count, sizeof(count)) != sizeof(count)) {
      return segment->read_error();
    }
    if (count >= max_build_degree_) return IndexError_InvalidArgument;
    const uint32_t next_count = count + 1;
    const std::array<IndexStorage::SegmentData, 2> writes{
        {{offset + sizeof(count) + count * sizeof(diskann_id_t),
          sizeof(neighbor_id), &neighbor_id},
         {offset, sizeof(next_count), &next_count}}};
    if (!segment->write_batch(writes.data(), writes.size()))
      return segment->write_error();
    observe_degree(next_count);
    return 0;
  }
  if (neighbors_compacted_) return IndexError_NoReady;
  const int ret = neighbor_pool_.append(id, neighbor_id);
  if (ret == 0) observe_degree(neighbor_pool_.get(id).first);
  return ret;
} catch (const std::exception &error) {
  LOG_ERROR("Failed to append DiskANN build adjacency: %s", error.what());
  return IndexError_Runtime;
}

int DiskAnnBuilderEntity::compact_neighbors() {
  // Pageable adjacency already avoids resident capacity for unused edges.
  if (buffered_state_) return 0;
  if (neighbors_compacted_) return 0;
  const int ret = neighbor_pool_.compact(max_degree_);
  if (ret == 0) neighbors_compacted_ = true;
  return ret;
}

int DiskAnnBuilderEntity::prepare_codes(size_t bytes) {
  if (bytes > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
    return IndexError_InvalidArgument;
  }
  code_bytes_ = 0;
  code_bytes_written_ = 0;
  try {
    if (buffered_state_) {
      auto &state = *buffered_state_;
      const int ret = state.ensure_directory();
      if (ret != 0) return ret;
      state.codes.close();
      if (!state.codes.create(state.code_path, 0)) return IndexError_OpenFile;
      std::vector<uint8_t>().swap(block_compressed_data_);
    } else {
      block_compressed_data_.resize(bytes);
    }
    code_bytes_ = bytes;
    return 0;
  } catch (const std::exception &error) {
    LOG_ERROR("Failed to prepare DiskANN PQ codes: %s", error.what());
    return IndexError_Runtime;
  }
}

int DiskAnnBuilderEntity::append_codes(const void *data, size_t bytes) {
  if (code_bytes_written_ > code_bytes_ ||
      bytes > code_bytes_ - code_bytes_written_ || (bytes != 0 && !data)) {
    return IndexError_InvalidArgument;
  }
  if (bytes == 0) return 0;
  if (buffered_state_) {
    if (!buffered_state_->codes.is_valid() ||
        buffered_state_->codes.write(static_cast<ssize_t>(code_bytes_written_),
                                     data, bytes) != bytes) {
      return IndexError_WriteData;
    }
  } else {
    if (block_compressed_data_.size() < code_bytes_) return IndexError_Runtime;
    memcpy(block_compressed_data_.data() + code_bytes_written_, data, bytes);
  }
  code_bytes_written_ += bytes;
  return 0;
}

int64_t DiskAnnBuilderEntity::dump_segment(const IndexDumper::Pointer &dumper,
                                           const std::string &segment_id,
                                           const void *data,
                                           size_t size) const {
  size_t len = dumper->write(data, size);
  if (len != size) {
    LOG_ERROR("Dump segment %s data failed, expect: %lu, actual: %lu",
              segment_id.c_str(), size, len);
    return IndexError_WriteData;
  }

  size_t padding_size = AlignSize(size) - size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  uint32_t crc = ailego::Crc32c::Hash(data, size);
  int ret = dumper->append(segment_id, size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d", segment_id.c_str(), ret);
    return ret;
  }

  return 0;
}

void DiskAnnBuilderEntity::set_pq_quantizer(
    std::shared_ptr<const turbo::Quantizer> quantizer) {
  pq_quantizer_ = std::move(quantizer);
  pq_meta_.quantizer_meta_buffer_size =
      pq_quantizer_ ? pq_quantizer_->serialized_size() : 0;
}

int DiskAnnBuilderEntity::dump_pq_meta_segment(
    const IndexDumper::Pointer &dumper) const {
  if (!pq_quantizer_ || pq_meta_.quantizer_meta_buffer_size == 0) {
    return IndexError_InvalidArgument;
  }
  uint32_t crc = 0U;

  // write meta
  size_t size_pq_meta = dumper->write(&pq_meta_, sizeof(DiskAnnPqMeta));
  if (size_pq_meta != sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("Failed to dump PQ meta data, expect: %lu, actual: %lu",
              sizeof(DiskAnnPqMeta), size_pq_meta);
    return IndexError_WriteData;
  }

  crc = ailego::Crc32c::Hash(&pq_meta_, sizeof(DiskAnnPqMeta), crc);

  // Write bounded chunks directly from the retained trained quantizer.
  size_t size_meta_buffer = 0;
  int ret =
      pq_quantizer_->serialize_to([&](const void *data, size_t bytes) -> int {
        if (bytes > pq_meta_.quantizer_meta_buffer_size - size_meta_buffer ||
            dumper->write(data, bytes) != bytes) {
          return IndexError_WriteData;
        }
        crc = ailego::Crc32c::Hash(data, bytes, crc);
        size_meta_buffer += bytes;
        return 0;
      });
  if (ret != 0) return ret;
  if (size_meta_buffer != pq_meta_.quantizer_meta_buffer_size) {
    return IndexError_WriteData;
  }

  // write size
  size_t size_total = size_pq_meta + size_meta_buffer;

  // write pad
  size_t padding_size = AlignSize(size_total) - size_total;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  ret = dumper->append(kDiskAnnPqMetaSegmentId, size_total, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump PQ segment failed, ret %d", ret);
    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_pq_data_segment(
    const IndexDumper::Pointer &dumper) const try {
  uint64_t doc_cnt = meta_header_.doc_cnt;
  uint64_t chunk_num = pq_meta_.chunk_num;

  if (chunk_num != 0 &&
      doc_cnt > std::numeric_limits<size_t>::max() / chunk_num) {
    return IndexError_InvalidArgument;
  }
  const size_t size_total = static_cast<size_t>(doc_cnt * chunk_num);
  if ((code_bytes_ != 0 &&
       (code_bytes_ != size_total || code_bytes_written_ != size_total)) ||
      (buffered_state_ &&
       (code_bytes_ != size_total || code_bytes_written_ != size_total)) ||
      (!buffered_state_ && block_compressed_data_.size() < size_total)) {
    return IndexError_Mismatch;
  }

  uint32_t crc = 0U;

  if (buffered_state_) {
    constexpr size_t kCopyBytes = 1024UL * 1024UL;
    ailego::BudgetedMappedBuffer buffer(std::min(size_total, kCopyBytes));
    for (size_t offset = 0; offset < size_total;) {
      const size_t bytes = std::min(buffer.size(), size_total - offset);
      if (buffered_state_->codes.read(static_cast<ssize_t>(offset),
                                      buffer.data(), bytes) != bytes)
        return IndexError_ReadData;
      if (dumper->write(buffer.data(), bytes) != bytes)
        return IndexError_WriteData;
      crc = ailego::Crc32c::Hash(buffer.data(), bytes, crc);
      offset += bytes;
    }
  } else {
    if (dumper->write(block_compressed_data_.data(), size_total) !=
        size_total) {
      return IndexError_WriteData;
    }
    crc = ailego::Crc32c::Hash(block_compressed_data_.data(), size_total, crc);
  }

  // write pad
  size_t padding_size = AlignSize(size_total) - size_total;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  int ret =
      dumper->append(kDiskAnnPqDataSegmentId, size_total, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump PQ data segment failed, ret %d", ret);
    return ret;
  }

  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &error) {
  LOG_ERROR("Failed to dump DiskANN PQ codes: %s", error.what());
  return IndexError_Runtime;
}

int DiskAnnBuilderEntity::dump_dummy_segment(
    const IndexDumper::Pointer &dumper) const {
  // to make offset aligned with 4K
  size_t dumper_header_size = dumper->size();

  size_t dummy_size =
      DiskAnnUtil::round_up(dumper_header_size, DiskAnnUtil::kSectorSize) -
      dumper_header_size;

  if (dummy_size != 0) {
    std::string dummpy_data(dummy_size, '\0');
    if (dumper->write(dummpy_data.data(), dummy_size) != dummy_size) {
      LOG_ERROR("write dummy failed, size %lu", dummy_size);
      return IndexError_WriteData;
    }

    int ret = dumper->append(kDiskAnnDummpySegmentId, dummy_size, 0, 0);
    if (ret != 0) {
      LOG_ERROR("Dump dummy data segment failed, ret %d", ret);
      return ret;
    }
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_key_segment(
    const IndexDumper::Pointer &dumper) const {
  if (buffered_state_) {
    ImmutableSegmentView keys;
    const size_t bytes = doc_cnt() * sizeof(diskann_key_t);
    int ret = keys.load(buffered_state_->graph_storage, "keys", 0, bytes);
    if (ret != 0) return ret;
    uint32_t crc = 0;
    ret = keys.copy_to([&](size_t, const void *data, size_t n) {
      crc = ailego::Crc32c::Hash(data, n, crc);
      return dumper->write(data, n) == n ? 0 : IndexError_WriteData;
    });
    if (ret != 0) return ret;
    const size_t padding = AlignSize(bytes) - bytes;
    const char zeros[4096]{};
    if (padding && dumper->write(zeros, padding) != padding)
      return IndexError_WriteData;
    return dumper->append(kDiskAnnKeySegmentId, bytes, padding, crc);
  }
  //! Dump keys
  size_t key_segment_size = doc_cnt() * sizeof(diskann_key_t);
  int64_t keys_size = dump_segment(dumper, kDiskAnnKeySegmentId,
                                   keys_buffer_.data(), key_segment_size);
  if (keys_size < 0) {
    return keys_size;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_key_mapping_segment(
    const IndexDumper::Pointer &dumper) const {
  if (buffered_state_) {
    PagedBuildBuffer mapping;
    int ret = BuildPagedKeyMapping(
        doc_cnt(),
        [this](size_t id, uint64_t *key) {
          return read_key(static_cast<diskann_id_t>(id), key);
        },
        &mapping);
    if (ret != 0) return ret;
    return mapping.dump(dumper, kDiskAnnKeyMappingSegmentId, 4096);
  }
  ailego::BudgetedVector<diskann_id_t> mapping(doc_cnt());

  const diskann_key_t *keys = reinterpret_cast<diskann_key_t *>(
      const_cast<char *>(keys_buffer_.data()));

  std::iota(mapping.begin(), mapping.end(), 0U);
  std::sort(mapping.begin(), mapping.end(),
            [&](diskann_id_t i, diskann_id_t j) { return keys[i] < keys[j]; });

  size_t size = mapping.size() * sizeof(diskann_id_t);
  int64_t ret =
      dump_segment(dumper, kDiskAnnKeyMappingSegmentId, mapping.data(), size);

  if (ret != 0) {
    LOG_ERROR("Dump vectors segment failed");

    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_entrypoint_segment(
    const IndexDumper::Pointer &dumper) const {
  ailego::BudgetedVector<char> entrypoint_buffer;

  size_t size = sizeof(uint32_t) + entrypoints_.size() * sizeof(diskann_id_t);
  entrypoint_buffer.resize(size);

  uint8_t *buffer_ptr = reinterpret_cast<uint8_t *>(&entrypoint_buffer[0]);

  uint32_t point_cnt = entrypoints_.size();
  memcpy(buffer_ptr, &point_cnt, sizeof(uint32_t));
  memcpy(buffer_ptr + sizeof(uint32_t), entrypoints_.data(),
         entrypoints_.size() * sizeof(diskann_id_t));

  int64_t ret = dump_segment(dumper, kDiskAnnEntryPointSegmentId,
                             entrypoint_buffer.data(), size);

  if (ret != 0) {
    LOG_ERROR("Dump entrypoint segment failed");

    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump(IndexHolder::Pointer holder, IndexMeta &meta,
                               const IndexDumper::Pointer &dumper) try {
  if (!holder || holder->count() != this->doc_cnt() ||
      holder->element_size() != meta_.element_size()) {
    return IndexError_Mismatch;
  }
  uint64_t doc_cnt = holder->count();
  const uint32_t max_observed_degree =
      max_observed_degree_.load(std::memory_order_relaxed);
  uint64_t max_node_size =
      (uint64_t)max_observed_degree * sizeof(diskann_id_t) + sizeof(uint32_t) +
      meta_.element_size();
  uint64_t node_per_sector =
      DiskAnnUtil::kSectorSize /
      max_node_size;  // 0 if max_node_size > DiskAnnUtil::kSectorSize

  const uint64_t sectors_per_node =
      DiskAnnUtil::div_round_up(max_node_size, DiskAnnUtil::kSectorSize);
  if (sectors_per_node >
      (std::numeric_limits<size_t>::max)() / DiskAnnUtil::kSectorSize) {
    return IndexError_NoMemory;
  }
  const size_t buffer_bytes = sectors_per_node * DiskAnnUtil::kSectorSize;
  ailego::BudgetedAlignedBuffer sector_storage(buffer_bytes,
                                               AILEGO_CACHE_LINE_SIZE);
  auto *sector_buffer = static_cast<char *>(sector_storage.data());
  ailego::BudgetedVector<diskann_id_t> neighbors;
  neighbors.reserve(max_observed_degree);

  LOG_INFO(
      "Dump Data, medoid: %zu, max node size: %zu, node per sector: %zu, "
      "max observed degree: %zu",
      (size_t)medoid(), (size_t)max_node_size, (size_t)node_per_sector,
      (size_t)max_observed_degree);

  // write a dummy segment to make data align
  int ret = dump_dummy_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump dummy segment failed");

    return ret;
  }

  // dump data by sector
  size_t write_size = 0;
  uint32_t crc = 0U;
  size_t len = 0;

  // no need to write first sector
  OrdinalAccessHolder::Reader::Pointer reader;
  if (auto *source = dynamic_cast<OrdinalAccessHolder *>(holder.get())) {
    ret = source->create_ordinal_reader(&reader);
    if (ret != 0 && ret != IndexError_NotImplemented) return ret;
    if (ret == 0 && !reader) return IndexError_Runtime;
  }
  auto iter = reader ? nullptr : holder->create_iterator();
  if (!reader && !iter) return IndexError_Runtime;
  AILEGO_DEFER([&]() {
    if (reader) reader->reset();
  });
  auto read_vector = [&](size_t id, void *output) -> int {
    uint64_t key = 0;
    const void *data = nullptr;
    if (reader) {
      int result = reader->read(id, &key, &data);
      if (result != 0) return result;
    } else {
      if (!iter->is_valid()) return IndexError_Mismatch;
      key = iter->key();
      data = iter->data();
      // A deferred read failure can return a non-null placeholder and
      // invalidate the iterator. Reject it before copying, even on the last
      // vector where no subsequent iteration would detect the error.
      if (!iter->is_valid()) {
        return IndexError_ReadData;
      }
    }
    if (!data) return IndexError_ReadData;
    if (key != get_key(id)) return IndexError_Mismatch;
    memcpy(output, data, meta.element_size());
    if (iter) iter->next();
    return 0;
  };

  auto encode_node = [&](diskann_id_t id, char *out) -> int {
    int status = read_neighbors(id, &neighbors);
    if (status != 0) return status;
    status = read_vector(id, out);
    if (status != 0) return status;
    const auto count = static_cast<uint32_t>(neighbors.size());
    ailego_assert(count > 0 && count <= max_observed_degree);
    auto *adjacency = out + meta_.element_size();
    std::memcpy(adjacency, &count, sizeof(count));
    std::memcpy(adjacency + sizeof(count), neighbors.data(),
                neighbors.size() * sizeof(diskann_id_t));
    return 0;
  };

  // Pack small nodes into one sector, or give a large node its own aligned
  // extent. The same pre-admitted buffer covers both layouts.
  const uint64_t nodes_per_write = node_per_sector ? node_per_sector : 1;
  for (uint64_t first = 0; first < doc_cnt; first += nodes_per_write) {
    std::memset(sector_buffer, 0, buffer_bytes);
    const uint64_t count = std::min(nodes_per_write, doc_cnt - first);
    for (uint64_t i = 0; i < count; ++i) {
      ret = encode_node(static_cast<diskann_id_t>(first + i),
                        sector_buffer + i * max_node_size);
      if (ret != 0) return ret;
    }
    len = dumper->write(sector_buffer, buffer_bytes);
    if (len != buffer_bytes) return IndexError_WriteData;
    write_size += buffer_bytes;
    crc = ailego::Crc32c::Hash(sector_buffer, buffer_bytes, crc);
  }
  const uint64_t index_size = write_size;

  const size_t padding_size = AlignSize(write_size) - write_size;
  const char padding[4096]{};
  if (padding_size && dumper->write(padding, padding_size) != padding_size) {
    LOG_ERROR("Append padding failed, size %zu", padding_size);
    return IndexError_WriteData;
  }

  sector_storage = {};
  decltype(neighbors)().swap(neighbors);
  ret = dumper->append(kDiskAnnVectorSegmentId, write_size, 0UL, crc);
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment failed, ret %d", ret);
    return ret;
  }

  // dump diskann meta
  meta_header_.doc_cnt = doc_cnt;
  meta_header_.ndims = meta_.dimension();
  meta_header_.medoid = medoid();
  meta_header_.max_node_size = max_node_size;
  meta_header_.max_degree = max_observed_degree;
  meta_header_.node_per_sector = node_per_sector;
  meta_header_.vamana_frozen_num = 0;
  meta_header_.vamana_frozen_loc = medoid();
  meta_header_.append_reorder_data = 0;
  meta_header_.index_size = index_size;

  ret = dump_segment(dumper, kDiskAnnMetaSegmentId, &meta_header_,
                     sizeof(DiskAnnMetaHeader));
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment failed");

    return ret;
  }

  // dump pq meta
  ret = dump_pq_meta_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump pq meta segment failed");

    return ret;
  }

  // dump pq data
  ret = dump_pq_data_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump pq data segment failed");

    return ret;
  }

  // dump key
  ret = dump_key_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump key segment failed");

    return ret;
  }

  // dump key mapping
  ret = dump_key_mapping_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump key mapping segment failed");

    return ret;
  }

  // dump entrypint
  ret = dump_entrypoint_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump entrypoint segment failed");

    return ret;
  }

  LOG_INFO("DiskAnn Index File Dumped");

  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

}  // namespace core
}  // namespace zvec

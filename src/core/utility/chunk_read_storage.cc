// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include "chunk_read_storage.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/io/file.h>
#include "ailego/utility/memory_helper.h"
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_unpacker.h>
#include "utility_params.h"

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace zvec {
namespace core {
namespace {
constexpr uint64_t kNoKey = std::numeric_limits<uint64_t>::max();
constexpr size_t kMetadataReserve = 128UL << 10;

bool ResolveOffset(size_t size, int64_t offset, bool from_end, size_t *out) {
  if (offset > 0 || (offset == 0 && !from_end)) {
    if (static_cast<uint64_t>(offset) > size) return false;
    *out = static_cast<size_t>(offset);
    return true;
  }
  const uint64_t distance = offset == 0 ? 0 : uint64_t(-(offset + 1)) + 1;
  if (distance > size) return false;
  *out = size - static_cast<size_t>(distance);
  return true;
}

struct MetadataCharge {
  explicit MetadataCharge(size_t count) : bytes(count) {
    if (!ailego::MemoryLimitPool::get_instance().try_charge_metadata(bytes))
      throw std::bad_alloc();
  }
  ~MetadataCharge() {
    ailego::MemoryLimitPool::get_instance().release_metadata(bytes);
  }
  size_t bytes;
};

struct ChargedBuffer {
  ChargedBuffer(size_t count, std::shared_ptr<MetadataCharge> metadata,
                bool workspace = false)
      : bytes((count + ailego::MemoryHelper::PageSize() - 1) /
              ailego::MemoryHelper::PageSize() * ailego::MemoryHelper::PageSize()),
        metadata_owner(std::move(metadata)) {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    using Pending = ailego::MemoryLimitPool::PendingAllocation;
    Pending pending(pool, workspace ? Pending::Kind::kWorkspace
                                    : Pending::Kind::kExternal);
    if (!pending.reserve(bytes)) throw std::bad_alloc();
    data = static_cast<uint8_t *>(ailego::File::MemoryMap(bytes, 0));
    if (!data) throw std::bad_alloc();
    projection = pending.commit_mapping();
  }
  ~ChargedBuffer() { (void)release_backing(); }
  bool release_backing() {
    if (!data) return true;
    if (!ailego::File::TryMemoryUnmap(data, bytes)) return false;
    data = nullptr;
    ailego::MemoryLimitPool::get_instance().release_unmapped_external(
        bytes, projection);
    return true;
  }
  ChargedBuffer(const ChargedBuffer &) = delete;
  ChargedBuffer &operator=(const ChargedBuffer &) = delete;
  uint8_t *data{nullptr};
  size_t bytes;
  std::shared_ptr<MetadataCharge> metadata_owner;
  ailego::ProcessRssBudget::MaterializationStamp projection{};
  uint64_t key{kNoKey};
  bool loading{false}, referenced{false};
};
}  // namespace

struct ChunkReadStorage::State : ailego::EvictableBlockOwner {
  struct View {
    std::shared_ptr<ChargedBuffer> workspace;
    std::vector<std::shared_ptr<ChargedBuffer>> pins;
  };

  ~State() override {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    changed.notify_all();
    if (worker.joinable()) worker.join();
    if (registered) ailego::BlockEvictionQueue::get_instance().set_invalid(this);
    // A returned View or buffer may outlive the storage. Its own destructor
    // releases its budget, so close never credits live reader memory.
    views.clear();
    slots.clear();
  }

  bool initialize(const std::string &file_path, size_t chunk, size_t cache,
                  size_t max_read, size_t count, size_t wait, bool direct) {
    if (!meta_file.open(file_path, true, false)) return false;
    if (!data_file.open(file_path, true, direct)) {
      if (!direct || !data_file.open(file_path, true, false)) return false;
    }
#if !defined(_WIN32)
    struct stat a{}, b{};
    if (::fstat(meta_file.native_handle(), &a) ||
        ::fstat(data_file.native_handle(), &b) ||
        a.st_dev != b.st_dev || a.st_ino != b.st_ino) return false;
#else
    BY_HANDLE_FILE_INFORMATION a{}, b{};
    if (!GetFileInformationByHandle(meta_file.native_handle(), &a) ||
        !GetFileInformationByHandle(data_file.native_handle(), &b) ||
        a.dwVolumeSerialNumber != b.dwVolumeSerialNumber ||
        a.nFileIndexHigh != b.nFileIndexHigh ||
        a.nFileIndexLow != b.nFileIndexLow) return false;
#endif
    file_size = meta_file.size();
    chunk_bytes = chunk;
    max_read_bytes = max_read;
    wait_ms = wait;
    const size_t slots_count = cache / chunk;
    size_t table_size = 2;
    while (table_size < slots_count * 2 + 1) table_size <<= 1;
    // Includes the fixed hash table, lease descriptors, frequency sketch and
    // a conservative reserve for parsed segment names/maps.
    metadata_charge = kMetadataReserve + table_size * sizeof(int32_t) +
                      slots_count * (sizeof(ChargedBuffer) + 128) +
                      count * (256 + (max_read / chunk + 3) *
                                         sizeof(std::shared_ptr<ChargedBuffer>));
    metadata_owner = std::make_shared<MetadataCharge>(metadata_charge);
    directory.assign(table_size, -1);
    slots.reserve(slots_count);
    ready.reserve(slots_count);
    views.reserve(count);
    // Execution storage is reserved before cache storage and never borrowed
    // by the cache. Two edge chunks permit aligned, zero-copy range reads.
    for (size_t i = 0; i < count; ++i) {
      auto view = std::make_shared<View>();
      view->workspace =
          std::make_shared<ChargedBuffer>(max_read + 2 * chunk, metadata_owner,
                                           true);
      view->pins.reserve(max_read / chunk + 3);
      views.emplace_back(std::move(view));
    }
    for (size_t i = 0; i < slots_count; ++i) {
      slots.emplace_back(std::make_shared<ChargedBuffer>(chunk, metadata_owner));
      ready.emplace_back(i);
    }
    return true;
  }

  void start() {
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    version = queue.next_version();
    queue.set_valid(this);
    registered = true;
    for (size_t i = 0; i < slots.size(); ++i) requeue(i);
    worker = std::thread([this] { maintenance(); });
  }

  static uint64_t mix(uint64_t key) {
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    return key;
  }
  size_t hash(uint64_t key) const {
    return size_t(mix(key)) & (directory.size() - 1);
  }
  uint8_t observe(uint64_t key) {
    auto &frequency = frequencies[size_t(mix(key)) % frequencies.size()];
    if (frequency < 255) ++frequency;
    if (++observations % 4096 == 0) {
      age_requested = true;
      changed.notify_all();
    }
    return frequency;
  }
  int32_t find(uint64_t key) const {
    size_t index = hash(key);
    for (size_t n = 0; n < directory.size(); ++n) {
      int32_t value = directory[index];
      if (value < 0) return -1;
      if (slots[value] && slots[value]->key == key) return value;
      index = (index + 1) & (directory.size() - 1);
    }
    return -1;
  }
  void insert(size_t slot) {
    size_t index = hash(slots[slot]->key);
    while (directory[index] >= 0)
      index = (index + 1) & (directory.size() - 1);
    directory[index] = static_cast<int32_t>(slot);
  }
  void erase(uint64_t key) {
    size_t index = hash(key);
    while (directory[index] >= 0) {
      size_t value = size_t(directory[index]);
      if (slots[value] && slots[value]->key == key) break;
      index = (index + 1) & (directory.size() - 1);
    }
    if (directory[index] < 0) return;
    directory[index] = -1;
    // Backward cluster repair prevents unbounded tombstone accumulation.
    index = (index + 1) & (directory.size() - 1);
    while (directory[index] >= 0) {
      size_t value = size_t(directory[index]);
      directory[index] = -1;
      insert(value);
      index = (index + 1) & (directory.size() - 1);
    }
  }

  void reap_views_locked() {
    for (auto &view : views)
      if (view.use_count() == 1) view->pins.clear();
  }
  std::shared_ptr<View> acquire_view() {
    std::unique_lock<std::mutex> lock(mutex);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(wait_ms);
    for (;;) {
      for (auto &view : views) {
        if (view.use_count() == 1) {
          view->pins.clear();
          return view;
        }
      }
      workspace_waits.fetch_add(1, std::memory_order_relaxed);
      if (stopping || std::chrono::steady_clock::now() >= deadline) return {};
      changed.wait_for(lock, std::chrono::milliseconds(1));
    }
  }

  bool read_data(size_t offset, size_t requested, uint8_t *output,
                 size_t required) {
    if (offset >= file_size || required > file_size - offset) return false;
    // Full aligned reads may reach EOF. Only the logical range must exist.
    const size_t got =
        data_file.read(static_cast<ssize_t>(offset), output, requested);
    io_calls.fetch_add(1, std::memory_order_relaxed);
    io_bytes.fetch_add(got, std::memory_order_relaxed);
    return got >= required;
  }

  std::shared_ptr<ChargedBuffer> cached(uint64_t key) {
    std::unique_lock<std::mutex> lock(mutex);
    int32_t index = find(key);
    if (index >= 0) {
      auto slot = slots[index];
      if (slot->loading) {
        // Join an in-flight fill briefly; do not wait for cache eviction.
        changed.wait_for(lock, std::chrono::milliseconds(wait_ms),
                         [&] { return !slot->loading || stopping; });
      }
      if (slot->loading || stopping) {
        misses.fetch_add(1, std::memory_order_relaxed);
        return {};
      }
      if (!slot->loading && slot->key == key) {
        (void)observe(key);
        slot->referenced = true;
        hits.fetch_add(1, std::memory_order_relaxed);
        return slot;
      }
    }
    misses.fetch_add(1, std::memory_order_relaxed);
    const uint8_t frequency = observe(key);
    while (!ready.empty()) {
      const size_t candidate = ready.back();
      auto &slot = slots[candidate];
      if (!slot || slot->key != kNoKey) {
        ready.pop_back();
        continue;
      }
      // Failed loaders may still have waiters holding this slot briefly.
      // Keep it available for a later request instead of losing its capacity.
      if (slot.use_count() != 1) break;
      ready.pop_back();
      slot->key = key;
      slot->loading = true;
      slot->referenced = true;
      insert(candidate);
      auto result = slot;
      lock.unlock();
      const size_t offset = size_t(key) * chunk_bytes;
      const bool ok = read_data(offset, chunk_bytes, result->data,
                                 std::min(chunk_bytes, file_size - offset));
      lock.lock();
      result->loading = false;
      if (!ok) {
        erase(key);
        result->key = kNoKey;
        ready.push_back(candidate);
      } else {
        fills.fetch_add(1, std::memory_order_relaxed);
      }
      changed.notify_all();
      return ok ? result : std::shared_ptr<ChargedBuffer>{};
    }
    if (frequency >= 2 && !slots.empty()) {
      requested_ready = std::max<size_t>(requested_ready, 4);
      admission_threshold = std::max(admission_threshold, frequency);
      changed.notify_all();
    }
    return {};
  }

  size_t scatter(size_t offset, size_t len, Segment::ScatterBlock &out) {
    out.reset();
    if (!len) return 0;
    if (offset > file_size || len > file_size - offset ||
        len > max_read_bytes) return 0;
    const size_t first = offset / chunk_bytes;
    const size_t last = (offset + len - 1) / chunk_bytes;
    const size_t count = last - first + 1;
    // A single hit needs no batch lease or workspace acquisition.
    if (count == 1) {
      auto slot = cached(first);
      if (slot) {
        out.lease = slot;
        out.spans.push_back({slot->data + offset % chunk_bytes, len});
        out.size = len;
        return len;
      }
    }
    auto view = acquire_view();
    if (!view) return 0;
    out.lease = view;
    out.spans.reserve(count);
    size_t run_first = 0, run_count = 0;
    auto finish_run = [&]() {
      if (!run_count) return true;
      const size_t begin = std::max(run_first * chunk_bytes, offset);
      const size_t end = std::min(
          (run_first + run_count) * chunk_bytes, offset + len);
      const size_t page = ailego::MemoryHelper::PageSize();
      const size_t physical = begin / page * page;
      const size_t requested = (end - physical + page - 1) / page * page;
      auto *destination = view->workspace->data +
                          physical - first * chunk_bytes;
      // Bypass only the requested range plus I/O alignment. Reading whole
      // cache chunks here rereads the shared edge of consecutive IVF batches.
      if (!read_data(physical, requested, destination, end - physical))
        return false;
      out.spans.push_back({destination + begin - physical, end - begin});
      run_count = 0;
      return true;
    };
    for (size_t key = first; key <= last; ++key) {
      // The single-chunk attempt above already observed this miss.
      auto slot = count == 1 ? std::shared_ptr<ChargedBuffer>{} : cached(key);
      if (slot) {
        if (!finish_run()) { out.reset(); return 0; }
        const size_t begin = std::max(key * chunk_bytes, offset);
        const size_t end = std::min((key + 1) * chunk_bytes, offset + len);
        out.spans.push_back({slot->data + begin - key * chunk_bytes,
                             end - begin});
        view->pins.emplace_back(std::move(slot));
      } else {
        if (!run_count) run_first = key;
        ++run_count;
      }
    }
    if (!finish_run()) { out.reset(); return 0; }
    out.size = len;
    return len;
  }

  void requeue(size_t key) {
    ailego::BlockEvictionQueue::BlockType item;
    item.owner = this;
    item.owner_key = key;
    item.version = version;
    ailego::BlockEvictionQueue::get_instance().add_single_block(item, 0);
  }
  bool is_dead_block(ailego::eviction_key_t key,
                     ailego::version_t generation) override {
    std::lock_guard<std::mutex> lock(mutex);
    return generation != version || key >= slots.size() || !slots[key];
  }
  bool evict_block(ailego::eviction_key_t key) override {
    std::lock_guard<std::mutex> lock(mutex);
    reap_views_locked();
    if (key >= slots.size() || !slots[key]) return false;
    auto &slot = slots[key];
    if (slot.use_count() != 1 || slot->loading) {
      requeue(key);
      return false;
    }
    if (!slot->release_backing()) {
      requeue(key);
      return false;
    }
    if (slot->key != kNoKey) erase(slot->key);
    // This is real global-budget reclamation, unlike ordinary replacement:
    // destroy the allocation before reporting success.
    slot.reset();
    trims.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  void maintenance() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stopping) {
      changed.wait(lock, [&] {
        return stopping || requested_ready != 0 || age_requested;
      });
      if (stopping) break;
      reap_views_locked();
      if (age_requested) {
        for (auto &frequency : frequencies) frequency /= 2;
        age_requested = false;
      }
      const size_t target = requested_ready;
      const uint8_t threshold = admission_threshold;
      requested_ready = 0;
      admission_threshold = 0;
      size_t attempts = slots.size() * 2;
      while (ready.size() < target && attempts-- && !slots.empty()) {
        size_t index = hand++ % slots.size();
        auto &slot = slots[index];
        if (!slot || slot.use_count() != 1 || slot->loading ||
            slot->key == kNoKey) continue;
        // A repeating cold scan must not evict a frequently reused chunk.
        // Equal estimated utility keeps the existing resident entry.
        if (frequencies[size_t(mix(slot->key)) % frequencies.size()] >= threshold)
          continue;
        if (slot->referenced) {
          slot->referenced = false;
          continue;
        }
        erase(slot->key);
        slot->key = kNoKey;
        ready.push_back(index);
        replacements.fetch_add(1, std::memory_order_relaxed);
      }
      changed.notify_all();
    }
  }

  ailego::File data_file, meta_file;
  std::shared_ptr<MetadataCharge> metadata_owner;
  size_t file_size{0}, chunk_bytes{0}, max_read_bytes{0}, wait_ms{0};
  size_t index_offset{0}, metadata_charge{0}, hand{0}, requested_ready{0};
  uint64_t observations{0};
  uint8_t admission_threshold{0};
  uint32_t file_magic{0};
  std::string path;
  IndexUnpacker::SegmentMap segments;
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::shared_ptr<ChargedBuffer>> slots;
  std::vector<std::shared_ptr<View>> views;
  std::vector<int32_t> directory;
  std::vector<size_t> ready;
  std::array<uint8_t, 4096> frequencies{};
  std::thread worker;
  bool stopping{false}, registered{false}, age_requested{false};
  ailego::version_t version{0};
  std::atomic<uint64_t> hits{0}, misses{0}, fills{0}, replacements{0};
  std::atomic<uint64_t> io_calls{0}, io_bytes{0}, copied{0}, workspace_waits{0};
  std::atomic<uint64_t> trims{0};
};

class ChunkReadStorage::ReadSegment : public IndexStorage::Segment {
 public:
  ReadSegment(std::shared_ptr<State> state,
              const IndexUnpacker::SegmentMeta &meta)
      : state_(std::move(state)), offset_(state_->index_offset +
                                         meta.data_offset()),
        size_(meta.data_size()), padding_(meta.padding_size()),
        crc_(meta.data_crc()) {}
  size_t data_size() const override { return size_; }
  size_t data_offset() const override { return offset_; }
  uint32_t data_crc() const override { return crc_; }
  size_t padding_size() const override { return padding_; }
  size_t capacity() const override { return size_ + padding_; }
  size_t read_scatter(size_t offset, ScatterBlock &out, size_t len) override {
    if (offset > capacity()) { out.reset(); return 0; }
    len = std::min(len, capacity() - offset);
    return state_->scatter(offset_ + offset, len, out);
  }
  size_t read(size_t offset, MemoryBlock &out, size_t len) override {
    out.reset();
    ScatterBlock block;
    const size_t got = read_scatter(offset, block, len);
    if (!got) return 0;
    if (block.spans.size() == 1) {
      out = MemoryBlock::MakeSharedView(
          const_cast<uint8_t *>(block.spans.front().data), block.lease);
      return got;
    }
    // Contiguous callers (e.g. legacy column-major kernels) retain one copy.
    // Row-major IVF consumes scatter spans and never executes this branch.
    auto view = std::static_pointer_cast<State::View>(block.lease);
    auto *destination = view->workspace->data +
                        (offset_ + offset) % state_->chunk_bytes;
    size_t copied = 0;
    for (const auto &span : block.spans) {
      std::memmove(destination + copied, span.data, span.size);
      copied += span.size;
    }
    state_->copied.fetch_add(got, std::memory_order_relaxed);
    out = MemoryBlock::MakeSharedView(destination, block.lease);
    return got;
  }
  size_t fetch(size_t offset, void *destination, size_t len) const override {
    if (offset > capacity() || !destination) return 0;
    len = std::min(len, capacity() - offset);
    size_t done = 0;
    while (done < len) {
      ScatterBlock block;
      size_t count = std::min(len - done, state_->max_read_bytes);
      if (state_->scatter(offset_ + offset + done, count, block) != count)
        return done;
      size_t cursor = done;
      for (const auto &span : block.spans) {
        std::memcpy(static_cast<uint8_t *>(destination) + cursor,
                    span.data, span.size);
        cursor += span.size;
      }
      state_->copied.fetch_add(count, std::memory_order_relaxed);
      done += count;
    }
    return done;
  }
  size_t write(size_t, const void *, size_t) override { return 0; }
  size_t resize(size_t) override { return capacity(); }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override { return std::make_shared<ReadSegment>(*this); }

 private:
  std::shared_ptr<State> state_;
  size_t offset_, size_, padding_;
  uint32_t crc_;
};

ChunkReadStorage::ChunkReadStorage() = default;
ChunkReadStorage::~ChunkReadStorage() = default;
int ChunkReadStorage::init(const ailego::Params &params) {
  params.get(CHUNK_READ_STORAGE_CHUNK_BYTES, &chunk_bytes_);
  params.get(CHUNK_READ_STORAGE_CACHE_BYTES, &cache_bytes_);
  params.get(CHUNK_READ_STORAGE_MAX_READ_BYTES, &max_read_bytes_);
  params.get(CHUNK_READ_STORAGE_WORKSPACE_COUNT, &workspace_count_);
  params.get(CHUNK_READ_STORAGE_WAIT_MS, &wait_ms_);
  params.get(CHUNK_READ_STORAGE_DIRECT_IO, &direct_io_);
  params.get(BUFFER_READ_STORAGE_HEADER_OFFSET, &header_offset_);
  params.get(BUFFER_READ_STORAGE_FOOTER_OFFSET, &footer_offset_);
  params.get(BUFFER_READ_STORAGE_CHECKSUM_VALIDATION, &checksum_);
  const size_t page = ailego::MemoryHelper::PageSize();
  if (chunk_bytes_ < page || chunk_bytes_ % page ||
      (chunk_bytes_ & (chunk_bytes_ - 1)) || workspace_count_ == 0 ||
      workspace_count_ > 1024 || max_read_bytes_ == 0 ||
      max_read_bytes_ > (1UL << 30) || chunk_bytes_ > (64UL << 20) ||
      cache_bytes_ / chunk_bytes_ > (1UL << 20) || wait_ms_ > 60000)
    return IndexError_InvalidArgument;
  return 0;
}
int ChunkReadStorage::open(const std::string &path, bool) {
  try {
    auto candidate = std::make_shared<State>();
    if (!candidate->initialize(path, chunk_bytes_, cache_bytes_,
                               max_read_bytes_, workspace_count_, wait_ms_,
                               direct_io_))
      return IndexError_OpenFile;
    size_t end = 0;
    if (!ResolveOffset(candidate->file_size, header_offset_, false,
                       &candidate->index_offset) ||
        !ResolveOffset(candidate->file_size, footer_offset_, true, &end) ||
        candidate->index_offset >= end) return IndexError_InvalidArgument;
    const size_t container_size = end - candidate->index_offset;
    auto scratch = candidate->views.front()->workspace;
    auto read = [&](size_t offset, const void **data, size_t length) {
      if (offset > container_size || length > container_size - offset ||
          length > scratch->bytes) return size_t{0};
      *data = scratch->data;
      return candidate->meta_file.read(
          static_cast<ssize_t>(candidate->index_offset + offset),
          scratch->data, length);
    };
    IndexUnpacker unpacker;
    if (!unpacker.unpack(read, container_size, checksum_))
      return IndexError_UnpackIndex;
    // Bound immutable directory allocation within the charged metadata reserve.
    size_t estimated = 0;
    for (const auto &item : unpacker.segments()) {
      const auto &meta = item.second;
      if (meta.data_offset() > container_size ||
          meta.data_size() > container_size - meta.data_offset() ||
          meta.padding_size() >
              container_size - meta.data_offset() - meta.data_size())
        return IndexError_InvalidLength;
      estimated += 512 + item.first.size();
      if (estimated > kMetadataReserve) return IndexError_NoMemory;
    }
    candidate->segments = std::move(*unpacker.mutable_segments());
    candidate->file_magic = unpacker.magic();
    candidate->path = path;
    candidate->start();
    state_ = std::move(candidate);
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &error) {
    LOG_ERROR("ChunkReadStorage open failed: %s", error.what());
    return IndexError_Runtime;
  }
}
int ChunkReadStorage::cleanup() { return close(); }
int ChunkReadStorage::close() { state_.reset(); return 0; }
int ChunkReadStorage::flush() { return 0; }
int ChunkReadStorage::append(const std::string &, size_t) {
  return IndexError_NotImplemented;
}
void ChunkReadStorage::refresh(uint64_t) {}
uint64_t ChunkReadStorage::check_point() const { return 0; }
IndexStorage::Segment::Pointer ChunkReadStorage::get(const std::string &id, int) {
  if (!state_) return {};
  auto it = state_->segments.find(id);
  return it == state_->segments.end() ? Segment::Pointer{} :
         std::make_shared<ReadSegment>(state_, it->second);
}
std::map<std::string, IndexStorage::Segment::Pointer>
ChunkReadStorage::get_all() const {
  std::map<std::string, Segment::Pointer> result;
  if (state_)
    for (const auto &item : state_->segments)
      result.emplace(std::string(item.first.data(), item.first.size()),
          std::make_shared<ReadSegment>(state_, item.second));
  return result;
}
bool ChunkReadStorage::has(const std::string &id) const {
  return state_ && state_->segments.find(id) != state_->segments.end();
}
uint32_t ChunkReadStorage::magic() const {
  return state_ ? state_->file_magic : 0;
}
IndexStorage::MemoryBlock::MemoryBlockType
ChunkReadStorage::memory_block_type() const {
  return MemoryBlock::MBT_SHARED_SCRATCH;
}
std::string ChunkReadStorage::file_path() const {
  return state_ ? state_->path : std::string{};
}
ChunkReadStorage::Stats ChunkReadStorage::stats() const {
  Stats result;
  if (!state_) return result;
  auto &s = *state_;
  result.cache_hits = s.hits.load();
  result.cache_misses = s.misses.load();
  result.fills = s.fills.load();
  result.replacements = s.replacements.load();
  result.io_calls = s.io_calls.load();
  result.io_bytes = s.io_bytes.load();
  result.copied_bytes = s.copied.load();
  result.workspace_waits = s.workspace_waits.load();
  result.trims = s.trims.load();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (const auto &slot : s.slots)
    if (slot) result.cache_bytes += slot->bytes;
  for (const auto &view : s.views)
    result.workspace_bytes += view->workspace->bytes;
  result.metadata_bytes = s.metadata_charge;
  return result;
}

INDEX_FACTORY_REGISTER_STORAGE(ChunkReadStorage);
}  // namespace core
}  // namespace zvec

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
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_unpacker.h>
#include "ailego/utility/memory_helper.h"
#include "utility_params.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include <ailego/io/libaio_loader.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <sys/stat.h>
#endif

namespace zvec {
namespace core {
namespace {
constexpr size_t kMetadataReserve = 128UL << 10;
constexpr int kUnloaded = std::numeric_limits<int>::min();
constexpr int kLoading = kUnloaded + 1;
constexpr int kEvicting = std::numeric_limits<int>::min() / 2;
using Pool = ailego::MemoryLimitPool;

size_t AlignPage(size_t size) {
  const size_t page = ailego::MemoryHelper::PageSize();
  return (size + page - 1) / page * page;
}
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
  explicit MetadataCharge(size_t count, bool optional = false) : bytes(count) {
    auto &pool = Pool::get_instance();
    if (!(optional ? pool.try_charge_metadata_without_reclaim(bytes)
                   : pool.try_charge_metadata(bytes)))
      throw std::bad_alloc();
  }
  ~MetadataCharge() {
    Pool::get_instance().release_metadata(bytes);
  }
  size_t bytes;
};
struct WorkspaceBuffer {
  explicit WorkspaceBuffer(size_t count) : bytes(AlignPage(count)) {
    Pool::PendingAllocation pending(Pool::get_instance(),
                                    Pool::PendingAllocation::Kind::kWorkspace);
    if (!pending.reserve(bytes)) throw std::bad_alloc();
    data = static_cast<uint8_t *>(ailego::File::MemoryMap(bytes, 0));
    if (!data) throw std::bad_alloc();
    stamp = pending.commit_mapping();
  }
  ~WorkspaceBuffer() {
    if (data && ailego::File::TryMemoryUnmap(data, bytes))
      Pool::get_instance().release_unmapped_external(bytes, stamp);
  }
  uint8_t *data{nullptr};
  size_t bytes;
  ailego::ProcessRssBudget::MaterializationStamp stamp{};
};
}  // namespace

struct ChunkReadStorage::State : ailego::EvictableBlockOwner,
                                 std::enable_shared_from_this<State> {
  // Negative ref values exclusively own loading/eviction transitions. Pinning
  // precedes pointer access; only a successful 0 -> kEvicting may free backing.
  struct Entry {
    std::atomic<uint8_t *> data{nullptr};
    std::atomic<int> refs{kUnloaded};
    std::atomic<bool> queued{false}, referenced{false};
    std::atomic<uint8_t> priority{0};
    std::atomic<uint32_t> requests{0};
    std::atomic<bool> transient{false};
    bool valid{false}, ever_loaded{false}, filter_readmission{false};
    size_t offset{0}, bytes{0}, allocation_bytes{0};
    size_t reuse_next{0};  // immutable capacity-class ring after layout setup
    Pool::CacheBuffer *buffer{nullptr};
    std::atomic<size_t> capacity{0};
  };
  static constexpr size_t kEntries = 256;
  static constexpr size_t kReuseClasses =
      std::numeric_limits<size_t>::digits * 4;
  static constexpr size_t kNoEntry = std::numeric_limits<size_t>::max();
  struct TableSegment {
    MetadataCharge charge{sizeof(TableSegment), true};
    std::array<Entry, kEntries> entries;
  };
  struct Layout {
    explicit Layout(size_t count)
        : charge(sizeof(Layout) + count * sizeof(Entry)),
          entries(new Entry[count]),
          size(count) {
      reuse_clock.fill(kNoEntry);
    }
    MetadataCharge charge;
    std::unique_ptr<Entry[]> entries;
    size_t size;
    size_t clock{0};  // advanced only under the storage's read-ahead mutex
    std::array<size_t, kReuseClasses> reuse_clock;
    std::array<size_t, kReuseClasses> reuse_count{};
    std::unique_ptr<Layout> next;
    void add_reuse_entry(size_t index, size_t bucket) {
      size_t &cursor = reuse_clock[bucket];
      if (cursor == kNoEntry) {
        cursor = index;
        entries[index].reuse_next = index;
      } else {
        entries[index].reuse_next = entries[cursor].reuse_next;
        entries[cursor].reuse_next = index;
      }
      ++reuse_count[bucket];
    }
    Entry *find(size_t offset) const {
      size_t lo = 0, hi = size;
      while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (entries[mid].offset <= offset)
          lo = mid + 1;
        else
          hi = mid;
      }
      if (!lo) return nullptr;
      auto *item = &entries[lo - 1];
      return offset - item->offset < item->bytes ? item : nullptr;
    }
  };
  struct View {
    std::atomic<bool> busy{false};
    std::atomic<size_t> capacity{0};
    std::unique_ptr<WorkspaceBuffer> workspace;
    ailego::BudgetedVector<size_t> pins;
  };
  struct ReadSlot {
    std::atomic<bool> busy{false};
    Segment::ScatterBlock block;
  };
  struct Lease : Segment::ScatterBlock::Workspace, MemoryBlock::ScratchLease {
    Lease(std::shared_ptr<State> owner_arg, View *view_arg, Entry *single_arg)
        : owner(std::move(owner_arg)), view(view_arg), single(single_arg) {}
    ~Lease() override {
      reset();
    }
    void retain() noexcept override {
      readers.fetch_add(1, std::memory_order_relaxed);
    }
    void release() noexcept override {
      const size_t previous = readers.fetch_sub(1, std::memory_order_acq_rel);
      assert(previous != 0);
      if (previous == 1) reset();
    }
    bool reusable(size_t, size_t) const noexcept override {
      return false;
    }
    void reset() noexcept override {
      if (single) owner->unpin(*single, single_key);
      if (view) {
        for (size_t key : view->pins)
          owner->unpin(*reinterpret_cast<Entry *>(key), key);
        view->pins.clear();
        view->busy.store(false, std::memory_order_release);
        owner->changed.notify_one();
      }
      single = nullptr;
      view = nullptr;
      owner.reset();
    }
    std::shared_ptr<State> owner;
    View *view;
    Entry *single;
    size_t single_key{0};
    std::atomic<size_t> readers{0};
  };

#if defined(__linux__) && !defined(__ANDROID__)
  // No worker threads or unbounded task queue. One storage-wide native AIO
  // context owns a fixed number of descriptors and charged cache buffers.
  // A slot retains its own pin until a demand reader consumes it, or a later
  // hint retires a completed read outside its window.
  struct ReadAhead {
    struct Slot {
      iocb control{};
      Entry *item{nullptr};
      size_t bytes{0}, required{0};
      bool done{false}, submitted{false}, resident{false};
    };
    ReadAhead(State &state, std::unique_ptr<MetadataCharge> reservation,
              size_t count, size_t limit)
        : owner(state),
          charge(std::move(reservation)),
          depth(count),
          byte_limit(limit) {
      auto &aio = LibAioLoader::Instance();
      if (!aio.load() || aio.setup_budgeted(static_cast<int>(depth), context,
                                            ring_memory) != 0)
        return;
      // setup_budgeted admits the kernel mapping to the RSS budget. Charge
      // its actual size to the managed budget too, before any submission.
      try {
        ring_charge = std::make_unique<MetadataCharge>(ring_memory.bytes);
      } catch (const std::bad_alloc &) {
        aio.destroy_budgeted(context, ring_memory);
        return;
      }
      ready = true;
    }
    ~ReadAhead() {
      while (inflight) reap(true);
      for (auto &slot : slots)
        if (slot.item) {
          if (!slot.done) cancel(slot);
          release(slot);
        }
      if (context &&
          LibAioLoader::Instance().destroy_budgeted(context, ring_memory)) {
        LOG_ERROR("%s", "ChunkReadStorage: failed to destroy idle AIO context");
      }
    }
    void cancel(Slot &slot) {
      assert(!slot.submitted);
      slot.item->valid = false;
      slot.item->transient.store(true, std::memory_order_relaxed);
      slot.item->refs.store(1, std::memory_order_release);
      slot.done = true;
    }
    void flush() {
      if (!ready) return;
      std::array<iocb *, 64> controls{};
      size_t count = 0;
      for (size_t i = 0; i < depth; ++i)
        if (slots[i].item && !slots[i].submitted && !slots[i].done)
          controls[count++] = &slots[i].control;
      if (!count) return;
      int submitted;
      do {
        submitted =
            LibAioLoader::Instance().io_submit(context, count, controls.data());
      } while (submitted == -EINTR);
      if (submitted > 0) {
        for (int i = 0; i < submitted; ++i)
          static_cast<Slot *>(controls[i]->data)->submitted = true;
        inflight += submitted;
        owner.io_calls.fetch_add(submitted, std::memory_order_relaxed);
        owner.prefetch_submitted.fetch_add(submitted,
                                           std::memory_order_relaxed);
        owner.prefetch_submit_batches.fetch_add(1, std::memory_order_relaxed);
      } else if (submitted != -EAGAIN || !inflight) {
        // No kernel ownership for these requests: publish failed reads and
        // let demand retry synchronously. Partial submissions stay queued.
        for (size_t i = 0; i < count; ++i)
          finish(*static_cast<Slot *>(controls[i]->data), -1);
      }
    }
    void reap(bool wait) {
      if (!inflight) return;
      std::array<io_event, 64> events{};
      timespec timeout{};
      int count;
      do {
        count = LibAioLoader::Instance().io_getevents(
            context, wait ? 1 : 0, static_cast<long>(inflight), events.data(),
            wait ? nullptr : &timeout);
      } while (count == -EINTR);
      if (count < 0) {
        // A successful destroy waits for all submitted operations. Never
        // publish or free a destination while the kernel can still write it.
        const int destroyed =
            LibAioLoader::Instance().destroy_budgeted(context, ring_memory);
        if (destroyed) {
          LOG_ERROR(
              "ChunkReadStorage: AIO completion failed (%d), teardown failed "
              "(%d)",
              count, destroyed);
          std::terminate();
        }
        ready = false;
        for (auto &slot : slots)
          if (slot.item && !slot.done) finish(slot, -1);
        return;
      }
      for (int i = 0; i < count; ++i) {
        auto *slot = static_cast<Slot *>(events[i].data);
        finish(*slot, events[i].res2 ? -1 : static_cast<long>(events[i].res));
      }
    }
    void finish(Slot &slot, long got) {
      if (slot.submitted) --inflight;
      Entry &item = *slot.item;
      if (got > 0) {
        owner.io_bytes.fetch_add(got, std::memory_order_relaxed);
        owner.prefetch_read_bytes.fetch_add(got, std::memory_order_relaxed);
      }
      item.valid = got >= 0 && static_cast<size_t>(got) >= slot.required;
      if (!item.valid) {
        item.transient.store(true, std::memory_order_relaxed);
        owner.prefetch_failed.fetch_add(1, std::memory_order_relaxed);
      } else if (!item.transient.load(std::memory_order_relaxed)) {
        if (item.ever_loaded)
          owner.replacements.fetch_add(1, std::memory_order_relaxed);
        item.ever_loaded = true;
        item.referenced.store(false, std::memory_order_relaxed);
        item.queued.store(true, std::memory_order_relaxed);
        owner.fills.fetch_add(1, std::memory_order_relaxed);
      }
      item.refs.store(1, std::memory_order_release);  // the slot's pin
      slot.done = true;
      if (item.valid && !item.transient.load(std::memory_order_relaxed))
        owner.enqueue(item, reinterpret_cast<size_t>(&item));
    }
    void release(Slot &slot) {
      assert(slot.done);
      owner.unpin(*slot.item, reinterpret_cast<size_t>(slot.item));
      bytes -= slot.bytes;
      --occupied;
      slot = {};
      owner.prefetch_pending_bytes.store(bytes, std::memory_order_relaxed);
      owner.prefetch_pending_requests.store(occupied,
                                            std::memory_order_release);
    }
    Entry *take(Entry *item, bool record_miss = true) {
      std::lock_guard<std::mutex> lock(mutex);
      for (size_t i = 0; i < depth; ++i) {
        auto &slot = slots[i];
        if (slot.item != item) continue;
        const auto start = std::chrono::steady_clock::now();
        while (!slot.done) {
          flush();
          if (!slot.done) reap(true);
        }
        owner.prefetch_wait_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count(),
            std::memory_order_relaxed);
        Entry *result =
            owner.pin(*item, reinterpret_cast<size_t>(item), slot.resident)
                ? item
                : nullptr;
        if (result && !slot.resident) {
          if (record_miss) owner.misses.fetch_add(1, std::memory_order_relaxed);
          owner.prefetch_consumed.fetch_add(1, std::memory_order_relaxed);
        }
        release(slot);
        return result;
      }
      return nullptr;
    }
    void account_slot(size_t requested) {
      ++occupied;
      bytes += requested;
      owner.prefetch_pending_bytes.store(bytes, std::memory_order_relaxed);
      owner.prefetch_pending_requests.store(occupied,
                                            std::memory_order_release);
      owner.prefetch_peak_requests.store(
          std::max<uint64_t>(owner.prefetch_peak_requests.load(), occupied));
      owner.prefetch_peak_bytes.store(
          std::max<uint64_t>(owner.prefetch_peak_bytes.load(), bytes));
    }
    bool hint(size_t offset, size_t length, Layout *layout, bool retire_stale) {
      std::lock_guard<std::mutex> lock(mutex);
      reap(false);
      if (!ready) return false;
      size_t begin = offset;
      for (; begin < offset + length;) {
        Entry *item = layout->find(begin);
        if (!item) break;
        const size_t end =
            std::min(item->offset + item->bytes, offset + length);
        bool present = false;
        for (size_t i = 0; i < depth; ++i) present |= slots[i].item == item;
        const int refs = item->refs.load(std::memory_order_acquire);
        if (present || (refs < 0 && refs != kUnloaded)) {
          begin = end;
          continue;
        }
        Slot *free = nullptr;
        for (size_t i = 0; i < depth; ++i)
          if (!slots[i].item) {
            free = &slots[i];
            break;
          }
        if (!free && retire_stale) {
          // Bounded stale lookahead must not strand the queue after an early
          // search exit. In-flight reads remain owned until they complete.
          for (size_t i = 0; i < depth; ++i) {
            auto &slot = slots[i];
            if ((slot.done || !slot.submitted) &&
                (slot.item->offset < offset ||
                 slot.item->offset >= offset + length)) {
              if (!slot.done) cancel(slot);
              release(slot);
              free = &slot;
              break;
            }
          }
        }
        if (!free) break;
        const size_t physical =
            item->offset / owner.io_alignment * owner.io_alignment;
        const size_t required = item->offset + item->bytes - physical;
        const size_t requested = AlignPage(required);
        const size_t capacity = item->allocation_bytes
                                    ? item->allocation_bytes
                                    : owner.cache_capacity(requested);
        if (capacity > byte_limit - bytes) break;
        // Keep cache hits in the same bounded plan. Loading other selected
        // chains must not evict a chain that this query has already pinned.
        if (owner.pin(*item, reinterpret_cast<size_t>(item), false)) {
          free->item = item;
          free->bytes = capacity;
          free->done = true;
          free->resident = true;
          account_slot(capacity);
          begin = end;
          continue;
        }
        int expected = kUnloaded;
        if (!item->refs.compare_exchange_strong(expected, kLoading,
                                                std::memory_order_acq_rel)) {
          begin = end;
          continue;
        }
        const uint32_t requests =
            item->requests.fetch_add(1, std::memory_order_relaxed);
        const bool transient =
            item->filter_readmission && item->ever_loaded && !requests;
        auto *buffer = owner.acquire_backing(*layout, capacity);
        if (!buffer) {
          item->refs.store(kUnloaded, std::memory_order_release);
          break;
        }
        item->buffer = buffer;
        item->capacity.store(capacity, std::memory_order_relaxed);
        item->data.store(buffer->data + item->offset - physical,
                         std::memory_order_relaxed);
        item->transient.store(transient, std::memory_order_relaxed);
        free->item = item;
        free->bytes = capacity;
        free->required = required;
        io_prep_pread(&free->control, owner.data_file.native_handle(),
                      buffer->data, requested, physical);
        free->control.data = free;
        account_slot(capacity);
        begin = end;
      }
      // Low-priority hints collect the selected lists. The first demand or
      // normal-priority hint submits the whole prepared batch in one syscall.
      if (retire_stale) flush();
      return begin == offset + length;
    }
    State &owner;
    std::unique_ptr<MetadataCharge> charge;
    std::unique_ptr<MetadataCharge> ring_charge;
    size_t depth, byte_limit, occupied{0}, inflight{0}, bytes{0};
    io_context_t context{nullptr};
    LibAioLoader::RingMemory ring_memory{};
    bool ready{false};
    std::array<Slot, 64> slots{};
    std::mutex mutex;
  };
#endif

  ~State() override {
#if defined(__linux__) && !defined(__ANDROID__)
    read_ahead.reset();  // join all kernel writes before freeing buffers/FDs
#endif
    if (registered)
      ailego::BlockEvictionQueue::get_instance().set_invalid(this);
    for (auto *layout = layouts.get(); layout; layout = layout->next.get())
      for (size_t i = 0; i < layout->size; ++i)
        release_backing(layout->entries[i]);
    for (size_t i = 0; table && i < table_count; ++i) {
      auto *segment = table[i].load(std::memory_order_relaxed);
      if (!segment) continue;
      for (auto &item : segment->entries) {
        assert(item.refs.load(std::memory_order_relaxed) <= 0);
        release_backing(item);
      }
      delete segment;
    }
  }

  bool initialize(const std::string &file_path, size_t chunk, size_t cache,
                  size_t max_read, size_t count, size_t wait, bool direct,
                  bool adapt, size_t prefetch_depth, size_t prefetch_bytes) {
    if (!meta_file.open(file_path, true, false)) return false;
    if (!data_file.open(file_path, true, direct)) {
      if (!direct || !data_file.open(file_path, true, false)) return false;
      direct = false;
    }
    // F_NOCACHE accepts unaligned requests on macOS, but page-aligned I/O
    // also avoids its slower partial-page path. Logical chunk bounds remain
    // independent of this physical prefix/suffix padding.
    io_alignment = direct ? ailego::MemoryHelper::PageSize() : 1;
#if !defined(_WIN32)
    struct stat a {
    }, b{};
    if (::fstat(meta_file.native_handle(), &a) ||
        ::fstat(data_file.native_handle(), &b) || a.st_dev != b.st_dev ||
        a.st_ino != b.st_ino)
      return false;
#else
    BY_HANDLE_FILE_INFORMATION a{}, b{};
    if (!GetFileInformationByHandle(meta_file.native_handle(), &a) ||
        !GetFileInformationByHandle(data_file.native_handle(), &b) ||
        a.dwVolumeSerialNumber != b.dwVolumeSerialNumber ||
        a.nFileIndexHigh != b.nFileIndexHigh ||
        a.nFileIndexLow != b.nFileIndexLow)
      return false;
#endif
    file_size = meta_file.size();
    chunk_bytes = chunk;
    cache_limit = cache;
    adaptive = adapt;
    wait_ms = wait;
    const size_t page = ailego::MemoryHelper::PageSize();
    const size_t execution_read = std::max(page, std::min(max_read, file_size));
    if (adaptive) {
      auto &pool = Pool::get_instance();
      const size_t external = pool.external_used();
      const size_t fixed = pool.metadata_used() + external -
                           std::min(external, pool.evictable_cache_used());
      const size_t budget = pool.capacity() - std::min(pool.capacity(), fixed);
      count = std::min(count, std::max<size_t>(4, budget / (8 * 3 * page)));
      max_read =
          std::max(page, std::min({max_read, file_size, size_t{64UL << 10},
                                   budget / (count * 16)}));
    }
    max_read_bytes = AlignPage(max_read);
    chunks = file_size / chunk + (file_size % chunk != 0);
    table_count = chunks / kEntries + (chunks % kEntries != 0);
    if (table_count > (std::numeric_limits<size_t>::max() - kMetadataReserve) /
                          sizeof(std::atomic<TableSegment *>))
      throw std::bad_alloc();
    metadata_charge = kMetadataReserve +
                      table_count * sizeof(std::atomic<TableSegment *>) +
                      count * (sizeof(View) + sizeof(ReadSlot) +
                               sizeof(std::unique_ptr<ReadSlot>));
    metadata_owner = std::make_unique<MetadataCharge>(metadata_charge);
    table = std::make_unique<std::atomic<TableSegment *>[]>(table_count);
    for (size_t i = 0; i < table_count; ++i)
      table[i].store(nullptr, std::memory_order_relaxed);
    views.reserve(count);
    read_slots.reserve(count);
    // Execution backing is independent of cache chunk size. Two I/O pages,
    // rather than two chunks, cover the aligned edges of any missing range.
    for (size_t i = 0; i < count; ++i) {
      auto view = std::make_unique<View>();
      view->pins.reserve(max_read_bytes / chunk + 3);
      if (!adaptive || i < 4) {
        view->workspace = std::make_unique<WorkspaceBuffer>(
            (adaptive && i == 0 ? execution_read : max_read_bytes) + 2 * page);
        view->capacity.store(view->workspace->bytes);
        workspace_bytes.fetch_add(view->workspace->bytes);
      }
      views.emplace_back(std::move(view));
      read_slots.emplace_back(std::make_unique<ReadSlot>());
    }
#if defined(__linux__) && !defined(__ANDROID__)
    if (prefetch_depth && prefetch_bytes && direct) {
      try {
        auto charge = std::make_unique<MetadataCharge>(sizeof(ReadAhead));
        auto queue = std::make_unique<ReadAhead>(
            *this, std::move(charge), prefetch_depth, prefetch_bytes);
        if (queue->ready) read_ahead = std::move(queue);
      } catch (const std::bad_alloc &) {
      }  // optional lookahead; demand still works
    }
#else
    (void)prefetch_depth;
    (void)prefetch_bytes;
#endif
    return true;
  }
  void start() {
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    version = queue.next_version();
    queue.set_valid(this);
    registered = true;
  }
  Entry *entry(size_t key, bool create) {
    if (key >= chunks) return nullptr;
    auto &slot = table[key / kEntries];
    auto *segment = slot.load(std::memory_order_acquire);
    if (!segment && create) {
      std::lock_guard<std::mutex> lock(table_mutex);
      segment = slot.load(std::memory_order_relaxed);
      if (!segment) {
        try {
          segment = new TableSegment();
        } catch (const std::bad_alloc &) {
          return nullptr;
        }
        for (size_t i = 0; i < kEntries; ++i) {
          auto &item = segment->entries[i];
          item.offset = (key / kEntries * kEntries + i) * chunk_bytes;
          item.bytes = item.offset < file_size
                           ? std::min(chunk_bytes, file_size - item.offset)
                           : 0;
        }
        table_metadata.fetch_add(sizeof(TableSegment),
                                 std::memory_order_relaxed);
        slot.store(segment, std::memory_order_release);
      }
    }
    return segment ? &segment->entries[key % kEntries] : nullptr;
  }
  void enqueue(Entry &item, size_t key) {
    ailego::BlockEvictionQueue::BlockType block;
    block.owner = this;
    block.owner_key = key;
    block.version = version;
    if (!ailego::BlockEvictionQueue::get_instance().add_single_block(
            block, item.priority.load(std::memory_order_relaxed)))
      eviction_requeue_failed(key, version);
  }
  void unpin(Entry &item, size_t key) {
    const int previous = item.refs.fetch_sub(1, std::memory_order_release);
    assert(previous > 0);
    if (previous == 1 && item.transient.load(std::memory_order_acquire)) {
      int expected = 0;
      if (item.refs.compare_exchange_strong(expected, kEvicting,
                                            std::memory_order_acq_rel)) {
        release_backing(item, false);
        item.transient.store(false, std::memory_order_relaxed);
        item.refs.store(kUnloaded, std::memory_order_release);
      }
      return;
    }
    if (previous == 1 && !item.queued.load(std::memory_order_relaxed)) {
      bool expected = false;
      if (item.queued.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel))
        enqueue(item, key);
    }
  }
  uint8_t *pin(Entry &item, size_t key, bool record_hit = true) {
    int refs = item.refs.load(std::memory_order_acquire);
    while (refs >= 0 && refs < std::numeric_limits<int>::max()) {
      if (item.refs.compare_exchange_weak(refs, refs + 1,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
        if (!item.valid) {
          unpin(item, key);
          return nullptr;
        }
        const auto sample =
            record_hit ? hits.fetch_add(1, std::memory_order_relaxed) : 1;
        if ((sample & 7) == 0) {
          item.referenced.store(true, std::memory_order_relaxed);
          item.priority.store(ailego::BlockEvictionQueue::kProtectedPriority,
                              std::memory_order_relaxed);
        }
        return item.data.load(std::memory_order_acquire);
      }
    }
    return nullptr;
  }
  bool release_backing(Entry &item, bool reset_requests = true) {
    if (!item.buffer) return true;
    const size_t bytes = item.capacity.load(std::memory_order_relaxed);
    Pool::get_instance().release_cache_buffer(item.buffer);
    item.buffer = nullptr;
    item.data.store(nullptr, std::memory_order_relaxed);
    item.capacity.store(0, std::memory_order_relaxed);
    if (reset_requests) item.requests.store(0, std::memory_order_relaxed);
    cache_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    return true;
  }
  bool claim_capacity(size_t count) {
    size_t bytes = cache_bytes.load(std::memory_order_relaxed);
    while (bytes <= cache_limit && count <= cache_limit - bytes) {
      if (cache_bytes.compare_exchange_weak(bytes, bytes + count,
                                            std::memory_order_relaxed))
        return true;
    }
    return false;
  }
  size_t cache_capacity(size_t requested) const {
    // Four classes per doubling bound padding to 25%, independently of the
    // index's longest list. Buffers and direct I/O remain whole-list sized.
    const size_t smallest_class = 4 * ailego::MemoryHelper::PageSize();
    if (!adaptive || requested < smallest_class) return AlignPage(requested);
    size_t base = smallest_class;
    while (base <= requested / 2) base *= 2;
    const size_t quantum = base / 4;
    if (requested > std::numeric_limits<size_t>::max() - (quantum - 1))
      return requested;  // callers already include physical page alignment
    return (requested + quantum - 1) / quantum * quantum;
  }
  static size_t reuse_class(size_t capacity) {
    const size_t page = ailego::MemoryHelper::PageSize();
    if (capacity < 4 * page) return capacity / page;
    size_t base = 4 * page, bucket = 4;
    while (base <= capacity / 2) {
      base *= 2;
      bucket += 4;
    }
    return bucket + (capacity - base) / (base / 4);
  }
  Pool::CacheBuffer *reuse_backing(Layout &layout, size_t required) {
    // Called under the read-ahead mutex. Transfer an unpinned, cold buffer
    // without passing through the idle pool: its mapping, descriptor and
    // global/local capacity charges remain live throughout the handoff.
    if (!Pool::get_instance().can_reuse_cache_backing()) return nullptr;
    const size_t bucket = reuse_class(required);
    size_t &cursor = layout.reuse_clock[bucket];
    const size_t attempts =
        std::min<size_t>(1024, layout.reuse_count[bucket] * 3);
    for (size_t i = 0; i < attempts; ++i) {
      Entry &item = layout.entries[cursor];
      cursor = item.reuse_next;
      if (item.capacity.load(std::memory_order_relaxed) != required) continue;
      int expected = 0;
      if (!item.refs.compare_exchange_strong(expected, kEvicting,
                                             std::memory_order_acq_rel))
        continue;
      if (item.capacity.load(std::memory_order_relaxed) != required ||
          !item.buffer ||
          item.referenced.exchange(false, std::memory_order_relaxed) ||
          item.priority.exchange(0, std::memory_order_relaxed) != 0) {
        item.refs.store(0, std::memory_order_release);
        continue;
      }
      auto *buffer = item.buffer;
      item.buffer = nullptr;
      item.data.store(nullptr, std::memory_order_relaxed);
      item.capacity.store(0, std::memory_order_relaxed);
      item.requests.store(0, std::memory_order_relaxed);
      item.queued.store(false, std::memory_order_relaxed);
      item.refs.store(kUnloaded, std::memory_order_release);
      trims.fetch_add(1, std::memory_order_relaxed);
      backing_reuses.fetch_add(1, std::memory_order_relaxed);
      backing_reuse_scans.fetch_add(i + 1, std::memory_order_relaxed);
      return buffer;
    }
    backing_reuse_scans.fetch_add(attempts, std::memory_order_relaxed);
    return nullptr;
  }
  Pool::CacheBuffer *acquire_backing(Layout &layout, size_t capacity) {
    auto &pool = Pool::get_instance();
    const bool claimed = claim_capacity(capacity);
    if (claimed) {
      if (auto *buffer = pool.try_acquire_cache_buffer(capacity)) return buffer;
    }
    // A transfer already owns one capacity charge; do not debit it twice.
    if (auto *buffer = reuse_backing(layout, capacity)) {
      if (claimed) cache_bytes.fetch_sub(capacity, std::memory_order_relaxed);
      return buffer;
    }
    if (!claimed) return nullptr;
    reclaim_layout(layout, capacity);
    auto *buffer = pool.try_acquire_cache_buffer(capacity);
    for (size_t attempt = 0; !buffer && attempt < 4; ++attempt) {
      pool.request_cache_reclaim(capacity);
      ailego::BlockEvictionQueue::get_instance().reclaim_for_allocation(
          capacity);
      buffer = pool.try_acquire_cache_buffer(capacity);
    }
    if (!buffer) cache_bytes.fetch_sub(capacity, std::memory_order_relaxed);
    return buffer;
  }
  void reclaim_layout(Layout &layout, size_t required) {
    // A whole-chain allocation should replace cold posting payload before it
    // drives the global CLOCK through thousands of small, hot key/meta pages.
    // Use the normal pin/CLOCK transition, with bounded storage-local work.
    size_t released = 0;
    const size_t attempts = std::min<size_t>(1024, layout.size * 3);
    for (size_t i = 0; i < attempts && released < required; ++i) {
      Entry &item = layout.entries[layout.clock++ % layout.size];
      if (item.refs.load(std::memory_order_acquire) != 0) continue;
      const size_t bytes = item.capacity.load(std::memory_order_relaxed);
      if (evict_block(reinterpret_cast<size_t>(&item))) released += bytes;
    }
  }
  bool read_data(size_t offset, size_t requested, uint8_t *output,
                 size_t required) {
    if (offset >= file_size || required > file_size - offset) return false;
    const size_t got =
        data_file.read(static_cast<ssize_t>(offset), output, requested);
    io_calls.fetch_add(1, std::memory_order_relaxed);
    io_bytes.fetch_add(got, std::memory_order_relaxed);
    return got >= required;
  }
  Entry *cached(Entry *item) {
    const size_t key = reinterpret_cast<size_t>(item);
    if (!item) {
      misses.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
#if defined(__linux__) && !defined(__ANDROID__)
    if (read_ahead && prefetch_pending_requests.load(std::memory_order_acquire))
      if (auto *prefetched = read_ahead->take(item)) return prefetched;
#endif
    if (pin(*item, key)) return item;
    misses.fetch_add(1, std::memory_order_relaxed);
    const uint32_t requests =
        item->requests.fetch_add(1, std::memory_order_relaxed);
    int expected = kUnloaded;
    if (!item->refs.compare_exchange_strong(expected, kLoading,
                                            std::memory_order_acq_rel)) {
#if defined(__linux__) && !defined(__ANDROID__)
      // A hint can claim the entry between take() and this CAS. Join it
      // rather than issuing a duplicate synchronous read into scratch.
      if (expected == kLoading && read_ahead)
        if (auto *prefetched = read_ahead->take(item, false)) return prefetched;
#endif
      return nullptr;  // Another loader/evictor owns this residency cycle.
    }
    // A just-evicted posting batch must demonstrate reuse before spending
    // cache backing again. One-pass reads still execute in reserved scratch.
    // Metadata and the first residency are admitted immediately.
    if (item->filter_readmission && item->ever_loaded && requests == 0) {
      item->refs.store(kUnloaded, std::memory_order_release);
      return nullptr;
    }
    const size_t physical = item->offset / io_alignment * io_alignment;
    const size_t requested =
        (item->offset + item->bytes - physical + io_alignment - 1) /
        io_alignment * io_alignment;
    const size_t capacity = item->allocation_bytes ? item->allocation_bytes
                                                   : cache_capacity(requested);
    if (!claim_capacity(capacity)) {
      item->refs.store(kUnloaded, std::memory_order_release);
      return nullptr;
    }
    auto *buffer = Pool::get_instance().try_acquire_cache_buffer(capacity);
    if (!buffer) {
      cache_bytes.fetch_sub(capacity, std::memory_order_relaxed);
      item->refs.store(kUnloaded, std::memory_order_release);
      if (requests != 0) Pool::get_instance().request_cache_reclaim(capacity);
      return nullptr;
    }
    const bool valid = read_data(physical, requested, buffer->data,
                                 item->offset + item->bytes - physical);
    item->buffer = buffer;
    item->capacity.store(capacity, std::memory_order_relaxed);
    item->data.store(buffer->data + item->offset - physical,
                     std::memory_order_relaxed);
    item->valid = valid;
    item->transient.store(false, std::memory_order_relaxed);
    if (!valid && release_backing(*item)) {
      item->refs.store(kUnloaded, std::memory_order_release);
      return nullptr;
    }
    if (item->ever_loaded) replacements.fetch_add(1, std::memory_order_relaxed);
    item->ever_loaded = true;
    item->referenced.store(false, std::memory_order_relaxed);
    item->queued.store(true, std::memory_order_relaxed);
    item->refs.store(1, std::memory_order_release);
    enqueue(*item, key);
    if (!valid) {
      unpin(*item, key);
      return nullptr;
    }
    fills.fetch_add(1, std::memory_order_relaxed);
    return item;
  }
  View *acquire_view(size_t bytes) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
    for (;;) {
      View *fit = nullptr, *fallback = nullptr;
      size_t fit_size = std::numeric_limits<size_t>::max(), fallback_size = 0;
      for (auto &view : views) {
        if (view->busy.load(std::memory_order_relaxed)) continue;
        const size_t size = view->capacity.load(std::memory_order_relaxed);
        if (size >= bytes && size < fit_size) {
          fit = view.get();
          fit_size = size;
        }
        if (!fallback || size > fallback_size) {
          fallback = view.get();
          fallback_size = size;
        }
      }
      if (auto *candidate = fit ? fit : fallback) {
        bool expected = false;
        if (candidate->busy.compare_exchange_strong(expected, true,
                                                    std::memory_order_acquire))
          return candidate;
        continue;
      }
      workspace_waits.fetch_add(1, std::memory_order_relaxed);
      std::unique_lock<std::mutex> lock(wait_mutex);
      if (std::chrono::steady_clock::now() >= deadline) return nullptr;
      changed.wait_for(lock, std::chrono::milliseconds(1));
    }
  }
  bool ensure_workspace(View &view, size_t bytes) {
    if (view.workspace && view.workspace->bytes >= bytes) return true;
    if (!adaptive) return false;
    if (view.workspace) {
      view.capacity.store(0, std::memory_order_relaxed);
      workspace_bytes.fetch_sub(view.workspace->bytes,
                                std::memory_order_relaxed);
      view.workspace.reset();
    }
    try {
      view.workspace = std::make_unique<WorkspaceBuffer>(bytes);
      view.capacity.store(view.workspace->bytes, std::memory_order_relaxed);
      workspace_bytes.fetch_add(view.workspace->bytes,
                                std::memory_order_relaxed);
      return true;
    } catch (const std::bad_alloc &) {
      return false;
    }
  }
  std::shared_ptr<Lease> lease(Segment::ScatterBlock &out, View *view,
                               Entry *single = nullptr, size_t key = 0) {
    try {
      auto result = std::dynamic_pointer_cast<Lease>(out.workspace);
      if (result) {
        result->owner = shared_from_this();
        result->view = view;
        result->single = single;
      } else {
        result = std::allocate_shared<Lease>(ailego::BudgetedAllocator<Lease>{},
                                             shared_from_this(), view, single);
        out.workspace = result;
      }
      result->single_key = key;
      return result;
    } catch (...) {
      if (view) view->busy.store(false, std::memory_order_release);
      if (single) unpin(*single, key);
      throw;
    }
  }
  size_t scatter(size_t offset, size_t len, Segment::ScatterBlock &out,
                 Layout *layout = nullptr) {
    out.reset();
    if (!len || offset > file_size || len > file_size - offset ||
        len > (adaptive ? (1UL << 30) : max_read_bytes))
      return 0;
    const auto locate = [&](size_t position) -> Entry * {
      return layout ? layout->find(position)
                    : entry(position / chunk_bytes, cache_limit != 0);
    };
    auto *first = locate(offset);
    const bool single = first && len <= first->bytes - (offset - first->offset);
    if (single) {
      if (auto *item = cached(first)) {
        out.lease = lease(out, nullptr, item, reinterpret_cast<size_t>(item));
        out.spans.push_back(
            {item->data.load(std::memory_order_relaxed) + offset - item->offset,
             len});
        out.size = len;
        return len;
      }
    }
    const size_t page = ailego::MemoryHelper::PageSize();
    auto *view = acquire_view(len + 2 * page);
    if (!view) return 0;
    auto owner = lease(out, view);
    out.lease = owner;
    out.spans.reserve(single ? 1 : len / chunk_bytes + 2);
    const size_t physical_base = offset / io_alignment * io_alignment;
    size_t run_first = 0, run_end = 0;
    const auto finish_run = [&]() {
      if (run_first == run_end) return true;
      if (!ensure_workspace(*view, len + 2 * page)) return false;
      const size_t begin = run_first, end = run_end;
      const size_t physical = begin / io_alignment * io_alignment;
      const size_t requested =
          (end - physical + io_alignment - 1) / io_alignment * io_alignment;
      auto *destination = view->workspace->data + physical - physical_base;
      if (!read_data(physical, requested, destination, end - physical))
        return false;
      out.spans.push_back({destination + begin - physical, end - begin});
      run_first = run_end = 0;
      return true;
    };
    for (size_t begin = offset; begin < offset + len;) {
      Entry *source = locate(begin);
      const size_t end =
          source
              ? std::min(source->offset + source->bytes, offset + len)
              : std::min((begin / chunk_bytes + 1) * chunk_bytes, offset + len);
      Entry *item = single ? nullptr : cached(source);
      if (item) {
        // Record the pin before any descriptor growth or I/O can throw.
        const size_t key = reinterpret_cast<size_t>(item);
        try {
          view->pins.push_back(key);
        } catch (...) {
          unpin(*item, key);
          throw;
        }
        if (!finish_run()) {
          out.reset();
          return 0;
        }
        out.spans.push_back(
            {item->data.load(std::memory_order_relaxed) + begin - item->offset,
             end - begin});
      } else {
        if (run_first == run_end) run_first = begin;
        run_end = end;
      }
      begin = end;
    }
    if (!finish_run()) {
      out.reset();
      return 0;
    }
    out.size = len;
    return len;
  }

  bool is_dead_block(ailego::eviction_key_t key,
                     ailego::version_t generation) override {
    if (generation != version) return true;
    auto *item = reinterpret_cast<Entry *>(key);
    return !item || !item->queued.load(std::memory_order_relaxed);
  }
  uint8_t eviction_priority(ailego::eviction_key_t key) const override {
    auto *item = reinterpret_cast<Entry *>(key);
    return item ? item->priority.load(std::memory_order_relaxed) : 0;
  }
  size_t eviction_bytes(ailego::eviction_key_t key) const override {
    auto *item = reinterpret_cast<Entry *>(key);
    return item ? item->capacity.load(std::memory_order_relaxed) : 0;
  }
  void eviction_requeue_failed(ailego::eviction_key_t key,
                               ailego::version_t generation) override {
    if (generation != version) return;
    if (auto *item = reinterpret_cast<Entry *>(key)) {
      item->queued.store(false, std::memory_order_relaxed);
      recovery_needed.store(true, std::memory_order_release);
      // A failed queue allocation must not strand an idle mapping. Pinned
      // chunks retry registration on final unpin; idle chunks can retire now.
      int expected = 0;
      if (item->refs.compare_exchange_strong(expected, kEvicting,
                                             std::memory_order_acq_rel)) {
        if (release_backing(*item)) {
          item->queued.store(false, std::memory_order_relaxed);
          item->refs.store(kUnloaded, std::memory_order_release);
          trims.fetch_add(1, std::memory_order_relaxed);
        } else {
          item->refs.store(0, std::memory_order_release);
        }
      }
    }
  }
  size_t recover_eviction_queue() override {
    if (!recovery_needed.exchange(false, std::memory_order_acq_rel)) return 0;
    size_t recovered = 0;
    const auto recover = [&](Entry &item) {
      if (item.refs.load(std::memory_order_acquire) != 0) return;
      bool expected = false;
      if (item.queued.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel)) {
        enqueue(item, reinterpret_cast<size_t>(&item));
        ++recovered;
      }
    };
    for (size_t segment_id = 0; segment_id < table_count; ++segment_id) {
      auto *segment = table[segment_id].load(std::memory_order_acquire);
      if (!segment) continue;
      for (size_t i = 0; i < kEntries; ++i) {
        recover(segment->entries[i]);
      }
    }
    std::lock_guard<std::mutex> lock(table_mutex);
    for (auto *layout = layouts.get(); layout; layout = layout->next.get())
      for (size_t i = 0; i < layout->size; ++i) recover(layout->entries[i]);
    return recovered;
  }
  bool evict_block(ailego::eviction_key_t key) override {
    auto *item = reinterpret_cast<Entry *>(key);
    if (!item) return false;
    int expected = 0;
    if (!item->refs.compare_exchange_strong(expected, kEvicting,
                                            std::memory_order_acq_rel)) {
      if (expected > 0) enqueue(*item, key);
      return false;
    }
    if (item->referenced.exchange(false, std::memory_order_relaxed)) {
      item->refs.store(0, std::memory_order_release);
      enqueue(*item, key);
      return false;
    }
    if (item->priority.exchange(0, std::memory_order_relaxed) != 0) {
      item->refs.store(0, std::memory_order_release);
      enqueue(*item, key);
      return false;
    }
    if (!release_backing(*item)) {
      item->refs.store(0, std::memory_order_release);
      enqueue(*item, key);
      return false;
    }
    item->queued.store(false, std::memory_order_relaxed);
    item->refs.store(kUnloaded, std::memory_order_release);
    trims.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  ailego::File data_file, meta_file;
  size_t file_size{0}, chunk_bytes{0}, max_read_bytes{0}, wait_ms{0};
  size_t io_alignment{1};
  size_t index_offset{0}, metadata_charge{0}, cache_limit{0};
  size_t chunks{0}, table_count{0};
  uint32_t file_magic{0};
  std::string path;
  IndexUnpacker::SegmentMap segments;
  std::unique_ptr<MetadataCharge> metadata_owner;
  std::unique_ptr<std::atomic<TableSegment *>[]> table;
  std::unique_ptr<Layout> layouts;
  std::vector<std::unique_ptr<View>> views;
  // Bounded, precharged descriptors. The slot is held only during read();
  // returned MemoryBlocks independently retain reader leases and page pins.
  std::vector<std::unique_ptr<ReadSlot>> read_slots;
  std::mutex table_mutex, wait_mutex;
  std::condition_variable changed;
  bool registered{false}, adaptive{false};
  ailego::version_t version{0};
  std::atomic<bool> recovery_needed{false};
  std::atomic<size_t> cache_bytes{0}, workspace_bytes{0}, table_metadata{0};
  std::atomic<uint64_t> hits{0}, misses{0}, fills{0}, replacements{0};
  std::atomic<uint64_t> io_calls{0}, io_bytes{0}, copied{0}, workspace_waits{0};
  std::atomic<uint64_t> trims{0};
  std::atomic<bool> prefetch_enabled{true};
  std::atomic<uint64_t> prefetch_submitted{0}, prefetch_consumed{0},
      prefetch_failed{0};
  std::atomic<uint64_t> prefetch_wait_ns{0}, prefetch_peak_requests{0},
      prefetch_peak_bytes{0};
  std::atomic<uint64_t> backing_reuses{0}, backing_reuse_scans{0};
  std::atomic<size_t> prefetch_pending_requests{0}, prefetch_pending_bytes{0};
  std::atomic<uint64_t> prefetch_read_bytes{0}, prefetch_submit_batches{0};
#if defined(__linux__) && !defined(__ANDROID__)
  std::unique_ptr<ReadAhead> read_ahead;
#endif
};

class ChunkReadStorage::ReadSegment : public IndexStorage::Segment {
 public:
  ReadSegment(std::shared_ptr<State> state,
              const IndexUnpacker::SegmentMeta &meta)
      : state_(std::move(state)),
        offset_(state_->index_offset + meta.data_offset()),
        size_(meta.data_size()),
        padding_(meta.padding_size()),
        crc_(meta.data_crc()) {}
  size_t data_size() const override {
    return size_;
  }
  size_t data_offset() const override {
    return offset_;
  }
  uint32_t data_crc() const override {
    return crc_;
  }
  size_t padding_size() const override {
    return padding_;
  }
  size_t capacity() const override {
    return size_ + padding_;
  }
  size_t read_scatter(size_t offset, ScatterBlock &out, size_t len) override {
    if (offset > capacity()) {
      out.reset();
      return 0;
    }
    return state_->scatter(offset_ + offset, std::min(len, capacity() - offset),
                           out, layout_);
  }
  bool supports_read_chunks() const override {
    return true;
  }
  void prefetch(size_t offset, size_t len, CachePriority priority) override {
      (void)try_prefetch(offset, len, priority);
    }
    bool try_prefetch(size_t offset, size_t len, CachePriority priority) override {
      if (offset > size_ || len > size_ - offset) return false;
#if defined(__linux__) && !defined(__ANDROID__)
    if (layout_ && state_->read_ahead && state_->prefetch_enabled.load() &&
        offset <= size_ && len <= size_ - offset)
        return state_->read_ahead->hint(offset_ + offset, len, layout_,
                                       priority != CachePriority::kLow);
#else
    (void)offset;
    (void)len;
    (void)priority;
#endif
      return true;
  }
  int set_read_chunks(const ReadChunks &chunks) override try {
    if (layout_) return IndexError_InvalidArgument;
    if (chunks.size() >
        (std::numeric_limits<size_t>::max() - sizeof(State::Layout)) /
            sizeof(State::Entry))
      return IndexError_NoMemory;
    size_t end = 0;
    for (const auto &chunk : chunks) {
      if (!chunk.size || chunk.offset < end || chunk.offset > size_ ||
          chunk.size > size_ - chunk.offset)
        return IndexError_InvalidArgument;
      end = chunk.offset + chunk.size;
    }
    auto layout = std::make_unique<State::Layout>(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
      layout->entries[i].offset = offset_ + chunks[i].offset;
      layout->entries[i].bytes = chunks[i].size;
      layout->entries[i].filter_readmission = true;
      const size_t prefix = layout->entries[i].offset % state_->io_alignment;
      const size_t allocation =
          state_->cache_capacity(AlignPage(prefix + chunks[i].size));
      layout->entries[i].allocation_bytes = allocation;
      layout->add_reuse_entry(i, State::reuse_class(allocation));
      // Registered adaptive chunks reuse charged backing at every size. A
      // first-revisit transient returns its mapping to the generic idle pool,
      // where pressure may unmap it before another same-sized read arrives.
      // Metadata pages retain their ordinary admission policy.
      if (state_->adaptive)
        layout->entries[i].filter_readmission = false;
    }
    std::lock_guard<std::mutex> lock(state_->table_mutex);
    const size_t charged = layout->charge.bytes;
    auto *published = layout.get();
    layout->next = std::move(state_->layouts);
    state_->layouts = std::move(layout);
    state_->table_metadata.fetch_add(charged);
    layout_ = published;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
  size_t read(size_t offset, MemoryBlock &out, size_t len) override {
    out.reset();
    State::ReadSlot *slot = nullptr;
    for (auto &candidate : state_->read_slots) {
      bool expected = false;
      if (candidate->busy.compare_exchange_strong(expected, true,
                                                  std::memory_order_acquire)) {
        // Nested metadata/key reads commonly retain an earlier result. Keep
        // its descriptor cached and try another idle slot before allocating.
        if (candidate->block.workspace &&
            candidate->block.workspace.use_count() != 1) {
          candidate->busy.store(false, std::memory_order_release);
          continue;
        }
        slot = candidate.get();
        break;
      }
    }
    ScatterBlock local;
    auto &block = slot ? slot->block : local;
    auto release_slot = ailego::ScopeGuard::Make([&] {
      if (!slot) return;
      block.spans.clear();
      block.lease.reset();
      block.fallback.reset();
      block.size = 0;
      // Descriptor ownership is not a reader. The last MemoryBlock copy
      // releases its pins immediately, even when this slot caches the Lease.
      if (block.workspace && block.workspace.use_count() == 1)
        block.workspace->reset();
      slot->busy.store(false, std::memory_order_release);
    });
    const size_t got = read_scatter(offset, block, len);
    if (!got) return 0;
    auto owner = std::static_pointer_cast<State::Lease>(block.lease);
    if (block.spans.size() == 1) {
      owner->retain();
      out = MemoryBlock::MakeReusedScratch(
          const_cast<uint8_t *>(block.spans.front().data), owner, false);
      return got;
    }
    const size_t page = ailego::MemoryHelper::PageSize();
    if (!state_->ensure_workspace(*owner->view, got + 2 * page)) return 0;
    auto *destination = owner->view->workspace->data +
                        (offset_ + offset) % state_->io_alignment;
    size_t copied = 0;
    for (const auto &span : block.spans) {
      std::memmove(destination + copied, span.data, span.size);
      copied += span.size;
    }
    state_->copied.fetch_add(got, std::memory_order_relaxed);
    owner->retain();
    out = MemoryBlock::MakeReusedScratch(destination, owner, false);
    return got;
  }
  size_t fetch(size_t offset, void *destination, size_t len) const override {
    if (offset > capacity() || !destination) return 0;
    len = std::min(len, capacity() - offset);
    size_t done = 0;
    while (done < len) {
      ScatterBlock block;
      const size_t count = std::min(len - done, state_->max_read_bytes);
      if (state_->scatter(offset_ + offset + done, count, block, layout_) !=
          count)
        return done;
      size_t cursor = done;
      for (const auto &span : block.spans) {
        std::memcpy(static_cast<uint8_t *>(destination) + cursor, span.data,
                    span.size);
        cursor += span.size;
      }
      state_->copied.fetch_add(count, std::memory_order_relaxed);
      done += count;
    }
    return done;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return capacity();
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<ReadSegment>(*this);
  }

 private:
  std::shared_ptr<State> state_;
  size_t offset_, size_, padding_;
  uint32_t crc_;
  State::Layout *layout_{nullptr};
};

ChunkReadStorage::ChunkReadStorage()
    : chunk_bytes_(ailego::MemoryHelper::PageSize()) {}
ChunkReadStorage::~ChunkReadStorage() = default;
int ChunkReadStorage::init(const ailego::Params &params) {
  params.get(CHUNK_READ_STORAGE_CHUNK_BYTES, &chunk_bytes_);
  params.get(CHUNK_READ_STORAGE_CACHE_BYTES, &cache_bytes_);
  params.get(CHUNK_READ_STORAGE_MAX_READ_BYTES, &max_read_bytes_);
  params.get(CHUNK_READ_STORAGE_WORKSPACE_COUNT, &workspace_count_);
  params.get(CHUNK_READ_STORAGE_WAIT_MS, &wait_ms_);
  params.get(CHUNK_READ_STORAGE_DIRECT_IO, &direct_io_);
  params.get(CHUNK_READ_STORAGE_ADAPTIVE, &adaptive_);
  params.get(CHUNK_READ_STORAGE_PREFETCH_DEPTH, &prefetch_depth_);
  params.get(CHUNK_READ_STORAGE_PREFETCH_BYTES, &prefetch_bytes_);
  params.get(BUFFER_READ_STORAGE_HEADER_OFFSET, &header_offset_);
  params.get(BUFFER_READ_STORAGE_FOOTER_OFFSET, &footer_offset_);
  params.get(BUFFER_READ_STORAGE_CHECKSUM_VALIDATION, &checksum_);
  const size_t page = ailego::MemoryHelper::PageSize();
  if (!chunk_bytes_) chunk_bytes_ = page;
  if (chunk_bytes_ < page || chunk_bytes_ % page ||
      (chunk_bytes_ & (chunk_bytes_ - 1)) || workspace_count_ == 0 ||
      workspace_count_ > 1024 || max_read_bytes_ == 0 ||
      max_read_bytes_ > (1UL << 30) || chunk_bytes_ > (64UL << 20) ||
      wait_ms_ > 60000 || prefetch_depth_ > 64 ||
      prefetch_bytes_ > (768UL << 20))
    return IndexError_InvalidArgument;
  return 0;
}
int ChunkReadStorage::open(const std::string &path, bool) {
  try {
    auto candidate = std::make_shared<State>();
    if (!candidate->initialize(
            path, chunk_bytes_, cache_bytes_, max_read_bytes_, workspace_count_,
            wait_ms_, direct_io_, adaptive_, prefetch_depth_, prefetch_bytes_))
      return IndexError_OpenFile;
    size_t end = 0;
    if (!ResolveOffset(candidate->file_size, header_offset_, false,
                       &candidate->index_offset) ||
        !ResolveOffset(candidate->file_size, footer_offset_, true, &end) ||
        candidate->index_offset >= end)
      return IndexError_InvalidArgument;
    const size_t container_size = end - candidate->index_offset;
    auto &view = *candidate->views.front();
    auto read = [&](size_t offset, const void **data, size_t length) {
      if (offset > container_size || length > container_size - offset ||
          !candidate->ensure_workspace(view, length))
        return size_t{0};
      *data = view.workspace->data;
      return candidate->meta_file.read(
          static_cast<ssize_t>(candidate->index_offset + offset),
          view.workspace->data, length);
    };
    IndexUnpacker unpacker;
    if (!unpacker.unpack(read, container_size, checksum_))
      return IndexError_UnpackIndex;
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
int ChunkReadStorage::cleanup() {
  return close();
}
int ChunkReadStorage::close() {
  state_.reset();
  return 0;
}
int ChunkReadStorage::flush() {
  return 0;
}
int ChunkReadStorage::append(const std::string &, size_t) {
  return IndexError_NotImplemented;
}
void ChunkReadStorage::refresh(uint64_t) {}
uint64_t ChunkReadStorage::check_point() const {
  return 0;
}
IndexStorage::Segment::Pointer ChunkReadStorage::get(const std::string &id,
                                                     int) {
  if (!state_) return {};
  auto it = state_->segments.find(id);
  return it == state_->segments.end()
             ? Segment::Pointer{}
             : std::make_shared<ReadSegment>(state_, it->second);
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
IndexStorage::MemoryBlock::MemoryBlockType ChunkReadStorage::memory_block_type()
    const {
  return MemoryBlock::MBT_SHARED_SCRATCH;
}
std::string ChunkReadStorage::file_path() const {
  return state_ ? state_->path : std::string{};
}
void ChunkReadStorage::enable_prefetch(bool enabled) {
  if (state_) state_->prefetch_enabled.store(enabled);
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
  result.cache_bytes = s.cache_bytes.load();
  result.workspace_bytes = s.workspace_bytes.load();
  result.metadata_bytes = s.metadata_charge + s.table_metadata.load();
  result.prefetch_submitted = s.prefetch_submitted.load();
  result.prefetch_consumed = s.prefetch_consumed.load();
  result.prefetch_failed = s.prefetch_failed.load();
  result.prefetch_wait_ns = s.prefetch_wait_ns.load();
  result.prefetch_peak_requests = s.prefetch_peak_requests.load();
  result.prefetch_peak_bytes = s.prefetch_peak_bytes.load();
  result.prefetch_pending_requests = s.prefetch_pending_requests.load();
  result.prefetch_pending_bytes = s.prefetch_pending_bytes.load();
  result.prefetch_read_bytes = s.prefetch_read_bytes.load();
  result.prefetch_submit_batches = s.prefetch_submit_batches.load();
  result.backing_reuses = s.backing_reuses.load();
  result.backing_reuse_scans = s.backing_reuse_scans.load();
#if defined(__linux__) && !defined(__ANDROID__)
  if (s.read_ahead) {
    result.prefetch_depth = s.read_ahead->depth;
    result.prefetch_limit_bytes = s.read_ahead->byte_limit;
    result.metadata_bytes +=
        s.read_ahead->charge->bytes +
        (s.read_ahead->ring_charge ? s.read_ahead->ring_charge->bytes : 0);
  }
#endif
  return result;
}
INDEX_FACTORY_REGISTER_STORAGE(ChunkReadStorage);
}  // namespace core
}  // namespace zvec

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
#include "rocksdb_memtable_factory.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/io/file.h>
// RocksDB 8.1 exposes the allocator hook in MemTableRepFactory but only
// forward-declares its type in the public header. Keep this version-pinned
// dependency private to this adapter; do not replace RocksDB's implementation.
#include "memory/allocator.h"

namespace zvec {
namespace {

class MappedMemtableAllocator final : public rocksdb::Allocator {
 public:
  explicit MappedMemtableAllocator(
      std::shared_ptr<rocksdb::WriteBufferManager> manager)
      : manager_(std::move(manager)), tracker_(manager_.get()) {
    if (!manager_) throw std::bad_alloc();
  }

  ~MappedMemtableAllocator() override {
    // No allocation or reader remains when the owning MemTableRep dies.
    // Release the backing before returning its charge to the shared WBM.
    auto *block = current_.load(std::memory_order_relaxed);
    while (block != nullptr) {
      auto *next = block->previous;
      const size_t bytes = block->bytes;
      block->~Block();
      if (ailego::File::TryMemoryUnmap(block, bytes))
        ailego::MemoryLimitPool::get_instance().release_process_backing(bytes);
      block = next;
    }
    tracker_.FreeMem();
  }

  char *Allocate(size_t bytes) override {
    return AllocateAligned(bytes);
  }

  char *AllocateAligned(size_t bytes, size_t /*huge_page_size*/ = 0,
                        rocksdb::Logger * /*logger*/ = nullptr) override {
    constexpr size_t kAlign = alignof(std::max_align_t);
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    if (bytes == 0 || bytes > kMax - (kAlign - 1)) throw std::bad_alloc();
    const size_t aligned = (bytes + kAlign - 1) / kAlign * kAlign;
    if (char *result = take(current_.load(std::memory_order_acquire), aligned))
      return result;
    std::lock_guard<std::mutex> lock(mutex_);
    if (char *result = take(current_.load(std::memory_order_relaxed), aligned))
      return result;
    if (aligned > kMax - sizeof(Block) - (kBlockBytes - 1))
      throw std::bad_alloc();
    const size_t mapped =
        (aligned + sizeof(Block) + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
    if (mapped > kMax - allocated_.load(std::memory_order_relaxed))
      throw std::bad_alloc();
    auto &pool = ailego::MemoryLimitPool::get_instance();
    // WBM owns admission, but RSS also needs to see backing before pages
    // are faulted in. Do not reject mandatory memtable work on RSS pressure.
    if (!pool.account_pending_process_backing(mapped)) throw std::bad_alloc();
    void *memory = ailego::File::MemoryMap(mapped, 0);
    if (memory == nullptr) {
      pool.finish_process_backing(mapped, false);
      throw std::bad_alloc();
    }
    try {
      tracker_.Allocate(mapped);
    } catch (...) {
      pool.finish_process_backing(
          mapped, !ailego::File::TryMemoryUnmap(memory, mapped));
      throw;
    }
    pool.finish_process_backing(mapped, true);
    auto *block = new (memory)
        Block{current_.load(std::memory_order_relaxed), mapped, sizeof(Block)};
    allocated_.fetch_add(mapped, std::memory_order_relaxed);
    // Claim our slice before publishing the block to concurrent writers.
    char *result = take(block, aligned);
    current_.store(block, std::memory_order_release);
    return result;
  }

  size_t BlockSize() const override {
    return kBlockBytes;
  }

  size_t allocated_bytes() const {
    return allocated_.load(std::memory_order_relaxed);
  }

  void seal() {
    tracker_.DoneAllocating();
  }

 private:
  struct alignas(std::max_align_t) Block {
    Block *previous;
    size_t bytes;
    std::atomic<size_t> used;
  };
  static constexpr size_t kBlockBytes = 64 * 1024;

  static char *take(Block *block, size_t bytes) {
    if (block == nullptr) return nullptr;
    size_t used = block->used.load(std::memory_order_relaxed);
    while (bytes <= block->bytes - used) {
      if (block->used.compare_exchange_weak(used, used + bytes,
                                            std::memory_order_relaxed))
        return reinterpret_cast<char *>(block) + used;
    }
    return nullptr;
  }

  std::shared_ptr<rocksdb::WriteBufferManager> manager_;
  rocksdb::AllocTracker tracker_;
  std::mutex mutex_;
  std::atomic<Block *> current_{nullptr};
  std::atomic<size_t> allocated_{0};
};

class MappedMemtableRep final : public rocksdb::MemTableRep {
 public:
  MappedMemtableRep(const KeyComparator &compare,
                    const rocksdb::SliceTransform *transform,
                    rocksdb::Logger *logger,
                    std::shared_ptr<rocksdb::WriteBufferManager> manager)
      : MemTableRep(nullptr),
        backing_(std::move(manager)),
        table_(rocksdb::SkipListFactory().CreateMemTableRep(
            compare, &backing_, transform, logger)) {
    allocator_ = &backing_;
  }

  rocksdb::KeyHandle Allocate(size_t len, char **buffer) override {
    return table_->Allocate(len, buffer);
  }
  void Insert(rocksdb::KeyHandle handle) override {
    table_->Insert(handle);
  }
  bool InsertKey(rocksdb::KeyHandle handle) override {
    return table_->InsertKey(handle);
  }
  void InsertWithHint(rocksdb::KeyHandle handle, void **hint) override {
    table_->InsertWithHint(handle, hint);
  }
  bool InsertKeyWithHint(rocksdb::KeyHandle handle, void **hint) override {
    return table_->InsertKeyWithHint(handle, hint);
  }
  void InsertConcurrently(rocksdb::KeyHandle handle) override {
    table_->InsertConcurrently(handle);
  }
  bool InsertKeyConcurrently(rocksdb::KeyHandle handle) override {
    return table_->InsertKeyConcurrently(handle);
  }
  void InsertWithHintConcurrently(rocksdb::KeyHandle handle,
                                  void **hint) override {
    table_->InsertWithHintConcurrently(handle, hint);
  }
  bool InsertKeyWithHintConcurrently(rocksdb::KeyHandle handle,
                                     void **hint) override {
    return table_->InsertKeyWithHintConcurrently(handle, hint);
  }
  bool Contains(const char *key) const override {
    return table_->Contains(key);
  }
  void MarkReadOnly() override {
    table_->MarkReadOnly();
    backing_.seal();
  }
  void MarkFlushed() override {
    table_->MarkFlushed();
  }
  void Get(const rocksdb::LookupKey &key, void *args,
           bool (*callback)(void *, const char *)) override {
    table_->Get(key, args, callback);
  }
  uint64_t ApproximateNumEntries(const rocksdb::Slice &start,
                                 const rocksdb::Slice &end) override {
    return table_->ApproximateNumEntries(start, end);
  }
  void UniqueRandomSample(uint64_t entries, uint64_t target,
                          std::unordered_set<const char *> *sample) override {
    table_->UniqueRandomSample(entries, target, sample);
  }
  size_t ApproximateMemoryUsage() override {
    // These mappings are outside the Arena supplied by RocksDB. Include the
    // full backing, not just live entry bytes, in its per-memtable flush test.
    return backing_.allocated_bytes();
  }
  Iterator *GetIterator(rocksdb::Arena *arena = nullptr) override {
    return table_->GetIterator(arena);
  }
  Iterator *GetDynamicPrefixIterator(rocksdb::Arena *arena = nullptr) override {
    return table_->GetDynamicPrefixIterator(arena);
  }
  bool IsMergeOperatorSupported() const override {
    return table_->IsMergeOperatorSupported();
  }
  bool IsSnapshotSupported() const override {
    return table_->IsSnapshotSupported();
  }

 private:
  // Reverse member destruction keeps the allocator alive through destruction
  // of the underlying SkipList. MarkFlushed alone must not release either.
  MappedMemtableAllocator backing_;
  std::unique_ptr<rocksdb::MemTableRep> table_;
};

}  // namespace

RocksdbMemtableFactory::RocksdbMemtableFactory(
    std::shared_ptr<rocksdb::WriteBufferManager> manager)
    : manager_(std::move(manager)) {}

rocksdb::MemTableRep *RocksdbMemtableFactory::CreateMemTableRep(
    const rocksdb::MemTableRep::KeyComparator &compare,
    rocksdb::Allocator * /*allocator*/,
    const rocksdb::SliceTransform *transform, rocksdb::Logger *logger) {
  return new MappedMemtableRep(compare, transform, logger, manager_);
}

}  // namespace zvec

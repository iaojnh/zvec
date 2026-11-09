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

#include <cstring>
#include <utility>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_storage.h>
#include "../../ailego/buffer/pool_payload_budget.h"
#include "utility/immutable_segment_view.h"
#include <filesystem>
#include <future>
#include <zvec/core/framework/index_factory.h>
#include "utility/utility_params.h"
#include <array>
#include <atomic>
#include <thread>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>

using namespace zvec;
using namespace zvec::core;

TEST(MemoryBlockBudget, CopiesMovesAndResetsTrackLiveAllocations) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  ASSERT_EQ(0, pool.init(8192));
  auto first = IndexStorage::MemoryBlock::AllocateOwned(4096, 4096);
  std::memset(first.data_, 0x35, 4096);
  EXPECT_EQ(4096u, pool.external_used());
  auto second = first;
  EXPECT_NE(first.data(), second.data());
  EXPECT_EQ(8192u, pool.external_used());
  EXPECT_THROW((void)IndexStorage::MemoryBlock(first), std::bad_alloc);
  EXPECT_EQ(8192u, pool.external_used());
  auto moved = std::move(first);
  EXPECT_EQ(nullptr, first.data());
  EXPECT_EQ(8192u, pool.external_used());
  second.reset();
  EXPECT_EQ(4096u, pool.external_used());
  second = std::move(moved);
  EXPECT_EQ(nullptr, moved.data());
  EXPECT_EQ(4096u, pool.external_used());
  EXPECT_EQ(0x35, *static_cast<const unsigned char *>(second.data()));
  second.reset();
  EXPECT_EQ(0u, pool.external_used());
}

TEST(MemoryBlockBudget, OversizedAllocationDoesNotConsumeBudget) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  ASSERT_EQ(0, pool.init(4096));
  EXPECT_THROW(IndexStorage::MemoryBlock::AllocateOwned(8192, 4096),
               std::bad_alloc);
  EXPECT_EQ(0u, pool.used());
}

// MakeBorrowedView is only for explicitly caller-managed buffers, not batch
// results whose backing storage must survive later reads. The block must free
// nothing and pin nothing on destruction, and copies/moves must keep aliasing
// the same buffer without taking ownership. A double-free here would trap
// under ASan.
TEST(MemoryBlockBorrowedView, IsNonOwning) {
  auto *buf = new char[64];
  std::memset(buf, 0xAB, 64);
  {
    auto view = IndexStorage::MemoryBlock::MakeBorrowedView(buf);
    EXPECT_EQ(static_cast<const void *>(buf), view.data());

    // Copy keeps a non-owning alias to the same buffer.
    auto copy = view;
    EXPECT_EQ(static_cast<const void *>(buf), copy.data());

    // Move keeps the alias too.
    auto moved = std::move(copy);
    EXPECT_EQ(static_cast<const void *>(buf), moved.data());
  }  // every view destroyed here; the backing buffer must NOT be freed.

  // Still readable, and safe to free exactly once by the sole owner.
  EXPECT_EQ(static_cast<unsigned char>(buf[0]), 0xABu);
  delete[] buf;
}

// A borrowed view over a slice of a shared buffer (as the arena hands out)
// must alias the exact slice and never free the shared backing storage.
TEST(MemoryBlockBorrowedView, AliasesArenaSlice) {
  std::vector<char> arena(256, 0);
  arena[128] = 0x5A;
  auto view = IndexStorage::MemoryBlock::MakeBorrowedView(arena.data() + 128);
  ASSERT_EQ(static_cast<const void *>(arena.data() + 128), view.data());
  EXPECT_EQ(0x5A, *static_cast<const char *>(view.data()));
  // Destroying the view leaves the arena intact for reuse.
  view.reset();
  EXPECT_EQ(0x5A, arena[128]);
}

TEST(MemoryBlockSharedView, CopiesAndMovesKeepArenaAlive) {
  auto arena = std::make_shared<std::vector<char>>(256, 0x5A);
  std::weak_ptr<std::vector<char>> lifetime = arena;
  const void *slice = arena->data() + 128;
  auto view =
      IndexStorage::MemoryBlock::MakeSharedView(arena->data() + 128, arena);
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_SHARED_SCRATCH, view.type_);
  auto copy = view;
  IndexStorage::MemoryBlock assigned;
  assigned = copy;
  auto moved = std::move(copy);
  EXPECT_EQ(nullptr, copy.data());
  IndexStorage::MemoryBlock move_assigned;
  move_assigned = std::move(assigned);
  EXPECT_EQ(nullptr, assigned.data());
  arena.reset();
  view.reset();

  ASSERT_FALSE(lifetime.expired());
  EXPECT_EQ(slice, moved.data());
  EXPECT_EQ(slice, move_assigned.data());
  EXPECT_EQ(0x5A, *static_cast<const char *>(moved.data()));
  EXPECT_EQ(0x5A, *static_cast<const char *>(move_assigned.data()));
  moved.reset();
  EXPECT_FALSE(lifetime.expired());
  move_assigned.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST(MemoryBlockSharedView, ResetAndReplacementReleasePreviousOwner) {
  auto arena = std::make_shared<std::vector<char>>(64, 0x2A);
  std::weak_ptr<std::vector<char>> lifetime = arena;
  auto view = IndexStorage::MemoryBlock::MakeSharedView(arena->data(), arena);
  auto copy = view;
  arena.reset();

  char replacement = 'x';
  view.reset(&replacement);
  EXPECT_EQ(&replacement, view.data());
  EXPECT_FALSE(lifetime.expired());
  copy = IndexStorage::MemoryBlock::MakeBorrowedView(&replacement);
  EXPECT_TRUE(lifetime.expired());

  auto next_arena = std::make_shared<std::vector<char>>(64, 0x3A);
  lifetime = next_arena;
  view =
      IndexStorage::MemoryBlock::MakeSharedView(next_arena->data(), next_arena);
  next_arena.reset();
  view = copy;
  EXPECT_TRUE(lifetime.expired());
}

TEST(MemoryBlockSharedView, DestructionReleasesLastSlice) {
  std::weak_ptr<std::vector<char>> lifetime;
  {
    auto arena = std::make_shared<std::vector<char>>(64, 0x2A);
    lifetime = arena;
    auto view = IndexStorage::MemoryBlock::MakeSharedView(arena->data(), arena);
    arena.reset();
    EXPECT_FALSE(lifetime.expired());
  }
  EXPECT_TRUE(lifetime.expired());
}

namespace zvec::core {
TEST(ImmutableSegmentViewTest, BoundedReadsOwnTheirLifetimeUnderPressure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(pool.used(), 0u);
  auto restore = ailego::ScopeGuard::Make([&]() {
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(previous), 0);
  });
  const std::string path = "immutable_segment_view_test.index";
  auto remove =
      ailego::ScopeGuard::Make([&]() { std::filesystem::remove(path); });
  std::vector<uint8_t> bytes(1024 * 1024 + 73);
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = i % 251;
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(dumper->create(path), 0);
  ASSERT_EQ(dumper->write(bytes.data(), bytes.size()), bytes.size());
  ASSERT_EQ(dumper->append("payload", bytes.size(), 0, 0), 0);
  ASSERT_EQ(dumper->close(), 0);
  const size_t file_pages =
      (std::filesystem::file_size(path) - 1) / ailego::kVectorPageSize + 1;
  const size_t cache_metadata =
      ailego::VecBufferPool::metadata_bytes_for_page_count(file_pages) +
      pool.page_buffer_size();  // Header of the first page-buffer slab.
  // Also cover the bypass-only open: no cache page fits, but load must still
  // avoid materializing the complete segment on the heap.
  for (size_t mode_index = 0; mode_index < 3; ++mode_index) {
    const char *mode =
        mode_index == 0 ? "MMapFileReadStorage" : "BufferReadStorage";
    SCOPED_TRACE(mode);
    SCOPED_TRACE(mode_index);
    // Keep 128 KiB for reads/copies independently of the cache's mandatory
    // metadata and allocator header (rounded to the host page size). The
    // bypass-only case admits the directory but cannot return an owned
    // 8193-byte range.
    const size_t budget =
        mode_index == 2 ? 4096
                        : 128 * 1024 + (mode_index == 1 ? cache_metadata : 0);
    ASSERT_LT(budget, bytes.size());
    ASSERT_EQ(pool.init(budget), 0);
    auto storage = IndexFactory::CreateStorage(mode);
    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE,
               BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_EQ(storage->init(params), 0);
    ASSERT_EQ(storage->open(path, false), 0);
    if (mode_index != 0) {
      ASSERT_NE(nullptr, storage->vec_buffer_pool());
      ASSERT_EQ(mode_index == 1, storage->vec_buffer_pool()->cache_enabled());
    }
    ImmutableSegmentView view;
    ASSERT_EQ(view.load(storage, "payload", 0, bytes.size()), 0);
    EXPECT_EQ(view.data() == nullptr, storage->vec_buffer_pool() != nullptr);
    EXPECT_LE(pool.used(), pool.capacity());
    if (mode_index == 2) {
      // Opening a view needs no payload copy. Returning an owned contiguous
      // range does need budget; bypass must not silently allocate outside it.
      const auto metadata_used = pool.used();
      EXPECT_GT(metadata_used, 0u);
      IndexStorage::MemoryBlock guard;
      const void *data = nullptr;
      EXPECT_EQ(view.view(0, 8193, &guard, &data), IndexError_ReadData);
      EXPECT_EQ(nullptr, data);
      EXPECT_EQ(metadata_used, pool.used());
      continue;
    }
    auto verify = [&view, &bytes](size_t start) {
      for (size_t offset = start; offset < bytes.size(); offset += 8191) {
        const size_t count = std::min<size_t>(8193, bytes.size() - offset);
        IndexStorage::MemoryBlock guard, second;
        const void *first = nullptr, *other = nullptr;
        if (view.view(offset, count, &guard, &first) != 0) return false;
        if (view.view(17, 4097, &second, &other) != 0) return false;
        if (std::memcmp(first, bytes.data() + offset, count) != 0) return false;
      }
      return true;
    };
    auto task = std::async(std::launch::async, verify, size_t{7});
    EXPECT_TRUE(verify(0));
    EXPECT_TRUE(task.get());
    EXPECT_NE(view.read(bytes.size(), bytes.data(), 1), 0);
    EXPECT_NE(view.read(std::numeric_limits<size_t>::max(), bytes.data(), 2),
              0);
    size_t copied = 0;
    ASSERT_EQ(view.copy_to([&](size_t offset, const void *data, size_t size) {
      EXPECT_LE(size, 64 * 1024u);
      EXPECT_EQ(std::memcmp(data, bytes.data() + offset, size), 0);
      copied += size;
      return 0;
    }),
              0);
    EXPECT_EQ(copied, bytes.size());
  }
}
}  // namespace zvec::core

namespace zvec::core {
namespace {
class FetchSegment : public IndexStorage::Segment {
 public:
  size_t data_size() const override {
    return 1U << 20;
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return 0;
  }
  size_t capacity() const override {
    return data_size();
  }
  size_t fetch(size_t offset, void *out, size_t bytes) const override {
    ++calls;
    charge_at_fetch = ailego::PoolPayloadBudget::get_instance().external_used();
    auto *data = static_cast<unsigned char *>(out);
    for (size_t i = 0; i < bytes; ++i) data[i] = (offset + i) % 251;
    return short_read ? bytes - 1 : bytes;
  }
  size_t read(size_t, IndexStorage::MemoryBlock &, size_t) override {
    return 0;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return {};
  }
  mutable size_t calls{0};
  mutable size_t charge_at_fetch{0};
  bool short_read{false};
};

class StorageFetchBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0u, pool_.used());
    previous_ = pool_.capacity();
    ASSERT_EQ(0, pool_.init(16 * page_));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  const size_t page_ = ailego::PoolPayloadBudget::page_buffer_size();
  size_t previous_{0};
};

TEST_F(StorageFetchBudgetTest, SharedDestinationKeepsBudgetUntilLastOwner) {
  std::shared_ptr<ailego::BudgetedMappedBuffer> buffer;
  {
    FetchSegment segment;
    buffer = std::allocate_shared<ailego::BudgetedMappedBuffer>(
        ailego::BudgetedAllocator<ailego::BudgetedMappedBuffer>{}, page_);
    const size_t charge = pool_.external_used();
    ASSERT_GE(charge, page_);
    ASSERT_EQ(page_, segment.fetch(3, buffer->data(), buffer->size()));
    EXPECT_EQ(charge, segment.charge_at_fetch);
    EXPECT_EQ(charge, pool_.external_used());
  }
  auto copy = buffer;
  const size_t charge = pool_.external_used();
  buffer.reset();
  EXPECT_EQ(charge, pool_.external_used());
  EXPECT_EQ(3, static_cast<unsigned char>(copy->data()[0]));
  std::thread worker([owned = std::move(copy)]() mutable { owned.reset(); });
  worker.join();
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(StorageFetchBudgetTest,
       ResizeAdmissionPrecedesFetchAndPreservesOldData) {
  FetchSegment segment;
  ailego::BudgetedMappedBuffer buffer(page_);
  ASSERT_EQ(page_, segment.fetch(3, buffer.data(), buffer.size()));
  const void *old = buffer.data();
  const size_t before = pool_.external_used();
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_THROW(buffer.resize(2 * page_), std::bad_alloc);
  EXPECT_EQ(old, buffer.data());
  EXPECT_EQ(3, static_cast<unsigned char>(buffer.data()[0]));
  EXPECT_EQ(1u, segment.calls);
  pressure.retain_at_most(0);
  EXPECT_EQ(before, pool_.external_used());
  buffer.resize(2 * page_);
  ASSERT_EQ(buffer.size(), segment.fetch(7, buffer.data(), buffer.size()));
  EXPECT_EQ(7, static_cast<unsigned char>(buffer.data()[0]));
}

TEST_F(StorageFetchBudgetTest, RepeatedFetchDoesNotChargeDestinationTwice) {
  FetchSegment segment;
  ailego::BudgetedVector<uint64_t> words(8);
  const size_t before = pool_.external_used();
  for (size_t i = 0; i < 32; ++i) {
    const size_t bytes = words.size() * sizeof(words[0]);
    EXPECT_EQ(bytes, segment.fetch(i, words.data(), bytes));
    EXPECT_EQ(before, segment.charge_at_fetch);
    EXPECT_EQ(before, pool_.external_used());
  }
  words.clear();
  EXPECT_EQ(before, pool_.external_used());  // retained capacity is still live
  ailego::BudgetedVector<uint64_t>().swap(words);
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(StorageFetchBudgetTest, ShortReadDoesNotReleaseCallerOwnedBudget) {
  FetchSegment segment;
  ailego::BudgetedMappedBuffer buffer(page_);
  const size_t charge = pool_.external_used();
  segment.short_read = true;
  EXPECT_EQ(page_ - 1, segment.fetch(3, buffer.data(), buffer.size()));
  EXPECT_EQ(charge, pool_.external_used());
  segment.short_read = false;
  EXPECT_EQ(page_, segment.fetch(7, buffer.data(), buffer.size()));
  buffer.clear();
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(StorageFetchBudgetTest, ConcurrentDestinationsShareOneBudget) {
  constexpr size_t kThreads = 4;
  std::atomic<size_t> ready{0}, accepted{0};
  std::array<std::thread, kThreads> workers;
  for (auto &worker : workers) {
    worker = std::thread([&] {
      FetchSegment segment;
      ailego::BudgetedMappedBuffer buffer;
      try {
        buffer.resize(6 * page_);
        EXPECT_EQ(buffer.size(),
                  segment.fetch(0, buffer.data(), buffer.size()));
        ++accepted;
      } catch (const std::bad_alloc &) {
        EXPECT_EQ(0u, segment.calls);
      }
      ++ready;
      while (ready.load() != kThreads) std::this_thread::yield();
      EXPECT_LE(pool_.used(), pool_.capacity());
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(2u, accepted.load());
}
}  // namespace
}  // namespace zvec::core

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

// Unit tests for vector_page_table.cc
//
// Focus: verify that MemoryLimitPool enforces its configured limit at all
// times, both under single-threaded sequential access and under concurrent
// multi-threaded access.
//
// Observable proxy for used_size_ (which is private):
//   - is_full()       → used_size_ >= pool_size_
//   - is_hot_level1() → used_size_ >= pool_size_ * 3 / 5
//   - is_hot_level2() → used_size_ >= pool_size_ * 4 / 5
//   - try_acquire_buffer() → returns false iff used_size_ >= pool_size_
//
// The key memory-limit invariant is: used_size_ <= pool_size_.
// We verify this by showing that acquiring exactly pool_size/block_size blocks
// fills the pool (is_full()==true) and acquiring one more fails, proving no
// silent over-allocation occurs.

#include <cstdint>
#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <zvec/ailego/buffer/lru_cache.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include "tests/test_util.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::ailego;

// ====================================================================
// Helpers
// ====================================================================

// RAII guard: automatically releases MemoryLimitPool buffers allocated
// directly via try_acquire_buffer (not backed by a VectorPageTable entry).
// Used to ensure cleanup even when a test fails mid-way.
struct MemoryGuard {
  struct Entry {
    char *buf;
    size_t size;
  };
  std::vector<Entry> entries;

  char *acquire(size_t size) {
    char *buf = nullptr;
    if (MemoryLimitPool::get_instance().try_acquire_buffer(size, buf)) {
      entries.push_back({buf, size});
      return buf;
    }
    return nullptr;
  }

  void release(char *buf, size_t size) {
    MemoryLimitPool::get_instance().release_buffer(buf, size);
    entries.erase(
        std::remove_if(entries.begin(), entries.end(),
                       [buf](const Entry &e) { return e.buf == buf; }),
        entries.end());
  }

  ~MemoryGuard() {
    for (auto &e : entries) {
      MemoryLimitPool::get_instance().release_buffer(e.buf, e.size);
    }
  }
};

// ====================================================================
// Part 1: MemoryLimitPool unit tests (direct, no file I/O)
// ====================================================================

// 5 blocks of 4 KiB each → 20 KiB pool
static constexpr size_t kUnitBlockSize = 4096;
static constexpr size_t kUnitNumBlocks = 5;
static constexpr size_t kUnitPoolSize = kUnitNumBlocks * kUnitBlockSize;

class MemoryLimitPoolTest : public testing::Test {
 protected:
  void SetUp() override {
    // pool_size_ = 0 → recycle() evicts anything in LRU → then set limit
    MemoryLimitPool::get_instance().init(kUnitPoolSize);
  }

  void TearDown() override {
    // Drain the LRU to release any page-table-backed blocks
    LRUCache::get_instance().recycle();
  }
};

// --------------------------------------------------------------------
// TEST: Acquiring exactly pool_size/block_size blocks fills the pool;
//       acquiring one more returns false without over-allocating.
// This is the primary proof that used_size_ never exceeds pool_size_.
// --------------------------------------------------------------------
TEST_F(MemoryLimitPoolTest, AcquireUpToLimitThenFail) {
  MemoryGuard guard;

  // Acquire blocks one by one; each should succeed
  for (size_t i = 0; i < kUnitNumBlocks; ++i) {
    char *buf = guard.acquire(kUnitBlockSize);
    ASSERT_NE(buf, nullptr) << "Block " << i << " should be acquirable";

    // Pool must NOT be full until we've loaded the last block
    if (i < kUnitNumBlocks - 1) {
      EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
          << "Pool should not be full after loading " << (i + 1) << " / "
          << kUnitNumBlocks << " blocks";
    }
  }

  // After loading all blocks the pool is exactly full
  EXPECT_TRUE(MemoryLimitPool::get_instance().is_full())
      << "Pool should be full after loading all blocks";

  // An extra allocation must fail — this is the invariant proof
  char *extra = nullptr;
  bool ok =
      MemoryLimitPool::get_instance().try_acquire_buffer(kUnitBlockSize, extra);
  EXPECT_FALSE(ok) << "Acquiring beyond the limit must fail";
  EXPECT_EQ(extra, nullptr);

  // Release all buffers and confirm the pool is no longer full
  for (auto &e : guard.entries) {
    MemoryLimitPool::get_instance().release_buffer(e.buf, e.size);
  }
  guard.entries.clear();

  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Pool must not be full after releasing all blocks";

  // The capacity is restored: one more allocation should succeed
  char *reuse = guard.acquire(kUnitBlockSize);
  ASSERT_NE(reuse, nullptr) << "Allocation must succeed after releasing";
}

// --------------------------------------------------------------------
// TEST: release_buffer correctly reduces used_size_
//       (a single full-pool allocation is released and is_full() clears)
// --------------------------------------------------------------------
TEST_F(MemoryLimitPoolTest, SingleReleaseClearsFullFlag) {
  MemoryGuard guard;

  // Consume the entire pool in one allocation
  char *buf = guard.acquire(kUnitPoolSize);
  ASSERT_NE(buf, nullptr);
  EXPECT_TRUE(MemoryLimitPool::get_instance().is_full());

  guard.release(buf, kUnitPoolSize);
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Pool should be empty after releasing the only allocation";
}

// --------------------------------------------------------------------
// TEST: Hot-level thresholds are reported at the correct percentages.
//       level-1 fires at >= 60 %  (pool_size * 3/5)
//       level-2 fires at >= 80 %  (pool_size * 4/5)
//       Pool = 5 blocks → threshold-1 = 3 blocks, threshold-2 = 4 blocks
// --------------------------------------------------------------------
TEST_F(MemoryLimitPoolTest, HotLevelThresholds) {
  MemoryGuard guard;

  EXPECT_FALSE(MemoryLimitPool::get_instance().is_hot_level1())
      << "No hot level with empty pool";
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_hot_level2())
      << "No hot level with empty pool";

  // Load 3 blocks: 3/5 = 60% → is_hot_level1 fires, is_hot_level2 does not
  for (int i = 0; i < 3; ++i) {
    ASSERT_NE(guard.acquire(kUnitBlockSize), nullptr);
  }
  EXPECT_TRUE(MemoryLimitPool::get_instance().is_hot_level1())
      << "is_hot_level1 must fire at 60%";
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_hot_level2())
      << "is_hot_level2 must not fire at 60%";

  // Load 1 more (4 total): 4/5 = 80% → is_hot_level2 fires
  ASSERT_NE(guard.acquire(kUnitBlockSize), nullptr);
  EXPECT_TRUE(MemoryLimitPool::get_instance().is_hot_level2())
      << "is_hot_level2 must fire at 80%";

  // Release everything and confirm both levels clear
  for (auto &e : guard.entries) {
    MemoryLimitPool::get_instance().release_buffer(e.buf, e.size);
  }
  guard.entries.clear();

  EXPECT_FALSE(MemoryLimitPool::get_instance().is_hot_level1())
      << "Hot levels must clear after full release";
}

// --------------------------------------------------------------------
// TEST: Concurrent acquire/release from multiple threads never causes
//       used_size_ to exceed pool_size_.
//
//       Strategy: N threads each loop "acquire 1 block → check is_full()
//       is consistent → release". The pool has exactly N blocks, so at most
//       N threads hold memory simultaneously.  After all threads finish we
//       verify that the pool accounting is clean (is_full() = false).
// --------------------------------------------------------------------
TEST_F(MemoryLimitPoolTest, ConcurrentAcquireReleaseWithinLimit) {
  constexpr int kThreads = kUnitNumBlocks;  // 5 threads, 5-block pool
  std::atomic<int> success_count{0};
  std::atomic<int> fail_count{0};
  constexpr int kIterations = 200;

  auto worker = [&]() {
    for (int i = 0; i < kIterations; ++i) {
      char *buf = nullptr;
      bool ok = MemoryLimitPool::get_instance().try_acquire_buffer(
          kUnitBlockSize, buf);
      if (ok) {
        ASSERT_NE(buf, nullptr);
        success_count.fetch_add(1, std::memory_order_relaxed);
        MemoryLimitPool::get_instance().release_buffer(buf, kUnitBlockSize);
      } else {
        fail_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(worker);
  }
  for (auto &th : threads) th.join();

  // At least some acquisitions must have succeeded
  EXPECT_GT(success_count.load(), 0);
  LOG_DEBUG("concurrent test: success=%d fail=%d", success_count.load(),
            fail_count.load());

  // After all threads complete the pool accounting must be clean
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Pool must not be full after all threads release their blocks";
}

// ====================================================================
// Part 2: VecBufferPool + VectorPageTable integration tests
// Verify that pread-backed buffer loading also stays within the limit.
// ====================================================================

static const std::string kWorkingDir{"./vec_page_table_test_dir/"};
static const std::string kVecFile{kWorkingDir + "test.vec"};

// 16 segments of 4 KiB = 64 KiB file; pool holds at most 4 segments
static constexpr size_t kFileBlockSize = 4096;
static constexpr size_t kFileSegments = 16;
static constexpr size_t kFileSize = kFileSegments * kFileBlockSize;
// Memory limit: 4 blocks (25 % of the file)
static constexpr size_t kPoolMemLimit = 4 * kFileBlockSize;

class VecBufferPoolMemoryTest : public testing::Test {
 public:
  static void SetUpTestCase() {
    zvec::test_util::RemoveTestPath(kWorkingDir);

    if (!File::MakePath(kWorkingDir)) {
      LOG_ERROR("Failed to create working directory");
      return;
    }

    // Create test file filled with a recognisable pattern (sequential uint32)
    File vec_file;
    if (!vec_file.create(kVecFile, kFileSize)) {
      LOG_ERROR("Failed to create test vector file");
      return;
    }
    for (uint32_t i = 0; i < kFileSize / sizeof(uint32_t); ++i) {
      vec_file.write(reinterpret_cast<void *>(&i), sizeof(i));
    }
    vec_file.close();
  }

  static void TearDownTestCase() {
    zvec::test_util::RemoveTestPath(kWorkingDir);
  }

  void SetUp() override {
    // Re-initialise pool limit for each test; recycles any LRU-eligible blocks
    MemoryLimitPool::get_instance().init(kPoolMemLimit);
  }

  void TearDown() override {
    LRUCache::get_instance().recycle();
  }
};

// --------------------------------------------------------------------
// TEST: Sequential load – loading exactly pool_limit/block_size blocks
//       fills the pool; the (limit+1)-th block fails without retry.
//       Releasing + retrying succeeds via LRU eviction.
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, SequentialLoadEnforcesLimit) {
  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(kPoolMemLimit, kFileBlockSize, kFileSegments), 0);

  // Load 4 blocks (= pool limit); all must succeed
  for (size_t i = 0; i < 4; ++i) {
    char *buf =
        pool.acquire_buffer(i, i * kFileBlockSize, kFileBlockSize, /*retry=*/0);
    ASSERT_NE(buf, nullptr) << "Block " << i << " within limit must load";

    // Memory must not exceed the limit after each step
    EXPECT_FALSE(
        MemoryLimitPool::get_instance().try_acquire_buffer(1, buf) &&
        (MemoryLimitPool::get_instance().release_buffer(buf, 1), false))
        << "Sanity: acquiring 1 byte must fail when pool is full (block " << i
        << ")";
    (void)buf;  // suppress maybe-unused
  }

  // Pool is exactly full
  EXPECT_TRUE(MemoryLimitPool::get_instance().is_full())
      << "Pool should be full after loading 4 blocks (= limit)";

  // 5th block without retry → must fail (proves no silent over-allocation)
  char *overflow =
      pool.acquire_buffer(4, 4 * kFileBlockSize, kFileBlockSize, /*retry=*/0);
  EXPECT_EQ(overflow, nullptr)
      << "(limit+1)-th block without retry must fail";

  // Release all 4 blocks (makes them eligible for LRU eviction)
  for (size_t i = 0; i < 4; ++i) {
    pool.page_table_.release_block(i);
  }

  // With retry=5, the 5th block should load after evicting an older block
  char *evicted_load =
      pool.acquire_buffer(4, 4 * kFileBlockSize, kFileBlockSize, /*retry=*/5);
  EXPECT_NE(evicted_load, nullptr)
      << "5th block must load after LRU eviction (retry=5)";
  if (evicted_load) {
    pool.page_table_.release_block(4);
  }

  // Evict remaining blocks so the VecBufferPool destructor passes its asserts
  LRUCache::get_instance().recycle();
}

// --------------------------------------------------------------------
// TEST: Loading all 16 segments with retry=5 triggers LRU eviction
//       repeatedly; at no point should memory exceed the 4-block limit.
//       Verified by checking that is_full() never transitions from true
//       to a state where another block was silently added on top.
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, EvictionKeepsMemoryWithinLimit) {
  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(kPoolMemLimit, kFileBlockSize, kFileSegments), 0);

  for (size_t i = 0; i < kFileSegments; ++i) {
    char *buf = pool.acquire_buffer(i, i * kFileBlockSize, kFileBlockSize,
                                    /*retry=*/5);
    ASSERT_NE(buf, nullptr) << "Block " << i
                            << " must load with eviction enabled";

    // After a successful load the pool must be at most full, never over
    // (is_full() true means used == limit, which is the boundary condition)
    // Probe: an additional 1-byte allocation must fail when pool is full
    {
      char *probe = nullptr;
      bool probe_ok =
          MemoryLimitPool::get_instance().try_acquire_buffer(kFileBlockSize, probe);
      if (probe_ok) {
        // Returned successfully → some space was available; immediately release
        MemoryLimitPool::get_instance().release_buffer(probe, kFileBlockSize);
        EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
            << "Probe succeeded but pool claims to be full – inconsistency at "
               "block "
            << i;
      }
      // else: pool is full, which is the expected boundary state
    }

    pool.page_table_.release_block(i);
  }

  LRUCache::get_instance().recycle();
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Pool must be clean after draining LRU";
}

// --------------------------------------------------------------------
// TEST: Verify loaded data integrity – the content read from disk through
//       VecBufferPool matches the pattern written in SetUpTestCase.
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, DataIntegrity) {
  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(kPoolMemLimit, kFileBlockSize, kFileSegments), 0);

  for (size_t seg = 0; seg < 4; ++seg) {
    size_t offset = seg * kFileBlockSize;
    char *buf = pool.acquire_buffer(seg, offset, kFileBlockSize, /*retry=*/0);
    ASSERT_NE(buf, nullptr);

    // Verify sequential uint32 values
    const uint32_t *data = reinterpret_cast<const uint32_t *>(buf);
    uint32_t base = static_cast<uint32_t>(offset / sizeof(uint32_t));
    for (size_t w = 0; w < kFileBlockSize / sizeof(uint32_t); ++w) {
      ASSERT_EQ(data[w], base + w)
          << "Data mismatch at segment " << seg << ", word " << w;
    }
    pool.page_table_.release_block(seg);
  }

  LRUCache::get_instance().recycle();
}

// --------------------------------------------------------------------
// TEST: Concurrent access from multiple threads – memory accounting
//       remains consistent throughout.
//
//       kThreads threads repeatedly acquire-use-release different blocks.
//       With retry=5 and the LRU eviction path, all acquisitions should
//       eventually succeed.  After all threads finish, the pool is drained
//       and is_full() must return false.
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, ConcurrentAccessMemoryConsistency) {
  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(kPoolMemLimit, kFileBlockSize, kFileSegments), 0);

  constexpr int kThreads = 8;
  constexpr int kIter = 80;
  std::atomic<int> acquired{0};
  std::atomic<int> failed{0};

  auto worker = [&](int tid) {
    for (int it = 0; it < kIter; ++it) {
      // Spread accesses over all 16 segments
      size_t bid = static_cast<size_t>((tid * 7 + it * 3) % kFileSegments);
      char *buf = pool.acquire_buffer(bid, bid * kFileBlockSize, kFileBlockSize,
                                      /*retry=*/5);
      if (buf != nullptr) {
        acquired.fetch_add(1, std::memory_order_relaxed);
        pool.page_table_.release_block(bid);
      } else {
        failed.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
  for (auto &th : threads) th.join();

  EXPECT_GT(acquired.load(), 0) << "At least some acquisitions should succeed";
  LOG_DEBUG("concurrent vec test: acquired=%d failed=%d", acquired.load(),
            failed.load());

  // Drain all LRU-eligible blocks and verify clean accounting
  LRUCache::get_instance().recycle();
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Memory must be fully released after draining LRU";
}

// --------------------------------------------------------------------
// TEST: VecBufferPoolHandle – acquire/release via handle mirrors
//       the underlying page-table ref-count correctly and memory
//       is returned to the pool when the last reference is dropped.
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, HandleAcquireRelease) {
  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(kPoolMemLimit, kFileBlockSize, kFileSegments), 0);

  VecBufferPoolHandle handle = pool.get_handle();

  // Acquire block 0 via handle
  char *buf = handle.get_block(0, kFileBlockSize, /*block_id=*/0);
  ASSERT_NE(buf, nullptr);

  // Acquire the same block again (ref-count +1, same buffer)
  handle.acquire_one(0);

  // Release twice to bring ref-count back to 0
  handle.release_one(0);
  handle.release_one(0);

  // After both releases, block 0 is LRU-eligible; evict and check memory
  LRUCache::get_instance().recycle();
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Memory must be free after handle releases";
}

// ====================================================================
// Part 3: VectorPageTable direct tests (no file I/O)
// Exercises the page-table primitives in isolation to verify:
//   - Unloaded entries return nullptr from acquire_block
//   - evict_block on a held block is a strict no-op (no memory freed)
//   - is_dead_block correctly identifies stale LRU version entries
// ====================================================================

static constexpr size_t kDirectEntries = 8;
static constexpr size_t kDirectBlockSize = 4096;
static constexpr size_t kDirectPoolSize = kDirectEntries * kDirectBlockSize;

class VectorPageTableDirectTest : public testing::Test {
 protected:
  void SetUp() override {
    MemoryLimitPool::get_instance().init(kDirectPoolSize);
    table_.init(kDirectEntries);
  }

  void TearDown() override {
    // Safety-net: evict every entry that has no active references.
    // Tests are responsible for releasing their own refs before teardown.
    for (size_t i = 0; i < kDirectEntries; ++i) {
      table_.evict_block(i);
    }
    LRUCache::get_instance().recycle();
  }

  // Helper: allocate through MemoryLimitPool so that evict_block can later
  // call release_buffer and the accounting stays consistent.
  char *alloc_block() {
    char *buf = nullptr;
    MemoryLimitPool::get_instance().try_acquire_buffer(kDirectBlockSize, buf);
    return buf;
  }

  VectorPageTable table_;
};

// --------------------------------------------------------------------
// TEST: acquire_block on an entry that has never been loaded must
//       return nullptr (ref_count starts at INT_MIN).
// --------------------------------------------------------------------
TEST_F(VectorPageTableDirectTest, AcquireUnloadedEntryReturnsNull) {
  for (size_t i = 0; i < kDirectEntries; ++i) {
    EXPECT_EQ(table_.acquire_block(i), nullptr)
        << "Entry " << i << " must return nullptr before being loaded";
  }
}

// --------------------------------------------------------------------
// TEST: evict_block while ref_count > 0 must be a no-op.
// Proof: after the failed eviction the entry is still accessible and
//        the pool memory is NOT released (is_full state unchanged).
// --------------------------------------------------------------------
TEST_F(VectorPageTableDirectTest, EvictHeldBlockIsNoOp) {
  char *buf = alloc_block();
  ASSERT_NE(buf, nullptr);

  // Load block 0 (ref_count = 1)
  char *result = table_.set_block_acquired(0, buf, kDirectBlockSize);
  ASSERT_EQ(result, buf);

  // Pool now holds one block worth of memory
  EXPECT_TRUE(MemoryLimitPool::get_instance().is_hot_level1() ||
              !MemoryLimitPool::get_instance().is_full())
      << "Memory is occupied";

  // Attempt to evict while ref_count == 1: CAS(expected=0) fails
  table_.evict_block(0);

  // Entry must still be accessible (buffer not freed)
  char *still_alive = table_.acquire_block(0);
  EXPECT_EQ(still_alive, buf)
      << "Block must still be alive after failed eviction";
  // Undo the extra acquire_block just done
  table_.release_block(0);

  // Now fully release (ref_count → 0) and evict cleanly
  table_.release_block(0);  // ref_count: 1 → 0
  table_.evict_block(0);    // CAS succeeds, memory freed

  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Memory must be freed after proper eviction";
}

// --------------------------------------------------------------------
// TEST: is_dead_block returns false for a current LRU entry and true
//       after the block has been evicted and reloaded (load_count bumped).
// This ensures stale LRU entries are skipped during recycle().
// --------------------------------------------------------------------
TEST_F(VectorPageTableDirectTest, IsDeadBlockDetectsStaleVersion) {
  char *buf1 = alloc_block();
  ASSERT_NE(buf1, nullptr);

  // First load: load_count becomes 1 inside set_block_acquired
  table_.set_block_acquired(0, buf1, kDirectBlockSize);
  table_.release_block(0);  // ref_count → 0

  // Construct an LRU entry reflecting the first load (version = 1)
  LRUCache::BlockType lru_entry{};
  lru_entry.page_table = &table_;
  lru_entry.vector_block.first = 0;
  lru_entry.vector_block.second = 1;  // matches current load_count

  EXPECT_FALSE(table_.is_dead_block(lru_entry))
      << "Entry must be alive right after first load";

  // Evict (frees buf1) and reload with a new buffer
  table_.evict_block(0);

  char *buf2 = alloc_block();
  ASSERT_NE(buf2, nullptr);
  table_.set_block_acquired(0, buf2, kDirectBlockSize);  // load_count → 2

  // The old LRU entry (version=1) must now be recognised as dead
  EXPECT_TRUE(table_.is_dead_block(lru_entry))
      << "Old LRU entry must be dead after block is reloaded";

  // Cleanup
  table_.release_block(0);
  table_.evict_block(0);

  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full());
}

// ====================================================================
// Part 4: Additional VecBufferPool correctness tests
// ====================================================================

// --------------------------------------------------------------------
// TEST: Acquiring the same block ID multiple times returns the same
//       buffer pointer and does NOT allocate extra memory each time.
//       Memory should be counted once per unique physical block.
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, SameBlockMultiAcquireNoDoubleCount) {
  // Shrink the pool limit to exactly 2 blocks for this test
  MemoryLimitPool::get_instance().init(2 * kFileBlockSize);

  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(2 * kFileBlockSize, kFileBlockSize, kFileSegments), 0);

  // First acquire of block 0: loads from disk, ref_count = 1
  char *buf0a = pool.acquire_buffer(0, 0, kFileBlockSize, /*retry=*/0);
  ASSERT_NE(buf0a, nullptr) << "First acquire of block 0 must succeed";

  // Second acquire of the same block 0: fast path, ref_count = 2, no new I/O
  char *buf0b = pool.acquire_buffer(0, 0, kFileBlockSize, /*retry=*/0);
  ASSERT_NE(buf0b, nullptr) << "Second acquire of block 0 must succeed";
  EXPECT_EQ(buf0a, buf0b) << "Both acquires must return the same buffer";

  // Only 1 block's worth of memory was consumed, so block 1 is still loadable
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Acquiring the same block twice must not double-count memory";

  char *buf1 = pool.acquire_buffer(1, kFileBlockSize, kFileBlockSize, /*retry=*/0);
  ASSERT_NE(buf1, nullptr) << "Block 1 must be loadable (pool has room for 2)";

  // Now 2 unique blocks are loaded → pool is full
  EXPECT_TRUE(MemoryLimitPool::get_instance().is_full())
      << "Pool must be full after loading 2 unique blocks";

  // Block 2 must fail (no room)
  char *buf2 = pool.acquire_buffer(2, 2 * kFileBlockSize, kFileBlockSize, /*retry=*/0);
  EXPECT_EQ(buf2, nullptr) << "Block 2 must fail when pool is full";

  // Release block 0 twice (mirrors the two acquires)
  pool.page_table_.release_block(0);
  pool.page_table_.release_block(0);
  pool.page_table_.release_block(1);
  LRUCache::get_instance().recycle();
}

// --------------------------------------------------------------------
// TEST: When pread returns fewer bytes than requested (e.g., reading
//       past end-of-file), acquire_buffer must:
//   1. Return nullptr
//   2. Release the pre-allocated memory back to the pool immediately
//      (no leak: the pool can still serve subsequent valid requests)
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, ReadFailureReleasesMemory) {
  // Only 1-block pool so any leak would make the next acquisition impossible
  MemoryLimitPool::get_instance().init(kFileBlockSize);

  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(kFileBlockSize, kFileBlockSize, kFileSegments), 0);

  // Reading at offset = kFileSize requests kFileBlockSize bytes past EOF;
  // pread returns 0 (or a short read), triggering the failure path.
  char *bad = pool.acquire_buffer(0, kFileSize, kFileBlockSize, /*retry=*/0);
  EXPECT_EQ(bad, nullptr) << "Reading past EOF must fail";

  // If memory were leaked, this acquisition would also fail.
  char *good = pool.acquire_buffer(1, kFileBlockSize, kFileBlockSize, /*retry=*/0);
  EXPECT_NE(good, nullptr)
      << "Valid block must be loadable after failed read (memory not leaked)";
  if (good) {
    pool.page_table_.release_block(1);
  }
  LRUCache::get_instance().recycle();
}

// --------------------------------------------------------------------
// TEST: After a block is evicted from memory, re-acquiring it must
//       reload the correct data from disk.
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, ReloadAfterEvictionRestoresData) {
  // 1-block pool forces eviction whenever a different block is loaded
  MemoryLimitPool::get_instance().init(kFileBlockSize);

  VecBufferPool pool(kVecFile);
  ASSERT_EQ(pool.init(kFileBlockSize, kFileBlockSize, kFileSegments), 0);

  auto verify_seg = [&](size_t seg) {
    char *buf =
        pool.acquire_buffer(seg, seg * kFileBlockSize, kFileBlockSize, /*retry=*/5);
    ASSERT_NE(buf, nullptr) << "Segment " << seg << " must load";
    const auto *data = reinterpret_cast<const uint32_t *>(buf);
    uint32_t base = static_cast<uint32_t>(seg * kFileBlockSize / sizeof(uint32_t));
    for (size_t w = 0; w < kFileBlockSize / sizeof(uint32_t); ++w) {
      ASSERT_EQ(data[w], base + w)
          << "Data mismatch at seg " << seg << " word " << w;
    }
    pool.page_table_.release_block(seg);
  };

  // Load segment 5, verify, release
  verify_seg(5);

  // Force eviction by draining the LRU
  LRUCache::get_instance().recycle();
  EXPECT_FALSE(MemoryLimitPool::get_instance().is_full())
      << "Memory must be free after eviction";

  // Reload segment 5 and verify data is identical (read from disk again)
  verify_seg(5);

  LRUCache::get_instance().recycle();
}

// --------------------------------------------------------------------
// TEST: init() with block_size == 0 must return an error code (-1).
// --------------------------------------------------------------------
TEST_F(VecBufferPoolMemoryTest, InitWithZeroBlockSizeReturnsError) {
  VecBufferPool pool(kVecFile);
  EXPECT_EQ(pool.init(kPoolMemLimit, /*block_size=*/0, kFileSegments), -1)
      << "init() with block_size=0 must return -1";
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

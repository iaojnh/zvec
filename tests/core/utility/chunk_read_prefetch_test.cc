// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <core/utility/chunk_read_storage.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <unistd.h>

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static void stop_background(MemoryLimitPool &pool) {
    pool.stop_background_evictor();
  }
};
}  // namespace zvec::ailego

namespace zvec::core {
namespace {
class ChunkReadPrefetchTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool.init(32UL << 20));
    ailego::MemoryLimitPoolTestPeer::stop_background(pool);
    ASSERT_TRUE(pool.configure_process_rss_limit(0));
    payload.resize(32 * page);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = char(i % 251);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(payload.size(), dumper->write(payload.data(), payload.size()));
    ASSERT_EQ(0, dumper->append("payload", payload.size(), 0, 0));
    ASSERT_EQ(0, dumper->close());
    storage = std::make_shared<ChunkReadStorage>();
    ailego::Params params;
    params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, 4 * page);
    params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{4});
    params.set(CHUNK_READ_STORAGE_PREFETCH_DEPTH, size_t{3});
    params.set(CHUNK_READ_STORAGE_PREFETCH_BYTES, 6 * page);
    ASSERT_EQ(0, storage->init(params));
    ASSERT_EQ(0, storage->open(path, false));
    segment = storage->get("payload");
    ASSERT_NE(nullptr, segment);
    IndexStorage::Segment::ReadChunks chunks;
    for (size_t offset = 0; offset < payload.size(); offset += page)
      chunks.push_back({offset, page});
    ASSERT_EQ(0, segment->set_read_chunks(chunks));
    if (storage->stats().prefetch_depth == 0)
      GTEST_SKIP() << "Native direct AIO unavailable; ordinary reader tests "
                      "cover fallback";
    ASSERT_EQ(3u, storage->stats().prefetch_depth);
  }
  void TearDown() override {
    segment.reset();
    storage.reset();
    EXPECT_EQ(0u, pool.evictable_cache_used());
    ailego::BlockEvictionQueue::get_instance().cleanup_dead_entries(
        std::numeric_limits<size_t>::max());
    EXPECT_EQ(0u, pool.external_used());
    EXPECT_EQ(0, pool.init(0));
    std::remove(path);
  }
  void reclaim_all() {
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    for (size_t i = 0; i < 256 && storage->stats().cache_bytes; ++i)
      queue.reclaim_for_allocation(page);
  }
  ailego::MemoryLimitPool &pool = ailego::MemoryLimitPool::get_instance();
  size_t page = ailego::MemoryLimitPool::page_buffer_size();
  const char *path = "chunk-read-prefetch-test.bin";
  std::vector<char> payload;
  std::shared_ptr<ChunkReadStorage> storage;
  IndexStorage::Segment::Pointer segment;
};

TEST_F(ChunkReadPrefetchTest, JoinsReadsAndBoundsCountAndBytes) {
  segment->prefetch(0, payload.size());
  for (size_t i = 0; i < payload.size(); i += page) {
    segment->prefetch(i, payload.size() - i);
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(64u, segment->read(i + 17, block, 64));
    ASSERT_EQ(0, std::memcmp(block.data(), payload.data() + i + 17, 64));
  }
  const auto stats = storage->stats();
  EXPECT_EQ(32u, stats.io_calls);
  EXPECT_EQ(32u, stats.prefetch_submitted);
  EXPECT_EQ(32u, stats.prefetch_consumed);
  EXPECT_EQ(0u, stats.prefetch_failed);
  EXPECT_LE(stats.prefetch_peak_requests, 3u);
  EXPECT_LE(stats.prefetch_peak_bytes, 6 * page);
  EXPECT_GT(stats.prefetch_peak_requests, 1u);
  EXPECT_EQ(0u, stats.prefetch_pending_requests);
  EXPECT_EQ(0u, stats.prefetch_pending_bytes);
}

TEST_F(ChunkReadPrefetchTest, EvictedOnceReadIsTransientThenAdmittedOnReuse) {
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read(17, block, 64));
  block.reset();
  reclaim_all();
  ASSERT_EQ(0u, storage->stats().cache_bytes);
  const auto before = storage->stats();
  segment->prefetch(0, page);
  ASSERT_EQ(64u, segment->read(17, block, 64));
  ASSERT_EQ(0, std::memcmp(block.data(), payload.data() + 17, 64));
  block.reset();
  EXPECT_EQ(0u, storage->stats().cache_bytes);
  EXPECT_EQ(before.fills, storage->stats().fills);
  segment->prefetch(0, page);
  ASSERT_EQ(64u, segment->read(17, block, 64));
  block.reset();
  EXPECT_EQ(before.fills + 1, storage->stats().fills);
  EXPECT_GT(storage->stats().cache_bytes, 0u);
}

TEST_F(ChunkReadPrefetchTest, ByteLimitIncludesPhysicalAlignmentPadding) {
  segment.reset();
  ASSERT_EQ(0, storage->close());
  ailego::Params params;
  params.set(CHUNK_READ_STORAGE_PREFETCH_BYTES, 3 * page);
  ASSERT_EQ(0, storage->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  segment = storage->get("payload");
  IndexStorage::Segment::ReadChunks chunks;
  for (size_t i = 0; i < payload.size(); i += page) chunks.push_back({i, page});
  ASSERT_EQ(0, segment->set_read_chunks(chunks));
  segment->prefetch(0, payload.size());
  const size_t alignment_padding = segment->data_offset() % page ? page : 0;
  const size_t maximum =
      std::min<size_t>(3, 3 * page / (page + alignment_padding));
  EXPECT_EQ(maximum, storage->stats().prefetch_peak_requests);
  EXPECT_LE(storage->stats().prefetch_peak_bytes, 3 * page);
}

TEST_F(ChunkReadPrefetchTest,
       WholeListLeaseRetainsTransientDataThroughAllBatches) {
  segment.reset();
  ASSERT_EQ(0, storage->close());
  ailego::Params params;
  params.set(CHUNK_READ_STORAGE_PREFETCH_BYTES, 2 * payload.size());
  params.set(CHUNK_READ_STORAGE_ADAPTIVE, true);
  ASSERT_EQ(0, storage->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  segment = storage->get("payload");
  IndexStorage::Segment::ReadChunks chunks;
  chunks.push_back({0, payload.size()});
  ASSERT_EQ(0, segment->set_read_chunks(chunks));
  IndexStorage::MemoryBlock whole;
  ASSERT_EQ(payload.size(), segment->read(0, whole, payload.size()));
  whole.reset();
  reclaim_all();
  ASSERT_EQ(0u, storage->stats().cache_bytes);
  const auto before = storage->stats();
  segment->prefetch(0, payload.size());
  ASSERT_EQ(payload.size(), segment->read(0, whole, payload.size()));
  for (size_t offset = 0; offset < payload.size(); offset += page) {
    ailego::BlockEvictionQueue::get_instance().reclaim_for_allocation(page);
    ASSERT_EQ(0, std::memcmp(static_cast<const char *>(whole.data()) + offset,
                             payload.data() + offset, page));
  }
  EXPECT_EQ(before.io_calls + 1, storage->stats().io_calls);
  EXPECT_EQ(before.fills, storage->stats().fills);
  EXPECT_EQ(0u, storage->stats().prefetch_pending_requests);
  whole.reset();
  EXPECT_EQ(0u, storage->stats().cache_bytes);
}

TEST_F(ChunkReadPrefetchTest, FullBudgetUsesExistingSynchronousWorkspace) {
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read(17, block, 64));
  block.reset();
  reclaim_all();
  ASSERT_EQ(0u, storage->stats().cache_bytes);
  const size_t remaining = pool.capacity() - pool.used();
  ASSERT_TRUE(pool.try_charge_external(remaining));
  auto refund =
      ailego::ScopeGuard::Make([&] { pool.release_external(remaining); });
  const auto before = storage->stats();
  segment->prefetch(page, page);
  ASSERT_EQ(before.prefetch_submitted, storage->stats().prefetch_submitted);
  ASSERT_EQ(64u, segment->read(page + 17, block, 64));
  EXPECT_EQ(0, std::memcmp(block.data(), payload.data() + page + 17, 64));
  EXPECT_LE(pool.used(), pool.capacity());
}

TEST_F(ChunkReadPrefetchTest, DifferentListLengthsShareChargedCapacityClasses) {
  segment.reset();
  ASSERT_EQ(0, storage->close());
  const size_t first = 5UL << 20, second = 9UL << 20;
  payload.resize(2 * (first + second));
  for (size_t i = 0; i < payload.size(); ++i) payload[i] = char(i % 251);
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(payload.size(), dumper->write(payload.data(), payload.size()));
  ASSERT_EQ(0, dumper->append("payload", payload.size(), 0, 0));
  ASSERT_EQ(0, dumper->close());
  ailego::Params params;
  params.set(CHUNK_READ_STORAGE_ADAPTIVE, true);
  params.set(CHUNK_READ_STORAGE_PREFETCH_BYTES, size_t{24UL << 20});
  params.set(CHUNK_READ_STORAGE_PREFETCH_DEPTH, size_t{2});
  ASSERT_EQ(0, storage->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  segment = storage->get("payload");
  IndexStorage::Segment::ReadChunks chunks;
  chunks.push_back({0, first});
  chunks.push_back({first, second});
  chunks.push_back({first + second, first});
  chunks.push_back({2 * first + second, second});
  ASSERT_EQ(0, segment->set_read_chunks(chunks));
  segment->prefetch(0, payload.size());
  IndexStorage::MemoryBlock a, b;
  ASSERT_EQ(first, segment->read(0, a, first));
  ASSERT_EQ(second, segment->read(first, b, second));
  const void *pa = a.data(), *pb = b.data();
  EXPECT_NE(pa, pb);  // a live chain's backing cannot be recycled
  EXPECT_EQ(0, std::memcmp(pa, payload.data(), first));
  EXPECT_EQ(0, std::memcmp(pb, payload.data() + first, second));
  const size_t capacity = second + (segment->data_offset() % page ? page : 0);
  EXPECT_EQ(2 * capacity, storage->stats().prefetch_peak_bytes);
  EXPECT_EQ(2 * capacity, storage->stats().cache_bytes);
  EXPECT_EQ(1u, storage->stats().prefetch_submit_batches);
  a.reset();
  b.reset();
  // A fourth buffer would exceed the 32MiB pool. The next pair must replace
  // unpinned payload without reusing either of its own live reader buffers.
  segment->prefetch(first + second, first + second);
  ASSERT_EQ(first, segment->read(first + second, a, first));
  ASSERT_EQ(second, segment->read(2 * first + second, b, second));
  const void *pc = a.data(), *pd = b.data();
  EXPECT_NE(pc, pd);
  EXPECT_EQ(0, std::memcmp(pc, payload.data() + first + second, first));
  EXPECT_EQ(0, std::memcmp(pd, payload.data() + 2 * first + second, second));
  EXPECT_LE(pool.used(), pool.capacity());
  a.reset();
  b.reset();
  reclaim_all();
  ASSERT_EQ(0u, storage->stats().cache_bytes);
  segment->prefetch(0, first);
  ASSERT_EQ(first, segment->read(0, a, first));
  EXPECT_TRUE(a.data() == pa || a.data() == pb || a.data() == pc ||
              a.data() == pd);
  EXPECT_EQ(0, std::memcmp(a.data(), payload.data(), first));
  a.reset();
  EXPECT_EQ(capacity, storage->stats().cache_bytes);
}

TEST_F(ChunkReadPrefetchTest, ConcurrentOverlappingHintsPreserveReaders) {
  std::atomic<bool> correct{true};
  std::atomic<bool> finished{false};
  std::thread reclaimer([&] {
    while (!finished.load()) {
      ailego::BlockEvictionQueue::get_instance().reclaim_for_allocation(page);
      std::this_thread::yield();
    }
  });
  std::vector<std::thread> readers;
  for (size_t t = 0; t < 8; ++t)
    readers.emplace_back([&, t] {
      for (size_t i = 0; i < 64; ++i) {
        const size_t offset = ((i + t) % 32) * page;
        segment->prefetch(offset, std::min(3 * page, payload.size() - offset));
        IndexStorage::MemoryBlock block;
        if (segment->read(offset + 17, block, 64) != 64) {
          correct = false;
          return;
        }
        auto copy = block;
        block.reset();
        if (std::memcmp(copy.data(), payload.data() + offset + 17, 64))
          correct = false;
      }
    });
  for (auto &reader : readers) reader.join();
  finished = true;
  reclaimer.join();
  EXPECT_TRUE(correct.load());
  EXPECT_LE(storage->stats().prefetch_peak_requests, 3u);
  EXPECT_LE(storage->stats().prefetch_peak_bytes, 6 * page);
}

TEST_F(ChunkReadPrefetchTest, RetainedReadAndPendingIoSurviveClose) {
  segment->prefetch(0, 3 * page);
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read(17, block, 64));
  auto retained = block;
  block.reset();
  segment.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  EXPECT_EQ(0, std::memcmp(retained.data(), payload.data() + 17, 64));
  retained.reset();
  EXPECT_EQ(0u, pool.evictable_cache_used());
}

TEST_F(ChunkReadPrefetchTest, ShortReadReleasesBackingAndReportsFailure) {
  ASSERT_EQ(0, ::truncate(path, 0));
  segment->prefetch(0, page);
  IndexStorage::MemoryBlock block;
  EXPECT_EQ(0u, segment->read(17, block, 64));
  EXPECT_GT(storage->stats().prefetch_failed, 0u);
  EXPECT_EQ(0u, storage->stats().cache_bytes);
}
}  // namespace
}  // namespace zvec::core
#endif

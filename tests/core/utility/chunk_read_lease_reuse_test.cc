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
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "utility/utility_params.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static void stop_background(MemoryLimitPool &pool) {
    pool.stop_background_evictor();
  }
  static ProcessRssBudget::ResidentSample resident(bool background) {
    return background ? MemoryLimitPool::read_background_resident_sample()
                      : MemoryLimitPool::read_slab_resident_sample();
  }
};
}  // namespace zvec::ailego
namespace zvec::core {
namespace {
class ChunkReadLeaseReuseTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool.init(32UL << 20));
    ailego::MemoryLimitPoolTestPeer::stop_background(pool);
    ASSERT_TRUE(pool.configure_process_rss_limit(0));
    payload.resize(8 * page);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = char(i % 251);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(payload.size(), dumper->write(payload.data(), payload.size()));
    ASSERT_EQ(0, dumper->append("payload", payload.size(), 0, 0));
    ASSERT_EQ(0, dumper->close());
    storage = std::make_shared<ChunkReadStorage>();
    ailego::Params params;
    params.set(CHUNK_READ_STORAGE_CHUNK_BYTES, page);
    params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, 4 * page);
    params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{2});
    params.set(CHUNK_READ_STORAGE_DIRECT_IO, false);
    ASSERT_EQ(0, storage->init(params));
    ASSERT_EQ(0, storage->open(path, false));
    segment = storage->get("payload");
    ASSERT_NE(nullptr, segment);
  }
  void TearDown() override {
    segment.reset();
    storage.reset();
    EXPECT_EQ(0u, pool.evictable_cache_used());
    ailego::BlockEvictionQueue::get_instance().cleanup_dead_entries(
        std::numeric_limits<size_t>::max());
    EXPECT_EQ(0, pool.init(0));
    std::remove(path);
  }
  void check(const IndexStorage::MemoryBlock &block, size_t offset,
             size_t size) {
    EXPECT_EQ(0, std::memcmp(block.data(), payload.data() + offset, size));
  }
  ailego::MemoryLimitPool &pool = ailego::MemoryLimitPool::get_instance();
  size_t page = ailego::MemoryLimitPool::page_buffer_size();
  const char *path = "chunk-read-lease-reuse-test.bin";
  std::vector<char> payload;
  std::shared_ptr<ChunkReadStorage> storage;
  IndexStorage::Segment::Pointer segment;
};
TEST_F(ChunkReadLeaseReuseTest, WarmHitReusesChargedDescriptorAtFullBudget) {
  IndexStorage::MemoryBlock block, other;
  ASSERT_EQ(64u, segment->read(32, block, 64));
  ASSERT_EQ(64u, segment->read(page + 48, other, 64));
  block.reset();
  other.reset();
  const size_t remaining = pool.capacity() - pool.used();
  ASSERT_TRUE(pool.try_charge_external(remaining));
  auto refund =
      ailego::ScopeGuard::Make([&] { pool.release_external(remaining); });
  const auto io_calls = storage->stats().io_calls;
  for (size_t i = 0; i < 16; ++i) {
    ASSERT_EQ(64u, segment->read(32, block, 64));
    ASSERT_EQ(64u, segment->read(page + 48, other, 64));
    check(block, 32, 64);
    check(other, page + 48, 64);
    block.reset();
    other.reset();
  }
  EXPECT_EQ(io_calls, storage->stats().io_calls);
}
TEST_F(ChunkReadLeaseReuseTest, CopiesAndMovesSurviveReuseAndStorageClose) {
  IndexStorage::MemoryBlock first, second;
  ASSERT_EQ(64u, segment->read(32, first, 64));
  auto retained = first;
  first.reset();
  ASSERT_EQ(64u, segment->read(page + 48, second, 64));
  auto moved = std::move(second);
  auto copy = moved;
  moved.reset();
  segment.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  check(retained, 32, 64);
  check(copy, page + 48, 64);
  retained.reset();
  copy.reset();
}

TEST_F(ChunkReadLeaseReuseTest, ReadReportsBudgetDenialAndCanRetry) {
  IndexStorage::Segment::ScatterBlock scatter;
  {
    const size_t remaining = pool.capacity() - pool.used();
    ASSERT_TRUE(pool.try_charge_external(remaining));
    auto refund =
        ailego::ScopeGuard::Make([&] { pool.release_external(remaining); });

    EXPECT_EQ(0u, segment->read_scatter(page - 32, scatter, 64));
    EXPECT_EQ(IndexError_NoMemory, segment->read_error());
    EXPECT_TRUE(scatter.spans.empty());
    EXPECT_EQ(nullptr, scatter.lease);
  }

  ASSERT_EQ(64u, segment->read_scatter(page - 32, scatter, 64));
  size_t copied = 0;
  for (const auto &span : scatter.spans) {
    EXPECT_EQ(0, std::memcmp(span.data, payload.data() + page - 32 + copied,
                             span.size));
    copied += span.size;
  }
  EXPECT_EQ(64u, copied);
}
TEST_F(ChunkReadLeaseReuseTest, CachedDescriptorDoesNotRetainReaderPins) {
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read(32, block, 64));
  auto retained = block;
  block.reset();
  auto &queue = ailego::BlockEvictionQueue::get_instance();
  for (size_t i = 0; i < 8; ++i) queue.reclaim_for_allocation(page);
  EXPECT_GT(storage->stats().cache_bytes, 0u);
  check(retained, 32, 64);
  retained.reset();
  for (size_t i = 0; i < 16 && storage->stats().cache_bytes; ++i)
    queue.reclaim_for_allocation(page);
  EXPECT_EQ(0u, storage->stats().cache_bytes);
}
TEST_F(ChunkReadLeaseReuseTest, ConcurrentReadersUseIndependentLeases) {
  std::atomic<bool> correct{true};
  std::vector<std::thread> readers;
  for (size_t t = 0; t < 8; ++t)
    readers.emplace_back([&, t] {
      for (size_t i = 0; i < 64; ++i) {
        const size_t offset = (t % 4) * page + (i * 17) % 256;
        IndexStorage::MemoryBlock block;
        if (segment->read(offset, block, 64) != 64) {
          correct = false;
          return;
        }
        auto retained = block;
        block.reset();
        if (std::memcmp(retained.data(), payload.data() + offset, 64))
          correct = false;
      }
    });
  for (auto &reader : readers) reader.join();
  EXPECT_TRUE(correct.load());
}

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST_F(ChunkReadLeaseReuseTest, WorkspaceWitnessFollowsEscapedReadAndUnmap) {
  segment.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  storage = std::make_shared<ChunkReadStorage>();
  ailego::Params params;
  params.set(CHUNK_READ_STORAGE_CACHE_BYTES, size_t{0});
  params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, 4 * page);
  params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{2});
  params.set(CHUNK_READ_STORAGE_DIRECT_IO, false);
  ASSERT_EQ(0, storage->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  segment = storage->get("payload");
  ASSERT_NE(nullptr, segment);
  // Hold the first workspace busy so the measured read uses a separate,
  // untouched workspace, independently of open-time metadata parsing.
  IndexStorage::MemoryBlock preceding;
  ASSERT_EQ(1u, segment->read(0, preceding, 1));
  const auto before = ailego::MemoryLimitPoolTestPeer::resident(false);
  const auto background_before =
      ailego::MemoryLimitPoolTestPeer::resident(true);
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(4 * page, segment->read(0, block, 4 * page));
  check(block, 0, 4 * page);
  const auto touched = ailego::MemoryLimitPoolTestPeer::resident(false);
  const auto background = ailego::MemoryLimitPoolTestPeer::resident(true);
  EXPECT_GE(touched.owned_resident, before.owned_resident + 4 * page);
  EXPECT_GE(background.owned_resident,
            background_before.owned_resident + 4 * page);
  EXPECT_EQ(0u, storage->stats().cache_bytes);
  preceding.reset();
  segment.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  // The escaped block still owns the workspace after its storage is closed.
  check(block, 0, 4 * page);
  EXPECT_GE(ailego::MemoryLimitPoolTestPeer::resident(false).owned_resident,
            4 * page);
  const size_t external = pool.external_used();
  block.reset();
  EXPECT_LT(pool.external_used(), external);
  EXPECT_LE(ailego::MemoryLimitPoolTestPeer::resident(false).owned_resident +
                4 * page,
            touched.owned_resident);
  EXPECT_LE(
      ailego::MemoryLimitPoolTestPeer::resident(true).owned_resident + 4 * page,
      background.owned_resident);
}
#endif

class MetadataAdmissionTest : public testing::Test {
 protected:
  class Pressure {
   public:
    explicit Pressure(size_t remaining) {
      auto &pool = ailego::MemoryLimitPool::get_instance();
      const size_t used = pool.used();
      char *sample = allocator_.allocate(1);
      const size_t header = pool.used() - used - 1;
      allocator_.deallocate(sample, 1);
      if (pool.available() < remaining + header) throw std::bad_alloc();
      bytes_ = pool.available() - remaining - header;
      data_ = allocator_.allocate(bytes_);
      std::memset(data_, 'p', bytes_);
    }
    ~Pressure() {
      allocator_.deallocate(data_, bytes_);
    }

   private:
    ailego::BudgetedAllocator<char> allocator_;
    char *data_{nullptr};
    size_t bytes_{0};
  };
  void SetUp() override {
    ASSERT_EQ(0, pool.init(32UL << 20));
    ailego::MemoryLimitPoolTestPeer::stop_background(pool);
    ASSERT_TRUE(pool.configure_process_rss_limit(0));
  }
  void TearDown() override {
    ailego::BlockEvictionQueue::get_instance().cleanup_dead_entries(
        std::numeric_limits<size_t>::max());
    EXPECT_EQ(0u, pool.external_used());
    EXPECT_EQ(0, pool.init(0));
    for (const auto &file : files) std::remove(file.c_str());
  }
  void make_file(size_t count, size_t name_bytes, std::string &path,
                 std::vector<std::string> &names) {
    path = "metadata-admission-" + std::to_string(files.size()) + ".bin";
    files.push_back(path);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_TRUE(dumper);
    ASSERT_EQ(0, dumper->create(path));
    const char value = 'v';
    names.clear();
    for (size_t i = 0; i < count; ++i) {
      names.push_back(std::to_string(i) + '-' + std::string(name_bytes, 'n'));
      ASSERT_EQ(1u, dumper->write(&value, 1));
      ASSERT_EQ(0, dumper->append(names.back(), 1, 0, 0));
    }
    ASSERT_EQ(0, dumper->close());
  }
  IndexStorage::Pointer create(const char *backend, bool alone = false,
                               size_t query_bytes = 64UL << 10) {
    auto storage = IndexFactory::CreateStorage(backend);
    if (!storage) return {};
    ailego::Params params;
    params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, query_bytes);
    params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{1});
    params.set(CHUNK_READ_STORAGE_CACHE_BYTES, size_t{0});
    params.set(CHUNK_READ_STORAGE_DIRECT_IO, false);
    params.set(FILE_READ_STORAGE_ALONE_FILE_HANDLE, alone);
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE,
               BUFFER_READ_STORAGE_WARMUP_NONE);
    EXPECT_EQ(0, storage->init(params));
    return storage;
  }
  void check(IndexStorage &storage, const std::string &name) {
    auto segment = storage.get(name, 1);
    ASSERT_TRUE(segment);
    char value = 0;
    ASSERT_EQ(1u, segment->fetch(0, &value, 1));
    EXPECT_EQ('v', value);
  }
  ailego::MemoryLimitPool &pool = ailego::MemoryLimitPool::get_instance();
  std::vector<std::string> files;
};

TEST_F(MetadataAdmissionTest, ManySegmentsAndLongNamesUseSharedBudget) {
  const struct Shape {
    size_t count, name_bytes, query_bytes;
  } shapes[] = {
      {300, 8, 64UL << 10},
      {3, 80UL << 10, ailego::MemoryLimitPool::page_buffer_size()},
      {1, 300UL << 10, ailego::MemoryLimitPool::page_buffer_size()},
  };
  for (const auto &shape : shapes) {
    std::string path;
    std::vector<std::string> names;
    make_file(shape.count, shape.name_bytes, path, names);
    ASSERT_EQ(shape.count, names.size());
    for (const char *backend : {"ChunkReadStorage", "BufferReadStorage",
                                "FileReadStorage", "MMapFileReadStorage"}) {
      SCOPED_TRACE(::testing::Message()
                   << backend << ':' << shape.count << ':' << shape.name_bytes);
      auto storage = create(backend, false, shape.query_bytes);
      ASSERT_TRUE(storage);
      const int result = storage->open(path, false);
      EXPECT_EQ(0, result);
      if (result != 0) continue;
      EXPECT_EQ(path, storage->file_path());
      EXPECT_EQ(shape.count + 1, storage->get_all().size());
      check(*storage, names.front());
      check(*storage, names.back());
      EXPECT_LE(pool.used(), pool.capacity());
      EXPECT_EQ(0, storage->close());
    }
  }
}

TEST_F(MetadataAdmissionTest, RealBudgetDenialPreservesDirectoryAndCanRetry) {
  std::string small, large;
  std::vector<std::string> old_names, new_names;
  make_file(1, 3, small, old_names);
  make_file(6000, 8, large, new_names);
  for (const char *backend : {"FileReadStorage", "ChunkReadStorage",
                              "BufferReadStorage", "MMapFileReadStorage"}) {
    for (size_t remaining : {size_t{0}, size_t{512} << 10}) {
      SCOPED_TRACE(::testing::Message() << backend << ':' << remaining);
      auto storage = create(backend);
      ASSERT_TRUE(storage);
      ASSERT_EQ(0, storage->open(small, false));
      check(*storage, old_names.front());
      auto old_segment = storage->get(old_names.front(), 1);
      ASSERT_TRUE(old_segment);
      IndexStorage::Segment::ReadChunks ranges{{0, 1}};
      std::shared_ptr<IndexSegmentStorage::Segment> wrapper;
      if (old_segment->supports_read_chunks() && !remaining)
        wrapper = std::make_shared<IndexSegmentStorage::Segment>(old_segment, 0,
                                                                 1, 0, 0);
      {
        Pressure pressure(remaining);
        ASSERT_EQ(remaining, pool.available());
        const auto used = pool.used();
        EXPECT_EQ(IndexError_NoMemory, storage->open(large, false));
        // Admission may reclaim old cache pages, but failed parsing must not
        // retain any new allocation. The published directory stays usable.
        EXPECT_LE(pool.used(), used);
        EXPECT_TRUE(storage->has(old_names.front()));
        EXPECT_FALSE(storage->has(new_names.back()));
        EXPECT_EQ(small, storage->file_path());
        if (wrapper) {
          int error = 0;
          EXPECT_NO_THROW(error = wrapper->set_read_chunks(ranges));
          EXPECT_EQ(IndexError_NoMemory, error);
        }
      }
      check(*storage, old_names.front());
      char value = 0;
      ASSERT_EQ(1u, old_segment->fetch(0, &value, 1));
      EXPECT_EQ('v', value);
      if (wrapper) EXPECT_EQ(0, wrapper->set_read_chunks(ranges));
      ASSERT_EQ(0, storage->open(large, false));
      check(*storage, new_names.back());
      EXPECT_EQ(large, storage->file_path());
      EXPECT_LE(pool.used(), pool.capacity());
      EXPECT_EQ(0, storage->close());
    }
  }
  // The separate-handle mode must preserve its path/directory too.
  auto storage = create("FileReadStorage", true);
  ASSERT_EQ(0, storage->open(small, false));
  {
    Pressure pressure(0);
    EXPECT_EQ(IndexError_NoMemory, storage->open(large, false));
    EXPECT_TRUE(storage->has(old_names.front()));
  }
  check(*storage, old_names.front());
  ASSERT_EQ(0, storage->open(large, false));
  check(*storage, new_names.back());
}

TEST_F(MetadataAdmissionTest,
       VersionMetadataAllocationIsAdmittedAndRolledBack) {
  std::string path;
  std::vector<std::string> names;
  make_file(1, 3, path, names);
  ailego::File file;
  ASSERT_TRUE(file.open(path, true, false));
  std::string image(file.size(), '\0');
  ASSERT_EQ(image.size(), file.read(0, image.data(), image.size()));
  file.close();
  const auto read_image = [&](size_t offset, const void **data, size_t size) {
    if (offset > image.size() || size > image.size() - offset) return size_t{0};
    *data = image.data() + offset;
    return size;
  };
  size_t version_offset = 0, version_size = 0;
  {
    IndexUnpacker parsed;
    ASSERT_TRUE(parsed.unpack(read_image, image.size(), false));
    const auto version = parsed.segments().find("IndexVersion");
    ASSERT_NE(parsed.segments().end(), version);
    version_offset = version->second.data_offset();
    version_size = version->second.data_size();
  }
  ASSERT_GT(version_size, std::string{}.capacity());
  const size_t before = pool.used();
  std::unique_ptr<Pressure> pressure;
  {
    IndexUnpacker denied;
    const auto read = [&](size_t offset, const void **data, size_t size) {
      const auto got = read_image(offset, data, size);
      if (offset == version_offset && size == version_size)
        pressure = std::make_unique<Pressure>(0);
      return got;
    };
    EXPECT_THROW(denied.unpack(read, image.size(), false), std::bad_alloc);
  }
  pressure.reset();
  EXPECT_EQ(before, pool.used());
  IndexUnpacker retry;
  ASSERT_TRUE(retry.unpack(read_image, image.size(), false));
  EXPECT_EQ(version_size, retry.version().size());
}
}  // namespace
}  // namespace zvec::core

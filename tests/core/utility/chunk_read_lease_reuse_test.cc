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
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_memory.h>
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
TEST_F(ChunkReadLeaseReuseTest, CopiedResultsReleaseOnlySourcePins) {
  segment.reset();
  ASSERT_EQ(0, storage->close());
  ailego::Params params;
  params.set(CHUNK_READ_STORAGE_CACHE_BYTES, page);
  params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, 4 * page);
  params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{2});
  params.set(CHUNK_READ_STORAGE_DIRECT_IO, true);
  ASSERT_EQ(0, storage->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  segment = storage->get("payload");
  IndexStorage::Segment::ReadChunks chunks{{0, 100}, {100, 100}, {200, 100}};
  ASSERT_EQ(0, segment->set_read_chunks(chunks));
  IndexStorage::MemoryBlock borrowed, copied;
  ASSERT_EQ(100u, segment->read(0, borrowed, 100));
  const auto io_calls = storage->stats().io_calls;
  // One cached source and two cold, unaligned chunks. The misses still form
  // one I/O range; copying must preserve the cached prefix and its live view.
  ASSERT_EQ(300u, segment->read(0, copied, 300));
  EXPECT_EQ(io_calls + 1, storage->stats().io_calls);
  auto retained = copied;
  copied.reset();
  auto &queue = ailego::BlockEvictionQueue::get_instance();
  for (size_t i = 0; i < 16; ++i) queue.reclaim_for_allocation(page);
  EXPECT_EQ(page, storage->stats().cache_bytes);
  check(borrowed, 0, 100);
  borrowed.reset();
  for (size_t i = 0; i < 32 && storage->stats().cache_bytes; ++i)
    queue.reclaim_for_allocation(page);
  EXPECT_EQ(0u, storage->stats().cache_bytes);
  check(retained, 0, 300);
  segment.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  check(retained, 0, 300);
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
  // Inject a failed metadata read while all successful reads still use the
  // real backing storage. Returned child segments clone that real backing.
  class FailingParent : public IndexStorage::Segment {
   public:
    explicit FailingParent(Pointer parent) : parent_(std::move(parent)) {}
    size_t data_size() const override {
      return parent_->data_size();
    }
    size_t padding_size() const override {
      return parent_->padding_size();
    }
    size_t capacity() const override {
      return parent_->capacity();
    }
    uint32_t data_crc() const override {
      return parent_->data_crc();
    }
    int read_error() const override {
      return failed_ ? IndexError_ReadData : parent_->read_error();
    }
    size_t fetch(size_t offset, void *data, size_t size) const override {
      failed_ = calls++ >= fail_at;
      return failed_ ? 0 : parent_->fetch(offset, data, size);
    }
    size_t read(size_t offset, IndexStorage::MemoryBlock &data,
                size_t size) override {
      failed_ = calls++ >= fail_at;
      return failed_ ? 0 : parent_->read(offset, data, size);
    }
    size_t read_for_parsing(size_t offset, IndexStorage::MemoryBlock &data,
                            size_t size) override {
      failed_ = calls++ >= fail_at;
      return failed_ ? 0 : parent_->read_for_parsing(offset, data, size);
    }
    size_t write(size_t, const void *, size_t) override {
      return 0;
    }
    size_t resize(size_t) override {
      return capacity();
    }
    void update_data_crc(uint32_t) override {}
    Pointer clone() override {
      return parent_->clone();
    }
    mutable size_t calls{0};
    size_t fail_at{std::numeric_limits<size_t>::max()};

   private:
    Pointer parent_;
    mutable bool failed_{false};
  };
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
    for (const auto &file : files) {
      IndexMemory::Instance()->remove(file);
      std::remove(file.c_str());
    }
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
  void make_nested_file(size_t count, size_t name_bytes, std::string &path,
                        std::vector<std::string> &names, size_t prefix = 0,
                        bool twice = true, bool mirror_memory = true) {
    path = "metadata-admission-" + std::to_string(files.size()) + ".bin";
    files.push_back(path);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_TRUE(dumper);
    ASSERT_EQ(0, dumper->create(path));
    if (prefix) {
      const std::string padding(prefix, 'p');
      ASSERT_EQ(prefix, dumper->write(padding.data(), prefix));
      ASSERT_EQ(0, dumper->append("prefix", prefix, 0, 0));
    }
    auto middle = std::make_shared<IndexSegmentDumper>(dumper, "nested");
    auto leaf = twice ? std::make_shared<IndexSegmentDumper>(middle, "nested")
                      : middle;
    names.clear();
    for (size_t i = 0; i < count; ++i) {
      names.push_back(std::to_string(i) + '-' + std::string(name_bytes, 'n'));
      const char value = 'v';
      ASSERT_EQ(1u, leaf->write(&value, 1));
      ASSERT_EQ(0, leaf->append(names.back(), 1, 0, 0));
    }
    ASSERT_EQ(0, leaf->close());
    if (twice) ASSERT_EQ(0, middle->close());
    ASSERT_EQ(0, dumper->close());
    if (!mirror_memory) return;
    // The same package also exercises the in-memory parent implementation.
    ailego::File file;
    ASSERT_TRUE(file.open(path, true, false));
    std::string image(file.size(), '\0');
    ASSERT_EQ(image.size(), file.read(0, image.data(), image.size()));
    auto rope = IndexMemory::Instance()->create(path);
    ASSERT_TRUE(rope);
    ASSERT_EQ(image.size(), rope->append(0).append(image.data(), image.size()));
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

TEST_F(MetadataAdmissionTest, NestedDirectoryAdmissionRollbackAndRetry) {
  for (const auto shape :
       {std::pair<size_t, size_t>{300, 8}, {3, 80UL << 10}, {1, 300UL << 10}}) {
    std::string path;
    std::vector<std::string> names;
    make_nested_file(shape.first, shape.second, path, names);
    for (const char *backend :
         {"ChunkReadStorage", "FileReadStorage", "BufferReadStorage",
          "MMapFileReadStorage", "MemoryReadStorage"}) {
      SCOPED_TRACE(::testing::Message()
                   << backend << ':' << shape.first << ':' << shape.second);
      const size_t page = ailego::MemoryLimitPool::page_buffer_size();
      auto storage = create(backend, false, page);
      ASSERT_TRUE(storage);
      ASSERT_EQ(0, storage->open(path, false));
      IndexSegmentStorage middle(storage->get("nested", 1));
      ASSERT_EQ(0, middle.open("", false));
      auto parent = std::make_shared<FailingParent>(middle.get("nested", 1));
      IndexSegmentStorage leaf(parent);
      ASSERT_EQ(0, leaf.open("", false));
      EXPECT_EQ(shape.first + 1, leaf.get_all().size());
      check(leaf, names.back());
      const size_t directory_bytes = pool.external_used();
      for (size_t remaining : {size_t{0}, page}) {
        SCOPED_TRACE(remaining);
        {
          Pressure pressure(remaining);
          const auto used = pool.external_used();
          EXPECT_EQ(IndexError_NoMemory, leaf.open("", false));
          EXPECT_EQ(used, pool.external_used());
          EXPECT_TRUE(leaf.has(names.back()));
        }
        EXPECT_EQ(directory_bytes, pool.external_used());
        check(leaf, names.front());
        ASSERT_EQ(0, leaf.open("", false));
      }
      // Cover every read attempt, including any bounded-read fallback.
      // A persistent I/O failure must not publish a partial replacement.
      parent->calls = 0;
      ASSERT_EQ(0, leaf.open("", false));
      const size_t attempts = parent->calls;
      for (size_t fail_at = 0; fail_at < attempts; ++fail_at) {
        SCOPED_TRACE(fail_at);
        parent->calls = 0;
        parent->fail_at = fail_at;
        const auto used = pool.external_used();
        EXPECT_EQ(IndexError_UnpackIndex, leaf.open("", false));
        EXPECT_EQ(used, pool.external_used());
        EXPECT_TRUE(leaf.has(names.back()));
        parent->fail_at = std::numeric_limits<size_t>::max();
        check(leaf, names.back());
        ASSERT_EQ(0, leaf.open("", false));
      }
      auto retained = leaf.get(names.back(), 1);
      ASSERT_TRUE(retained);
      EXPECT_EQ(0, leaf.close());
      EXPECT_EQ(0, middle.close());
      EXPECT_EQ(0, storage->close());
      char value = 0;
      ASSERT_EQ(1u, retained->fetch(0, &value, 1));
      EXPECT_EQ('v', value);
    }
  }
}

TEST_F(MetadataAdmissionTest, NestedDirectReadsPreserveTightAdmission) {
  const struct Shape {
    size_t count, name_bytes, quota;
  } shapes[] = {{1, 3, 1024}, {3, 80UL << 10, 320UL << 10}};
  for (const auto &shape : shapes) {
    std::string path;
    std::vector<std::string> names;
    make_nested_file(shape.count, shape.name_bytes, path, names);
    for (const char *backend : {"MMapFileReadStorage", "MemoryReadStorage"}) {
      SCOPED_TRACE(::testing::Message() << backend << ':' << shape.quota);
      auto storage = create(backend);
      ASSERT_TRUE(storage);
      ASSERT_EQ(0, storage->open(path, false));
      IndexSegmentStorage middle(storage->get("nested", 1));
      ASSERT_EQ(0, middle.open("", false));
      IndexSegmentStorage leaf(middle.get("nested", 1));
      for (size_t attempt = 0; attempt < 2; ++attempt) {
        {
          // The directory fits, but a mandatory page/directory copy does not.
          // Verify both the first publication and a transactional reopen.
          Pressure pressure(shape.quota);
          ASSERT_EQ(shape.quota, pool.available());
          ASSERT_EQ(0, leaf.open("", false));
          EXPECT_TRUE(leaf.has(names.back()));
        }
        check(leaf, names.front());
        check(leaf, names.back());
      }
    }
  }
}

TEST_F(MetadataAdmissionTest, CrossPageParserPreservesFetchAdmission) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  std::string path;
  std::vector<std::string> names;
  // The leaf header crosses a page; all package/version bytes are produced
  // by the real dumpers. A long name makes later directory admission tight.
  make_nested_file(1, 300UL << 10, path, names,
                   page - 16 - 2 * sizeof(IndexFormat::MetaHeader));
  auto trial = [&](size_t quota, bool parser) {
    auto storage = create("BufferReadStorage");
    EXPECT_EQ(0, storage->open(path, false));
    IndexSegmentStorage middle(storage->get("nested", 1));
    EXPECT_EQ(0, middle.open("", false));
    auto parent = middle.get("nested", 1);
    int result = IndexError_NoMemory;
    try {
      Pressure pressure(quota);
      EXPECT_EQ(quota, pool.available());
      if (parser) {
        IndexSegmentStorage leaf(parent);
        result = leaf.open("", false);
        if (result == 0) EXPECT_TRUE(leaf.has(names.back()));
      } else {
        // Reference the earlier streamed-copy success contract, without
        // assuming allocator headers, native page size or version length.
        ailego::BudgetedMappedBuffer copy;
        IndexUnpacker reference;
        const auto fetch = [&](size_t offset, const void **data, size_t len) {
          if (copy.size() < len && copy.resize(len) != 0) return size_t{0};
          *data = copy.data();
          return parent->fetch(offset, copy.data(), len);
        };
        if (reference.unpack(fetch, parent->data_size(), false)) {
          EXPECT_NE(reference.segments().end(),
                    reference.segments().find(names.back()));
          result = 0;
        }
      }
    } catch (const std::bad_alloc &) {
    }
    return result;
  };
  // Initialize shared registries equally, then find the real reference
  // boundary. Every trial constructs a fresh parent and releases it fully.
  size_t low = 1, high = 2UL << 20;
  ASSERT_EQ(0, trial(high, false));
  while (low < high) {
    const size_t quota = low + (high - low) / 2;
    if (trial(quota, false) == 0)
      high = quota;
    else
      low = quota + 1;
  }
  ASSERT_EQ(0, trial(low, false));
  EXPECT_EQ(0, trial(low, true));
}

TEST_F(MetadataAdmissionTest, ChunkParserPreservesSuccessfulBudgets) {
  ASSERT_EQ(0, pool.init(128UL << 20));
  ailego::MemoryLimitPoolTestPeer::stop_background(pool);
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  const struct Shape {
    bool adaptive, warm, funded;
    size_t window, cache, count, name_bytes;
    size_t chunk_bytes{0};
  } shapes[] = {
      {true, false, false, page, 0, 1, 32UL << 20},
      {true, false, false, page, 8UL << 20, 1, 2UL << 20},
      {true, true, false, page, 8UL << 20, 1, 2UL << 20},
      {false, false, false, 8UL << 20, 8UL << 20, 1, 2UL << 20},
      {false, true, false, 8UL << 20, 8UL << 20, 1, 2UL << 20},
      {true, false, false, page, 8UL << 20, 512, 4096},
      {true, false, true, page, 0, 1, 2UL << 20},
      {false, true, false, 8UL << 20, 8UL << 20, 1, 2UL << 20,
       4UL << 20}};
  for (const auto &shape : shapes) {
    SCOPED_TRACE(::testing::Message()
                 << shape.adaptive << ':' << shape.warm << ':' << shape.funded
                 << ':' << shape.window << ':' << shape.cache << ':'
                 << shape.count << ':' << shape.name_bytes << ':'
                 << shape.chunk_bytes);
    std::string path;
    std::vector<std::string> names;
    // One level keeps all cold/warm preparation explicit, before pressure.
    make_nested_file(shape.count, shape.name_bytes, path, names, 0, false,
                     false);
    auto trial = [&](size_t quota, bool parser) {
      auto storage = std::make_shared<ChunkReadStorage>();
      ailego::Params params;
      params.set(CHUNK_READ_STORAGE_ADAPTIVE, shape.adaptive);
      params.set(CHUNK_READ_STORAGE_CHUNK_BYTES,
                 shape.chunk_bytes ? shape.chunk_bytes : page);
      params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, shape.window);
      params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{1});
      params.set(CHUNK_READ_STORAGE_CACHE_BYTES, shape.cache);
      params.set(CHUNK_READ_STORAGE_DIRECT_IO, false);
      EXPECT_EQ(0, storage->init(params));
      EXPECT_EQ(0, storage->open(path, false));
      auto parent = storage->get("nested", 1);
      if (shape.warm) {
        std::string data(parent->data_size(), '\0');
        EXPECT_EQ(data.size(), parent->fetch(0, data.data(), data.size()));
      }
      if (shape.funded) {
        IndexStorage::MemoryBlock data;
        EXPECT_EQ(parent->data_size(),
                  parent->read(0, data, parent->data_size()));
      }
      const auto workspace = storage->stats().workspace_bytes;
      int result = IndexError_NoMemory;
      {
        Pressure pressure(quota);
        EXPECT_EQ(quota, pool.available());
        try {
          if (parser) {
            IndexSegmentStorage nested(parent);
            result = nested.open("", false);
            if (result == 0) EXPECT_TRUE(nested.has(names.back()));
          } else {
            ailego::BudgetedMappedBuffer copy;
            IndexUnpacker reference;
            const auto fetch = [&](size_t offset, const void **data, size_t len) {
              if (copy.size() < len && copy.resize(len) != 0) return size_t{0};
              *data = copy.data();
              return parent->fetch(offset, copy.data(), len);
            };
            if (reference.unpack(fetch, parent->data_size(), false)) {
              EXPECT_NE(reference.segments().end(),
                        reference.segments().find(names.back()));
              result = 0;
            }
          }
        } catch (const std::bad_alloc &) {
        }
      }
      if (shape.funded) EXPECT_EQ(workspace, storage->stats().workspace_bytes);
      return result;
    };
    if (shape.funded) {
      // Only the private directory fits. Reusing its funded input succeeds;
      // unconditionally making another copy would reject the same operation.
      const size_t quota = names.back().size() + 8 * page;
      EXPECT_EQ(IndexError_NoMemory, trial(quota, false));
      EXPECT_EQ(0, trial(quota, true));
      continue;
    }
    size_t low = 1, high = 2 * shape.count * (shape.name_bytes + 256) +
                           (2UL << 20);
    ASSERT_EQ(0, trial(high, false));
    while (low < high) {
      const size_t quota = low + (high - low) / 2;
      if (trial(quota, false) == 0)
        high = quota;
      else
        low = quota + 1;
    }
    ASSERT_EQ(0, trial(low, false));
    EXPECT_EQ(0, trial(low, true));
    // A funded destination and reclaimable warm sources can also admit the
    // directory with zero initially available bytes. Cold/small workspaces
    // still report a real refusal, then succeed again after pressure lifts.
    EXPECT_EQ(shape.warm && !shape.adaptive ? 0 : IndexError_NoMemory,
              trial(0, true));
    EXPECT_EQ(0u, pool.external_used());
    EXPECT_EQ(0, trial(high, true));
  }
}

#if defined(__linux__) && !defined(__ANDROID__)
TEST_F(MetadataAdmissionTest, ReadAheadPreservesDirectoryAdmission) {
  std::string path;
  std::vector<std::string> names;
  make_file(1, 300UL << 10, path, names);
  size_t depth = 0;
  auto trial = [&](size_t quota, bool prefetch) {
    auto storage = std::make_shared<ChunkReadStorage>();
    ailego::Params params;
    params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES,
               ailego::MemoryLimitPool::page_buffer_size());
    params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{1});
    params.set(CHUNK_READ_STORAGE_CACHE_BYTES, size_t{0});
    params.set(CHUNK_READ_STORAGE_DIRECT_IO, true);
    params.set(CHUNK_READ_STORAGE_PREFETCH_DEPTH, prefetch ? size_t{3} : 0);
    EXPECT_EQ(0, storage->init(params));
    Pressure pressure(quota);
    EXPECT_EQ(quota, pool.available());
    const int ret = storage->open(path, false);
    depth = storage->stats().prefetch_depth;
    if (ret == 0) EXPECT_TRUE(storage->has(names.back()));
    return ret;
  };
  size_t low = 1, high = 2UL << 20;
  ASSERT_EQ(0, trial(high, true));
  if (!depth) GTEST_SKIP() << "Native direct AIO unavailable";
  while (low < high) {
    const size_t quota = low + (high - low) / 2;
    if (trial(quota, false) == 0)
      high = quota;
    else
      low = quota + 1;
  }
  ASSERT_EQ(0, trial(low, false));
  // An optional read-ahead queue must not shrink the directory's success
  // budget. Admit it only once all mandatory open work has completed.
  EXPECT_EQ(0, trial(low, true));
}
#endif

TEST_F(MetadataAdmissionTest, MappingAllocationBoundariesRollBackAndRetry) {
  const std::string small = "metadata-mapping-small.bin";
  const std::string large = "metadata-mapping-large.bin";
  files.push_back(small);
  files.push_back(large);
  size_t mapping_bytes = 0;
  for (bool many : {false, true}) {
    IndexMapping mapping;
    ASSERT_EQ(0, mapping.create(many ? large : small, 4096));
    mapping_bytes = mapping.header().content_offset;
    const size_t count = many ? 24 : 1;
    for (size_t i = 0; i < count; ++i) {
      const std::string name =
          many ? std::to_string(i) + std::string(1000, 'n') : "old";
      ASSERT_EQ(0, mapping.append(name, 1));
    }
    mapping.refresh(0);
    ASSERT_EQ(0, mapping.flush());
  }
  // Reproduce the exact late registration boundary without relying on a
  // platform-specific node size. This mirrors the private mapping value.
  struct Record {
    IndexFormat::MetaHeader *header;
    size_t length, charge;
    ailego::ProcessRssBudget::MaterializationStamp projection{};
  };
  using HeaderMap =
      std::map<uint64_t, Record, std::less<uint64_t>,
               ailego::BudgetedAllocator<std::pair<const uint64_t, Record>>>;
  size_t node_bytes = 0;
  {
    HeaderMap pending;
    const auto before = pool.used();
    pending.emplace(0, Record{});
    node_bytes = pool.used() - before;
  }
  size_t required_bytes = 0;
  {
    IndexMapping mapping;
    const auto before = pool.external_used();
    ASSERT_EQ(0, mapping.open(large, true, false));
    required_bytes = pool.external_used() - before;
  }
  std::vector<size_t> remaining{0,
                                mapping_bytes - 1,
                                mapping_bytes + node_bytes,
                                required_bytes - 1,
                                required_bytes,
                                required_bytes + 512};
  // Walk the directory/node allocations at every metadata-chain boundary.
  // The 32-byte step is smaller than a managed map/node allocation.
  for (size_t base = mapping_bytes; base <= required_bytes;
       base += mapping_bytes) {
    for (size_t extra = 0; extra <= 4096; extra += 32)
      remaining.push_back(base + extra);
  }
  bool saw_denial = false, saw_success = false;
  for (bool cow : {false, true}) {
    for (size_t available : remaining) {
      SCOPED_TRACE(::testing::Message() << cow << ':' << available);
      IndexMapping published;
      ASSERT_EQ(0, published.open(small, cow, !cow));
      const auto old_size = published.index_size();
      const auto old_magic = published.magic();
      {
        Pressure pressure(available);
        const auto used = pool.external_used();
        const int ret = published.open(large, cow, !cow);
        if (ret == IndexError_NoMemory) {
          saw_denial = true;
          ASSERT_EQ(used, pool.external_used());
          EXPECT_TRUE(published.has("old"));
          EXPECT_EQ(old_size, published.index_size());
          EXPECT_EQ(old_magic, published.magic());
        } else {
          ASSERT_EQ(0, ret);
          saw_success = true;
          EXPECT_EQ(24u, published.segment_count());
        }
      }
      // Every refused partial candidate must be retryable after real pressure
      // is removed, including failure after mapping an earlier chain.
      ASSERT_EQ(0, published.open(large, cow, !cow));
      EXPECT_EQ(24u, published.segment_count());
      published.close();
      EXPECT_EQ(0u, pool.external_used());
    }
  }
  EXPECT_TRUE(saw_denial);
  EXPECT_TRUE(saw_success);
}
}  // namespace
}  // namespace zvec::core

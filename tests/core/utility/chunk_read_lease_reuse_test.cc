// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
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
      if (offset == 0 && size == sizeof(IndexFormat::MetaHeader))
        ++fetched_headers;
      failed_ = calls++ >= fail_at;
      return failed_ ? 0 : parent_->fetch(offset, data, size);
    }
    size_t fetch_for_parsing(size_t offset, void *data,
                             size_t size) const override {
      if (offset == 0 && size == sizeof(IndexFormat::MetaHeader))
        ++fetched_headers;
      failed_ = calls++ >= fail_at;
      return failed_ ? 0 : parent_->fetch_for_parsing(offset, data, size);
    }
    size_t read(size_t offset, IndexStorage::MemoryBlock &data,
                size_t size) override {
      failed_ = calls++ >= fail_at;
      return failed_ ? 0 : parent_->read(offset, data, size);
    }
    size_t read_for_parsing(size_t offset, IndexStorage::MemoryBlock &data,
                            size_t size) override {
      if (offset == 0 && size == sizeof(IndexFormat::MetaHeader))
        ++parsed_headers;
      failed_ = calls++ >= fail_at;
      const size_t got =
          failed_ ? 0 : parent_->read_for_parsing(offset, data, size);
      if (retain_input && got > ailego::MemoryLimitPool::page_buffer_size()) {
        escaped = data;
        escaped_offset = offset;
        escaped_size = got;
      }
      return got;
    }
    size_t read_for_parsing_with_strategy(size_t offset,
                                          IndexStorage::MemoryBlock &data,
                                          size_t size,
                                          ParsingReadStrategy strategy,
                                          int &error) override {
      ++strategy_reads[static_cast<size_t>(strategy)];
      if (offset == 0 && size == sizeof(IndexFormat::MetaHeader)) {
        ++parsed_headers;
        ++strategy_headers[static_cast<size_t>(strategy)];
      }
      failed_ = calls++ >= fail_at;
      if (failed_) {
        data.reset();
        error = IndexError_ReadData;
        return 0;
      }
      const size_t got = parent_->read_for_parsing_with_strategy(
          offset, data, size, strategy, error);
      if (retain_input && got > ailego::MemoryLimitPool::page_buffer_size()) {
        escaped = data;
        escaped_offset = offset;
        escaped_size = got;
      }
      return got;
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
    mutable size_t calls{0}, parsed_headers{0}, fetched_headers{0};
    std::array<size_t, 4> strategy_headers{}, strategy_reads{};
    size_t fail_at{std::numeric_limits<size_t>::max()};
    bool retain_input{false};
    IndexStorage::MemoryBlock escaped;
    size_t escaped_offset{0}, escaped_size{0};

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
    auto leaf =
        twice ? std::make_shared<IndexSegmentDumper>(middle, "nested") : middle;
    names.clear();
    for (size_t i = 0; i < count; ++i) {
      names.push_back(std::to_string(i) + '-' + std::string(name_bytes, 'n'));
      const char value = 'v';
      ASSERT_EQ(1u, leaf->write(&value, 1));
      ASSERT_EQ(0, leaf->append(names.back(), 1, 0, 0));
    }
    ASSERT_EQ(0, leaf->close());
    if (twice) {
      ASSERT_EQ(0, middle->close());
    }
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
  struct Shape {
    bool adaptive, warm, funded;
    size_t window, cache, count, name_bytes;
    size_t chunk_bytes{0};
    bool descriptor{false}, held{false}, empty{false}, busy{false};
  };
  std::vector<Shape> shapes;
  for (bool adaptive : {false, true})
    for (bool warm : {false, true})
      for (size_t window : {page, size_t{8UL << 20}})
        for (size_t chunk : {page, size_t{4UL << 20}})
          shapes.push_back(
              {adaptive, warm, false, window, 8UL << 20, 1, 2UL << 20, chunk});
  shapes.push_back({true, false, false, page, 0, 1, 32UL << 20});
  shapes.push_back({true, false, false, page, 8UL << 20, 512, 4096});
  shapes.push_back({true, false, true, page, 0, 1, 2UL << 20});
  const Shape cached{true,      true, false,     page,
                     8UL << 20, 1,    2UL << 20, 4UL << 20};
  shapes.push_back(cached);
  shapes.back().descriptor = true;
  shapes.push_back(cached);
  shapes.back().held = true;
  shapes.push_back({true, false, false, page, 0, 1, 2UL << 20});
  shapes.back().empty = true;
  shapes.push_back(cached);
  shapes.back().window = 8UL << 20;
  shapes.back().busy = true;
  for (const auto &shape : shapes) {
    SCOPED_TRACE(::testing::Message()
                 << shape.adaptive << ':' << shape.warm << ':' << shape.funded
                 << ':' << shape.window << ':' << shape.cache << ':'
                 << shape.count << ':' << shape.name_bytes << ':'
                 << shape.chunk_bytes << ':' << shape.descriptor << ':'
                 << shape.held << ':' << shape.empty << ':' << shape.busy);
    std::string path;
    std::vector<std::string> names;
    // One level keeps all cold/warm preparation explicit, before pressure.
    make_nested_file(shape.count, shape.name_bytes, path, names,
                     shape.busy ? 8UL << 20 : 0, false, false);
    auto trial = [&](size_t quota, bool parser, bool expect_retry = false) {
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
      IndexStorage::MemoryBlock held;
      if (shape.busy) {
        auto prefix = storage->get("prefix", 1);
        EXPECT_EQ(5UL << 20, prefix->read(0, held, 5UL << 20));
        auto &queue = ailego::BlockEvictionQueue::get_instance();
        for (size_t i = 0; i < 64 && storage->stats().cache_bytes; ++i)
          queue.reclaim_for_allocation(8UL << 20);
      }
      if (shape.warm) {
        std::string data(parent->data_size(), '\0');
        EXPECT_EQ(data.size(), parent->fetch(0, data.data(), data.size()));
      }
      if (shape.funded) {
        IndexStorage::MemoryBlock data;
        EXPECT_EQ(parent->data_size(),
                  parent->read(0, data, parent->data_size()));
      }
      if (shape.descriptor || shape.held || shape.empty) {
        IndexStorage::MemoryBlock data;
        EXPECT_EQ(32u, parent->read(0, data, 32));
        if (shape.held) held = std::move(data);
      }
      if (shape.empty) {
        Pressure pressure(0);
        IndexStorage::MemoryBlock data;
        EXPECT_EQ(0u, parent->read(0, data, parent->data_size()));
        EXPECT_EQ(IndexError_NoMemory, parent->read_error());
        EXPECT_EQ(0u, storage->stats().workspace_bytes);
      }
      auto observed = std::make_shared<FailingParent>(parent);
      parent = observed;
      const auto workspace = storage->stats().workspace_bytes;
      int result = IndexError_NoMemory;
      {
        Pressure pressure(quota);
        EXPECT_EQ(quota, pool.available());
        try {
          if (parser) {
            IndexSegmentStorage nested(parent);
            result = nested.open("", false);
            if (result == 0) {
              EXPECT_TRUE(nested.has(names.back()));
            }
          } else {
            ailego::BudgetedMappedBuffer copy;
            IndexUnpacker reference;
            const auto fetch = [&](size_t offset, const void **data,
                                   size_t len) {
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
      if (shape.funded) {
        EXPECT_EQ(workspace, storage->stats().workspace_bytes);
      }
      // Each callback probes each strategy once; a private restart excludes
      // its failed strategy. Four whole attempts bound all header probes.
      EXPECT_LE(observed->parsed_headers, 12u);
      for (auto count : observed->strategy_headers) EXPECT_LE(count, 4u);
      EXPECT_LE(observed->fetched_headers, 4u);
      if (parser && shape.held && quota == page) {
        EXPECT_EQ(0u, observed->fetched_headers);
      }
      if (expect_retry) {
        EXPECT_GT(observed->strategy_reads[2], 0u);
      }
      if (shape.busy) {
        EXPECT_EQ('p', *static_cast<const char *>(held.data()));
      }
      return result;
    };
    if (shape.empty) {
      // A denied growth released the old workspace. The reference must now
      // rebuild its small fetch window as well as the directory input.
      const size_t quota = 2 * names.back().size() + 4 * page + 2048;
      ASSERT_EQ(0, trial(quota, false));
      EXPECT_EQ(0, trial(quota, true));
      continue;
    }
    if (shape.funded || shape.held) {
      // Only the private directory fits. Reusing its funded input succeeds;
      // unconditionally making another copy would reject the same operation.
      const size_t quota = names.back().size() + 8 * page;
      EXPECT_EQ(IndexError_NoMemory, trial(quota, false));
      EXPECT_EQ(0, trial(quota, true));
      if (shape.held) {
        EXPECT_EQ(IndexError_NoMemory, trial(page, true));
      }
      continue;
    }
    size_t low = 1,
           high = 2 * shape.count * (shape.name_bytes + 256) + (2UL << 20);
    ASSERT_EQ(0, trial(high, false));
    while (low < high) {
      const size_t quota = low + (high - low) / 2;
      if (trial(quota, false) == 0)
        high = quota;
      else
        low = quota + 1;
    }
    ASSERT_EQ(0, trial(low, false));
    const bool retry = shape.adaptive && shape.warm && shape.window == page &&
                       shape.chunk_bytes > page;
    EXPECT_EQ(0, trial(low, true, retry));
    // Zero available quota may still succeed by reusing/retiring funded input.
    // Preserve the established large-window warm-cache successes; cold
    // input has no backing to reclaim and must report a real refusal.
    const int zero = trial(0, true);
    EXPECT_TRUE(zero == 0 || zero == IndexError_NoMemory);
    if (!shape.warm) {
      EXPECT_EQ(IndexError_NoMemory, zero);
    } else if (!shape.adaptive && shape.window > page) {
      EXPECT_EQ(0, zero);
    }
    EXPECT_EQ(0u, pool.external_used());
    EXPECT_EQ(0, trial(high, true));
  }
}

TEST_F(MetadataAdmissionTest, ParserRetryPreservesEscapedInput) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  std::string path;
  std::vector<std::string> names;
  make_nested_file(1, 2UL << 20, path, names, 0, false, false);
  for (bool funded : {false, true}) {
    SCOPED_TRACE(funded);
    auto storage = std::make_shared<ChunkReadStorage>();
    ailego::Params params;
    params.set(CHUNK_READ_STORAGE_ADAPTIVE, !funded);
    params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, funded ? 8UL << 20 : page);
    params.set(CHUNK_READ_STORAGE_CHUNK_BYTES, 4UL << 20);
    params.set(CHUNK_READ_STORAGE_CACHE_BYTES, 8UL << 20);
    params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{1});
    params.set(CHUNK_READ_STORAGE_DIRECT_IO, false);
    ASSERT_EQ(0, storage->init(params));
    ASSERT_EQ(0, storage->open(path, false));
    auto parent = storage->get("nested", 1);
    std::string image(parent->data_size(), '\0');
    ASSERT_EQ(image.size(), parent->fetch(0, image.data(), image.size()));
    // A workspace copy can leave its cache source independently evictable.
    // Hold that real source as well so this negative case cannot fund private
    // names by reclaiming unrelated input backing.
    IndexStorage::Segment::ScatterBlock source_alias;
    if (funded) {
      ASSERT_EQ(parent->data_size(),
                parent->read_scatter(0, source_alias, parent->data_size()));
    }
    auto observed = std::make_shared<FailingParent>(parent);
    observed->retain_input = true;
    const auto workspace = storage->stats().workspace_bytes;
    {
      Pressure pressure(page);
      IndexSegmentStorage nested(observed);
      EXPECT_EQ(IndexError_NoMemory, nested.open("", false));
      EXPECT_FALSE(nested.has(names.back()));
    }
    // The same input lease has an escaped reader. A failed private allocation
    // must not unmap its workspace or unlock its actual cache backing.
    ASSERT_GT(observed->escaped_size, page);
    EXPECT_EQ(0u, observed->fetched_headers);
    EXPECT_EQ(workspace, storage->stats().workspace_bytes);
    auto retained = std::move(observed->escaped);
    const size_t offset = observed->escaped_offset,
                 size = observed->escaped_size;
    observed.reset();
    parent.reset();
    ASSERT_EQ(0, storage->close());
    storage.reset();
    EXPECT_EQ(0, std::memcmp(retained.data(), image.data() + offset, size));
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

namespace zvec::core {
namespace {
class ParserOwnershipTest : public MetadataAdmissionTest {
 protected:
  void package(std::string &path, const std::string &name, size_t prefix,
               size_t payload) {
    path = "closure-ownership-" + std::to_string(files.size()) + ".bin";
    files.push_back(path);
    auto file = IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, file->create(path));
    if (prefix) {
      const std::string bytes(prefix, 'p');
      ASSERT_EQ(prefix, file->write(bytes.data(), prefix));
      ASSERT_EQ(0, file->append("prefix", prefix, 0, 0));
    }
    auto nested = std::make_shared<IndexSegmentDumper>(file, "nested");
    const std::string bytes(payload, 'v');
    ASSERT_EQ(payload, nested->write(bytes.data(), payload));
    ASSERT_EQ(0, nested->append(name, payload, 0, 0));
    ASSERT_EQ(0, nested->close());
    ASSERT_EQ(0, file->close());
  }
  int reference(const IndexStorage::Segment::Pointer &parent) {
    try {
      ailego::BudgetedMappedBuffer input;
      IndexUnpacker unpacker;
      auto fetch = [&](size_t offset, const void **data, size_t len) {
        if (input.size() < len && input.resize(len) != 0) return size_t{0};
        *data = input.data();
        return parent->fetch(offset, input.data(), len);
      };
      return unpacker.unpack(fetch, parent->data_size(), false) ? 0
             : parent->read_error() == IndexError_NoMemory
                 ? IndexError_NoMemory
                 : IndexError_UnpackIndex;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
  }
  int reference_borrow(const IndexStorage::Segment::Pointer &parent) {
    IndexStorage::Segment::ScatterBlock input;
    IndexUnpacker unpacker;
    try {
      const auto read = [&](size_t offset, const void **data, size_t len) {
        input.reset();
        const size_t got = parent->read_scatter(offset, input, len);
        if (got != len || input.spans.size() != 1) return size_t{0};
        *data = input.spans.front().data;
        return got;
      };
      return unpacker.unpack(read, parent->data_size(), false)
                 ? 0
                 : IndexError_NoMemory;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
  }
  int parse(const IndexStorage::Segment::Pointer &parent,
            const std::string &name) {
    IndexSegmentStorage nested(parent);
    const int result = nested.open("", false);
    if (!result) EXPECT_TRUE(nested.has(name));
    return result;
  }
  ailego::Params chunk_params(size_t page) {
    ailego::Params params;
    params.set(CHUNK_READ_STORAGE_ADAPTIVE, true);
    params.set(CHUNK_READ_STORAGE_CHUNK_BYTES, page);
    params.set(CHUNK_READ_STORAGE_CACHE_BYTES, 8UL << 20);
    params.set(CHUNK_READ_STORAGE_MAX_READ_BYTES, page);
    params.set(CHUNK_READ_STORAGE_WORKSPACE_COUNT, size_t{1});
    params.set(CHUNK_READ_STORAGE_WAIT_MS, size_t{0});
    params.set(CHUNK_READ_STORAGE_DIRECT_IO, false);
    return params;
  }
  class FetchOnly : public FailingParent {
   public:
    using FailingParent::FailingParent;
    size_t read_for_parsing(size_t, IndexStorage::MemoryBlock &out,
                            size_t) override {
      out.reset();
      return 0;
    }
    int read_error() const override {
      return IndexError_ReadData;
    }
    size_t read_for_parsing_with_strategy(size_t,
                                          IndexStorage::MemoryBlock &out,
                                          size_t, ParsingReadStrategy,
                                          int &error) override {
      out.reset();
      error = IndexError_Unsupported;
      return 0;
    }
  };
};
TEST_F(ParserOwnershipTest, BusyWorkspacePreservesOwnedInputAdmission) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  const size_t h = sizeof(IndexFormat::MetaHeader);
  const size_t v = (std::strlen(IndexVersion::Details()) + 31) / 32 * 32;
  const std::string name(2UL << 20, 'n');
  // Cache population, page alignment and workspace ownership independently
  // affect representation availability. The reference owns its exact input
  // and reads the same real file without borrowing a query workspace.
  const struct Shape {
    bool aligned, warm, cache;
  } shapes[] = {{true, true, true},
                {true, false, true},
                {false, true, true},
                {false, false, true},
                {false, false, false}};
  for (const auto &shape : shapes) {
    SCOPED_TRACE(::testing::Message()
                 << shape.aligned << ':' << shape.warm << ':' << shape.cache);
    std::string path;
    package(path, name, 8 * page - h, page - h - v + !shape.aligned);
    auto trial = [&](size_t quota, int mode) {
      auto storage = std::make_shared<ChunkReadStorage>();
      auto params = chunk_params(page);
      if (!shape.cache) params.set(CHUNK_READ_STORAGE_CACHE_BYTES, size_t{0});
      EXPECT_EQ(0, storage->init(params));
      EXPECT_EQ(0, storage->open(path, false));
      auto prefix = storage->get("prefix", 1),
           parent = storage->get("nested", 1);
      if (shape.warm) {
        std::string image(parent->data_size(), '\0');
        EXPECT_EQ(image.size(), parent->fetch(0, image.data(), image.size()));
      }
      IndexStorage::MemoryBlock busy;
      EXPECT_EQ(2 * page, prefix->read(0, busy, 2 * page));
      const auto workspace = storage->stats().workspace_bytes;
      ailego::File file;
      EXPECT_TRUE(file.open(path, true, false));
      int result = IndexError_NoMemory;
      {
        Pressure pressure(quota);
        EXPECT_EQ(quota, pool.available());
        if (mode == 2) {
          busy.reset();
          EXPECT_EQ(quota, pool.available());
        }
        if (mode)
          result = parse(parent, name);
        else
          try {
            ailego::BudgetedMappedBuffer input;
            IndexUnpacker unpacker;
            const auto fetch = [&](size_t offset, const void **data,
                                   size_t len) {
              if (input.size() < len && input.resize(len) != 0)
                return size_t{0};
              *data = input.data();
              return file.read(parent->data_offset() + offset, input.data(),
                               len);
            };
            if (unpacker.unpack(fetch, parent->data_size(), false)) result = 0;
          } catch (const std::bad_alloc &) {
          }
      }
      if (busy.data()) {
        EXPECT_EQ('p', *static_cast<const char *>(busy.data()));
        EXPECT_EQ(workspace, storage->stats().workspace_bytes);
      }
      return result;
    };
    size_t low = 1, high = 6UL << 20;
    ASSERT_EQ(0, trial(high, 0));
    while (low < high) {
      const size_t quota = low + (high - low) / 2;
      if (trial(quota, 0) == 0)
        high = quota;
      else
        low = quota + 1;
    }
    EXPECT_EQ(0, trial(low, 0));
    EXPECT_EQ(0, trial(low, 1));
    EXPECT_EQ(0, trial(low, 2));
  }
}
TEST_F(ParserOwnershipTest, DescriptorAdmissionPreservesOnlyAvailableInput) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  const size_t h = sizeof(IndexFormat::MetaHeader);
  const size_t v = (std::strlen(IndexVersion::Details()) + 31) / 32 * 32;
  std::string path;
  const std::string name(512, 'n');
  package(path, name, 8 * page - h, page - h - v);
  auto trial = [&](int mode) {
    auto storage = std::make_shared<ChunkReadStorage>();
    EXPECT_EQ(0, storage->init(chunk_params(page)));
    EXPECT_EQ(0, storage->open(path, false));
    auto prefix = storage->get("prefix", 1), parent = storage->get("nested", 1);
    IndexStorage::Segment::ScatterBlock busy, warm;
    EXPECT_EQ(2 * page, prefix->read_scatter(0, busy, 2 * page));
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    for (size_t i = 0; i < 64 && storage->stats().cache_bytes; ++i)
      queue.reclaim_for_allocation(8UL << 20);
    EXPECT_EQ(0, prefix->set_read_chunks({{0, prefix->data_size()}}));
    EXPECT_EQ(prefix->data_size(),
              prefix->read_scatter(0, warm, prefix->data_size()));
    warm.reset();
    EXPECT_EQ(prefix->data_size(),
              prefix->read_scatter(0, warm, prefix->data_size()));
    warm.reset();
    EXPECT_EQ(0, parent->set_read_chunks({{0, parent->data_size()}}));
    EXPECT_EQ(parent->data_size(),
              parent->read_scatter(0, warm, parent->data_size()));
    warm.reset();
    if (mode == 1) {
      // Touch only the already-hot unrelated chunk. Fund the descriptor but
      // leave every execution workspace busy and the source entry cold.
      IndexStorage::MemoryBlock descriptor;
      EXPECT_EQ(16u, prefix->read(0, descriptor, 16));
    }
    const auto cache = storage->stats().cache_bytes;
    int result;
    {
      Pressure pressure(0);
      result = mode == 2 ? reference_borrow(parent) : parse(parent, name);
      std::printf("LEASE mode=%d result=%d cache=%llu->%llu io=%llu\n", mode,
                  result, static_cast<unsigned long long>(cache),
                  static_cast<unsigned long long>(storage->stats().cache_bytes),
                  static_cast<unsigned long long>(storage->stats().io_calls));
    }
    EXPECT_FALSE(busy.spans.empty());
    if (!busy.spans.empty()) EXPECT_EQ('p', busy.spans.front().data[0]);
    return result;
  };
  const int candidate = trial(0), control = trial(1), borrowed = trial(2);
  RecordProperty("candidate", candidate);
  EXPECT_EQ(0, candidate);
  EXPECT_EQ(0, control);
  EXPECT_EQ(0, borrowed);
}
TEST_F(ParserOwnershipTest, ReusedWorkerInputAllowsRequiredDirectory) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  const std::string name(page, 'n');
  std::string path;
  package(path, name, 0, 64UL << 10);
  auto trial = [&](int mode) {
    auto storage = IndexFactory::CreateStorage("BufferReadStorage");
    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE,
               BUFFER_READ_STORAGE_WARMUP_NONE);
    EXPECT_EQ(0, storage->init(params));
    EXPECT_EQ(0, storage->open(path, false));
    auto parent = storage->get("nested", 1);
    IndexStorage::MemoryBlock snapshot;
    EXPECT_EQ(64UL << 10, parent->read(0, snapshot, 64UL << 10));
    EXPECT_EQ(IndexStorage::MemoryBlock::MBT_REUSABLE_SCRATCH, snapshot.type_);
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    for (size_t i = 0; i < 64; ++i) queue.reclaim_for_allocation(8UL << 20);
    IndexStorage::MemoryBlock header, footer, version;
    EXPECT_EQ(sizeof(IndexFormat::MetaHeader),
              parent->read(0, header, sizeof(IndexFormat::MetaHeader)));
    EXPECT_EQ(
        sizeof(IndexFormat::MetaFooter),
        parent->read(parent->data_size() - sizeof(IndexFormat::MetaFooter),
                     footer, sizeof(IndexFormat::MetaFooter)));
    EXPECT_EQ(1u, parent->read(sizeof(IndexFormat::MetaHeader) + (64UL << 10),
                               version, 1));
    snapshot.reset();
    auto copy_parent = std::make_shared<FetchOnly>(parent);
    int result;
    {
      Pressure pressure(0);
      EXPECT_EQ(0u, pool.available());
      result = mode == 0 ? reference(parent)
                         : parse(mode == 2 ? copy_parent : parent, name);
      std::printf("WORKER mode=%d result=%d available_after=%zu\n", mode,
                  result, pool.available());
    }
    return result;
  };
  const int ref = trial(0), candidate = trial(1), control = trial(2);
  RecordProperty("candidate", candidate);
  EXPECT_EQ(0, ref);
  EXPECT_EQ(0, candidate);
  EXPECT_EQ(0, control);
}
TEST_F(ParserOwnershipTest, OwnedMetadataReadsPreserveDirtyNestedData) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  std::string path;
  std::vector<std::string> names;
  make_nested_file(1, page, path, names);
  const auto &name = names.back();
  auto source = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, source->init(ailego::Params{}));
  ASSERT_EQ(0, source->open(path, false));
  auto segment = source->get("nested", 1);
  std::string image(segment->data_size(), '\0');
  ASSERT_EQ(image.size(), segment->fetch(0, image.data(), image.size()));
  segment.reset();
  source.reset();

  const std::string writable = path + ".writable";
  files.push_back(writable);
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  ASSERT_EQ(0, storage->open(writable, IndexStorage::OpenOptions{true, false}));
  ASSERT_EQ(0, storage->append("nested", image.size()));
  auto parent = storage->get("nested", 1);
  ASSERT_EQ(image.size(), parent->write(0, image.data(), image.size()));
  // Do not flush: the owned-input path must observe dirty resident metadata.
  auto owned = std::make_shared<FetchOnly>(parent);
  {
    Pressure pressure(3 * page + 1024);
    EXPECT_EQ(0, parse(owned, "nested"));
  }
  IndexSegmentStorage middle(owned);
  ASSERT_EQ(0, middle.open("", false));
  auto inner = std::make_shared<FetchOnly>(middle.get("nested", 1));
  {
    Pressure pressure(3 * page + 1024);
    EXPECT_EQ(0, parse(inner, name));
  }
}

TEST_F(ParserOwnershipTest, TemporaryInputRetryPreservesAliases) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  std::string path;
  const std::string name(2 * page, 'n');
  package(path, name, 0, 64UL << 10);
  auto storage = IndexFactory::CreateStorage("BufferReadStorage");
  ailego::Params params;
  params.set(BUFFER_READ_STORAGE_WARMUP_MODE, BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_EQ(0, storage->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  auto parent = storage->get("nested", 1);
  std::string image(parent->data_size(), '\0');
  ASSERT_EQ(image.size(), parent->fetch(0, image.data(), image.size()));
  IndexStorage::MemoryBlock snapshot;
  ASSERT_EQ(64UL << 10, parent->read(0, snapshot, 64UL << 10));
  auto &queue = ailego::BlockEvictionQueue::get_instance();
  for (size_t i = 0; i < 64; ++i) queue.reclaim_for_allocation(8UL << 20);
  IndexStorage::MemoryBlock header, footer, version;
  ASSERT_EQ(sizeof(IndexFormat::MetaHeader),
            parent->read(0, header, sizeof(IndexFormat::MetaHeader)));
  ASSERT_EQ(sizeof(IndexFormat::MetaFooter),
            parent->read(parent->data_size() - sizeof(IndexFormat::MetaFooter),
                         footer, sizeof(IndexFormat::MetaFooter)));
  ASSERT_EQ(1u, parent->read(sizeof(IndexFormat::MetaHeader) + (64UL << 10),
                             version, 1));
  snapshot.reset();
  auto observed = std::make_shared<FailingParent>(parent);
  observed->retain_input = true;
  {
    Pressure pressure(0);
    IndexSegmentStorage nested(observed);
    EXPECT_EQ(IndexError_NoMemory, nested.open("", false));
    EXPECT_FALSE(nested.has(name));
    EXPECT_EQ(0u, observed->fetched_headers);
  }
  ASSERT_GE(observed->escaped_size, 2 * page);
  auto retained = std::move(observed->escaped);
  const auto offset = observed->escaped_offset, size = observed->escaped_size;
  observed.reset();
  parent.reset();
  header.reset();
  footer.reset();
  version.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  EXPECT_EQ(0, std::memcmp(retained.data(), image.data() + offset, size));
}

#if defined(__linux__) && !defined(__ANDROID__)
TEST_F(ParserOwnershipTest, PrefetchPinsYieldWithoutInvalidatingReaders) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  std::string path;
  for (bool large_directory : {true, false}) {
    SCOPED_TRACE(large_directory);
    const std::string name(large_directory ? 2UL << 20 : page, 'n');
    package(path, name, 8 * page - sizeof(IndexFormat::MetaHeader),
            large_directory ? 1 : 2UL << 20);
    for (bool busy : {false, true}) {
      for (bool warm : {false, true}) {
        SCOPED_TRACE(::testing::Message() << busy << ':' << warm);
        auto trial = [&](size_t quota, int mode, bool external = false) {
          auto storage = std::make_shared<ChunkReadStorage>();
          auto params = chunk_params(page);
          params.set(CHUNK_READ_STORAGE_PREFETCH_DEPTH, size_t{2});
          params.set(CHUNK_READ_STORAGE_PREFETCH_BYTES, 8UL << 20);
          params.set(CHUNK_READ_STORAGE_DIRECT_IO, true);
          EXPECT_EQ(0, storage->init(params));
          EXPECT_EQ(0, storage->open(path, false));
          EXPECT_GT(storage->stats().prefetch_depth, 0u);
          auto parent = storage->get("nested", 1);
          ailego::File file;
          EXPECT_TRUE(file.open(path, true, false));
          std::string image(parent->data_size(), '\0');
          EXPECT_EQ(image.size(), file.read(parent->data_offset(), image.data(),
                                            image.size()));
          IndexFormat::MetaFooter footer;
          std::memcpy(&footer, image.data() + image.size() - sizeof(footer),
                      sizeof(footer));
          const size_t begin =
              busy ? image.size() - sizeof(footer) - footer.segments_meta_size
                   : 0;
          const size_t length = busy ? footer.segments_meta_size : image.size();
          EXPECT_EQ(0, parent->set_read_chunks({{begin, length}}));
          IndexStorage::Segment::ScatterBlock alias;
          if (warm || external) {
            EXPECT_EQ(length, parent->read_scatter(begin, alias, length));
            if (!external) alias.reset();
          }
          IndexStorage::MemoryBlock occupied;
          if (busy)
            EXPECT_EQ(2 * page,
                      storage->get("prefix", 1)->read(0, occupied, 2 * page));
          EXPECT_TRUE(parent->try_prefetch(
              begin, length, IndexStorage::Segment::CachePriority::kNormal));
          EXPECT_EQ(1u, storage->stats().prefetch_pending_requests);
          if (mode != 1) {
            // Reference/control consume the optional slot, without releasing
            // any mapping or execution workspace. The candidate must do so
            // itself.
            char header[sizeof(IndexFormat::MetaHeader)];
            EXPECT_EQ(sizeof(header),
                      parent->fetch(begin, header, sizeof(header)));
            EXPECT_EQ(0u, storage->stats().prefetch_pending_requests);
          }
          int result = IndexError_NoMemory;
          {
            Pressure pressure(quota);
            if (mode)
              result = parse(parent, name);
            else
              try {
                ailego::BudgetedMappedBuffer input;
                IndexUnpacker unpacker;
                const auto fetch = [&](size_t offset, const void **data,
                                       size_t len) {
                  if (input.size() < len && input.resize(len) != 0)
                    return size_t{0};
                  *data = input.data();
                  return file.read(parent->data_offset() + offset, input.data(),
                                   len);
                };
                if (unpacker.unpack(fetch, parent->data_size(), false))
                  result = 0;
              } catch (const std::bad_alloc &) {
              }
          }
          EXPECT_EQ(0u, storage->stats().prefetch_pending_requests);
          if (occupied.data())
            EXPECT_EQ('p', *static_cast<const char *>(occupied.data()));
          if (external) {
            EXPECT_FALSE(alias.spans.empty());
            if (!alias.spans.empty())
              EXPECT_EQ(0, std::memcmp(alias.spans.front().data,
                                       image.data() + begin, length));
          }
          return result;
        };
        if (!large_directory) {
          // The optional data hint can fund the whole parse at zero available
          // quota. Retire it before admitting even the first input descriptor.
          EXPECT_EQ(0, trial(0, 2));
          EXPECT_EQ(0, trial(0, 1));
          continue;
        }
        size_t low = 1, high = 6UL << 20;
        ASSERT_EQ(0, trial(high, 0));
        while (low < high) {
          const size_t quota = low + (high - low) / 2;
          if (trial(quota, 0) == 0)
            high = quota;
          else
            low = quota + 1;
        }
        EXPECT_EQ(0, trial(low, 0));
        EXPECT_EQ(0, trial(low, 1));
        EXPECT_EQ(0, trial(low, 2));
        if (warm) {
          // Real external references survive both a successful borrowed parse
          // and an actual refusal. Only the optional prefetch reference
          // retires.
          EXPECT_EQ(0, trial(name.size() + 8 * page, 1, true));
          EXPECT_EQ(IndexError_NoMemory, trial(page, 1, true));
        }
      }
    }
  }
}

#endif

}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {
class IndependentParserAdmissionTest : public ParserOwnershipTest {
 protected:
  enum class Reference { kReuse, kBorrow, kGrow, kOwned, kCandidate };
  struct Shape {
    bool warm{true}, adaptive{true};
    size_t chunk{4UL << 20}, cache{8UL << 20};
    bool busy{false}, funded{false}, empty{false}, alias{false};
    bool prefetch{false}, directory_hint{false};
    size_t name_bytes{64UL << 10};
    Reference witness{Reference::kGrow};
  };

  // This oracle never calls open() or the production strategy selector.
  // Its four fixed plans use the established scatter/legacy read/caller-copy
  // APIs. Workspace plans evict idle sources before each read, so a warm loan
  // cannot hide a cheaper funded-workspace path. All trials start afresh.
  int fixed_parse(Reference plan, const IndexStorage::Segment::Pointer &parent,
                  ChunkReadStorage *chunk, bool may_grow, size_t hint_offset,
                  bool pending) {
    using Block = IndexStorage::MemoryBlock;
    IndexStorage::Segment::ScatterBlock scatter;
    Block loan;
    ailego::BudgetedMappedBuffer copy;
    IndexUnpacker unpacker;
    const size_t initial_workspace = chunk ? chunk->stats().workspace_bytes : 0;
    const size_t page = ailego::MemoryLimitPool::page_buffer_size();
    try {
      if (plan == Reference::kOwned && pending) {
        // Retire the already-funded hint before admitting copied input. This
        // ordinary demand read transfers only the hint pin, then drops it.
        char byte;
        if (parent->fetch(hint_offset, &byte, 1) != 1)
          return IndexError_NoMemory;
      }
      const auto read = [&](size_t offset, const void **data, size_t len) {
        scatter.reset();
        loan.reset();
        copy.clear();
        if (plan == Reference::kBorrow) {
          const size_t got = parent->read_scatter(offset, scatter, len);
          if (got != len || scatter.spans.size() != 1) return size_t{0};
          *data = scatter.spans.front().data;
          return got;
        }
        if (plan == Reference::kReuse || plan == Reference::kGrow) {
          if (!chunk || initial_workspace == 0 ||
              (plan == Reference::kGrow && !may_grow))
            return size_t{0};
          if (plan == Reference::kReuse && len + 2 * page > initial_workspace)
            return size_t{0};
          auto &queue = ailego::BlockEvictionQueue::get_instance();
          for (size_t i = 0; i < 64 && chunk->stats().cache_bytes; ++i)
            queue.reclaim_for_allocation(8UL << 20);
          const size_t got = parent->read_for_parsing(offset, loan, len);
          if (got == len) *data = loan.data();
          return got;
        }
        if (copy.resize(len) != 0) return size_t{0};
        *data = copy.data();
        return parent->fetch_for_parsing(offset, copy.data(), len);
      };
      return unpacker.unpack(read, parent->data_size(), false)
                 ? 0
                 : IndexError_NoMemory;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
  }

  int trial(const Shape &shape, const std::string &path,
            const std::string &name, size_t quota, Reference plan) {
    const size_t page = ailego::MemoryLimitPool::page_buffer_size();
    auto storage = std::make_shared<ChunkReadStorage>();
    auto params = chunk_params(page);
    params.set(CHUNK_READ_STORAGE_CHUNK_BYTES, shape.chunk);
    params.set(CHUNK_READ_STORAGE_CACHE_BYTES, shape.cache);
    params.set(CHUNK_READ_STORAGE_ADAPTIVE, shape.adaptive);
    if (shape.prefetch) {
      params.set(CHUNK_READ_STORAGE_PREFETCH_DEPTH, size_t{2});
      params.set(CHUNK_READ_STORAGE_PREFETCH_BYTES, 8UL << 20);
      params.set(CHUNK_READ_STORAGE_DIRECT_IO, true);
    }
    EXPECT_EQ(0, storage->init(params));
    EXPECT_EQ(0, storage->open(path, false));
    auto parent = storage->get("nested", 1);
    IndexStorage::MemoryBlock occupied, prepared;
    IndexStorage::Segment::ScatterBlock alias;
    if (shape.funded) {
      EXPECT_EQ(parent->data_size(),
                parent->read(0, prepared, parent->data_size()));
      prepared.reset();
    }
    std::string image(parent->data_size(), '\0');
    // Inspect the canonical real file without warming query caches.
    ailego::File file;
    EXPECT_TRUE(file.open(path, true, false));
    EXPECT_EQ(image.size(),
              file.read(parent->data_offset(), image.data(), image.size()));
    IndexFormat::MetaFooter footer;
    std::memcpy(&footer, image.data() + image.size() - sizeof(footer),
                sizeof(footer));
    const size_t begin = shape.directory_hint ? image.size() - sizeof(footer) -
                                                    footer.segments_meta_size
                                              : 0;
    if (shape.prefetch) {
      EXPECT_GT(storage->stats().prefetch_depth, 0u);
      EXPECT_EQ(0, parent->set_read_chunks(
                       {{begin, shape.directory_hint ? footer.segments_meta_size
                                                     : image.size()}}));
    }
    if (shape.warm || shape.alias) {
      EXPECT_EQ(image.size(), parent->fetch(0, image.data(), image.size()));
    }
    if (shape.alias) {
      EXPECT_EQ(parent->data_size(),
                parent->read_scatter(0, alias, parent->data_size()));
    }
    // Cache the same descriptor in every trial, without retaining a reader.
    EXPECT_EQ(16u, parent->read(0, prepared, 16));
    prepared.reset();
    if (shape.busy) {
      EXPECT_EQ(2 * page,
                storage->get("prefix", 1)->read(0, occupied, 2 * page));
    }
    if (shape.empty) {
      Pressure pressure(0);
      EXPECT_EQ(0u, parent->read(0, prepared, parent->data_size()));
      EXPECT_EQ(0u, storage->stats().workspace_bytes);
    }
    if (shape.prefetch) {
      EXPECT_TRUE(parent->try_prefetch(
          begin,
          shape.directory_hint ? footer.segments_meta_size : image.size(),
          IndexStorage::Segment::CachePriority::kNormal));
    }
    int result;
    {
      Pressure pressure(quota);
      EXPECT_EQ(quota, pool.available());
      result = plan == Reference::kCandidate
                   ? parse(parent, name)
                   : fixed_parse(plan, parent, storage.get(), shape.adaptive,
                                 begin, shape.prefetch);
      if (occupied.data()) {
        EXPECT_EQ('p', *static_cast<const char *>(occupied.data()));
      }
      if (!alias.spans.empty()) {
        EXPECT_EQ(0, std::memcmp(alias.spans.front().data, image.data(),
                                 image.size()));
      }
    }
    return result;
  }

  void compare(const Shape &shape, size_t index) {
    const size_t page = ailego::MemoryLimitPool::page_buffer_size();
    std::string path;
    const std::string name(shape.name_bytes, 'n');
    package(path, name, 8 * page - sizeof(IndexFormat::MetaHeader), 1);
    size_t low = 0, high = 2 * name.size() + 8 * page + 4096;
    ASSERT_EQ(0, trial(shape, path, name, high, shape.witness));
    while (low < high) {
      const size_t quota = low + (high - low) / 2;
      if (trial(shape, path, name, quota, shape.witness) == 0)
        high = quota;
      else
        low = quota + 1;
    }
    RecordProperty("boundary_" + std::to_string(index), std::to_string(low));
    // Check both sides of the independently discovered byte boundary, then
    // the neighboring page boundary where workspace padding changes peaks.
    std::vector<size_t> quotas{low, low + 1, low + page};
    if (low) quotas.push_back(low - 1);
    for (const size_t quota : quotas) {
      SCOPED_TRACE(::testing::Message() << index << ':' << quota);
      bool possible = false;
      for (auto plan : {Reference::kReuse, Reference::kBorrow, Reference::kGrow,
                        Reference::kOwned}) {
        const int result = trial(shape, path, name, quota, plan);
        RecordProperty("plan_" + std::to_string(index) + '_' +
                           std::to_string(quota) + '_' +
                           std::to_string(static_cast<int>(plan)),
                       result);
        possible |= result == 0;
      }
      const int actual = trial(shape, path, name, quota, Reference::kCandidate);
      if (possible)
        EXPECT_EQ(0, actual);
      else
        EXPECT_TRUE(actual == 0 || actual == IndexError_NoMemory);
    }
  }
};

TEST_F(IndependentParserAdmissionTest, ReadAndFormatErrorsStopImmediately) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  for (bool io_failure : {false, true}) {
    SCOPED_TRACE(io_failure);
    std::string path;
    package(path, "child", 0, 1);
    auto storage = std::make_shared<ChunkReadStorage>();
    ASSERT_EQ(0, storage->init(chunk_params(page)));
    ASSERT_EQ(0, storage->open(path, false));
    auto parent = storage->get("nested", 1);
    if (!io_failure) {
      ailego::File file;
      ASSERT_TRUE(file.open(path, false, false));
      IndexFormat::MetaHeader header;
      ASSERT_EQ(sizeof(header),
                file.read(parent->data_offset(), &header, sizeof(header)));
      header.header_crc ^= 1;
      ASSERT_EQ(sizeof(header),
                file.write(parent->data_offset(), &header, sizeof(header)));
    }
    auto observed = std::make_shared<FailingParent>(parent);
    if (io_failure) observed->fail_at = 0;
    IndexSegmentStorage nested(observed);
    EXPECT_EQ(IndexError_UnpackIndex, nested.open("", false));
    EXPECT_EQ(1u, observed->calls);
    EXPECT_EQ(0u, observed->fetched_headers);
    EXPECT_FALSE(nested.has("child"));
  }
}

TEST_F(IndependentParserAdmissionTest,
       InputAllocationRefusalReachesOwnedInput) {
  class RefusedInput : public FailingParent {
   public:
    using FailingParent::FailingParent;
    size_t read_for_parsing_with_strategy(size_t offset,
                                          IndexStorage::MemoryBlock &out,
                                          size_t len,
                                          ParsingReadStrategy strategy,
                                          int &) override {
      ++strategy_reads[static_cast<size_t>(strategy)];
      if (offset == 0 && len == sizeof(IndexFormat::MetaHeader))
        ++strategy_headers[static_cast<size_t>(strategy)];
      out.reset();
      throw std::bad_alloc();
    }
  };
  std::string path;
  package(path, "child", 0, 1);
  auto storage = std::make_shared<ChunkReadStorage>();
  ASSERT_EQ(0, storage->init(chunk_params(pool.page_buffer_size())));
  ASSERT_EQ(0, storage->open(path, false));
  auto parent = std::make_shared<RefusedInput>(storage->get("nested", 1));
  EXPECT_EQ(0, parse(parent, "child"));
  EXPECT_EQ(1u, parent->fetched_headers);
  for (size_t i = 0; i < 3; ++i) EXPECT_EQ(1u, parent->strategy_headers[i]);
}

TEST_F(IndependentParserAdmissionTest, FourPlansCoverBudgetBoundaries) {
  const size_t page = ailego::MemoryLimitPool::page_buffer_size();
  std::vector<Shape> shapes(8);
  shapes[0].name_bytes = 2UL << 20;  // Previously missed one-page interval.
  shapes[1].warm = false;
  shapes[1].chunk = page;
  shapes[2].adaptive = false;
  shapes[2].witness = Reference::kOwned;
  shapes[3].warm = false;
  shapes[3].cache = 0;
  shapes[3].funded = true;
  shapes[3].witness = Reference::kReuse;
  shapes[4].chunk = page;
  shapes[4].busy = true;
  shapes[4].witness = Reference::kOwned;
  shapes[5].warm = false;
  shapes[5].cache = 0;
  shapes[5].busy = true;
  shapes[5].witness = Reference::kOwned;
  shapes[6].warm = false;
  shapes[6].cache = 0;
  shapes[6].empty = true;
  shapes[6].witness = Reference::kOwned;
  shapes[7].alias = true;
  shapes[7].witness = Reference::kBorrow;
  for (size_t i = 0; i < shapes.size(); ++i) {
    SCOPED_TRACE(i);
    compare(shapes[i], i);
  }
}

#if defined(__linux__) && !defined(__ANDROID__)
TEST_F(IndependentParserAdmissionTest, FourPlansCoverPrefetchedInput) {
  for (bool warm : {false, true}) {
    for (bool busy : {false, true}) {
      Shape shape;
      shape.name_bytes = 2UL << 20;
      shape.warm = warm;
      shape.prefetch = true;
      shape.busy = busy;
      shape.directory_hint = busy;
      shape.witness = busy ? Reference::kOwned : Reference::kGrow;
      compare(shape, 8 + warm * 2 + busy);
    }
  }
}
#endif
}  // namespace
}  // namespace zvec::core

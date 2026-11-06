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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <ailego/pattern/defer.h>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "utility/utility_params.h"
#include "../../ailego/buffer/pool_payload_budget.h"

using namespace zvec;
using namespace zvec::core;

namespace {

constexpr size_t kPoolSize = 16UL * 1024UL * 1024UL;

class BufferReadStorageTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(kPoolSize));

    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(dumper, nullptr);
    ASSERT_EQ(0, dumper->create(file_path_));

    // Four native pages force read_range's bulk cold-read path on every
    // platform (macOS commonly uses 16 KiB pages, Linux 4 KiB).
    payload_.resize(4UL * ailego::kVectorPageSize);
    for (size_t i = 0; i < payload_.size(); ++i) {
      payload_[i] = static_cast<char>(i % 251);
    }
    ASSERT_EQ(payload_.size(), dumper->write(payload_.data(), payload_.size()));
    ASSERT_EQ(0, dumper->append("payload", payload_.size(), 0, 0));
    ASSERT_EQ(0, dumper->close());
  }

  void TearDown() override {
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
    std::remove(file_path_.c_str());
  }

  IndexStorage::Pointer create_storage(const std::string &warmup_mode) {
    auto storage = IndexFactory::CreateStorage("BufferReadStorage");
    EXPECT_NE(storage, nullptr);
    if (!storage) {
      return nullptr;
    }

    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE, warmup_mode);
    EXPECT_EQ(0, storage->init(params));
    return storage;
  }

  std::string file_path_{"buffer_read_storage_warmup_test_file"};
  std::string payload_;
};

TEST_F(BufferReadStorageTest, NoneDefersPagePopulationUntilFirstRead) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  auto &pool = ailego::PoolPayloadBudget::get_instance();
  EXPECT_EQ(0u, pool.stats().page_used);
  ASSERT_NE(nullptr, storage->vec_buffer_pool());
  EXPECT_TRUE(storage->vec_buffer_pool()->cache_enabled());
  EXPECT_GE(storage->vec_buffer_pool()->file_descriptor(), 0);

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  std::string actual(payload_.size(), '\0');
  ASSERT_EQ(actual.size(), segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(payload_, actual);
  EXPECT_GT(pool.stats().page_used, 0u);
}

TEST_F(BufferReadStorageTest, BudgetedFetchCopiesSurviveStorageClose) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(nullptr, segment);
  auto first = std::allocate_shared<ailego::BudgetedMappedBuffer>(
      ailego::BudgetedAllocator<ailego::BudgetedMappedBuffer>{},
      ailego::kVectorPageSize);
  ailego::BudgetedMappedBuffer second(ailego::kVectorPageSize);
  ASSERT_EQ(first->size(), segment->fetch(3, first->data(), first->size()));
  ASSERT_EQ(second.size(), segment->fetch(7, second.data(), second.size()));
  segment.reset();
  ASSERT_EQ(0, storage->close());
  EXPECT_EQ(0, std::memcmp(payload_.data() + 3, first->data(),
                           ailego::kVectorPageSize));
  EXPECT_EQ(0, std::memcmp(payload_.data() + 7, second.data(),
                           ailego::kVectorPageSize));
  auto copy = first;
  first.reset();
  EXPECT_EQ(0, std::memcmp(payload_.data() + 3, copy->data(),
                           ailego::kVectorPageSize));
}

TEST_F(BufferReadStorageTest, SequentialPreservesExistingWarmupBehavior) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  EXPECT_GT(ailego::PoolPayloadBudget::get_instance().stats().page_used, 0u);

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  std::string actual(payload_.size(), '\0');
  ASSERT_EQ(actual.size(), segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(payload_, actual);
}

TEST_F(BufferReadStorageTest, ResidentScatterReadReturnsPinnedPageSpans) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  const size_t offset = ailego::kVectorPageSize / 2;
  const size_t length = 2 * ailego::kVectorPageSize;
  IndexStorage::Segment::ScatterBlock block;
  ASSERT_EQ(length, segment->read_scatter(offset, block, length));
  EXPECT_GT(block.spans.size(), 1u);
  EXPECT_EQ(length, block.size);
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_UNKNOWN, block.fallback.type_);
  ASSERT_NE(nullptr, block.lease);

  std::string actual;
  actual.reserve(length);
  for (const auto &span : block.spans) {
    actual.append(reinterpret_cast<const char *>(span.data), span.size);
  }
  ASSERT_EQ(length, actual.size());
  EXPECT_EQ(0, std::memcmp(payload_.data() + offset, actual.data(), length));
}

TEST_F(BufferReadStorageTest, ColdScatterReadKeepsContiguousFallback) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  const size_t length = 2 * ailego::kVectorPageSize;
  IndexStorage::Segment::ScatterBlock block;
  ASSERT_EQ(length, segment->read_scatter(0, block, length));
  ASSERT_EQ(1u, block.spans.size());
  EXPECT_EQ(length, block.spans[0].size);
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_SHARED_SCRATCH,
            block.fallback.type_);
  EXPECT_EQ(nullptr, block.lease);
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.spans[0].data, length));
}

TEST_F(BufferReadStorageTest, ScatterCapacityStaysChargedAcrossReset) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(nullptr, segment);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t before = pool.external_used();
  {
    IndexStorage::Segment::ScatterBlock block;
    const size_t length = 2 * ailego::kVectorPageSize;
    ASSERT_EQ(length, segment->read_scatter(0, block, length));
    ASSERT_NE(nullptr, block.lease);
    const size_t capacity = block.spans.capacity();
    const size_t live = pool.external_used();
    EXPECT_GT(live, before + capacity * sizeof(block.spans[0]));
    block.reset();
    EXPECT_EQ(capacity, block.spans.capacity());
    EXPECT_GE(pool.external_used(), before + capacity * sizeof(block.spans[0]));
    EXPECT_LT(pool.external_used(), live);
  }
  EXPECT_EQ(before, pool.external_used());
}

TEST_F(BufferReadStorageTest,
       ScatterAdmissionFailureReleasesPartialDescriptors) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(nullptr, segment);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t length = 2 * ailego::kVectorPageSize;
  // Fail at different allocations, including after the descriptor arrays
  // have been admitted but before the lease can be created.
  for (const size_t headroom : {0u, 32u, 64u, 128u}) {
    ailego::MemoryReservation pressure;
    ASSERT_GT(pool.available(), headroom);
    pressure.add(pool.available() - headroom);
    const size_t charged = pool.external_used();
    IndexStorage::Segment::ScatterBlock block;
    EXPECT_EQ(0u, segment->read_scatter(0, block, length));
    EXPECT_EQ(IndexError_NoMemory, segment->read_error());
    EXPECT_EQ(nullptr, block.lease);
    EXPECT_TRUE(block.spans.empty());
    EXPECT_EQ(0u, block.size);
    EXPECT_EQ(charged, pool.external_used());
    EXPECT_EQ(0u, pool.stats().page_used);
  }
  IndexStorage::Segment::ScatterBlock block;
  ASSERT_EQ(length, segment->read_scatter(0, block, length));
  ASSERT_EQ(1u, block.spans.size());
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.spans[0].data, length));
}

TEST_F(BufferReadStorageTest, ScatterLeaseSurvivesCloseAndThreadTransfer) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(nullptr, segment);
  IndexStorage::Segment::ScatterBlock block;
  const size_t length = 2 * ailego::kVectorPageSize;
  ASSERT_EQ(length, segment->read_scatter(0, block, length));
  ASSERT_NE(nullptr, block.lease);
  segment.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  std::thread reader([owned = std::move(block), this]() {
    size_t offset = 0;
    for (const auto &span : owned.spans) {
      EXPECT_EQ(0, std::memcmp(payload_.data() + offset, span.data, span.size));
      offset += span.size;
    }
    EXPECT_EQ(2 * ailego::kVectorPageSize, offset);
  });
  reader.join();
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
}

TEST_F(BufferReadStorageTest, LargeBatchPointerScratchUsesManagedBacking) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(nullptr, segment);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t before = pool.stats().retained_block_bytes;
  const size_t count =
      ailego::BudgetedAlignedBuffer::kRetainedThreshold / payload_.size() + 1;
  std::vector<IndexStorage::SegmentData> reads(
      count, IndexStorage::SegmentData(0, payload_.size()));
  ASSERT_TRUE(segment->read(reads.data(), reads.size()));
  EXPECT_GE(pool.stats().retained_block_bytes,
            before + ailego::MemoryLimitPool::retained_block_size(
                         count * payload_.size()));
  for (const auto &read : reads) {
    EXPECT_EQ(0, std::memcmp(payload_.data(), read.data, read.length));
  }
  IndexStorage::SegmentData small(7, 32);
  ASSERT_TRUE(segment->read(&small, 1));
  EXPECT_EQ(reads.front().data, small.data);
  EXPECT_EQ(0, std::memcmp(payload_.data() + 7, small.data, small.length));
  segment.reset();
  ASSERT_EQ(0, storage->close());
  EXPECT_EQ(0u, pool.used());
}

TEST_F(BufferReadStorageTest, HostRssPressureReclaimsCacheAndRecovers) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(nullptr, segment);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  IndexStorage::MemoryBlock pinned;
  ASSERT_EQ(32u, segment->read(0, pinned, 32));
  ASSERT_GT(pool.stats().page_used, ailego::kVectorPageSize);
  const size_t baseline = ailego::MemoryHelper::SelfRSS();
  ASSERT_GT(baseline, 0u);
  ASSERT_TRUE(pool.configure_process_rss_limit(baseline + (32UL << 20)));
  AILEGO_DEFER([&] { pool.configure_process_rss_limit(0); });
  // Host allocation deliberately bypasses all zvec accounting.
  constexpr size_t kHostBytes = 40UL << 20;
  void *host = ailego::File::MemoryMap(kHostBytes, 0);
  ASSERT_NE(nullptr, host);
  AILEGO_DEFER([&] {
    if (host) ailego::File::MemoryUnmap(host, kHostBytes);
  });
  std::memset(host, 0x57, kHostBytes);
  const auto wait_until = [](auto ready) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!ready() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return ready();
  };
  ASSERT_TRUE(wait_until([&] {
    return pool.under_process_rss_pressure() &&
           pool.stats().page_used == ailego::kVectorPageSize;
  }));
  EXPECT_EQ(0, std::memcmp(payload_.data(), pinned.data(), 32));
  EXPECT_FALSE(pool.try_charge_external(1UL << 20));
  ailego::File::MemoryUnmap(host, kHostBytes);
  host = nullptr;
  ASSERT_TRUE(wait_until([&] { return !pool.under_process_rss_pressure(); }));
  IndexStorage::MemoryBlock next;
  ASSERT_EQ(payload_.size(), segment->read(0, next, payload_.size()));
  EXPECT_EQ(0, std::memcmp(payload_.data(), next.data(), payload_.size()));
}

TEST_F(BufferReadStorageTest, LargeColdReadMappingFollowsLastReader) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  IndexStorage::MemoryBlock retained;
  const size_t length = 2 * ailego::kVectorPageSize + 1;
  const size_t rounded = ailego::BudgetedMappedBuffer::mapping_size(
      (length + 4095) & ~size_t{4095});
  {
    auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_EQ(0, storage->open(file_path_, false));
    auto segment = storage->get("payload");
    ASSERT_NE(nullptr, segment);
    IndexStorage::MemoryBlock first;
    ASSERT_EQ(length, segment->read(0, first, length));
    ASSERT_EQ(IndexStorage::MemoryBlock::MBT_SHARED_SCRATCH, first.type_);
    const size_t charged = pool.external_used();
    EXPECT_GE(charged, rounded);
    retained = first;
    EXPECT_EQ(first.data(), retained.data());
    EXPECT_EQ(charged, pool.external_used());
    first.reset();
    EXPECT_EQ(charged, pool.external_used());
    ASSERT_EQ(length, segment->read(7, first, length));
    EXPECT_NE(first.data(), retained.data());
    EXPECT_EQ(0, std::memcmp(payload_.data() + 7, first.data(), length));
    EXPECT_EQ(0, std::memcmp(payload_.data(), retained.data(), length));
    segment.reset();
    ASSERT_EQ(0, storage->close());
  }
  // Only the returned mapping and its small, charged control block survive.
  const size_t charge = pool.external_used();
  EXPECT_GE(charge, rounded);
  EXPECT_LT(charge, rounded + 1024);
  EXPECT_EQ(0, std::memcmp(payload_.data(), retained.data(), length));
  std::thread reader(
      [block = std::move(retained)]() mutable { block.reset(); });
  reader.join();
  EXPECT_EQ(0u, pool.external_used());
}

TEST_F(BufferReadStorageTest, RejectsUnknownWarmupMode) {
  auto storage = IndexFactory::CreateStorage("BufferReadStorage");
  ASSERT_NE(storage, nullptr);

  ailego::Params params;
  params.set(BUFFER_READ_STORAGE_WARMUP_MODE, "unknown");
  EXPECT_EQ(IndexError_InvalidArgument, storage->init(params));
}

TEST_F(BufferReadStorageTest, TinyPoolBypassHonorsAlignmentScratchBudget) {
  const size_t kTooSmall = ailego::kVectorPageSize - 1;
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(kTooSmall));

  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  EXPECT_EQ(0u,
            ailego::PoolPayloadBudget::get_instance().stats().metadata_used);
  ASSERT_NE(nullptr, storage->vec_buffer_pool());
  EXPECT_FALSE(storage->vec_buffer_pool()->cache_enabled());
  EXPECT_GE(storage->vec_buffer_pool()->file_descriptor(), 0);

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  std::string actual(payload_.size(), '\0');
  if (storage->vec_buffer_pool()->requires_aligned_io()) {
    // O_DIRECT needs one admitted intermediary page. The legacy path used an
    // unaccounted TLS allocation here; it must now reject instead of bypassing
    // the memory limit. F_NOCACHE can read directly into the destination.
    EXPECT_EQ(0u, segment->fetch(0, actual.data(), actual.size()));
    EXPECT_EQ(IndexError_NoMemory, segment->read_error());
    return;
  }
  ASSERT_EQ(actual.size(), segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(payload_, actual);
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().stats().page_used);
}

TEST_F(BufferReadStorageTest, PoolWithoutRoomForMetadataFallsBackToBypass) {
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(
                   ailego::kVectorPageSize));

  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  EXPECT_EQ(0u,
            ailego::PoolPayloadBudget::get_instance().stats().metadata_used);
  ASSERT_NE(nullptr, storage->vec_buffer_pool());
  EXPECT_FALSE(storage->vec_buffer_pool()->cache_enabled());
  EXPECT_GE(storage->vec_buffer_pool()->file_descriptor(), 0);

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  IndexStorage::MemoryBlock block;
  if (storage->vec_buffer_pool()->requires_aligned_io() ||
      ailego::PoolPayloadBudget::get_instance().available() < 4096) {
    // The segment's scratch-owner control block is charged too. On a 4 KiB
    // native-page platform, this one-page pool cannot also fit a 4 KiB copy.
    EXPECT_EQ(0u, segment->read(0, block, 64));
    EXPECT_EQ(nullptr, block.data());
    return;
  }
  ASSERT_EQ(64u, segment->read(0, block, 64));
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, block.type_);
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().stats().page_used);
}

TEST_F(BufferReadStorageTest, CachePressureFallsBackToOwnedRead) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t external_charge = pool.available() - 4096;
  ASSERT_GT(external_charge, 0u);
  ASSERT_TRUE(pool.try_charge_external(external_charge));
  AILEGO_DEFER([&] { pool.release_external(external_charge); });

  IndexStorage::MemoryBlock block;
  if (storage->vec_buffer_pool()->requires_aligned_io()) {
    EXPECT_EQ(0u, segment->read(0, block, 64));
    EXPECT_EQ(nullptr, block.data());
    return;
  }
  ASSERT_EQ(64u, segment->read(0, block, 64));
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, block.type_);
}

TEST_F(BufferReadStorageTest, ExhaustedScratchBudgetFailsAndCanRetry) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_TRUE(segment);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.available());
    IndexStorage::MemoryBlock block;
    EXPECT_EQ(0u, segment->read(0, block, payload_.size()));
    EXPECT_EQ(nullptr, block.data());
  }
  const size_t before = pool.external_used();
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(payload_.size(), segment->read(0, block, payload_.size()));
  EXPECT_GE(pool.external_used(), before + payload_.size());
  ASSERT_EQ(0, storage->close());
  segment.reset();
  storage.reset();
  EXPECT_GE(pool.external_used(), payload_.size());
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), payload_.size()));
  block.reset();
  EXPECT_EQ(0u, pool.external_used());
}

TEST_F(BufferReadStorageTest, MemoryBlockKeepsPoolAliveAfterStorageClose) {
  IndexStorage::MemoryBlock block;
  IndexStorage::MemoryBlock copy;
  {
    auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_NE(storage, nullptr);
    ASSERT_EQ(0, storage->open(file_path_, false));
    auto segment = storage->get("payload");
    ASSERT_NE(segment, nullptr);
    ASSERT_EQ(64u, segment->read(0, block, 64));
    ASSERT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, block.type_);
    EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
    copy = block;

    ASSERT_EQ(0, storage->close());
    segment.reset();
    storage.reset();
    EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
  }
  block.reset();
  EXPECT_EQ(0, std::memcmp(payload_.data(), copy.data(), 64));
  copy.reset();
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().stats().page_used);
}

TEST_F(BufferReadStorageTest, BorrowedReadAvoidsOwningHandleOnResidentPage) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read_borrowed(0, block, 64));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, block.type_);
  EXPECT_EQ(nullptr, block.buffer_pool_handle_owner_);
  EXPECT_NE(nullptr, block.buffer_pool_handle_);
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));

  // Borrowed blocks must be released while their Segment/storage owner is
  // still alive.
  block.reset();
}

TEST_F(BufferReadStorageTest, MemoryBlockPinsUntilReset) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  IndexStorage::MemoryBlock data;
  ASSERT_EQ(64u, segment->read(0, data, 64));
  ASSERT_NE(data.data(), nullptr);
  EXPECT_EQ(0, std::memcmp(payload_.data(), data.data(), 64));
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t external_charge = pool.available();
  ASSERT_GT(external_charge, 0u);
  ASSERT_TRUE(pool.try_charge_external(external_charge));
  EXPECT_FALSE(pool.try_charge_external(1));

  IndexStorage::SegmentData copied(0, 64);
  const bool copied_ok = segment->read(&copied, 1);
  // The batch-copy scratch must also be admitted while the page stays pinned.
  EXPECT_FALSE(copied_ok);
  if (copied_ok) EXPECT_EQ(0, std::memcmp(payload_.data(), copied.data, 64));
  EXPECT_EQ(0, std::memcmp(payload_.data(), data.data(), 64));
  data.reset();
  bool charged_after_pin_release = false;
  for (size_t attempt = 0; attempt < 100 && !charged_after_pin_release;
       ++attempt) {
    charged_after_pin_release = pool.try_charge_external(1);
    if (!charged_after_pin_release) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  if (charged_after_pin_release) {
    pool.release_external(1);
  }
  pool.release_external(external_charge);
  EXPECT_TRUE(charged_after_pin_release);
  if (!copied_ok) {
    ASSERT_TRUE(segment->read(&copied, 1));
    EXPECT_EQ(0, std::memcmp(payload_.data(), copied.data, 64));
  }
}

TEST_F(BufferReadStorageTest, LastMemoryBlockReleasesScratchBudget) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0u, pool.used());
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto first = storage->get("payload");
  ASSERT_NE(nullptr, first);
  auto second = first->clone();
  ASSERT_NE(nullptr, second);
  IndexStorage::MemoryBlock first_data, second_data;
  ASSERT_EQ(payload_.size(), first->read(0, first_data, payload_.size()));
  ASSERT_EQ(payload_.size(), second->read(0, second_data, payload_.size()));
  EXPECT_GE(pool.external_used(), 2 * payload_.size());
  first.reset();
  first_data.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  EXPECT_GE(pool.external_used(), payload_.size());
  EXPECT_EQ(0,
            std::memcmp(payload_.data(), second_data.data(), payload_.size()));
  second.reset();
  second_data.reset();
  EXPECT_EQ(0u, pool.used());
  EXPECT_EQ(0, pool.init(kPoolSize));
}
TEST_F(BufferReadStorageTest, MemoryBlockReadsAreIndependentAcrossThreads) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  constexpr size_t kReadSize = 64;
  constexpr size_t kOffsets[] = {0, 127};
  std::atomic<size_t> ready{0};
  std::atomic<bool> failed{false};
  std::array<std::thread, 2> readers;
  for (size_t i = 0; i < readers.size(); ++i) {
    readers[i] = std::thread([&, i] {
      IndexStorage::MemoryBlock data;
      if (segment->read(kOffsets[i], data, kReadSize) != kReadSize ||
          data.data() == nullptr) {
        failed.store(true, std::memory_order_release);
        ready.fetch_add(1, std::memory_order_release);
        return;
      }
      ready.fetch_add(1, std::memory_order_release);
      while (ready.load(std::memory_order_acquire) != readers.size()) {
        std::this_thread::yield();
      }
      if (std::memcmp(data.data(), payload_.data() + kOffsets[i], kReadSize) !=
          0) {
        failed.store(true, std::memory_order_release);
      }
    });
  }
  for (auto &reader : readers) {
    reader.join();
  }
  EXPECT_FALSE(failed.load(std::memory_order_acquire));
}

TEST_F(BufferReadStorageTest, MissingFileReturnsErrorWithoutThrowing) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);

  const std::string missing_path = file_path_ + ".missing";
  std::remove(missing_path.c_str());
  EXPECT_EQ(IndexError_OpenFile, storage->open(missing_path, false));
  EXPECT_TRUE(storage->file_path().empty());
  EXPECT_FALSE(storage->has("payload"));
}

TEST_F(BufferReadStorageTest, FailedReopenPreservesPublishedState) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  auto published_segment = storage->get("payload");
  ASSERT_NE(published_segment, nullptr);
  const std::string missing_path = file_path_ + ".missing";
  std::remove(missing_path.c_str());
  EXPECT_EQ(IndexError_OpenFile, storage->open(missing_path, false));

  EXPECT_EQ(file_path_, storage->file_path());
  EXPECT_TRUE(storage->has("payload"));
  std::array<char, 64> actual{};
  EXPECT_EQ(actual.size(),
            published_segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(0, std::memcmp(payload_.data(), actual.data(), actual.size()));
}

TEST_F(BufferReadStorageTest, RejectsOutOfRangeContainerOffset) {
  auto storage = IndexFactory::CreateStorage("BufferReadStorage");
  ASSERT_NE(storage, nullptr);

  ailego::Params params;
  params.set(BUFFER_READ_STORAGE_WARMUP_MODE, BUFFER_READ_STORAGE_WARMUP_NONE);
  params.set(BUFFER_READ_STORAGE_HEADER_OFFSET,
             std::numeric_limits<int64_t>::min());
  ASSERT_EQ(0, storage->init(params));
  EXPECT_EQ(IndexError_InvalidArgument, storage->open(file_path_, false));
}

TEST_F(BufferReadStorageTest, RangeChecksDoNotOverflow) {
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  std::string actual(payload_.size() - 1, '\0');
  EXPECT_EQ(actual.size(), segment->fetch(1, actual.data(),
                                          std::numeric_limits<size_t>::max()));
  EXPECT_EQ(payload_.substr(1), actual);

  IndexStorage::SegmentData invalid(std::numeric_limits<size_t>::max(), 2);
  EXPECT_FALSE(segment->read(&invalid, 1));
  segment->prefetch(std::numeric_limits<size_t>::max(),
                    std::numeric_limits<size_t>::max());
}

}  // namespace

TEST_F(BufferReadStorageTest,
       MetadataAdmissionRejectsThenRetainsNamesUntilClose) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(file_path_));
  const std::string prefix(256, 'm');
  for (size_t i = 0; i < 128; ++i) {
    const char data = static_cast<char>(i);
    ASSERT_EQ(1u, dumper->write(&data, 1));
    ASSERT_EQ(0, dumper->append(prefix + std::to_string(i), 1, 0, 0));
  }
  ASSERT_EQ(0, dumper->close());
  dumper.reset();
  const auto baseline = pool.used();
  for (const char *name :
       {"BufferReadStorage", "MMapFileReadStorage", "FileReadStorage"}) {
    auto storage = IndexFactory::CreateStorage(name);
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init({}));
    ailego::MemoryReservation pressure;
    pressure.add(pool.available());
    EXPECT_EQ(IndexError_NoMemory, storage->open(file_path_, false));
    EXPECT_EQ(pool.capacity(), pool.used());
    pressure.retain_at_most(0);
    EXPECT_EQ(baseline, pool.used());
    ASSERT_EQ(0, storage->open(file_path_, false));
    EXPECT_GT(pool.used() - baseline, 128 * prefix.size());
    EXPECT_TRUE(storage->has(prefix + "127"));
    ASSERT_EQ(0, storage->close());
    EXPECT_EQ(baseline, pool.used());
  }
}

TEST_F(BufferReadStorageTest, LargeMetadataScratchUsesManagedBacking) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(file_path_));
  const std::string prefix(4096, 'm');
  for (size_t i = 0; i < 128; ++i) {
    const char value = static_cast<char>(i);
    ASSERT_EQ(1u, dumper->write(&value, 1));
    ASSERT_EQ(0, dumper->append(prefix + std::to_string(i), 1, 0, 0));
  }
  ASSERT_EQ(0, dumper->close());
  dumper.reset();
  const auto before = pool.stats().retained_block_bytes;
  auto storage = create_storage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_EQ(0, storage->open(file_path_, false));
  // Opening releases the large temporary into the pool's bounded idle reserve.
  EXPECT_GE(pool.stats().retained_block_bytes, before + 128 * prefix.size());
  EXPECT_TRUE(storage->has(prefix + "127"));
  ASSERT_EQ(0, storage->close());
  EXPECT_EQ(0u, pool.used());
}

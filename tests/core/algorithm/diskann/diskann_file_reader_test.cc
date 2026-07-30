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

// Unit tests for diskann_buffer_pool_read -- the core sector->page bridge that
// BufferPoolAlignedFileReader forwards to. Verifies byte-identical reads, cache
// hits on repeated access, unaligned range stitching, correct handling of
// duplicate pages, eviction/reload, and no pin leaks on acquire failure.
//
// NOTE: this targets the shim rather than BufferPoolAlignedFileReader directly
// because the reader header pulls <ailego/io/libaio_loader.h>, which cannot
// coexist in one TU with vector_page_table.h's <zvec/ailego/io/...> copy. The
// reader is exercised end-to-end in diskann_buffer_pool_search_test.cc.

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "diskann_buffer_pool_shim.h"

using zvec::ailego::block_id_t;
using zvec::ailego::kVectorPageSize;
using zvec::ailego::MemoryLimitPool;
using zvec::ailego::VecBufferPool;
using namespace zvec::core;

namespace {

// Create a backing file of `num_pages` pages, page p filled with byte (p&0xff)
// so page content can be verified after arbitrary eviction/reload.
std::string MakeBackingFile(size_t num_pages) {
  static std::atomic<uint64_t> seq{0};
  const size_t ps = kVectorPageSize;
  std::string path =
      "diskann_reader_test_" + std::to_string(seq.fetch_add(1)) + ".bin";
  std::remove(path.c_str());
  FILE *f = std::fopen(path.c_str(), "wb");
  EXPECT_NE(f, nullptr);
  std::vector<char> page(ps);
  for (size_t p = 0; p < num_pages; ++p) {
    std::memset(page.data(), static_cast<int>(p & 0xff), ps);
    EXPECT_EQ(std::fwrite(page.data(), 1, ps, f), ps);
  }
  std::fclose(f);
  return path;
}

void ExpectPageContent(const char *buf, size_t page_id) {
  const size_t ps = kVectorPageSize;
  char expected = static_cast<char>(page_id & 0xff);
  ASSERT_EQ(buf[0], expected) << "page " << page_id << " head mismatch";
  ASSERT_EQ(buf[ps - 1], expected) << "page " << page_id << " tail mismatch";
}

class DiskAnnReaderTest : public ::testing::Test {
 protected:
  void InitPool(size_t capacity_pages) {
    MemoryLimitPool::get_instance().init(capacity_pages * kVectorPageSize);
  }
  std::string NewFile(size_t num_pages) {
    files_.push_back(MakeBackingFile(num_pages));
    return files_.back();
  }
  void TearDown() override {
    for (const auto &p : files_) std::remove(p.c_str());
    files_.clear();
  }
  // Convenience: issue one batch of (offset,len,buf) requests through the shim.
  int ReadReqs(VecBufferPool *pool, const std::vector<uint64_t> &offsets,
               const std::vector<uint64_t> &lens,
               const std::vector<void *> &bufs) {
    return diskann_buffer_pool_read(pool, offsets.data(), lens.data(),
                                    bufs.data(), offsets.size());
  }
  int ReadReqsBypass(VecBufferPool *pool, const std::vector<uint64_t> &offsets,
                     const std::vector<uint64_t> &lens,
                     const std::vector<void *> &bufs) {
    return diskann_buffer_pool_read_bypass(pool, offsets.data(), lens.data(),
                                           bufs.data(), offsets.size());
  }
  std::vector<std::string> files_;
};

}  // namespace

// A single 4KB sector read returns byte-identical content.
TEST_F(DiskAnnReaderTest, SingleSectorRead) {
  InitPool(/*capacity_pages=*/16);
  VecBufferPool pool(NewFile(/*num_pages=*/8), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  std::vector<char> buf(kVectorPageSize);
  ASSERT_EQ(
      ReadReqs(&pool, {3 * kVectorPageSize}, {kVectorPageSize}, {buf.data()}),
      kDiskAnnBufferPoolOk);
  ExpectPageContent(buf.data(), 3);
}

// One request that spans multiple contiguous sectors fills each page slot.
TEST_F(DiskAnnReaderTest, MultiSectorSingleRequest) {
  InitPool(/*capacity_pages=*/16);
  VecBufferPool pool(NewFile(/*num_pages=*/8), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  const size_t pages = 3;  // pages 2,3,4
  std::vector<char> buf(pages * kVectorPageSize);
  ASSERT_EQ(ReadReqs(&pool, {2 * kVectorPageSize}, {pages * kVectorPageSize},
                     {buf.data()}),
            kDiskAnnBufferPoolOk);
  for (size_t k = 0; k < pages; ++k) {
    ExpectPageContent(buf.data() + k * kVectorPageSize, 2 + k);
  }
}

// A batch of requests hitting scattered sectors, including a duplicate page.
TEST_F(DiskAnnReaderTest, ScatteredBatchWithDuplicatePage) {
  InitPool(/*capacity_pages=*/32);
  VecBufferPool pool(NewFile(/*num_pages=*/32), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  const size_t ids[] = {7, 1, 7, 31, 0};
  const size_t n = sizeof(ids) / sizeof(ids[0]);
  std::vector<std::vector<char>> bufs(n, std::vector<char>(kVectorPageSize));
  std::vector<uint64_t> offsets(n), lens(n);
  std::vector<void *> bufptrs(n);
  for (size_t i = 0; i < n; ++i) {
    offsets[i] = ids[i] * kVectorPageSize;
    lens[i] = kVectorPageSize;
    bufptrs[i] = bufs[i].data();
  }

  ASSERT_EQ(ReadReqs(&pool, offsets, lens, bufptrs), kDiskAnnBufferPoolOk);
  for (size_t i = 0; i < n; ++i) {
    ExpectPageContent(bufs[i].data(), ids[i]);
  }
  // All pins from this batch (including the duplicate 7) must be released.
  for (size_t id : ids) {
    EXPECT_TRUE(pool.page_table_.is_released(id));
  }
}

// First read misses; a second read of the same pages hits without new misses.
// `miss` is an exact counter, so the no-new-physical-reads check is precise.
// `hit` is sampled 1/64 and scaled, so the second round reads >=64 resident
// pages to guarantee at least one sampled hit lands and the counter moves.
TEST_F(DiskAnnReaderTest, MissThenHit) {
  const size_t working = 100;  // > kHitSampleRate(64) so a hit is sampled
  InitPool(/*capacity_pages=*/128);
  VecBufferPool pool(NewFile(/*num_pages=*/working), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  std::vector<char> buf(kVectorPageSize);
  auto read_page = [&](size_t p) {
    ASSERT_EQ(
        ReadReqs(&pool, {p * kVectorPageSize}, {kVectorPageSize}, {buf.data()}),
        kDiskAnnBufferPoolOk);
    ExpectPageContent(buf.data(), p);
  };

  for (size_t p = 0; p < working; ++p) read_page(p);
  VecBufferPool::Stats after_first = pool.stats();
  EXPECT_GT(after_first.miss, 0u);

  for (size_t p = 0; p < working; ++p) read_page(p);
  VecBufferPool::Stats after_second = pool.stats();
  EXPECT_EQ(after_second.miss, after_first.miss);  // no new physical reads
  EXPECT_GT(after_second.hit, after_first.hit);
}

// Under a small cap the pool evicts, but re-reading still returns correct data.
TEST_F(DiskAnnReaderTest, EvictionThenReReadCorrect) {
  const size_t num_pages = 64;
  InitPool(/*capacity_pages=*/8);  // working set >> capacity
  VecBufferPool pool(NewFile(num_pages), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  std::vector<char> buf(kVectorPageSize);
  for (int iter = 0; iter < 2; ++iter) {
    for (size_t p = 0; p < num_pages; ++p) {
      ASSERT_EQ(ReadReqs(&pool, {p * kVectorPageSize}, {kVectorPageSize},
                         {buf.data()}),
                kDiskAnnBufferPoolOk);
      ExpectPageContent(buf.data(), p);
    }
  }
  VecBufferPool::Stats s = pool.stats();
  EXPECT_GT(s.evict, 0u);
}

// FileDumper segment offsets may be unaligned even when DiskANN requests a
// whole sector. The bridge must stitch the range from both backing pages.
TEST_F(DiskAnnReaderTest, ReadUnalignedOffsetAcrossPages) {
  InitPool(/*capacity_pages=*/16);
  VecBufferPool pool(NewFile(/*num_pages=*/8), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  std::vector<char> buf(kVectorPageSize);
  ASSERT_EQ(ReadReqs(&pool, {/*offset=*/123}, {kVectorPageSize}, {buf.data()}),
            kDiskAnnBufferPoolOk);
  EXPECT_EQ(0, buf.front());
  EXPECT_EQ(0, buf[kVectorPageSize - 124]);
  EXPECT_EQ(1, buf[kVectorPageSize - 123]);
  EXPECT_EQ(1, buf.back());
}

// The shim also supports a final partial page for generic range reads.
TEST_F(DiskAnnReaderTest, ReadUnalignedLength) {
  InitPool(/*capacity_pages=*/16);
  VecBufferPool pool(NewFile(/*num_pages=*/8), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  std::vector<char> buf(kVectorPageSize + 100);
  ASSERT_EQ(ReadReqs(&pool, {0}, {kVectorPageSize + 100}, {buf.data()}),
            kDiskAnnBufferPoolOk);
  EXPECT_EQ(0, buf.front());
  EXPECT_EQ(0, buf[kVectorPageSize - 1]);
  EXPECT_EQ(1, buf[kVectorPageSize]);
  EXPECT_EQ(1, buf.back());
}

// An out-of-range page makes acquire fail; no pin may leak, and a subsequent
// valid read must still succeed and release all pins.
TEST_F(DiskAnnReaderTest, AcquireFailureLeavesNoPin) {
  const size_t num_pages = 4;
  InitPool(/*capacity_pages=*/8);
  VecBufferPool pool(NewFile(num_pages), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  std::vector<char> buf0(kVectorPageSize);
  std::vector<char> buf1(kVectorPageSize);
  // Page 1 is valid, page num_pages is one past the end -> acquire fails.
  EXPECT_NE(
      ReadReqs(&pool, {1 * kVectorPageSize, num_pages * kVectorPageSize},
               {kVectorPageSize, kVectorPageSize}, {buf0.data(), buf1.data()}),
      kDiskAnnBufferPoolOk);
  EXPECT_TRUE(pool.page_table_.is_released(1));

  // A valid read still works afterwards.
  ASSERT_EQ(
      ReadReqs(&pool, {1 * kVectorPageSize}, {kVectorPageSize}, {buf0.data()}),
      kDiskAnnBufferPoolOk);
  ExpectPageContent(buf0.data(), 1);
  EXPECT_TRUE(pool.page_table_.is_released(1));
}

// Static-cache construction reserves its bytes from the same hard pool. A
// normal cached read cannot allocate a page when that reservation fills the
// pool, while the direct bypass must still read successfully without admitting
// a page.
TEST_F(DiskAnnReaderTest, BypassReadSurvivesFullExternalReservation) {
  InitPool(/*capacity_pages=*/2);
  VecBufferPool pool(NewFile(/*num_pages=*/4), /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  auto &memory_pool = MemoryLimitPool::get_instance();
  ASSERT_TRUE(memory_pool.try_charge_external(memory_pool.capacity()));
  struct ExternalChargeGuard {
    MemoryLimitPool &pool;
    size_t bytes;
    ~ExternalChargeGuard() {
      pool.release_external(bytes);
    }
  } charge_guard{memory_pool, memory_pool.capacity()};

  std::vector<char> buf(kVectorPageSize);
  EXPECT_NE(
      ReadReqs(&pool, {2 * kVectorPageSize}, {kVectorPageSize}, {buf.data()}),
      kDiskAnnBufferPoolOk);
  ASSERT_EQ(ReadReqsBypass(&pool, {2 * kVectorPageSize}, {kVectorPageSize},
                           {buf.data()}),
            kDiskAnnBufferPoolOk);
  ExpectPageContent(buf.data(), 2);
  EXPECT_FALSE(pool.is_page_resident(2));
}

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstring>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/retained_block_allocator.h>
#include <zvec/ailego/buffer/vector_page_table.h>

namespace zvec::ailego {
namespace {
class RetainedBlockAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(64UL << 20));
  }
  void TearDown() override {
    EXPECT_EQ(pool_.queue_metadata_used(), pool_.used());
    EXPECT_LE(pool_.committed(), pool_.capacity());
    EXPECT_EQ(0, pool_.init(0));
    EXPECT_EQ(0u, pool_.stats().retained_block_bytes);
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  using Map = std::unordered_map<
      uint64_t, uint64_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
      RetainedBlockAllocator<std::pair<const uint64_t, uint64_t>>>;
};

TEST_F(RetainedBlockAllocatorTest,
       IdleBlockRetainsBudgetAndReusesAcrossOwners) {
  const size_t baseline = pool_.committed();
  const auto reuses_before = pool_.stats().retained_block_reuses;
  void *first = pool_.acquire_retained_block(100000, true);
  ASSERT_NE(nullptr, first);
  const size_t charge = MemoryLimitPool::retained_block_size(100000);
  EXPECT_EQ(baseline + charge, pool_.committed());
  pool_.release_retained_block(first);
  EXPECT_EQ(baseline, pool_.used());
  EXPECT_EQ(baseline + charge, pool_.committed());
  EXPECT_EQ(charge, pool_.stats().retained_idle_bytes);
  // A different consumer/category can reuse the same backing.
  void *second = pool_.acquire_retained_block(100000, false);
  ASSERT_EQ(first, second);
  EXPECT_EQ(charge, pool_.external_used());
  EXPECT_EQ(baseline + charge, pool_.committed());
  EXPECT_EQ(reuses_before + 1, pool_.stats().retained_block_reuses);
  pool_.release_retained_block(second);
}

TEST_F(RetainedBlockAllocatorTest, PressureUnmapsIdleBlocksBeforeRefunding) {
  const size_t baseline = pool_.committed();
  auto *data = pool_.acquire_retained_block(2UL << 20);
  ASSERT_NE(nullptr, data);
  std::memset(data, 1, 2UL << 20);
  pool_.release_retained_block(data);
  const size_t rest = pool_.capacity() - baseline;
  ASSERT_TRUE(pool_.try_charge_external(rest));
  EXPECT_EQ(0u, pool_.stats().retained_block_bytes);
  EXPECT_EQ(0u, pool_.stats().retained_idle_bytes);
  EXPECT_EQ(pool_.capacity(), pool_.committed());
  pool_.release_external(rest);
}

TEST_F(RetainedBlockAllocatorTest,
       RejectsOverflowAndPreservesActiveAllocation) {
  EXPECT_EQ(nullptr,
            pool_.acquire_retained_block(std::numeric_limits<size_t>::max()));
  auto *data = pool_.acquire_retained_block(100000);
  ASSERT_NE(nullptr, data);
  std::memset(data, 7, 100000);
  EXPECT_EQ(nullptr, pool_.acquire_retained_block(pool_.capacity()));
  EXPECT_EQ(7, static_cast<unsigned char *>(data)[99999]);
  EXPECT_NE(0, pool_.init(0));
  pool_.release_retained_block(data);
}

TEST_F(RetainedBlockAllocatorTest,
       SharedResourceSupportsConcurrentAllocations) {
  RetainedBlockAllocator<uint64_t> allocator;
  std::atomic<bool> okay{true};
  std::vector<std::thread> workers;
  for (uint64_t worker = 0; worker < 4; ++worker) {
    workers.emplace_back([allocator, worker, &okay]() mutable {
      for (size_t round = 0; round < 100; ++round) {
        std::array<uint64_t *, 32> values{};
        for (auto &value : values) {
          value = allocator.allocate(1);
          new (value) uint64_t(worker);
        }
        for (auto *value : values) {
          if (*value != worker) okay = false;
          allocator.deallocate(value, 1);
        }
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_TRUE(okay);
  EXPECT_EQ(0u, allocator.live_bytes());
}
}  // namespace
}  // namespace zvec::ailego

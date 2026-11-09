// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include "db/common/concurrent_roaring_bitmap.h"
#include "db/common/doc_id_buffer.h"
#include "db/index/column/fts_column/posting/posting_build_buffer.h"
#include "db/index/column/inverted_column/inverted_codec.h"
#include "db/index/common/doc_id_map.h"
#include "db/sqlengine/parser/zvec_cached_sql_parser.h"
#include "../ailego/buffer/pool_payload_budget.h"

// Jieba imports Windows macros such as ERROR/IN/CONST. Parse DB headers first.
// clang-format off
#include "db/index/column/fts_column/tokenizer/jieba_tokenizer.h"
// clang-format on

namespace zvec {
namespace {
class DbMemoryStorageTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_.init(8 * 1024 * 1024);
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
};

TEST_F(DbMemoryStorageTest, InvalidPostingPayloadDoesNotChangeBitmap) {
  auto *bitmap = roaring_bitmap_create();
  ASSERT_NE(nullptr, bitmap);
  auto cleanup = ailego::ScopeGuard::Make([&] { roaring_bitmap_free(bitmap); });
  roaring_bitmap_add_range(bitmap, 0, 10000);
  std::string encoded;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(bitmap, &encoded).ok());
  ASSERT_EQ(0, encoded[0]);
  roaring_bitmap_clear(bitmap);
  roaring_bitmap_add(bitmap, 19);
  for (size_t size : {size_t{1}, encoded.size() - 1}) {
    roaring_bitmap_t *decoded = nullptr;
    EXPECT_FALSE(
        InvertedIndexCodec::Deserialize(encoded.data(), size, &decoded).ok());
    EXPECT_EQ(nullptr, decoded);
    if (decoded) roaring_bitmap_free(decoded);
    EXPECT_FALSE(
        InvertedIndexCodec::merge_or(encoded.data(), size, false, bitmap).ok());
    EXPECT_FALSE(
        InvertedIndexCodec::merge_and(encoded.data(), size, bitmap).ok());
    EXPECT_EQ(1u, roaring_bitmap_get_cardinality(bitmap));
    EXPECT_TRUE(roaring_bitmap_contains(bitmap, 19));
  }
}

TEST_F(DbMemoryStorageTest, DeletionBitmapBudgetAndFailedLoadPreserveState) {
  const auto baseline = pool_.used();
  {
    ConcurrentRoaringBitmap64 bitmap;
    ASSERT_TRUE(bitmap.add(7).ok());
    EXPECT_GT(pool_.used(), baseline);
    EXPECT_FALSE(bitmap.contains((uint64_t{1} << 32) + 7));
    EXPECT_EQ(1u, bitmap.range_cardinality(0, uint64_t{1} << 40));
    EXPECT_EQ(0u,
              bitmap.range_cardinality(uint64_t{1} << 32, uint64_t{1} << 40));
    const auto path =
        std::filesystem::temp_directory_path() /
        ("zvec_bitmap_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    auto cleanup =
        ailego::ScopeGuard::Make([&] { std::filesystem::remove(path); });
    ASSERT_TRUE(bitmap.serialize(path.string(), false).ok());
    EXPECT_EQ(StatusCode::ALREADY_EXISTS,
              bitmap.serialize(path.string(), false).code());
    const auto available = pool_.capacity() - pool_.used();
    ASSERT_TRUE(pool_.try_charge_external(available));
    {
      auto release =
          ailego::ScopeGuard::Make([&] { pool_.release_external(available); });
      EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
                bitmap.serialize(path.string(), true).code());
      EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
                bitmap.deserialize(path.string()).code());
      EXPECT_TRUE(bitmap.contains(7));
      ConcurrentRoaringBitmap64 denied;
      EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, denied.add(19).code());
      EXPECT_EQ(0u, denied.cardinality());
      EXPECT_THROW(denied = bitmap, std::bad_alloc);
      EXPECT_EQ(0u, denied.cardinality());
    }
    ConcurrentRoaringBitmap64 loaded;
    ASSERT_TRUE(loaded.deserialize(path.string()).ok());
    EXPECT_TRUE(loaded.contains(7));  // Failed overwrite did not truncate file.
    ASSERT_TRUE(bitmap.add((uint64_t{1} << 40) + 7).ok());
    ASSERT_TRUE(bitmap.serialize(path.string(), true).ok());
    ASSERT_TRUE(loaded.deserialize(path.string()).ok());
    EXPECT_EQ(2u, loaded.cardinality());
    EXPECT_TRUE(loaded.contains((uint64_t{1} << 40) + 7));
    ConcurrentRoaringBitmap64 narrow;
    ASSERT_TRUE(narrow.add(11).ok());
    ASSERT_TRUE(narrow.serialize(path.string(), true).ok());
    ASSERT_TRUE(loaded.deserialize(path.string()).ok());
    EXPECT_FALSE(loaded.contains(7));
    EXPECT_TRUE(loaded.contains(11));
    EXPECT_FALSE(loaded.contains((uint64_t{1} << 40) + 7));
    // A corrupt replacement must not publish its type or erase loaded data.
    {
      std::fstream corrupt(path,
                           std::ios::in | std::ios::out | std::ios::binary);
      ASSERT_TRUE(corrupt.good());
      corrupt.seekg(-1, std::ios::end);
      const int last = corrupt.get();
      ASSERT_NE(std::char_traits<char>::eof(), last);
      corrupt.seekp(-1, std::ios::end);
      corrupt.put(static_cast<char>(last ^ 0x5a));
      ASSERT_TRUE(corrupt.good());
    }
    EXPECT_FALSE(loaded.deserialize(path.string()).ok());
    EXPECT_TRUE(loaded.contains(11));
  }
  EXPECT_EQ(baseline, pool_.used());
}

TEST_F(DbMemoryStorageTest, FailedRangeAdmissionLeavesExistingIdsIntact) {
  DocIdMap ids;
  ASSERT_TRUE(ids.append(10).ok());
  ailego::MemoryReservation blocker;
  blocker.add(pool_.capacity());
  EXPECT_FALSE(ids.prepare_append(20).ok());
  EXPECT_EQ(1u, ids.size());
  EXPECT_EQ(10u, *ids.at(0));
  EXPECT_FALSE(ids.find(20));
  blocker.retain_at_most(0);
  ASSERT_TRUE(ids.append(20).ok());
  EXPECT_EQ(20u, *ids.at(1));
}

TEST_F(DbMemoryStorageTest,
       MatchedIdsRejectInvalidInputAndStopOnConsumerError) {
  DocIdBuffer ids;
  EXPECT_TRUE(ids.append(nullptr, 0).ok());
  EXPECT_FALSE(ids.append(nullptr, 1).ok());
  const uint64_t values[] = {UINT64_MAX, 1, 0};
  EXPECT_FALSE(ids.append(values, SIZE_MAX).ok());
  ailego::MemoryReservation blocker;
  blocker.add(pool_.capacity());
  EXPECT_FALSE(ids.append(values, 3).ok());
  EXPECT_EQ(0u, ids.size());
  blocker.retain_at_most(0);
  ASSERT_TRUE(ids.append(values, 3).ok());
  size_t seen = 0;
  auto status = ids.for_each([&](uint64_t id) {
    EXPECT_EQ(values[seen++], id);
    return Status::InternalError("stop replay");
  });
  EXPECT_FALSE(status.ok());
  EXPECT_EQ("stop replay", status.message());
  EXPECT_EQ(1u, seen);
}

}  // namespace
}  // namespace zvec

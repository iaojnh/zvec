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
#include "db/index/column/fts_column/tokenizer/jieba_tokenizer.h"
#include "db/index/column/inverted_column/inverted_codec.h"
#include "db/index/common/doc_id_map.h"
#include "db/sqlengine/parser/zvec_cached_sql_parser.h"

namespace zvec {
namespace {
class DbMemoryStorageTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_.init(8 * 1024 * 1024);
  }
  ailego::MemoryLimitPool &pool_ = ailego::MemoryLimitPool::get_instance();
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

TEST_F(DbMemoryStorageTest, MillionContiguousIdsNeedOneInlineRange) {
  DocIdMap ids;
  for (size_t i = 0; i < 1000000; ++i) ASSERT_TRUE(ids.append(100 + i).ok());
  EXPECT_EQ(1u, ids.range_count());
  EXPECT_FALSE(ids.paged());
  EXPECT_EQ(0u, pool_.used());
  ASSERT_TRUE(ids.at(999999));
  EXPECT_EQ(1000099u, *ids.at(999999));
  EXPECT_EQ(1234u, *ids.find(1334));
  EXPECT_FALSE(ids.find(99));
  EXPECT_FALSE(ids.at(1000000));
}

TEST_F(DbMemoryStorageTest, OptionalParserCacheEvictsAndPinnedTreesKeepBudget) {
  sqlengine::ZVecCachedSQLParser parser(1), uncached(0);
  auto first = parser.parse_filter("memory_test_field = 1");
  ASSERT_TRUE(first);
  const auto charged = pool_.external_used();
  EXPECT_GT(charged, 0u);
  EXPECT_EQ(first, parser.parse_filter("memory_test_field = 1"));
  auto independent = uncached.parse_filter("memory_test_field = 1");
  ASSERT_TRUE(independent);
  EXPECT_NE(first, independent);
  EXPECT_EQ(charged, pool_.external_used());
  EXPECT_FALSE(pool_.try_charge_external(pool_.capacity()));
  EXPECT_EQ(charged, pool_.external_used());  // Caller still owns first tree.
  first.reset();
  EXPECT_EQ(0u, pool_.external_used());
  {
    ailego::MemoryReservation blocker;
    blocker.add(pool_.capacity());
    // Failing optional cache admission must not fail parsing.
    ASSERT_TRUE(parser.parse_filter("memory_test_field = 2"));
    EXPECT_EQ(pool_.capacity(), pool_.external_used());
  }
  const std::string large =
      "memory_test_field = '" + std::string(40 * 1024, 'x') + "'";
  ASSERT_TRUE(parser.parse_filter(large));
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(DbMemoryStorageTest, JiebaDictionaryAllowanceDenialRetryAndMove) {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("zvec_jieba_budget_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  ASSERT_TRUE(std::filesystem::create_directory(root));
  auto cleanup = ailego::ScopeGuard::Make([&] {
    std::filesystem::remove(root / "jieba.dict.utf8");
    std::filesystem::remove(root);
  });
  {
    std::ofstream dict(root / "jieba.dict.utf8");
    dict << "中文 10 n\n测试 5 n\n";
    ASSERT_TRUE(dict.good());
  }
  ailego::JsonValue parsed;
  const auto json = "{\"cut_mode\":\"full\",\"jieba_dict_dir\":\"" +
                    root.generic_string() + "\"}";
  ASSERT_TRUE(parsed.parse(json.c_str()));
  auto config = parsed.as_object();
  pool_.init(1024 * 1024);
  {
    fts::JiebaTokenizer tokenizer;
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, tokenizer.init(config).code());
    EXPECT_FALSE(tokenizer.is_valid());
    EXPECT_EQ(0u, pool_.external_used());
    pool_.init(32 * 1024 * 1024);
    ASSERT_TRUE(tokenizer.init(config).ok());
    EXPECT_GT(pool_.external_used(), 8u * 1024 * 1024);
    EXPECT_FALSE(tokenizer.tokenize("中文测试").empty());
    fts::JiebaTokenizer moved(std::move(tokenizer));
    EXPECT_FALSE(tokenizer.is_valid());
    EXPECT_TRUE(moved.is_valid());
    fts::JiebaTokenizer replaced;
    ASSERT_TRUE(replaced.init(config).ok());
    replaced = std::move(moved);
    EXPECT_FALSE(moved.is_valid());
    EXPECT_FALSE(replaced.tokenize("中文测试").empty());
    EXPECT_LT(pool_.external_used(), 9u * 1024 * 1024);
  }
  EXPECT_EQ(0u, pool_.external_used());
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

TEST_F(DbMemoryStorageTest, FragmentedIdsRoundTripAcrossPages) {
  DocIdMap ids;
  std::vector<uint64_t> expected;
  std::mt19937 random(19);
  uint64_t id = 9;
  for (size_t i = 0; i < 100000; ++i) {
    id += 1 + random() % 4;
    ASSERT_TRUE(ids.append(id).ok());
    expected.push_back(id);
  }
  ASSERT_TRUE(ids.paged());
  for (size_t i = 0; i < expected.size(); i += 37) {
    auto actual = ids.at(i);
    ASSERT_TRUE(actual);
    EXPECT_EQ(expected[i], *actual);
    auto position = ids.find(expected[i]);
    ASSERT_TRUE(position);
    EXPECT_EQ(i, *position);
    if (i && expected[i] > expected[i - 1] + 1)
      EXPECT_FALSE(ids.find(expected[i] - 1));
  }
  size_t filtered = 0;
  for (auto value : expected) filtered += value % 3 == 0;
  EXPECT_EQ(filtered,
            ids.count_filtered([](uint64_t value) { return value % 3 == 0; }));
}

TEST_F(DbMemoryStorageTest, AbandonedPreparationDoesNotPublishOrLoseRanges) {
  DocIdMap ids;
  ASSERT_TRUE(ids.append(5).ok());
  ASSERT_TRUE(ids.prepare_append(10).ok());
  EXPECT_EQ(1u, ids.size());
  EXPECT_FALSE(ids.find(10));
  ASSERT_TRUE(ids.append(6).ok());
  ASSERT_TRUE(ids.append(20).ok());
  EXPECT_EQ(2u, ids.range_count());
  EXPECT_EQ(20u, *ids.at(2));
  EXPECT_FALSE(ids.find(10));
  EXPECT_FALSE(ids.append(19).ok());
  EXPECT_EQ(3u, ids.size());
}

TEST_F(DbMemoryStorageTest, IdLimitsAndEmptyMapping) {
  DocIdMap ids;
  EXPECT_FALSE(ids.at(0));
  EXPECT_FALSE(ids.find(0));
  ASSERT_TRUE(ids.append(UINT64_MAX - 2).ok());
  ASSERT_TRUE(ids.append(UINT64_MAX).ok());
  EXPECT_EQ(UINT64_MAX, *ids.at(1));
  EXPECT_FALSE(ids.find(UINT64_MAX - 1));
  EXPECT_FALSE(ids.append(0).ok());
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

TEST_F(DbMemoryStorageTest, PostingBlocksMatchArrayEncodingAndReleaseBudget) {
  using namespace fts;
  BM25Scorer scorer;
  scorer.update_stats(100000, 5000000);
  for (size_t count : {0u, 1u, 127u, 128u, 129u, 257u, 10000u}) {
    PostingBuildBuffer rows;
    std::vector<uint32_t> ids, tfs, lengths;
    for (size_t i = 0; i < count; ++i) {
      ids.push_back(static_cast<uint32_t>(3 * i + 5));
      tfs.push_back(static_cast<uint32_t>(i % 7 + 1));
      lengths.push_back(static_cast<uint32_t>(i % 29 + 30));
      rows.append(ids.back(), tfs.back(), lengths.back());
    }
    const auto expected = BitPackedPostingList::encode(
        ids.data(), tfs.data(), lengths.data(), count, count, scorer);
    const size_t before = pool_.external_used();
    {
      auto encoded = rows.encode(count, scorer);
      EXPECT_EQ(expected, encoded.data);
      EXPECT_GE(encoded.reservation.bytes(), encoded.data.size());
      BitPackedPostingIterator iter;
      ASSERT_EQ(0, iter.open(encoded.data.data(), encoded.data.size()));
      for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(ids[i], iter.next_doc());
        EXPECT_EQ(tfs[i], iter.term_freq());
        EXPECT_EQ(lengths[i], iter.doc_len());
      }
      EXPECT_EQ(BitPackedPostingIterator::NO_MORE_DOCS, iter.next_doc());
    }
    EXPECT_EQ(before, pool_.external_used());
    rows.clear();
    rows.append(91, 2, 3);
    auto reused = rows.encode(1, scorer);
    BitPackedPostingIterator iter;
    ASSERT_EQ(0, iter.open(reused.data.data(), reused.data.size()));
    EXPECT_EQ(91u, iter.next_doc());
    EXPECT_EQ(2u, iter.term_freq());
    EXPECT_EQ(3u, iter.doc_len());
    EXPECT_EQ(BitPackedPostingIterator::NO_MORE_DOCS, iter.next_doc());
  }
}

TEST_F(DbMemoryStorageTest, EncodedOutputFailsBeforeUnbudgetedAllocation) {
  using namespace fts;
  pool_.init(1);
  PostingBuildBuffer rows;
  rows.append(1, 2, 3);
  BM25Scorer scorer;
  scorer.update_stats(10, 100);
  EXPECT_THROW(rows.encode(1, scorer), std::bad_alloc);
  EXPECT_EQ(0u, pool_.external_used());
  pool_.init(1024 * 1024);
  EXPECT_NO_THROW(rows.encode(1, scorer));
}

TEST_F(DbMemoryStorageTest, MillionPostingsUsePagedScratchUnderSmallPool) {
  using namespace fts;
  BM25Scorer scorer;
  constexpr uint32_t kCount = 1000003;
  scorer.update_stats(kCount, kCount * 3);
  PostingBuildBuffer rows;
  for (uint32_t i = 0; i < kCount; ++i) rows.append(i * 2, 2, 3);
  ASSERT_TRUE(rows.paged());
  auto encoded = rows.encode(kCount, scorer);
  BitPackedPostingIterator iter;
  ASSERT_EQ(0, iter.open(encoded.data.data(), encoded.data.size()));
  for (uint32_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(i * 2, iter.next_doc());
    ASSERT_EQ(2u, iter.term_freq());
    ASSERT_EQ(3u, iter.doc_len());
  }
  EXPECT_EQ(BitPackedPostingIterator::NO_MORE_DOCS, iter.next_doc());
  EXPECT_LE(pool_.used(), pool_.capacity());
}

TEST_F(DbMemoryStorageTest, ReplayFailureReleasesOutputReservation) {
  using namespace fts;
  BM25Scorer scorer;
  scorer.update_stats(1, 1);
  size_t calls = 0;
  EXPECT_THROW(
      BitPackedPostingList::encode_blocks(
          1, 1, scorer,
          [&](size_t, size_t, uint32_t *id, uint32_t *tf, uint32_t *length) {
            if (++calls == 2) throw std::runtime_error("injected read failure");
            *id = *tf = *length = 1;
          }),
      std::runtime_error);
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(DbMemoryStorageTest, DocLengthsKeepDefaultForHolesAndZeroValues) {
  fts::PostingDocLengths lengths;
  lengths.set(0, 19);
  lengths.set(1000000, 73);
  lengths.set(129, 0);
  EXPECT_EQ(19u, lengths.get(0));
  EXPECT_EQ(73u, lengths.get(1000000));
  EXPECT_EQ(1u, lengths.get(129));
  EXPECT_EQ(1u, lengths.get(1));
  EXPECT_EQ(1u, lengths.get(999999));
  EXPECT_EQ(1u, lengths.get(UINT32_MAX));
}

TEST_F(DbMemoryStorageTest, MatchedIdsLargerThanPoolReplayInOriginalOrder) {
  const size_t count = 2000003;
  DocIdBuffer ids;
  std::array<uint64_t, 513> batch;
  for (size_t offset = 0; offset < count;) {
    size_t n = std::min(batch.size(), count - offset);
    for (size_t i = 0; i < n; ++i) batch[i] = (offset + i) % 997;
    ASSERT_TRUE(ids.append(batch.data(), n).ok());
    offset += n;
  }
  EXPECT_EQ(count, ids.size());
  EXPECT_LE(pool_.used(), pool_.capacity());
  size_t seen = 0;
  ASSERT_TRUE(ids.for_each([&](uint64_t id) {
                   EXPECT_EQ(seen % 997, id);
                   ++seen;
                   return Status::OK();
                 })
                  .ok());
  EXPECT_EQ(count, seen);
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

TEST_F(DbMemoryStorageTest, MatchedIdsAlsoWorkWithoutConfiguredPool) {
  pool_.init(0);
  DocIdBuffer ids;
  const uint64_t value = UINT64_MAX;
  ASSERT_TRUE(ids.append(&value, 1).ok());
  size_t seen = 0;
  EXPECT_TRUE(ids.for_each([&](uint64_t id) {
                   EXPECT_EQ(value, id);
                   ++seen;
                   return Status::OK();
                 })
                  .ok());
  EXPECT_EQ(1u, seen);
}
}  // namespace
}  // namespace zvec

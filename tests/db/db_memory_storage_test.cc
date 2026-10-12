// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "db/common/doc_id_buffer.h"
#include "db/index/column/fts_column/posting/posting_build_buffer.h"
#include "db/index/common/doc_id_map.h"

namespace zvec {
namespace {
class DbMemoryStorageTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_.init(8 * 1024 * 1024);
  }
  ailego::MemoryLimitPool &pool_ = ailego::MemoryLimitPool::get_instance();
};

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

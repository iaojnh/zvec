// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <random>
#include <gtest/gtest.h>
#include "ailego/pattern/defer.h"
#include "db/index/column/fts_column/fts_rocksdb_merge.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec {
namespace {
using Bitmap =
    std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)>;
Bitmap bitmap() {
  return Bitmap(roaring_bitmap_create(), roaring_bitmap_free);
}
std::string encode(roaring_bitmap_t *value, bool scalar) {
  std::string out;
  if (scalar) {
    EXPECT_TRUE(InvertedIndexCodec::Serialize(value, &out).ok());
  } else {
    out.resize(roaring_bitmap_portable_size_in_bytes(value));
    roaring_bitmap_portable_serialize(value, out.data());
  }
  return out;
}
void verify(const std::string &out, bool scalar,
            const roaring_bitmap_t *expected) {
  roaring_bitmap_t *raw = nullptr;
  if (scalar)
    ASSERT_TRUE(
        InvertedIndexCodec::Deserialize(out.data(), out.size(), &raw).ok());
  else
    raw = roaring_bitmap_portable_deserialize_safe(out.data(), out.size());
  Bitmap actual(raw, roaring_bitmap_free);
  ASSERT_NE(nullptr, actual);
  EXPECT_TRUE(roaring_bitmap_equals(actual.get(), expected));
}

TEST(PostingMergeTest, MixedContainersMatchRoaringUnionAndPartialMerge) {
  std::mt19937 random(123);
  for (bool scalar : {false, true}) {
    for (uint32_t n : {1u, 3u, 4u, 17u}) {
      std::vector<std::string> values;
      auto expected = bitmap();
      for (uint32_t operand = 0; operand < 4; ++operand) {
        auto input = bitmap();
        for (uint32_t key = 0; key < n; ++key) {
          const uint32_t high = key == n - 1 ? 65535 : key;
          if (operand == 0)
            roaring_bitmap_add_range(input.get(), (uint64_t(high) << 16) + 10,
                                     (uint64_t(high) << 16) + 8010);
          else
            for (uint32_t i = 0; i < (operand == 1 ? 9000u : 80u); ++i)
              roaring_bitmap_add(input.get(),
                                 (high << 16) | (random() & 65535));
        }
        roaring_bitmap_add(input.get(), UINT32_MAX);
        if (operand % 2 == 0) roaring_bitmap_run_optimize(input.get());
        roaring_bitmap_or_inplace(expected.get(), input.get());
        values.push_back(encode(input.get(), scalar));
      }
      std::vector<rocksdb::Slice> inputs;
      for (const auto &value : values) inputs.emplace_back(value);
      std::string output;
      ASSERT_TRUE(detail::PostingMerge::merge(inputs.data(), inputs.size(),
                                              nullptr, scalar, &output));
      verify(output, scalar, expected.get());
      auto merger = scalar ? std::shared_ptr<rocksdb::MergeOperator>(
                                 new InvertedRocksdbValueMerger)
                           : std::shared_ptr<rocksdb::MergeOperator>(
                                 new fts::FtsPostingsMerge);
      std::string partial = values.front();
      for (size_t i = 1; i < values.size(); ++i) {
        std::string next;
        ASSERT_TRUE(
            merger->PartialMerge({}, partial, values[i], &next, nullptr));
        partial.swap(next);
      }
      verify(partial, scalar, expected.get());
    }
  }
}

TEST(PostingMergeTest, PackedFtsAndSmallScalarListsRemainCompatible) {
  auto expected = bitmap();
  std::vector<uint32_t> ids, frequencies(3000, 1), lengths(3000, 10);
  for (uint32_t i = 0; i < 3000; ++i) {
    ids.push_back(i * 333);
    roaring_bitmap_add(expected.get(), ids.back());
  }
  fts::BM25Scorer scorer;
  scorer.update_stats(3000, 30000);
  auto packed = fts::BitPackedPostingList::encode(
      ids.data(), frequencies.data(), lengths.data(), ids.size(), ids.size(),
      scorer);
  auto extra = bitmap();
  roaring_bitmap_add_range(extra.get(), 65000, 66000);
  auto encoded = encode(extra.get(), false);
  roaring_bitmap_or_inplace(expected.get(), extra.get());
  const rocksdb::Slice inputs[]{packed, encoded};
  std::string output;
  ASSERT_TRUE(detail::PostingMerge::merge(inputs, 2, nullptr, false, &output));
  verify(output, false, expected.get());
  for (uint32_t n : {1u, 3u, 4u}) {
    auto a = bitmap();
    roaring_bitmap_add(a.get(), 0);
    auto b = bitmap();
    for (uint32_t i = 0; i < n; ++i) roaring_bitmap_add(b.get(), i * 65536);
    auto first = encode(a.get(), true), second = encode(b.get(), true);
    const rocksdb::Slice pair[]{first, second};
    ASSERT_TRUE(detail::PostingMerge::merge(pair, 2, nullptr, true, &output));
    verify(output, true, b.get());
    EXPECT_EQ(n <= 3 ? n : 0, static_cast<unsigned char>(output[0]));
  }
}

TEST(PostingMergeTest, RunEmissionHandlesGapsAndWordBoundaries) {
  for (uint32_t end : {63u, 64u, 127u, 255u, 65535u}) {
    auto input = bitmap();
    roaring_bitmap_add_range(input.get(), 0, uint64_t(end) + 1);
    roaring_bitmap_remove_range(input.get(), 64, 128);
    roaring_bitmap_run_optimize(input.get());
    auto empty = bitmap();
    for (bool scalar : {false, true}) {
      if (scalar) roaring_bitmap_add(empty.get(), end);
      auto first = encode(input.get(), scalar);
      auto second = encode(empty.get(), scalar);
      auto expected = bitmap();
      roaring_bitmap_or_inplace(expected.get(), input.get());
      roaring_bitmap_or_inplace(expected.get(), empty.get());
      // Repeated operands force the container-streaming path, including when
      // both encoded inputs would otherwise fit the native small-merge path.
      const rocksdb::Slice pair[]{first, second, first, first};
      std::string out;
      ASSERT_TRUE(detail::PostingMerge::merge(pair, 4, nullptr, scalar, &out));
      verify(out, scalar, expected.get());
    }
  }
}

TEST(PostingMergeTest, FullMergeBorrowsSingleValueAndRejectsCorruption) {
  for (bool scalar : {false, true}) {
    auto merger = scalar ? std::shared_ptr<rocksdb::MergeOperator>(
                               new InvertedRocksdbValueMerger)
                         : std::shared_ptr<rocksdb::MergeOperator>(
                               new fts::FtsPostingsMerge);
    auto a = bitmap();
    roaring_bitmap_add(a.get(), 7);
    const auto value = encode(a.get(), scalar);
    const rocksdb::Slice key("key"), base(value);
    std::vector<rocksdb::Slice> operands{base};
    std::string owned;
    rocksdb::Slice borrowed;
    rocksdb::MergeOperator::MergeOperationOutput out(owned, borrowed);
    ASSERT_TRUE(merger->FullMergeV2(
        rocksdb::MergeOperator::MergeOperationInput{key, nullptr, operands,
                                                    nullptr},
        &out));
    EXPECT_EQ(base.data(), borrowed.data());
    EXPECT_TRUE(owned.empty());
    operands.clear();
    borrowed = {};
    ASSERT_TRUE(merger->FullMergeV2(
        rocksdb::MergeOperator::MergeOperationInput{key, &base, operands,
                                                    nullptr},
        &out));
    EXPECT_EQ(base.data(), borrowed.data());
    operands.emplace_back("broken");
    borrowed = {};
    EXPECT_FALSE(merger->FullMergeV2(
        rocksdb::MergeOperator::MergeOperationInput{key, &base, operands,
                                                    nullptr},
        &out));
    EXPECT_FALSE(merger->PartialMerge(key, base, "broken", &owned, nullptr));
  }
}

TEST(PostingMergeTest, CacheBudgetPressureDoesNotPoisonMandatoryMerge) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  pool.init(1);
  auto cleanup = ailego::ScopeGuard::Make([&] { pool.init(0); });
  auto a = bitmap();
  auto b = bitmap();
  for (uint32_t i = 0; i < 1000000; ++i) {
    roaring_bitmap_add(a.get(), i * 7);
    roaring_bitmap_add(b.get(), i * 7 + 1);
  }
  auto first = encode(a.get(), false), second = encode(b.get(), false);
  const rocksdb::Slice inputs[]{first, second};
  roaring_bitmap_or_inplace(a.get(), b.get());
  std::string out;
  ASSERT_TRUE(detail::PostingMerge::merge(inputs, 2, nullptr, false, &out));
  verify(out, false, a.get());
  EXPECT_EQ(0u, pool.external_used());
}
}  // namespace
}  // namespace zvec

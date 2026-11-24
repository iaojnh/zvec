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

#include <algorithm>
#include <cstring>
#include <optional>
#include <random>
#include <gtest/gtest.h>
#include <rocksdb/utilities/stackable_db.h>
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"
#include "tests/test_util.h"

namespace zvec {
namespace {

std::string Raw(int32_t value) {
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}
int32_t Decode(const std::string &value) {
  uint32_t bits = 0;
  for (unsigned char c : value) bits = (bits << 8) | c;
  bits ^= uint32_t{1} << 31;
  int32_t number;
  std::memcpy(&number, &bits, sizeof(number));
  return number;
}

class FailPrefixPut : public rocksdb::StackableDB {
 public:
  explicit FailPrefixPut(rocksdb::DB *db)
      : StackableDB(std::shared_ptr<rocksdb::DB>(db, [](rocksdb::DB *) {})) {}
  rocksdb::Status Put(const rocksdb::WriteOptions &options,
                      rocksdb::ColumnFamilyHandle *cf,
                      const rocksdb::Slice &key,
                      const rocksdb::Slice &value) override {
    if (cf->GetName() == "value$PREFIXES" && !key.empty() && ++writes_ == 3)
      return rocksdb::Status::IOError("injected prefix write failure");
    return StackableDB::Put(options, cf, key, value);
  }

 private:
  int writes_{0};
};

class InvertedRangePrefixTest : public testing::Test {
 protected:
  void SetUp() override {
    path_ = std::string("./range_prefix_") +
            testing::UnitTest::GetInstance()->current_test_info()->name();
    test_util::RemoveTestPath(path_);
  }
  void TearDown() override {
    column_.reset();
    if (context_.db_) context_.close();
    test_util::RemoveTestPath(path_);
  }
  void create(DataType type = DataType::INT32) {
    field_ = FieldSchema("value", type, true,
                         std::make_shared<InvertIndexParams>(true, false));
    ASSERT_TRUE(context_
                    .create(path_, {"value$TERMS", "value$RANGES", INVERT_CDF},
                            std::make_shared<InvertedRocksdbValueMerger>())
                    .ok());
    column_ = InvertedColumnIndexer::CreateAndOpen("prefix_test", field_,
                                                   context_, false);
    ASSERT_NE(column_, nullptr);
  }
  void reopen() {
    column_.reset();
    ASSERT_TRUE(context_.flush().ok());
    ASSERT_TRUE(context_.close().ok());
    ASSERT_TRUE(
        context_
            .open(path_, true, std::make_shared<InvertedRocksdbValueMerger>())
            .ok());
    column_ = InvertedColumnIndexer::CreateAndOpen("prefix_test", field_,
                                                   context_, true);
    ASSERT_NE(column_, nullptr);
  }
  std::vector<std::optional<int32_t>> values(bool nulls = true,
                                             bool hot = false) {
    std::vector<std::optional<int32_t>> data;
    for (int32_t i = 0; i < 8000; ++i)
      data.push_back(nulls && i % 97 == 0
                         ? std::optional<int32_t>()
                         : std::optional<int32_t>(
                               hot && i > 1000 && i < 7000 ? 0 : i - 4000));
    std::mt19937 random(881);
    std::shuffle(data.begin(), data.end(), random);
    for (uint32_t i = 0; i < data.size(); ++i)
      EXPECT_TRUE((data[i] ? column_->insert(i, Raw(*data[i]))
                           : column_->insert_null(i))
                      .ok());
    return data;
  }
  template <class T, class Encode>
  void verify(const std::vector<std::optional<T>> &data,
              const std::vector<T> &bounds, Encode encode) {
    for (const auto &bound : bounds)
      for (CompareOp op :
           {CompareOp::LT, CompareOp::LE, CompareOp::GT, CompareOp::GE}) {
        auto result = column_->search(encode(bound), op);
        ASSERT_NE(result, nullptr);
        auto it = result->create_iterator();
        size_t count = 0;
        for (uint32_t id = 0; id < data.size(); ++id) {
          if (!data[id]) continue;
          bool match = op == CompareOp::LT   ? *data[id] < bound
                       : op == CompareOp::LE ? *data[id] <= bound
                       : op == CompareOp::GT ? *data[id] > bound
                                             : *data[id] >= bound;
          if (!match) continue;
          ASSERT_TRUE(it->valid());
          ASSERT_EQ(it->doc_id(), id);
          it->next();
          ++count;
        }
        EXPECT_FALSE(it->valid());
        EXPECT_EQ(result->count(), count);
      }
  }
  std::unique_ptr<InvertedRangePrefix> metadata(
      const std::vector<std::optional<int32_t>> &data) {
    InvertedRangePrefix::BitmapPtr nulls(roaring_bitmap_create());
    for (uint32_t i = 0; i < data.size(); ++i)
      if (!data[i]) roaring_bitmap_add(nulls.get(), i);
    return InvertedRangePrefix::Load(context_,
                                     context_.get_cf("value$PREFIXES"),
                                     data.size() - 1, nulls.get());
  }
  std::string path_;
  FieldSchema field_;
  RocksdbContext context_;
  InvertedColumnIndexer::Ptr column_;
};

TEST_F(InvertedRangePrefixTest, PersistsSixteenQuantilesAndReopensReadOnly) {
  create();
  auto data = values();
  ASSERT_TRUE(column_->seal().ok());
  auto index = metadata(data);
  ASSERT_NE(index, nullptr);
  ASSERT_EQ(index->prefixes.size(), 15);
  std::vector<int32_t> bounds{-5000, -1, 0, 1, 5000};
  for (const auto &prefix : index->prefixes) {
    const auto boundary = Decode(index->buckets[prefix.end_bucket - 1].end);
    bounds.insert(bounds.end(), {boundary - 1, boundary, boundary + 1});
    std::string value;
    ASSERT_TRUE(context_.db_
                    ->Get(context_.read_opts_,
                          context_.get_cf("value$PREFIXES"),
                          index->buckets[prefix.end_bucket - 1].key, &value)
                    .ok());
    roaring_bitmap_t *raw = nullptr;
    ASSERT_TRUE(
        InvertedIndexCodec::Deserialize(value.data(), value.size(), &raw).ok());
    InvertedRangePrefix::BitmapPtr bitmap(raw);
    EXPECT_EQ(roaring_bitmap_get_cardinality(bitmap.get()),
              index->buckets[prefix.end_bucket - 1].cumulative_cardinality);
  }
  verify(data, bounds, Raw);
  reopen();
  verify(data, bounds, Raw);
}
TEST_F(InvertedRangePrefixTest, MissingPrefixReadFallsBackWithoutHeapSnapshot) {
  create();
  auto data = values(false);
  ASSERT_TRUE(column_->seal().ok());
  auto index = metadata(data);
  ASSERT_NE(index, nullptr);
  const auto &prefix = index->prefixes[7];
  const auto key = index->buckets[prefix.end_bucket - 1].key;
  const auto bound = Decode(index->buckets[prefix.end_bucket - 1].end);
  ASSERT_TRUE(
      context_.db_
          ->Delete(context_.write_opts_, context_.get_cf("value$PREFIXES"), key)
          .ok());
  verify(data, {bound}, Raw);
  ASSERT_TRUE(context_.db_
                  ->Put(context_.write_opts_, context_.get_cf("value$PREFIXES"),
                        key, "invalid")
                  .ok());
  EXPECT_EQ(column_->search(Raw(bound), CompareOp::LE), nullptr);
}
TEST_F(InvertedRangePrefixTest, OpensOldIndexWithoutPrefixColumnFamily) {
  create();
  auto data = values();
  ASSERT_TRUE(column_->seal().ok());
  column_.reset();
  ASSERT_TRUE(context_.drop_cf("value$PREFIXES").ok());
  reopen();
  verify(data, {-5000, -1, 0, 1, 5000}, Raw);
  EXPECT_EQ(context_.get_cf("value$PREFIXES"), nullptr);
}
TEST_F(InvertedRangePrefixTest,
       RepeatedValuesAndNullsDoNotDuplicateCheckpoints) {
  create();
  auto data = values(true, true);
  ASSERT_TRUE(column_->seal().ok());
  auto index = metadata(data);
  ASSERT_NE(index, nullptr);
  EXPECT_LT(index->prefixes.size(), 15);
  size_t previous = 0;
  for (const auto &prefix : index->prefixes) {
    EXPECT_GT(prefix.end_bucket, previous);
    previous = prefix.end_bucket;
  }
  verify(data, {-5000, -3000, -1, 0, 1, 3000, 5000}, Raw);
  reopen();
  verify(data, {-1, 0, 1}, Raw);
}
TEST_F(InvertedRangePrefixTest,
       FailedSealCanRetryWithoutPartialPrefixManifest) {
  create();
  auto data = values(false);
  auto db = std::move(context_.db_);
  context_.db_ = std::make_unique<FailPrefixPut>(db.get());
  const auto status = column_->seal();
  context_.db_ = std::move(db);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(column_->is_sealed());
  std::string value;
  EXPECT_TRUE(context_.db_
                  ->Get(context_.read_opts_, context_.get_cf("value$PREFIXES"),
                        "", &value)
                  .IsNotFound());
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(metadata(data), nullptr);
  reopen();
  verify(data, {-5000, -1, 0, 1, 5000}, Raw);
}
TEST_F(InvertedRangePrefixTest, BinaryStringsKeepRangeSemantics) {
  create(DataType::STRING);
  std::vector<std::optional<std::string>> data;
  for (size_t i = 0; i < 8000; ++i)
    data.emplace_back(std::string(1, char(i % 256)) + std::to_string(i) +
                      std::string("\0end", 4));
  std::mt19937 random(881);
  std::shuffle(data.begin(), data.end(), random);
  for (uint32_t i = 0; i < data.size(); ++i)
    ASSERT_TRUE(column_->insert(i, *data[i]).ok());
  ASSERT_TRUE(column_->seal().ok());
  std::vector<std::string> bounds{"", std::string(1, '\0'),
                                  std::string(1, char(255)), *data[23],
                                  *data[5000]};
  auto encode = [](const std::string &value) { return value; };
  verify(data, bounds, encode);
  reopen();
  verify(data, bounds, encode);
}
TEST_F(InvertedRangePrefixTest, DropFieldRemovesPersistedPrefixData) {
  FieldSchema field("value", DataType::INT32, true,
                    std::make_shared<InvertIndexParams>(true, false));
  auto index = InvertedIndexer::CreateAndOpen("prefix_test", path_, true,
                                              {field}, false);
  ASSERT_NE(index, nullptr);
  auto column = (*index)["value"];
  for (int32_t i = 0; i < 8000; ++i)
    ASSERT_TRUE(column->insert(i, Raw(i)).ok());
  ASSERT_TRUE(index->seal().ok());
  column.reset();
  ASSERT_TRUE(index->remove_column_indexer("value").ok());
  ASSERT_TRUE(index->create_column_indexer(field).ok());
  column = (*index)["value"];
  ASSERT_TRUE(column->insert(0, Raw(7)).ok());
  ASSERT_TRUE(index->seal().ok());
  auto result = column->search(Raw(100), CompareOp::LT);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count(), 1);
}

TEST_F(InvertedRangePrefixTest, UnsupportedBucketOrderKeepsSealCompatible) {
  create(DataType::STRING);
  for (uint32_t id = 0; id < 8000; ++id) {
    const auto value =
        id < 16 ? "a" + std::string(id, '\0') : "z" + std::to_string(id);
    ASSERT_TRUE(column_->insert(id, value).ok());
  }
  ASSERT_TRUE(column_->seal().ok());
  std::string value;
  EXPECT_TRUE(context_.db_
                  ->Get(context_.read_opts_, context_.get_cf("value$PREFIXES"),
                        "", &value)
                  .IsNotFound());
  reopen();
  auto result = column_->search("a", CompareOp::EQ);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count(), 1);
}

}  // namespace
}  // namespace zvec

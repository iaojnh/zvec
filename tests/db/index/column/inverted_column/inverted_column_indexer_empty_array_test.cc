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

#include <new>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <rocksdb/utilities/stackable_db.h>
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "tests/test_util.h"

namespace zvec {
namespace {

class EmptyArrayIndexTest : public testing::TestWithParam<DataType> {
 protected:
  void SetUp() override {
    test_util::RemoveTestPath(path_);
  }

  void TearDown() override {
    indexer_.reset();
    test_util::RemoveTestPath(path_);
  }

  void expect_ids(const InvertedSearchResult::Ptr &result,
                  const std::vector<uint32_t> &expected) {
    ASSERT_TRUE(result);
    ASSERT_EQ(result->count(), expected.size());
    std::vector<uint32_t> actual;
    result->extract_ids(&actual);
    EXPECT_EQ(actual, expected);
  }

  void verify(const std::string &term) {
    auto column = (*indexer_)["values"];
    ASSERT_TRUE(column);
    expect_ids(column->search_non_null(), {0, 1, 2});
    expect_ids(column->search_null(), {3});
    expect_ids(column->search_array_len(0, CompareOp::EQ), {0});
    expect_ids(column->multi_search({term}, CompareOp::CONTAIN_ANY), {1});
    expect_ids(column->multi_search({term}, CompareOp::CONTAIN_ALL), {1});
    expect_ids(column->multi_search({term}, CompareOp::NOT_CONTAIN_ANY),
               {0, 2});
    expect_ids(column->multi_search({term}, CompareOp::NOT_CONTAIN_ALL),
               {0, 2});
  }

  const std::string path_{"./empty_array_index"};
  InvertedIndexer::Ptr indexer_;
};

TEST_P(EmptyArrayIndexTest, NonNullAndNegatedContainAcrossReopenAndSeal) {
  const FieldSchema field{"values", GetParam(), true,
                          std::make_shared<InvertIndexParams>()};
  indexer_ =
      InvertedIndexer::CreateAndOpen("test", path_, true, {field}, false);
  ASSERT_TRUE(indexer_);
  auto column = (*indexer_)["values"];
  ASSERT_TRUE(column);
  std::string term;
  if (GetParam() == DataType::ARRAY_STRING) {
    ASSERT_TRUE(column->insert(0, std::vector<std::string>{}).ok());
    ASSERT_TRUE(column->insert(1, std::vector<std::string>{"a"}).ok());
    ASSERT_TRUE(column->insert(2, std::vector<std::string>{"b"}).ok());
    term = "a";
  } else if (GetParam() == DataType::ARRAY_BOOL) {
    ASSERT_TRUE(column->insert(0, std::vector<bool>{}).ok());
    ASSERT_TRUE(column->insert(1, std::vector<bool>{true}).ok());
    ASSERT_TRUE(column->insert(2, std::vector<bool>{false}).ok());
    term = "true";
  } else {
    const int32_t match = 1, other = 2;
    term.assign(reinterpret_cast<const char *>(&match), sizeof(match));
    ASSERT_TRUE(column->insert(0, std::string{}).ok());
    ASSERT_TRUE(column->insert(1, term).ok());
    ASSERT_TRUE(
        column
            ->insert(2, std::string(reinterpret_cast<const char *>(&other),
                                    sizeof(other)))
            .ok());
  }
  ASSERT_TRUE(column->insert_null(3).ok());
  column.reset();

  {
    SCOPED_TRACE("streaming");
    verify(term);
  }
  ASSERT_TRUE(indexer_->flush().ok());
  indexer_.reset();
  indexer_ =
      InvertedIndexer::CreateAndOpen("test", path_, false, {field}, false);
  ASSERT_TRUE(indexer_);
  {
    SCOPED_TRACE("reopened streaming");
    verify(term);
  }
  ASSERT_TRUE(indexer_->seal().ok());
  ASSERT_TRUE((*indexer_)["values"]->is_sealed());
  {
    SCOPED_TRACE("sealed");
    verify(term);
  }
  indexer_.reset();
  indexer_ =
      InvertedIndexer::CreateAndOpen("test", path_, false, {field}, true);
  ASSERT_TRUE(indexer_);
  {
    SCOPED_TRACE("reopened sealed");
    verify(term);
  }
}

INSTANTIATE_TEST_SUITE_P(ArrayTypes, EmptyArrayIndexTest,
                         testing::Values(DataType::ARRAY_STRING,
                                         DataType::ARRAY_INT32,
                                         DataType::ARRAY_BOOL));

class DeniedPostingIterator : public rocksdb::Iterator {
 public:
  explicit DeniedPostingIterator(rocksdb::Iterator *iterator)
      : iterator_(iterator) {}

  bool Valid() const override {
    return !denied_ && iterator_->Valid();
  }
  void SeekToFirst() override {
    denied_ = false;
    iterator_->SeekToFirst();
  }
  void SeekToLast() override {
    denied_ = false;
    iterator_->SeekToLast();
  }
  void Seek(const rocksdb::Slice &target) override {
    denied_ = false;
    iterator_->Seek(target);
  }
  void SeekForPrev(const rocksdb::Slice &target) override {
    denied_ = false;
    iterator_->SeekForPrev(target);
  }
  void Next() override {
    denied_ = true;
  }
  void Prev() override {
    denied_ = true;
  }
  rocksdb::Slice key() const override {
    return iterator_->key();
  }
  rocksdb::Slice value() const override {
    return iterator_->value();
  }
  rocksdb::Status status() const override {
    return denied_ ? rocksdb::Status::MemoryLimit() : iterator_->status();
  }

 private:
  std::unique_ptr<rocksdb::Iterator> iterator_;
  bool denied_{false};
};

class DenyPostingReadDB : public rocksdb::StackableDB {
 public:
  explicit DenyPostingReadDB(rocksdb::DB *db) : StackableDB(db) {}

  rocksdb::Status Get(const rocksdb::ReadOptions &options,
                      rocksdb::ColumnFamilyHandle *column,
                      const rocksdb::Slice &key,
                      rocksdb::PinnableSlice *value) override {
    if (deny_get) return rocksdb::Status::MemoryLimit();
    return GetBaseDB()->Get(options, column, key, value);
  }

  rocksdb::Iterator *NewIterator(
      const rocksdb::ReadOptions &options,
      rocksdb::ColumnFamilyHandle *column) override {
    auto *iterator = GetBaseDB()->NewIterator(options, column);
    return deny_next ? new DeniedPostingIterator(iterator) : iterator;
  }

  bool deny_get{false};
  bool deny_next{false};
};

TEST(InvertedPostingBudgetTest, RejectsPartialReadsAndAllowsSealRetry) {
  const std::string path = "./inverted_posting_budget";
  test_util::RemoveTestPath(path);
  AILEGO_DEFER([&]() { test_util::RemoveTestPath(path); });

  RocksdbContext context;
  const FieldSchema text{"text", DataType::STRING, true,
                         std::make_shared<InvertIndexParams>(true, true)};
  const FieldSchema tags{"tags", DataType::ARRAY_STRING, true,
                         std::make_shared<InvertIndexParams>()};
  ASSERT_TRUE(context
                  .create(path,
                          {"text" + INVERT_SUFFIX_TERMS,
                           "text" + INVERT_SUFFIX_RANGES,
                           "text" + INVERT_SUFFIX_REVERSED_TERMS,
                           "tags" + INVERT_SUFFIX_TERMS,
                           "tags" + INVERT_SUFFIX_ARRAY_LEN, INVERT_CDF},
                          std::make_shared<InvertedRocksdbValueMerger>())
                  .ok());
  AILEGO_DEFER([&]() { EXPECT_TRUE(context.close().ok()); });
  auto *db = new DenyPostingReadDB(context.db_.release());
  context.db_.reset(db);
  db->deny_get = true;
  Status open_status;
  EXPECT_FALSE(InvertedColumnIndexer::CreateAndOpen(
      "test", text, context, false, &open_status));
  EXPECT_EQ(open_status.code(), StatusCode::RESOURCE_EXHAUSTED);
  db->deny_get = false;

  auto column = InvertedColumnIndexer::CreateAndOpen("test", text, context);
  auto array = InvertedColumnIndexer::CreateAndOpen("test", tags, context);
  ASSERT_TRUE(column);
  ASSERT_TRUE(array);
  ASSERT_TRUE(column->insert(0, std::string("alpha")).ok());
  ASSERT_TRUE(column->insert(1, std::string("alfa")).ok());
  ASSERT_TRUE(column->insert(2, std::string("beta")).ok());
  ASSERT_TRUE(array->insert(0, std::vector<std::string>{"alpha"}).ok());
  ASSERT_TRUE(array->insert(1, std::vector<std::string>{"alfa", "beta"}).ok());

  db->deny_get = true;
  EXPECT_THROW(column->search("alpha", CompareOp::EQ), std::bad_alloc);
  EXPECT_THROW(array->multi_search({"alpha"}, CompareOp::CONTAIN_ANY),
               std::bad_alloc);
  EXPECT_THROW(array->search_array_len(1, CompareOp::EQ), std::bad_alloc);
  db->deny_get = false;

  db->deny_next = true;
  EXPECT_THROW(column->search("beta", CompareOp::NE), std::bad_alloc);
  EXPECT_THROW(column->search("beta", CompareOp::LT), std::bad_alloc);
  EXPECT_THROW(column->search("alfa", CompareOp::GT), std::bad_alloc);
  EXPECT_THROW(column->search("al", CompareOp::HAS_PREFIX), std::bad_alloc);
  EXPECT_THROW(column->search("a", CompareOp::HAS_SUFFIX), std::bad_alloc);
  EXPECT_THROW(array->search_array_len(1, CompareOp::NE), std::bad_alloc);
  EXPECT_THROW(array->search_array_len(2, CompareOp::LT), std::bad_alloc);
  EXPECT_THROW(array->search_array_len(0, CompareOp::GT), std::bad_alloc);
  EXPECT_THROW(array->search_non_null(), std::bad_alloc);
  EXPECT_EQ(column->seal().code(), StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_FALSE(column->is_sealed());
  db->deny_next = false;

  ASSERT_TRUE(column->seal().ok());
  auto matches = column->search("al", CompareOp::HAS_PREFIX);
  ASSERT_TRUE(matches);
  EXPECT_EQ(matches->count(), 2U);
  matches = array->search_non_null();
  ASSERT_TRUE(matches);
  EXPECT_EQ(matches->count(), 2U);
}

}  // namespace
}  // namespace zvec

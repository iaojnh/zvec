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

#include <atomic>
#include <thread>
#include <gtest/gtest.h>
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "tests/test_util.h"

namespace zvec {
namespace {

class InvertedRangeCacheTest : public testing::Test {
 protected:
  static constexpr uint32_t kRows = 20000;
  const std::string path_ = "./inverted_range_cache_test_dir";
  InvertedIndexer::Ptr index_;
  InvertedColumnIndexer::Ptr column_;

  static std::string Encode(int32_t value) {
    return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
  }

  static int32_t Value(uint32_t id) {
    return (static_cast<int32_t>((uint64_t(id) * 7919) % 20011) - 10000) / 4;
  }

  void SetUp() override {
    test_util::RemoveTestPath(path_);
    FieldSchema field{"number", DataType::INT32, true,
                      std::make_shared<InvertIndexParams>(true, false)};
    index_ = InvertedIndexer::CreateAndOpen("range_cache", path_, true, {field},
                                            false);
    ASSERT_TRUE(index_);
    column_ = (*index_)["number"];
    ASSERT_TRUE(column_);
    for (uint32_t id = 0; id < kRows; ++id) {
      auto status = id % 19 == 0 ? column_->insert_null(id)
                                 : column_->insert(id, Encode(Value(id)));
      ASSERT_TRUE(status.ok());
    }
  }

  void TearDown() override {
    column_.reset();
    index_.reset();
    test_util::RemoveTestPath(path_);
  }

  static bool Matches(int32_t value, int32_t bound, CompareOp op) {
    switch (op) {
      case CompareOp::LT:
        return value < bound;
      case CompareOp::LE:
        return value <= bound;
      case CompareOp::GT:
        return value > bound;
      case CompareOp::GE:
        return value >= bound;
      default:
        return false;
    }
  }

  void Verify(int32_t bound, CompareOp op) {
    auto result = column_->search(Encode(bound), op);
    ASSERT_TRUE(result);
    size_t count = 0;
    for (uint32_t id = 0; id < kRows; ++id) {
      const bool expected = id % 19 != 0 && Matches(Value(id), bound, op);
      ASSERT_EQ(result->contains(id), expected)
          << "id=" << id << " bound=" << bound << " op=" << int(op);
      count += expected;
    }
    ASSERT_EQ(result->count(), count);
  }
};

TEST_F(InvertedRangeCacheTest, RangeBoundariesNullsAndIndependentResults) {
  ASSERT_TRUE(column_->seal().ok());
  for (int32_t bound = -2700; bound <= 2700; bound += 47) {
    for (auto op :
         {CompareOp::LT, CompareOp::LE, CompareOp::GT, CompareOp::GE}) {
      Verify(bound, op);
    }
  }
  auto left = column_->search(Encode(500), CompareOp::LT);
  auto right = column_->search(Encode(-500), CompareOp::GT);
  ASSERT_TRUE(left && right);
  left->AND(*right);
  right->OR(*left);
  left->destroy_bitmap();
  Verify(500, CompareOp::LT);
  Verify(-500, CompareOp::GT);

  column_.reset();
  index_.reset();
  FieldSchema field{"number", DataType::INT32, true,
                    std::make_shared<InvertIndexParams>(true, false)};
  index_ = InvertedIndexer::CreateAndOpen("range_cache", path_, false, {field},
                                          true);
  ASSERT_TRUE(index_);
  column_ = (*index_)["number"];
  ASSERT_TRUE(column_);
  Verify(0, CompareOp::LE);
  Verify(0, CompareOp::GE);
}

TEST_F(InvertedRangeCacheTest, ConcurrentFirstUse) {
  ASSERT_TRUE(column_->seal().ok());
  std::atomic<bool> start{false};
  std::atomic<uint32_t> errors{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < 8; ++i) {
        int32_t bound = -2500 + (t * 8 + i) * 79;
        auto result = column_->search(Encode(bound), CompareOp::GT);
        if (!result) {
          ++errors;
          continue;
        }
        for (uint32_t id = 0; id < kRows; ++id) {
          if (result->contains(id) != (id % 19 != 0 && Value(id) > bound)) {
            ++errors;
            break;
          }
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(errors.load(), 0u);
}

TEST_F(InvertedRangeCacheTest, StreamingSearchDoesNotRetainStaleResults) {
  auto result = column_->search(Encode(0), CompareOp::GT);
  ASSERT_TRUE(result);
  ASSERT_FALSE(result->contains(kRows));
  ASSERT_TRUE(column_->insert(kRows, Encode(123)).ok());
  result = column_->search(Encode(0), CompareOp::GT);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->contains(kRows));
}

}  // namespace
}  // namespace zvec

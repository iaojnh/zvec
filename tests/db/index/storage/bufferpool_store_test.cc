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
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <zvec/db/doc.h>
#include "db/index/storage/bufferpool_forward_store.h"
#include "db/index/storage/parquet_buffer_pool.h"
#include "db/index/storage/selected_rows_plan.h"
#include "utils/utils.h"

using namespace zvec;

class BufferPoolStoreTest : public testing::Test {
 protected:
  void SetUp() override {
    auto s = test::TestHelper::WriteTestFile(parquet_path, FileFormat::PARQUET);
    if (!s.ok()) {
      std::cout << "err: " << s.message() << std::endl;
      exit(1);
    }
    zvec::ailego::MemoryLimitPool::get_instance().init(10 * 1024 * 1024);
  }

  void TearDown() override {
    if (std::filesystem::exists(parquet_path)) {
      std::filesystem::remove(parquet_path);
    }
  }
  std::string parquet_path = "test.parquet";
};


TEST_F(BufferPoolStoreTest, ParquetFetch) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name", "score"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
}


TEST_F(BufferPoolStoreTest, ParquetFetchWithSelectColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithUID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto table = store->fetch({USER_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto table = store->fetch({GLOBAL_DOC_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({}, std::vector<int>{});
  EXPECT_EQ(table, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name"}, std::vector<int>{});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithMoreIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 2, 3, 6, 2, 1, 7});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 8);
  EXPECT_EQ(table->num_columns(), 1);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithInvalidIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 30});
  ASSERT_TRUE(table == nullptr);
}

TEST_F(BufferPoolStoreTest, BatchFetchNullableListsAcrossRowGroups) {
  arrow::Int32Builder ids;
  arrow::StringBuilder names;
  arrow::BinaryBuilder bytes;
  auto values = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder items(arrow::default_memory_pool(), values);
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(ids.Append(i).ok());
    if (i % 3 == 0) {
      ASSERT_TRUE(names.AppendNull().ok());
      ASSERT_TRUE(bytes.AppendNull().ok());
      ASSERT_TRUE(items.AppendNull().ok());
    } else {
      ASSERT_TRUE(names.Append(std::string(i, 'a' + i)).ok());
      ASSERT_TRUE(bytes.Append(std::string("\0\xff", 2)).ok());
      ASSERT_TRUE(items.Append().ok());
      if (i % 3 == 1) {
        ASSERT_TRUE(values->Append(i).ok());
        ASSERT_TRUE(values->AppendNull().ok());
      }
    }
  }
  std::shared_ptr<arrow::Array> id_array, name_array, byte_array, item_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(names.Finish(&name_array).ok());
  ASSERT_TRUE(bytes.Finish(&byte_array).ok());
  ASSERT_TRUE(items.Finish(&item_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("id", arrow::int32()),
                     arrow::field("name", arrow::utf8()),
                     arrow::field("bytes", arrow::binary()),
                     arrow::field("items", arrow::list(arrow::int32()))}),
      {id_array, name_array, byte_array, item_array});
  auto output = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         *output, 4)
                  .ok());
  ASSERT_TRUE((*output)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"items", "bytes", LOCAL_ROW_ID, "id",
                                         "name"};
  const std::vector<std::string> scalar_columns{"items", "bytes", "id", "name"};
  for (int row = 0; row < 12; ++row) {
    auto result = store->fetch(scalar_columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->values.size(), scalar_columns.size());
    for (size_t col = 0; col < scalar_columns.size(); ++col) {
      auto expected =
          source->GetColumnByName(scalar_columns[col])->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  for (const std::vector<int> rows :
       {std::vector<int>{9, 0, 6, 3, 9, 4, 11, 1, 8},
        std::vector<int>{1, 3, 1, 0}}) {
    auto result = store->fetch(columns, rows);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->ValidateFull().ok());
    ASSERT_EQ(result->num_rows(), rows.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      for (size_t row = 0; row < rows.size(); ++row) {
        auto actual = result->column(col)->GetScalar(row);
        ASSERT_TRUE(actual.ok());
        if (columns[col] == LOCAL_ROW_ID) {
          EXPECT_TRUE((*actual)->Equals(arrow::UInt64Scalar(rows[row])));
        } else {
          auto expected =
              source->GetColumnByName(columns[col])->GetScalar(rows[row]);
          ASSERT_TRUE(expected.ok());
          EXPECT_TRUE((*actual)->Equals(**expected));
        }
      }
    }
  }
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanHandlesChangingColumnLayouts) {
  arrow::Int64Builder values;
  arrow::Int64Builder shifted_values;
  for (int64_t row = 0; row < 8; ++row) {
    ASSERT_TRUE(values.Append(row).ok());
    ASSERT_TRUE(shifted_values.Append(100 + row).ok());
  }
  auto array = values.Finish().ValueOrDie();
  auto shifted = shifted_values.Finish().ValueOrDie();
  const detail::SelectedRowsPlan::Rows rows{{0, 7}, {1, 0}, {2, 3},
                                            {3, 4}, {4, 7}, {5, 1}};
  detail::SelectedRowsPlan plan(rows);
  auto check = [&](const arrow::ArrayVector &chunks, int64_t base) {
    arrow::ChunkedArray column(chunks);
    ASSERT_TRUE(plan.Prepare(column).ok());
    std::vector<int64_t> actual;
    ASSERT_TRUE(plan.Visit<arrow::Int64Array>(
                        column, [&](const arrow::Int64Array &chunk,
                                    int64_t row, int) {
                          actual.push_back(chunk.Value(row));
                          return arrow::Status::OK();
                        })
                    .ok());
    ASSERT_EQ(actual.size(), rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      EXPECT_EQ(actual[i], base + rows[i].second);
    }
  };
  check({array->Slice(0, 3), array->Slice(3)}, 0);
  // Equal lengths reuse addressing but must bind the new column's data.
  check({shifted->Slice(0, 3), shifted->Slice(3)}, 100);
  // Equal total length and chunk count do not imply equal boundaries.
  check({shifted->Slice(0, 5), shifted->Slice(5)}, 100);
  check({array}, 0);
  check({shifted->Slice(0, 0), shifted->Slice(0, 2), shifted->Slice(2, 0),
         shifted->Slice(2), shifted->Slice(8, 0)},
        100);
  check({array->Slice(0, 2), array->Slice(2, 1), array->Slice(3)}, 0);
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanRechecksBoundsAfterLayoutChanges) {
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendValues({0, 1, 2, 3, 4, 5}).ok());
  auto array = builder.Finish().ValueOrDie();
  const detail::SelectedRowsPlan::Rows rows{{0, 5}, {1, 0}, {2, 5}};
  detail::SelectedRowsPlan plan(rows);
  arrow::ChunkedArray full(array);
  arrow::ChunkedArray short_column(array->Slice(0, 5));
  ASSERT_TRUE(plan.Prepare(full).ok());
  EXPECT_TRUE(plan.Prepare(short_column).IsIndexError());
  ASSERT_TRUE(plan.Prepare(full).ok());
  std::vector<int64_t> actual;
  ASSERT_TRUE(plan.Visit<arrow::Int64Array>(
                      full, [&](const arrow::Int64Array &chunk, int64_t row,
                                int) {
                        actual.push_back(chunk.Value(row));
                        return arrow::Status::OK();
                      })
                  .ok());
  EXPECT_EQ(actual, (std::vector<int64_t>{5, 0, 5}));
  const detail::SelectedRowsPlan::Rows huge{{0, uint64_t(-1)}};
  detail::SelectedRowsPlan invalid(huge);
  EXPECT_TRUE(invalid.Prepare(full).IsIndexError());
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanHandlesEmptySelectionsAndColumns) {
  const detail::SelectedRowsPlan::Rows rows;
  detail::SelectedRowsPlan plan(rows);
  arrow::ChunkedArray column(arrow::ArrayVector{}, arrow::int64());
  ASSERT_TRUE(plan.Prepare(column).ok());
  int visits = 0;
  ASSERT_TRUE(plan.Visit<arrow::Int64Array>(
                      column, [&](const arrow::Int64Array &, int64_t, int) {
                        ++visits;
                        return arrow::Status::OK();
                      })
                  .ok());
  EXPECT_EQ(visits, 0);
  const detail::SelectedRowsPlan::Rows nonempty{{0, 0}};
  detail::SelectedRowsPlan invalid(nonempty);
  EXPECT_TRUE(invalid.Prepare(column).IsIndexError());
}

TEST_F(BufferPoolStoreTest, TypedBatchAcrossLargeRowGroupsWithMixedNullColumns) {
  arrow::BooleanBuilder flags;
  arrow::Int32Builder i32;
  arrow::Int64Builder i64;
  arrow::UInt32Builder u32;
  arrow::UInt64Builder u64;
  arrow::FloatBuilder floats;
  arrow::DoubleBuilder doubles;
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  arrow::StringBuilder nullable;
  arrow::StringBuilder all_null;
  constexpr int count = 130001;
  for (int row = 0; row < count; ++row) {
    ASSERT_TRUE(flags.Append(row % 2 != 0).ok());
    ASSERT_TRUE(i32.Append(-row).ok());
    ASSERT_TRUE(i64.Append(-(int64_t{1} << 40) - row).ok());
    ASSERT_TRUE(u32.Append(4000000000u + row).ok());
    ASSERT_TRUE(u64.Append((uint64_t{1} << 63) + row).ok());
    ASSERT_TRUE(floats.Append(row + 0.5f).ok());
    ASSERT_TRUE(doubles.Append(row + 0.125).ok());
    const auto text = std::to_string(row);
    ASSERT_TRUE(strings.Append(text).ok());
    ASSERT_TRUE(binary.Append(std::string("\0\xff", 2) + text).ok());
    if (row % 3 == 0) {
      ASSERT_TRUE(nullable.AppendNull().ok());
    } else {
      ASSERT_TRUE(nullable.Append(text).ok());
    }
    ASSERT_TRUE(all_null.AppendNull().ok());
  }
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  std::vector<arrow::ArrayBuilder *> builders{
      &flags, &i32, &i64, &u32, &u64, &floats, &doubles,
      &strings, &binary, &nullable, &all_null};
  for (size_t col = 0; col < builders.size(); ++col) {
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builders[col]->Finish(&array).ok());
    if (col < 9) ASSERT_EQ(array->null_count(), 0);
    fields.push_back(arrow::field("n" + std::to_string(col), array->type()));
    arrays.push_back(std::move(array));
  }
  auto source = arrow::Table::Make(arrow::schema(fields), arrays);
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         *out, 100000)
                  .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{
      "n8", "n3", LOCAL_ROW_ID, "n9", "n0", "n7", "n10",
      "n1", "n2", "n4", "n5", "n6", "n7"};
  const std::vector<int> rows{99999, 0, 65536, 65535, 100000, 130000,
                              0, 66666, 99999, 100001};
  auto result = store->fetch(columns, rows);
  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(result->ValidateFull().ok());
  ASSERT_EQ(result->num_rows(), rows.size());
  for (size_t col = 0; col < columns.size(); ++col) {
    for (size_t row = 0; row < rows.size(); ++row) {
      auto actual = result->column(col)->GetScalar(row);
      ASSERT_TRUE(actual.ok());
      if (columns[col] == LOCAL_ROW_ID) {
        EXPECT_TRUE((*actual)->Equals(arrow::UInt64Scalar(rows[row])));
      } else {
        auto expected = source->GetColumnByName(columns[col])->GetScalar(rows[row]);
        ASSERT_TRUE(expected.ok());
        EXPECT_TRUE((*actual)->Equals(**expected));
      }
    }
  }
  EXPECT_EQ(store->fetch(columns, std::vector<int>{0, count}), nullptr);
  EXPECT_EQ(store->fetch(columns, std::vector<int>{-1, 0}), nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchCheckOrderWithLocalRowIDMiddle) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table =
      store->fetch({"id", "name", LOCAL_ROW_ID, "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 5);
  EXPECT_EQ(table->num_columns(), 4);
  auto field = table->schema()->field(2);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = table->column(2);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> expected_ids = {0, 3, 6, 1, 0};
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}


TEST_F(BufferPoolStoreTest, ParquetFetchCheckOrderWithLocalRowIDEnd) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table =
      store->fetch({"id", "name", "score", LOCAL_ROW_ID}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 5);
  EXPECT_EQ(table->num_columns(), 4);
  auto field = table->schema()->field(3);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = table->column(3);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> expected_ids = {0, 3, 6, 1, 0};
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}


TEST_F(BufferPoolStoreTest, ParquetScan) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({"id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 3);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithSelectColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({"id", "name"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 2);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithInvalidColumn) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({"id", "unknown_column"});
  ASSERT_TRUE(reader == nullptr);
}


TEST_F(BufferPoolStoreTest, ParquetScanWithUserID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({USER_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({GLOBAL_DOC_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name", "score"}, 0);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 1);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSpecificRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name", "score"}, 3);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 4);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithUserID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({USER_ID, "id", "name"}, 1);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto user_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(user_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::StringScalar>(user_id_scalar) !=
              nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({GLOBAL_DOC_ID, "id", "name"}, 4);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto global_doc_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(global_doc_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::UInt64Scalar>(
                  global_doc_id_scalar) != nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithNegativeIndex) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, -1);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithOutOfRangeIndex) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, 15);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithInvalidColumn) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "invalid_column"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, SingleRowProjectionPreservesOrderAndDuplicates) {
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto reader = parquet::arrow::OpenFile(*file, arrow::default_memory_pool());
  ASSERT_TRUE(reader.ok());
  std::shared_ptr<arrow::Table> source;
  ASSERT_TRUE((*reader)->ReadTable(&source).ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"name",  GLOBAL_DOC_ID, "id", "name",
                                         USER_ID, "score",       "id"};
  for (int row : {0, 3, 9}) {
    auto result = store->fetch(columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->length, 1);
    ASSERT_EQ(result->values.size(), columns.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      auto expected = source->GetColumnByName(columns[col])->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  auto before = ailego::MemoryLimitPool::get_instance().used();
  EXPECT_EQ(store->fetch({"name", "unknown", "id"}, 0), nullptr);
  EXPECT_EQ(store->fetch({LOCAL_ROW_ID}, 0), nullptr);
  EXPECT_EQ(ailego::MemoryLimitPool::get_instance().used(), before);
}

TEST_F(BufferPoolStoreTest, AllDataTypeFetchSingleRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "list_int32"}, 2);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 2);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 3);

  auto list_scalar = batch->values[1].scalar();
  ASSERT_TRUE(list_scalar != nullptr);
  auto list_value = std::dynamic_pointer_cast<arrow::ListScalar>(list_scalar);
  ASSERT_TRUE(list_value != nullptr);
  EXPECT_EQ(list_value->value->length(), 128);

  auto list_array =
      std::dynamic_pointer_cast<arrow::Int32Array>(list_value->value);
  ASSERT_TRUE(list_array != nullptr);
  for (int i = 0; i < 10 && i < list_array->length(); ++i) {
    EXPECT_EQ(list_array->Value(i), 2 * 10 + i);
  }
}

TEST_F(BufferPoolStoreTest, AllDataType) {
  auto mmap_store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());

  std::vector<std::string> columns = {"id", "list_int32"};
  std::vector<int> indices = {0, 3, 6, 1, 0};

  TablePtr mmap_table = mmap_store->fetch(columns, indices);
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 2);

  for (size_t j = 0; j < columns.size(); ++j) {
    auto column = mmap_table->column(j);
    for (int k = 0; k < column->num_chunks(); ++k) {
      auto array = column->chunk(k);
      if (array->type()->id() == arrow::Type::INT32) {
        auto int_array = std::static_pointer_cast<arrow::Int32Array>(array);
        for (int i = 0; i < array->length(); ++i) {
          int32_t value = int_array->Value(i);
          EXPECT_EQ(value, indices[i] + 1);
        }
      } else if (array->type()->id() == arrow::Type::LIST) {
        auto list_array = std::static_pointer_cast<arrow::ListArray>(array);
        for (int i = 0; i < array->length(); ++i) {
          auto list_value = list_array->value_slice(i);
          auto list_value_array =
              std::static_pointer_cast<arrow::Int32Array>(list_value);
          EXPECT_EQ(list_value_array->length(), 128);
          for (int m = 0; m < list_value_array->length(); ++m) {
            int32_t value = list_value_array->Value(m);
            EXPECT_EQ(value, indices[i] * 10 + m);
          }
        }
      }
    }
  }
}

TEST_F(BufferPoolStoreTest, DeleteDestructs) {
  BufferPoolForwardStore *store = new BufferPoolForwardStore(parquet_path);
  delete store;
}

TEST_F(BufferPoolStoreTest, PhysicSchema) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(store->open().ok());
  EXPECT_NE(store->physic_schema(), nullptr);
}

TEST_F(BufferPoolStoreTest,
       OwnedScalarAndTypedBatchReadsMatchNullableSourceTypes) {
  arrow::BooleanBuilder flags;
  arrow::Int32Builder i32;
  arrow::Int64Builder i64;
  arrow::UInt32Builder u32;
  arrow::UInt64Builder u64;
  arrow::FloatBuilder floats;
  arrow::DoubleBuilder doubles;
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  std::vector<arrow::ArrayBuilder *> builders{
      &flags, &i32, &i64, &u32, &u64, &floats, &doubles, &strings, &binary};
  for (int row = 0; row < 6; ++row) {
    if (row == 2) {
      for (auto *builder : builders) ASSERT_TRUE(builder->AppendNull().ok());
      continue;
    }
    ASSERT_TRUE(flags.Append(row % 2 != 0).ok());
    ASSERT_TRUE(i32.Append(-100 - row).ok());
    ASSERT_TRUE(i64.Append(-(int64_t{1} << 40) - row).ok());
    ASSERT_TRUE(u32.Append(4000000000u + row).ok());
    ASSERT_TRUE(u64.Append((uint64_t{1} << 63) + row).ok());
    ASSERT_TRUE(floats.Append(row + 0.5f).ok());
    ASSERT_TRUE(doubles.Append(row + 0.125).ok());
    ASSERT_TRUE(strings.Append(row ? std::string("a\0b", 3) : "").ok());
    ASSERT_TRUE(binary.Append(row ? std::string("\0\xff", 2) : "").ok());
  }
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  std::vector<std::string> columns;
  for (size_t col = 0; col < builders.size(); ++col) {
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builders[col]->Finish(&array).ok());
    columns.push_back("c" + std::to_string(col));
    fields.push_back(arrow::field(columns.back(), array->type()));
    arrays.push_back(std::move(array));
  }
  arrow::UInt64Builder doc_ids;
  arrow::StringBuilder primary_keys;
  for (int row = 0; row < 6; ++row) {
    ASSERT_TRUE(doc_ids.Append(100 + row).ok());
    ASSERT_TRUE(
        primary_keys.Append(std::string(256, 'p') + std::to_string(row)).ok());
  }
  std::shared_ptr<arrow::Array> id_array, pk_array;
  ASSERT_TRUE(doc_ids.Finish(&id_array).ok());
  ASSERT_TRUE(primary_keys.Finish(&pk_array).ok());
  columns.insert(columns.begin(), {GLOBAL_DOC_ID, USER_ID});
  arrays.insert(arrays.begin(), {id_array, pk_array});
  fields.insert(fields.begin(), {arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                                 arrow::field(USER_ID, arrow::utf8())});
  const auto path = std::filesystem::absolute(parquet_path).string();
  auto source = arrow::Table::Make(arrow::schema(fields), arrays);
  auto out = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 2)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  for (int row : {5, 0, 2, 3, 1, 4, 5}) {
    auto result = store->fetch(columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->length, 1);
    ASSERT_EQ(result->values.size(), columns.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      auto expected = source->column(col)->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      ASSERT_TRUE(result->values[col].is_scalar());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  for (const std::vector<int> rows :
       {std::vector<int>{5, 0, 2, 3, 1, 4, 5}, std::vector<int>{2, 2, 0},
        std::vector<int>{4, 5, 4}}) {
    auto table = store->fetch(columns, rows);
    ASSERT_NE(table, nullptr);
    ASSERT_TRUE(table->ValidateFull().ok());
    ASSERT_EQ(table->num_rows(), rows.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      for (size_t row = 0; row < rows.size(); ++row) {
        auto expected = source->column(col)->GetScalar(rows[row]);
        auto actual = table->column(col)->GetScalar(row);
        ASSERT_TRUE(expected.ok() && actual.ok());
        EXPECT_TRUE((*actual)->Equals(**expected));
      }
    }
  }
  std::vector<Doc::Ptr> retained;
  auto check_doc = [&](const Doc::Ptr &doc, int row) {
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->doc_id(), 100 + row);
    EXPECT_EQ(doc->pk_ref(), std::string(256, 'p') + std::to_string(row));
    if (row == 2) {
      for (int col = 0; col < 9; ++col)
        EXPECT_FALSE(doc->has("c" + std::to_string(col)));
      return;
    }
    EXPECT_EQ(doc->get<bool>("c0"), row % 2 != 0);
    EXPECT_EQ(doc->get<int32_t>("c1"), -100 - row);
    EXPECT_EQ(doc->get<int64_t>("c2"), -(int64_t{1} << 40) - row);
    EXPECT_EQ(doc->get<uint32_t>("c3"), 4000000000u + row);
    EXPECT_EQ(doc->get<uint64_t>("c4"), (uint64_t{1} << 63) + row);
    EXPECT_EQ(doc->get<float>("c5"), row + 0.5f);
    EXPECT_EQ(doc->get<double>("c6"), row + 0.125);
    EXPECT_EQ(doc->get<std::string>("c7"), row ? std::string("a\0b", 3) : "");
    EXPECT_EQ(doc->get<std::string>("c8"), row ? std::string("\0\xff", 2) : "");
  };
  for (int row = 0; row < 6; ++row) {
    retained.push_back(store->fetch_doc(columns, source->schema(), row));
    check_doc(retained.back(), row);
  }
  auto file = arrow::io::ReadableFile::Open(path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::Make(path, *file);
  ASSERT_TRUE(context.ok());
  std::weak_ptr<arrow::Buffer> cached_bytes;
  {
    auto key = (*context)->buffer_id;
    key.column = 1;
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(handle.data(), nullptr);
    cached_bytes = handle.data()->chunk(0)->data()->buffers[2];
  }
  store.reset();
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  EXPECT_TRUE(cached_bytes.expired());
  for (int row = 0; row < 6; ++row) check_doc(retained[row], row);
}

TEST_F(BufferPoolStoreTest, DirectDocDeclinesComplexTypesWithoutPinning) {
  arrow::UInt64Builder ids;
  arrow::StringBuilder keys;
  auto values = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  ASSERT_TRUE(ids.Append(1).ok());
  ASSERT_TRUE(keys.Append("pk").ok());
  ASSERT_TRUE(lists.Append().ok());
  ASSERT_TRUE(values->Append(11).ok());
  std::shared_ptr<arrow::Array> id_array, key_array, list_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(keys.Finish(&key_array).ok());
  ASSERT_TRUE(lists.Finish(&list_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                     arrow::field(USER_ID, arrow::utf8()),
                     arrow::field("items", list_array->type())}),
      {id_array, key_array, list_array});
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 1)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID, "items"};
  auto before = ailego::MemoryLimitPool::get_instance().used();
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 0), nullptr);
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), -1), nullptr);
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 1), nullptr);
  EXPECT_EQ(ailego::MemoryLimitPool::get_instance().used(), before);
  auto legacy = store->fetch(columns, 0);
  ASSERT_NE(legacy, nullptr);
  auto scalar =
      std::dynamic_pointer_cast<arrow::ListScalar>(legacy->values[2].scalar());
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(
      std::static_pointer_cast<arrow::Int32Array>(scalar->value)->Value(0), 11);
}

TEST_F(BufferPoolStoreTest, SingleRowStringsDoNotRetainCachedColumnBuffers) {
  const auto path = std::filesystem::absolute(parquet_path).string();
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  for (int row = 0; row < 8; ++row) {
    ASSERT_TRUE(strings.Append(std::string(16384, 'a' + row)).ok());
    ASSERT_TRUE(
        binary.Append(std::string("\0\xff", 2) + std::string(16384, 'a' + row))
            .ok());
  }
  std::shared_ptr<arrow::Array> text_array, binary_array;
  ASSERT_TRUE(strings.Finish(&text_array).ok());
  ASSERT_TRUE(binary.Finish(&binary_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("text", arrow::utf8()),
                     arrow::field("bytes", arrow::binary())}),
      {text_array, binary_array});
  auto out = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 4)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  auto retained = store->fetch({"text", "bytes"}, 1);
  ASSERT_NE(retained, nullptr);
  auto file = arrow::io::ReadableFile::Open(path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::Make(path, *file);
  ASSERT_TRUE(context.ok());
  std::vector<std::weak_ptr<arrow::Buffer>> cached_buffers;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  for (int col : {0, 1}) {
    auto key = (*context)->buffer_id;
    key.column = col;
    key.row_group = 0;
    auto before = pool.used();
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(key);
    ASSERT_NE(handle.data(), nullptr);
    ASSERT_EQ(pool.used(), before) << "Expected the scalar fetch's cache entry";
    cached_buffers.push_back(handle.data()->chunk(0)->data()->buffers[2]);
  }
  store.reset();
  size_t evicted = 0;
  for (int i = 0; i < 4; ++i) {
    evicted +=
        ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  EXPECT_GT(evicted, 0u);
  for (const auto &buffer : cached_buffers) EXPECT_TRUE(buffer.expired());
  EXPECT_EQ(std::static_pointer_cast<arrow::StringScalar>(
                retained->values[0].scalar())
                ->view(),
            std::string(16384, 'b'));
  EXPECT_EQ(std::static_pointer_cast<arrow::BinaryScalar>(
                retained->values[1].scalar())
                ->view(),
            std::string("\0\xff", 2) + std::string(16384, 'b'));
}

TEST_F(BufferPoolStoreTest,
       TypedBatchStringsOwnDataAfterReorderingAndEviction) {
  arrow::StringBuilder text;
  arrow::BinaryBuilder bytes;
  arrow::Int32Builder ids;
  for (int row = 0; row < 15; ++row) {
    ASSERT_TRUE(ids.Append(row).ok());
    if (row % 4 == 2) {
      ASSERT_TRUE(text.AppendNull().ok());
      ASSERT_TRUE(bytes.AppendNull().ok());
    } else {
      std::string value(row * 1100, 'a' + row);
      ASSERT_TRUE(text.Append(value).ok());
      ASSERT_TRUE(bytes.Append(std::string("\0\xff", 2) + value).ok());
    }
  }
  std::shared_ptr<arrow::Array> texts, binaries, numbers;
  ASSERT_TRUE(text.Finish(&texts).ok());
  ASSERT_TRUE(bytes.Finish(&binaries).ok());
  ASSERT_TRUE(ids.Finish(&numbers).ok());
  auto source =
      arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8()),
                                        arrow::field("bytes", arrow::binary()),
                                        arrow::field("id", arrow::int32())}),
                         {texts, binaries, numbers});
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 3)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<int> rows{14, 0, 3, 4, 8, 11, 5, 14, 1};
  const std::vector<std::string> columns{"bytes", "id", "text", "text"};
  auto result = store->fetch(columns, rows);
  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(result->ValidateFull().ok());
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::Make(parquet_path, *file);
  ASSERT_TRUE(context.ok());
  std::vector<std::weak_ptr<arrow::Buffer>> cached_buffers;
  for (int row_group = 0; row_group < 5; ++row_group) {
    auto key = (*context)->buffer_id;
    key.row_group = row_group;
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(handle.data(), nullptr);
    cached_buffers.push_back(handle.data()->chunk(0)->data()->buffers[2]);
  }
  store.reset();
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  for (const auto &buffer : cached_buffers) EXPECT_TRUE(buffer.expired());
  ASSERT_EQ(result->num_rows(), rows.size());
  ASSERT_EQ(result->num_columns(), columns.size());
  for (size_t col = 0; col < columns.size(); ++col) {
    for (size_t row = 0; row < rows.size(); ++row) {
      auto expected =
          source->GetColumnByName(columns[col])->GetScalar(rows[row]);
      auto actual = result->column(col)->GetScalar(row);
      ASSERT_TRUE(expected.ok() && actual.ok());
      EXPECT_TRUE((*actual)->Equals(**expected));
    }
  }
}

namespace {
arrow::Status WriteContextFile(const std::string &path, int64_t base) {
  arrow::Int64Builder values;
  for (int64_t row = 0; row < 8; ++row) {
    ARROW_RETURN_NOT_OK(values.Append(base + row));
  }
  ARROW_ASSIGN_OR_RAISE(auto array, values.Finish());
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::int64())}), {array});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  ARROW_RETURN_NOT_OK(
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 4));
  return out->Close();
}

void ExpectContextValue(const ExecBatchPtr &batch, int64_t value) {
  ASSERT_NE(batch, nullptr);
  ASSERT_EQ(batch->values.size(), 1u);
  EXPECT_EQ(
      std::static_pointer_cast<arrow::Int64Scalar>(batch->values[0].scalar())
          ->value,
      value);
}
}  // namespace

TEST_F(BufferPoolStoreTest, CopiedAndMovedHandlesKeepPinsUntilLastRelease) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 500).ok());
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::Make(parquet_path, *file);
  ASSERT_TRUE(context.ok());
  std::unique_ptr<ParquetBufferContextHandle> copied;
  std::weak_ptr<arrow::Buffer> cached_buffer;
  {
    auto key = (*context)->buffer_id;
    key.row_group = 1;
    auto original =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(original.data(), nullptr);
    cached_buffer = original.data()->chunk(0)->data()->buffers[1];
    copied = std::make_unique<ParquetBufferContextHandle>(original);
    ASSERT_EQ(copied->data(), original.data());
  }
  {
    ParquetBufferContextHandle moved(std::move(*copied));
    EXPECT_EQ(copied->data(), nullptr);
    copied.reset();
    for (int i = 0; i < 4; ++i) {
      ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
    }
    EXPECT_FALSE(cached_buffer.expired());
    ASSERT_NE(moved.data(), nullptr);
    auto scalar = moved.data()->GetScalar(2);
    ASSERT_TRUE(scalar.ok());
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(*scalar)->value,
              506);
  }
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  }
  EXPECT_TRUE(cached_buffer.expired());
}

#if !defined(_WIN32) && !defined(_WIN64)
TEST_F(BufferPoolStoreTest, OpenedFileSurvivesReplacementAndEviction) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 100).ok());
  auto original = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(original->open().ok());
  auto retained = original->fetch({"value"}, 0);
  ExpectContextValue(retained, 100);

  const auto replacement_path = parquet_path + ".new";
  ASSERT_TRUE(WriteContextFile(replacement_path, 200).ok());
  std::filesystem::rename(replacement_path, parquet_path);
  auto replacement = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(replacement->open().ok());

  // The old file's second row group has never been cached.
  ExpectContextValue(original->fetch({"value"}, 6), 106);
  ExpectContextValue(replacement->fetch({"value"}, 6), 206);
  auto scan = original->scan({"value"});
  ASSERT_NE(scan, nullptr);
  original.reset();
  size_t evicted = 0;
  for (int i = 0; i < 4; ++i) {
    evicted += ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  }
  EXPECT_GT(evicted, 0u);
  ExpectContextValue(retained, 100);
  ExpectContextValue(replacement->fetch({"value"}, 0), 200);
  std::shared_ptr<arrow::RecordBatch> batch;
  int64_t rows = 0;
  while (true) {
    batch.reset();
    ASSERT_TRUE(scan->ReadNext(&batch).ok());
    if (!batch) break;
    auto array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    for (int64_t row = 0; row < array->length(); ++row) {
      EXPECT_EQ(array->Value(row), 100 + rows++);
    }
  }
  EXPECT_EQ(rows, 8);
}

#endif

TEST_F(BufferPoolStoreTest, FileKeysShareIdentityWithoutRetainingFile) {
  auto first = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto second = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto context = std::make_shared<ParquetFileContext>(parquet_path, first);
  auto other = std::make_shared<ParquetFileContext>(parquet_path, second);
  auto key = context->buffer_id;
  EXPECT_TRUE(ParquetBufferIDEqual{}(key, other->buffer_id));
  EXPECT_EQ(ParquetBufferIDHash{}(key),
            ParquetBufferIDHash{}(other->buffer_id));
  {
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(key);
    auto shared =
        ParquetBufferPool::get_instance().acquire_buffer(other->buffer_id);
    ASSERT_NE(handle.data(), nullptr);
    EXPECT_EQ(handle.data(), shared.data());
  }
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  }
  std::weak_ptr<arrow::io::RandomAccessFile> lifetime = first;
  context.reset();
  first.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST_F(BufferPoolStoreTest, ConcurrentOpenedFileReads) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 300).ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  std::atomic<bool> correct{true};
  std::vector<std::thread> readers;
  for (int thread = 0; thread < 4; ++thread) {
    readers.emplace_back([&, thread]() {
      for (int i = 0; i < 64; ++i) {
        int row = (i + thread * 4) % 8;
        auto batch = store->fetch({"value"}, row);
        if (!batch || batch->values.size() != 1 ||
            std::static_pointer_cast<arrow::Int64Scalar>(
                batch->values[0].scalar())
                    ->value != 300 + row) {
          correct.store(false);
        }
      }
    });
  }
  for (auto &reader : readers) reader.join();
  EXPECT_TRUE(correct.load());
}

TEST_F(BufferPoolStoreTest, NonLocalContextLoadsWithoutOpeningPath) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 400).ok());
  auto source = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto bytes = source->ReadAt(0, source->GetSize().ValueOrDie()).ValueOrDie();
  auto file = std::make_shared<arrow::io::BufferReader>(bytes);
  auto context =
      ParquetFileContext::Make("no-such-file.parquet", file).ValueOrDie();
  auto another =
      ParquetFileContext::Make("no-such-file.parquet", file).ValueOrDie();
  EXPECT_NE(context->buffer_id.generation, 0u);
  EXPECT_FALSE(ParquetBufferIDEqual{}(context->buffer_id, another->buffer_id));
  auto handle =
      ParquetBufferPool::get_instance().acquire_buffer(context->buffer_id);
  ASSERT_NE(handle.data(), nullptr);
  auto scalar = handle.data()->GetScalar(0).ValueOrDie();
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value, 400);
}

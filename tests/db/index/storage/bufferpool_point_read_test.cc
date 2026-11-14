// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <filesystem>
#include <numeric>
#include <tuple>
#include <ailego/pattern/scope_guard.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/index/storage/bufferpool_forward_store.h"
#include "db/index/storage/parquet_memory_pool.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include "../../common/memory_budget_test_helper.h"

namespace zvec {
namespace {

class FlatBytePointReadTest
    : public testing::TestWithParam<std::tuple<bool, bool, bool>> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(0));
    static std::atomic<size_t> next{0};
    path = "flat-byte-point-read-" + std::to_string(next.fetch_add(1)) +
           ".parquet";
  }
  void TearDown() override {
    ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    EXPECT_EQ(0u, ailego::MemoryLimitPool::get_instance().external_used());
    EXPECT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(0));
    std::filesystem::remove(path);
  }
  static std::string value(int row, bool binary) {
    if (row % 13 == 0) return {};
    auto bytes = std::to_string(row % 7) + std::string(8192, 'x');
    if (binary) {
      bytes[1] = '\0';
      bytes[2] = static_cast<char>(0xff);
    }
    return bytes;
  }
  static bool is_null(int row, bool nullable) {
    return nullable && row % 11 == 0;
  }
  std::string path;
};

TEST_P(FlatBytePointReadTest, SelectionFitsWithoutDecodedOutputBatch) {
  const auto [dictionary, binary, nullable] = GetParam();
  constexpr int count = 512;
  auto type = binary ? arrow::binary() : arrow::utf8();
  {
    auto builder = arrow::MakeBuilder(type);
    ASSERT_TRUE(builder.ok());
    auto *bytes = static_cast<arrow::BinaryBuilder *>(builder->get());
    for (int row = 0; row < count; ++row)
      ASSERT_TRUE((is_null(row, nullable) ? bytes->AppendNull()
                                          : bytes->Append(value(row, binary)))
                      .ok());
    auto array = bytes->Finish();
    ASSERT_TRUE(array.ok());
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("value", type, nullable)}), {*array});
    auto file = arrow::io::FileOutputStream::Open(path);
    ASSERT_TRUE(file.ok());
    parquet::WriterProperties::Builder properties;
    if (!dictionary) properties.disable_dictionary();
    properties.data_pagesize(64 * 1024);
    properties.write_batch_size(4);
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *file, count, properties.build())
                    .ok());
    ASSERT_TRUE((*file)->Close().ok());
  }
  // A whole decoded column is about 4 MiB. Only six requested values and
  // the decoder's current page are needed, under this payload allowance.
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(768 * 1024));
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<int> rows{511, 0, 127, 127, 13, 499};
  auto selected = store->fetch_checked({"value"}, rows,
                                       detail::GetParquetMemoryPool().get());
  ASSERT_TRUE(selected.ok())
      << selected.status().ToString()
      << " used=" << ailego::MemoryLimitPool::get_instance().used()
      << " committed=" << ailego::MemoryLimitPool::get_instance().committed();
  ASSERT_EQ(rows.size(), static_cast<size_t>((*selected)->num_rows()));
  const auto &values = static_cast<const arrow::BinaryArray &>(
      *(*selected)->column(0)->chunk(0));
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(is_null(rows[i], nullable), values.IsNull(i));
    if (!values.IsNull(i)) EXPECT_EQ(value(rows[i], binary), values.GetView(i));
  }
  std::vector<int> all(count);
  std::iota(all.begin(), all.end(), 0);
  auto refused = store->fetch_checked({"value"}, all,
                                      detail::GetParquetMemoryPool().get());
  ASSERT_FALSE(refused.ok());
  EXPECT_TRUE(refused.status().IsOutOfMemory()) << refused.status().ToString();
  // A genuinely exhausted budget must remain a refusal, with no decoder
  // exception escaping or partially advancing future point reads.
  auto &budget = ailego::MemoryLimitPool::get_instance();
  {
    const size_t held = budget.capacity() - budget.used();
    ASSERT_TRUE(budget.try_charge_workspace(held));
    auto release =
        ailego::ScopeGuard::Make([&] { budget.release_external(held); });
    auto blocked = store->fetch_checked({"value"}, rows,
                                        detail::GetParquetMemoryPool().get());
    ASSERT_FALSE(blocked.ok());
    EXPECT_TRUE(blocked.status().IsOutOfMemory())
        << blocked.status().ToString();
  }
  auto retry = store->fetch_checked({"value"}, rows,
                                    detail::GetParquetMemoryPool().get());
  ASSERT_TRUE(retry.ok()) << retry.status().ToString();
  store.reset();
  EXPECT_EQ(value(rows.front(), binary), values.GetView(0));
}

INSTANTIATE_TEST_SUITE_P(Encodings, FlatBytePointReadTest,
                         testing::Combine(testing::Bool(), testing::Bool(),
                                          testing::Bool()));

TEST(ParquetBackingFailureTest, IncludesBudgetSnapshot) {
  ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
  auto &budget = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(64 * 1024));
  {
    detail::ParquetMemoryPool pool(arrow::default_memory_pool(), true);
    for (const auto &[size, alignment] :
         {std::pair<int64_t, int64_t>{64, 64}, {1024, 256}}) {
      const size_t held = budget.capacity() - budget.used() - size - alignment;
      ASSERT_TRUE(budget.try_charge_workspace(held));
      auto release =
          ailego::ScopeGuard::Make([&] { budget.release_external(held); });
      uint8_t *data = nullptr;
      const auto status = pool.Allocate(size, alignment, &data);
      EXPECT_TRUE(status.IsOutOfMemory()) << status.ToString();
      if (status.ok()) pool.Free(data, size, alignment);
      EXPECT_NE(std::string::npos, status.ToString().find("backing_bytes="));
      EXPECT_NE(std::string::npos, status.ToString().find("metadata="));
      EXPECT_NE(std::string::npos, status.ToString().find("rss_limit="));
    }
  }
  EXPECT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(0));
}

}  // namespace
}  // namespace zvec

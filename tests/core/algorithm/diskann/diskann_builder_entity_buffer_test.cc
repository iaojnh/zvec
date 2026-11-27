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
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <thread>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include "diskann_builder_entity.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

class DiskAnnBuilderEntityBufferTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Writable pools also charge their stable page table, lock stripes and
    // 128-page writeback staging. On macOS staging alone is 2 MiB per file.
    // Leave only 1 MiB for resident data across our two temporary storages,
    // irrespective of the platform's page size and metadata requirements.
    const size_t metadata =
        ailego::PoolPayloadBudget::storage_metadata_bytes(4096);
    ailego::PoolPayloadBudget::get_instance().init(2 * metadata +
                                                   1024UL * 1024UL);
  }

  void SetUp() override {
    static std::atomic<size_t> counter{0};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    directory_ = "diskann_builder_entity_buffer_" + std::to_string(stamp) +
                 "_" + std::to_string(counter.fetch_add(1));
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }

  void TearDown() override {
    // The entity owns and removes only its scratch files. No unknown files
    // are removed recursively, even in this test fixture.
    std::error_code error;
    EXPECT_TRUE(std::filesystem::is_empty(directory_, error));
    EXPECT_TRUE(std::filesystem::remove(directory_, error));
  }

  std::string directory_;
};

TEST_F(DiskAnnBuilderEntityBufferTest,
       MetadataPressureFailsCleanlyAndCanRetry) {
  DiskAnnBuilderEntity entity;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
  ASSERT_EQ(0, entity.init(meta, 8, 16, 0, 1));
  ASSERT_EQ(0, entity.enable_buffered_build(directory_ + "/build"));
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  {
    ASSERT_GT(pool.available(), ailego::kVectorPageSize);
    const size_t reservation = pool.available() - ailego::kVectorPageSize;
    ASSERT_TRUE(pool.try_charge_external(reservation));
    AILEGO_DEFER([&]() { pool.release_external(reservation); });
    EXPECT_EQ(IndexError_NoMemory, entity.reserve_space(8));
    EXPECT_TRUE(entity.buffered_build());
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
  }
  ASSERT_EQ(0, entity.reserve_space(8));
  const float vector[4] = {1, 2, 3, 4};
  ASSERT_EQ(0, entity.add_vector(17, vector));
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(0, entity.read_vector(0, block));
  EXPECT_FLOAT_EQ(4.0F, static_cast<const float *>(block.data())[3]);
}

}  // namespace
}  // namespace zvec::core

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

#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core_interface {
namespace {

class IVFBufferedBuildTest : public ::testing::Test {
 protected:
  const std::filesystem::path directory_{"ivf_buffered_build_data"};
  size_t capacity_{0};
  size_t writable_metadata_{0};
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    writable_metadata_ = ailego::PoolPayloadBudget::storage_metadata_bytes(4096);
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }
  void TearDown() override {
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
    EXPECT_TRUE(std::filesystem::remove(directory_));
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(capacity_));
  }
};

TEST_F(IVFBufferedBuildTest, ExternalLabelSortPreservesOriginalVectorOrder) {
  // More than 4096 records force an external merge of the bucket ordinals.
  static constexpr uint32_t kRows = 4099;
  static constexpr uint32_t kDim = 8;
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(pool.init(4 * writable_metadata_ + 128 * 1024), 0);
  for (auto quantizer : {QuantizerType::kNone, QuantizerType::kFP16}) {
    auto param = IVFIndexParamBuilder()
                     .with_dimension(kDim)
                     .with_metric_type(MetricType::kL2sq)
                     .with_data_type(DataType::DT_FP32)
                     .with_quantizer_param(QuantizerParam(quantizer))
                     .with_n_list(8)
                     .with_n_iters(2)
                     .build();
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);
    const auto path = (directory_ / "external-labels").string();
    ASSERT_EQ(
        index->open(path, {StorageOptions::StorageType::kBufferPool, true}), 0);
    for (uint32_t i = 0; i < kRows; ++i) {
      float row[kDim];
      for (uint32_t d = 0; d < kDim; ++d)
        row[d] = float((i * 17 + d * 31) % 1009) / 1024.0f;
      ASSERT_EQ(index->add(VectorData{DenseVector{row}}, kRows - i), 0);
    }
    ASSERT_EQ(index->train(), 0);
    EXPECT_LE(pool.used(), pool.capacity());
    ASSERT_EQ(index->close(), 0);
    index.reset();
    for (auto mode : {StorageOptions::StorageType::kMMAP,
                      StorageOptions::StorageType::kBufferPool}) {
      auto reopened = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_EQ(reopened->open(path, {mode, false, true}), 0);
      for (uint32_t i = 0; i < kRows; ++i) {
        VectorDataBuffer result;
        ASSERT_EQ(reopened->fetch(kRows - i, &result), 0);
        const auto &bytes =
            std::get<DenseVectorBuffer>(result.vector_buffer).data;
        ASSERT_EQ(bytes.size(), kDim * sizeof(float));
        for (uint32_t d = 0; d < kDim; ++d) {
          float value = 0;
          std::memcpy(&value, bytes.data() + d * sizeof(float), sizeof(value));
          EXPECT_FLOAT_EQ(value, float((i * 17 + d * 31) % 1009) / 1024.0f);
        }
      }
      ASSERT_EQ(reopened->close(), 0);
    }
    ASSERT_TRUE(std::filesystem::remove(path));
    EXPECT_EQ(pool.used(), 0u);
  }
}

}  // namespace
}  // namespace zvec::core_interface

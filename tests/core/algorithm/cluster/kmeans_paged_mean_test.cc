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

#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "cluster/paged_mean_table.h"

namespace zvec::core {
namespace {
class KmeansPagedMeanTest : public ::testing::Test {
 protected:
  size_t previous_{0};
  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    previous_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    ASSERT_EQ(pool.init(0), 0);
  }
  void TearDown() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(previous_), 0);
  }
};

TEST_F(KmeansPagedMeanTest,
       ExactStateSurvivesEvictionForNumericAndPackedMeans) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(pool.init(ailego::VecBufferPool::metadata_bytes_for_page_count(
                          4096, true) +
                      64 * 1024),
            0);
  std::vector<std::shared_ptr<VectorMean>> means{
      std::make_shared<NumericalVectorMean<float>>(16),
      std::make_shared<NumericalVectorMean<ailego::Float16>>(16),
      std::make_shared<NumericalVectorMean<int8_t>>(16),
      std::make_shared<NibbleVectorMean<uint8_t>>(16),
      std::make_shared<BinaryVectorMean>(16)};
  float floats[16];
  ailego::Float16 halves[16];
  int8_t ints[16];
  uint8_t packed[8]{0x23, 0xf1, 0x87, 0x10, 0x56, 0x32, 0x11, 0x81};
  for (size_t i = 0; i < 16; ++i) {
    floats[i] = float(i) / 13.0f;
    halves[i] = floats[i];
    ints[i] = static_cast<int8_t>(i - 8);
  }
  const void *inputs[]{floats, halves, ints, packed, packed};
  const size_t sizes[]{sizeof(floats), sizeof(halves), sizeof(ints), 8, 2};
  for (size_t t = 0; t < means.size(); ++t) {
    auto &mean = *means[t];
    ASSERT_TRUE(mean.plus(inputs[t], sizes[t]));
    ASSERT_TRUE(mean.plus(inputs[t], sizes[t]));
    std::vector<uint8_t> expected(mean.state_size()), scratch;
    mean.save_state(expected.data());
    PagedMeanTable table;
    ASSERT_EQ(table.init(4096, mean), 0);
    for (size_t i = 0; i < 4096; ++i)
      ASSERT_EQ(table.write(i, mean, &scratch), 0);
    for (size_t i = 0; i < 4096; i += 17) {
      mean.reset();
      ASSERT_EQ(table.read(i, &mean, &scratch), 0);
      EXPECT_EQ(scratch, expected);
      EXPECT_EQ(mean.count(), 2u);
    }
    EXPECT_LE(pool.used(), pool.capacity());
  }
}

TEST_F(KmeansPagedMeanTest, FixedCentroidsMatchHeapAndReleasePool) {
  static constexpr size_t kDimension = 137;
  IndexMeta meta(IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto features = std::make_shared<CompactIndexFeatures>(meta);
  for (size_t i = 0; i < 513; ++i) {
    float row[kDimension];
    for (size_t d = 0; d < kDimension; ++d)
      row[d] = float((i * 7 + d * 13) % 101) / 17;
    features->emplace(row);
  }
  for (bool batch : {false, true}) {
    IndexCluster::CentroidList reference;
    auto &pool = ailego::MemoryLimitPool::get_instance();
    const size_t budget_bytes =
        2 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
        64 * 1024;
    for (size_t budget : {size_t{0}, budget_bytes}) {
      ASSERT_EQ(pool.init(budget), 0);
      auto cluster = IndexFactory::CreateCluster("KmeansCluster");
      ailego::Params params;
      params.set("zvec.kmeans.cluster.max_iterations", 3);
      params.set("zvec.kmeans.cluster.batch", batch);
      ASSERT_EQ(cluster->init(meta, params), 0);
      ASSERT_EQ(cluster->mount(features), 0);
      IndexCluster::CentroidList centers;
      // Exceed the small inline limit so clustering exercises paged means,
      // including accumulator rows that cross page boundaries.
      for (size_t i = 0; i < 64; ++i) {
        centers.emplace_back(features->element(i), meta.element_size());
        centers.back().set_follows(batch ? 3 : 0);
      }
      // A single worker fixes summation order; the table's worker rows are
      // independently exercised by the concurrent build regressions.
      auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
      ASSERT_EQ(cluster->cluster(threads, centers), 0);
      EXPECT_EQ(pool.used(), 0u);
      if (!budget) {
        reference = centers;
      } else {
        ASSERT_EQ(centers.size(), reference.size());
        for (size_t i = 0; i < centers.size(); ++i) {
          EXPECT_EQ(centers[i].follows(), reference[i].follows());
          EXPECT_DOUBLE_EQ(centers[i].score(), reference[i].score());
          EXPECT_EQ(std::memcmp(centers[i].vector<float>().data(),
                                reference[i].vector<float>().data(),
                                meta.element_size()),
                    0);
        }
      }
    }
  }
}
}  // namespace
}  // namespace zvec::core

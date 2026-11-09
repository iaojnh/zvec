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

#include "../core/algorithm/cluster/stratified_cluster_storage_test_helper.h"

namespace zvec::core {
namespace {
class StratifiedClusterStorageAcceptanceTest
    : public StratifiedClusterStorageTest {};

TEST_F(StratifiedClusterStorageAcceptanceTest,
       GenericKmeansTrainsOnePassSourceLargerThanPool) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(
      0, pool.init(4 * ailego::PoolPayloadBudget::storage_metadata_bytes(4096) +
                   768 * 1024));
  auto metadata = meta(IndexMeta::DT_FP32, 137);
  auto source = std::make_shared<GeneratedStratifiedHolder>(
      metadata, pool.capacity() * 2 / metadata.element_size() + 1);
  auto cluster = IndexFactory::CreateCluster("StratifiedCluster");
  auto options = params("KmeansCluster");
  Params child;
  child.set(KMEANS_CLUSTER_MAX_ITERATIONS, 2u);
  options.set(STRATIFIED_CLUSTER_FIRST_PARAMS, child);
  options.set(STRATIFIED_CLUSTER_SECOND_PARAMS, child);
  ASSERT_EQ(0, cluster->init(metadata, options));
  IndexCluster::CentroidList cents;
  auto *streaming = dynamic_cast<HolderCluster *>(cluster.get());
  ASSERT_NE(nullptr, streaming);
  ASSERT_EQ(0, streaming->cluster_holder(threads, source, cents));
  EXPECT_GT(source->count() * source->element_size(), pool.capacity());
  EXPECT_EQ(1u, source->iterations);
  size_t follows = 0;
  for (const auto &centroid : cents) {
    EXPECT_TRUE(centroid.feature_stored());
    follows += centroid.follows();
  }
  EXPECT_EQ(source->count(), follows);
}

}  // namespace
}  // namespace zvec::core

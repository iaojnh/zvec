// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
TEST(SparseQueryWorkspace, MatchesLegacyAndAliasesUnchangedInput) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  auto reformer = IndexFactory::CreateReformer("HalfFloatSparseReformer");
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init({}));
  const uint32_t counts[]{0, 10000, 3, 0, 7};
  std::vector<float> values(10010);
  std::vector<uint32_t> indices(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = float(int(i % 19) - 9) / 4;
    indices[i] = static_cast<uint32_t>(i);
  }
  IndexQueryMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  IndexQueryMeta converted, legacy_meta;
  std::string legacy;
  ASSERT_EQ(0, reformer->transform(counts, indices.data(), values.data(), meta,
                                   5, &legacy, &legacy_meta));
  IndexReformer::SparseQueryBuffer workspace;
  const void *output = nullptr;
  ASSERT_EQ(0, pool.init(1024));
  EXPECT_THROW(reformer->transform_sparse_batch(
                   counts, indices.data(), values.data(), meta, 5, &workspace,
                   &output, &converted),
               std::bad_alloc);
  EXPECT_TRUE(workspace.empty());
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(64 * 1024));
  ASSERT_EQ(0, reformer->transform_sparse_batch(
                   counts, indices.data(), values.data(), meta, 5, &workspace,
                   &output, &converted));
  EXPECT_EQ(workspace.data(), output);
  ASSERT_EQ(legacy.size(), workspace.size());
  EXPECT_EQ(0, std::memcmp(legacy.data(), output, legacy.size()));
  EXPECT_EQ(legacy_meta.data_type(), converted.data_type());
  EXPECT_GE(pool.used(), workspace.size());
  IndexReformer::SparseQueryBuffer().swap(workspace);
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1));
  ASSERT_EQ(0, reformer->transform_sparse_batch(
                   counts, indices.data(), legacy.data(), converted, 5,
                   &workspace, &output, &converted));
  EXPECT_EQ(legacy.data(), output);
  EXPECT_EQ(0u, pool.used());
  const uint32_t overflow[]{UINT32_MAX, 1};
  EXPECT_EQ(IndexError_OutOfRange,
            reformer->transform_sparse_batch(overflow, indices.data(),
                                             values.data(), meta, 2, &workspace,
                                             &output, &converted));
}
}  // namespace zvec::core

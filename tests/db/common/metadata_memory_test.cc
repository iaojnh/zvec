// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/common/metadata_memory.h"
#include <gtest/gtest.h>

namespace zvec::detail {

TEST(RetainedMetadataMemoryTest, ReusedAllowanceObeysHostPressure) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  constexpr size_t kMiB = 1024 * 1024;
  ASSERT_EQ(0, pool.init(64 * kMiB));
  RetainedMetadataBudget budget;
  {
    RetainedMetadataBudget::Lease warmup(budget);
    warmup.acquire(kMiB);
  }
  const auto backing = budget.reserved();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  pool.set_memory_pressure(Pressure::kCritical);
  {
    RetainedMetadataBudget::Lease blocked(budget);
    EXPECT_THROW(blocked.acquire(kMiB), std::bad_alloc);
    EXPECT_EQ(0u, blocked.bytes());
  }
  pool.set_memory_pressure(Pressure::kBackground);
  {
    RetainedMetadataBudget::Lease recovered(budget);
    EXPECT_NO_THROW(recovered.acquire(kMiB));
  }
  pool.set_memory_pressure(Pressure::kNormal);
  EXPECT_EQ(backing, budget.reserved());
}

}  // namespace zvec::detail

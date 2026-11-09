// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/common/result_memory_allowance.h"
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include "db/common/query_result_memory.h"
#include "db/index/column/fts_column/fts_memory.h"
#include "db/index/column/inverted_column/inverted_search_result.h"
#include "db/index/common/doc_field_converter.h"
#include "../../ailego/buffer/pool_payload_budget.h"
#include "binding/c/c_memory.h"
#include "db/common/metadata_memory.h"

namespace zvec::ailego {

struct MemoryLimitPoolTestPeer {
  static bool configure(MemoryLimitPool &pool, size_t limit, size_t rss) {
    sample_ = rss;
    return pool.process_rss_.configure(limit, +[] { return sample_.load(); });
  }

  static void sample(MemoryLimitPool &pool, size_t rss) {
    sample_ = rss;
    pool.process_rss_.refresh(true);
  }

  static bool workspace_waiting(MemoryLimitPool &pool) {
    return pool.workspace_waiting_size_.load(std::memory_order_relaxed) != 0;
  }

  inline static std::atomic<size_t> sample_{0};
};

}  // namespace zvec::ailego

namespace zvec::detail {
namespace {

constexpr size_t kMiB = 1024 * 1024;

class ResultMemoryAllowanceTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(64 * kMiB));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
};

TEST_F(ResultMemoryAllowanceTest, RetainsAndReusesBackingAfterCallsFinish) {
  ResultMemoryAllowance allowance;
  for (int i = 0; i < 3; ++i) {
    {
      ResultMemoryAllowance::Lease lease(&allowance);
      lease.add(2 * kMiB);
      lease.add(kMiB);
      EXPECT_EQ(3 * kMiB, allowance.active_bytes());
      EXPECT_EQ(6 * kMiB, allowance.reserved_bytes());
    }
    EXPECT_EQ(0u, allowance.active_bytes());
    EXPECT_EQ(6 * kMiB, pool_.external_used());
  }
}

TEST_F(ResultMemoryAllowanceTest, PendingCopyRetainsHeadroomAcrossSamples) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Peer = ailego::MemoryLimitPoolTestPeer;
  ASSERT_TRUE(Peer::configure(native, 64 * kMiB, 16 * kMiB));
  auto restore = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.configure_process_rss_limit(0)); });
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease warmup(&allowance);
    ASSERT_NO_THROW(warmup.add(8 * kMiB));
  }
  ASSERT_EQ(16 * kMiB, allowance.reserved_bytes());
  Peer::sample(native, 57 * kMiB);
  ResultMemoryAllowance::Lease first(&allowance), next(&allowance);
  {
    ResultMemoryAllowance::PendingCopy pending;
    ASSERT_NO_THROW(first.add_pending(4 * kMiB, pending));
    Peer::sample(native, 57 * kMiB);
    EXPECT_THROW(next.add(4 * kMiB), std::bad_alloc);
    EXPECT_EQ(4 * kMiB, allowance.active_bytes());
    // The producer abandons the copy before touching the destination.
  }
  EXPECT_NO_THROW(next.add(4 * kMiB));
}

TEST_F(ResultMemoryAllowanceTest,
       ReusedBackingObeysCriticalPressureAndRecovery) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease warmup(&allowance);
    warmup.add(kMiB);
  }
  const auto backing = allowance.reserved_bytes();
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  native.set_memory_pressure(Pressure::kCritical);
  {
    ResultMemoryAllowance::Lease blocked(&allowance);
    EXPECT_THROW(blocked.add(kMiB), std::bad_alloc);
  }
  // Background still permits mandatory work within an existing reservation.
  native.set_memory_pressure(Pressure::kBackground);
  {
    ResultMemoryAllowance::Lease recovered(&allowance);
    EXPECT_NO_THROW(recovered.add(kMiB));
  }
  native.set_memory_pressure(Pressure::kNormal);
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(backing, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, ReturnedLeaseDoesNotKeepMaterializationLock) {
  ResultMemoryAllowance allowance;
  auto held = std::make_shared<ResultMemoryAllowance::Lease>(&allowance, true);
  held->add(kMiB);
  held->end_materialization();
  auto next = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease lease(&allowance, true);
    lease.add(kMiB);
    return allowance.active_bytes();
  });
  ASSERT_EQ(std::future_status::ready, next.wait_for(std::chrono::seconds(2)));
  EXPECT_EQ(2 * kMiB, next.get());
  EXPECT_EQ(kMiB, allowance.active_bytes());
  auto release =
      std::async(std::launch::async,
                 [owner = std::move(held)]() mutable { owner.reset(); });
  release.get();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, RejectedGrowthLeavesLeaseAndChargeUnchanged) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease lease(&allowance);
    lease.add(kMiB);
    const size_t pressure = pool_.capacity() - pool_.used();
    ASSERT_TRUE(pool_.try_charge_external(pressure));
    EXPECT_THROW(lease.add(kMiB), std::bad_alloc);
    EXPECT_EQ(kMiB, allowance.active_bytes());
    EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
    pool_.release_external(pressure);
    // Retry succeeds once the competing owner returns real capacity.
    lease.add(kMiB);
    EXPECT_EQ(2 * kMiB, allowance.active_bytes());
  }
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(4 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, FailedMaterializerUnblocksNextCall) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease oversized(&allowance, true);
    EXPECT_THROW(oversized.add(pool_.capacity()), std::bad_alloc);
  }
  auto next = std::async(std::launch::async, [&] {
    ResultMemoryAllowance::Lease lease(&allowance, true);
    lease.add(kMiB);
  });
  EXPECT_EQ(std::future_status::ready, next.wait_for(std::chrono::seconds(5)));
  next.get();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, DenseVectorCopyKeepsItsOwnAdmission) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  auto doc = MakeQueryDocument(memory);
  auto field = std::make_shared<FieldSchema>("vector", DataType::VECTOR_FP32);
  const std::vector<float> values{1.0f, 2.0f, -3.0f};
  const size_t expected =
      192 + field->name().size() * 2 + values.size() * sizeof(float);
  {
    vector_column_params::VectorDataBuffer source;
    auto &buffer =
        std::get<vector_column_params::DenseVectorBuffer>(source.vector_buffer);
    buffer.reserve(values.size() * sizeof(float));
    buffer.data.assign(reinterpret_cast<const char *>(values.data()),
                       values.size() * sizeof(float));
    ASSERT_TRUE(ConvertVectorDataBufferToDocField(field, source, doc.get(),
                                                  memory.get())
                    .ok());
    EXPECT_EQ(expected, allowance.active_bytes());
  }
  memory.reset();
  EXPECT_EQ(values, doc->get_ref<std::vector<float>>("vector"));
  EXPECT_EQ(expected, allowance.active_bytes());
  auto held = doc;
  doc.reset();
  EXPECT_EQ(expected, allowance.active_bytes());
  held.reset();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(expected * 2, allowance.reserved_bytes());
}

TEST_F(ResultMemoryAllowanceTest, QueryDocumentKeepsAdmissionUntilLastOwner) {
  ResultMemoryAllowance allowance;
  auto memory = std::make_shared<ResultMemoryAllowance::Lease>(&allowance);
  memory->add(kMiB);
  auto first = MakeQueryDocument(memory);
  first->set_pk("retained");
  auto second = first;
  memory.reset();
  first.reset();
  EXPECT_EQ(kMiB, allowance.active_bytes());
  EXPECT_EQ("retained", second->pk_ref());
  second.reset();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(2 * kMiB, allowance.reserved_bytes());
}

}  // namespace
}  // namespace zvec::detail

TEST(CMemoryOwnerTest, CopiesKeepAdmissionUntilCrossThreadFree) {
  using namespace zvec::detail;
  constexpr size_t kMiB = 1024 * 1024;
  auto &pool = zvec::ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(64 * kMiB));
  auto &allowance = SharedResultMemoryAllowance();
  const auto initial = allowance.active_bytes();
  auto *copy = AllocateCResult(kMiB);
  ASSERT_NE(nullptr, copy);
  EXPECT_GE(allowance.active_bytes() - initial, kMiB);
  const auto backing = allowance.reserved_bytes();
  std::async(std::launch::async, [copy] { FreeCResult(copy); }).get();
  EXPECT_EQ(initial, allowance.active_bytes());
  EXPECT_EQ(backing, allowance.reserved_bytes());
  copy = AllocateCResult(kMiB);
  EXPECT_EQ(backing, allowance.reserved_bytes());
  FreeCResult(copy);
  EXPECT_EQ(initial, allowance.active_bytes());
  EXPECT_THROW(AllocateCResult(64 * kMiB), std::bad_alloc);
  EXPECT_EQ(initial, allowance.active_bytes());
  EXPECT_THROW(AllocateCResultArray(SIZE_MAX, sizeof(void *)), std::bad_alloc);
  // Existing C allocations remain valid inputs to the public free path.
  FreeCResult(std::malloc(17));
  FreeCResult(nullptr);
}

namespace zvec::detail {

}  // namespace zvec::detail

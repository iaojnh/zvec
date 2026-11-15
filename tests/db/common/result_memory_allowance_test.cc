// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/common/result_memory_allowance.h"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static bool configure_rss(MemoryLimitPool &pool) {
    return pool.process_rss_.configure(
        size_t{128} * 1024 * 1024,
        +[]() -> size_t { return size_t{16} * 1024 * 1024; });
  }
};
}  // namespace zvec::ailego

namespace zvec::detail {
namespace {

class ResultMemoryAllowanceTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(32 * 1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
};

TEST_F(ResultMemoryAllowanceTest, ReleasedQuotaIsAvailableToNextAdmission) {
  ResultMemoryAllowance allowance;
  auto first = std::make_unique<ResultMemoryAllowance::Lease>(&allowance);
  first->add(4 * 1024 * 1024);
  first.reset();
  // Each result fits; retaining both logical charges would exceed the
  // quarter-budget limit. Do not query statistics before the next admission.
  ResultMemoryAllowance::Lease next(&allowance);
  EXPECT_NO_THROW(next.add(4 * 1024 * 1024));
  EXPECT_EQ(4u * 1024 * 1024, allowance.active_bytes());
}

TEST_F(ResultMemoryAllowanceTest, CopyRefundsAreVisibleToIdleReclaim) {
  ResultMemoryAllowance allowance;
  {
    ResultMemoryAllowance::Lease lease(&allowance);
    ResultMemoryAllowance::PendingCopy pending;
    lease.add_pending(1024, 256, 128, pending);
    pending.start();
  }
  allowance.reclaim_idle();
  EXPECT_EQ(0u, allowance.active_bytes());
  EXPECT_EQ(0u, allowance.reserved_bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, CrossThreadDestructionPreservesPinnedQuota) {
  ResultMemoryAllowance allowance;
  ResultMemoryAllowance::Lease pinned(&allowance);
  pinned.add(128);
  constexpr size_t count = 64;
  std::vector<std::unique_ptr<ResultMemoryAllowance::Lease>> leases;
  for (size_t i = 0; i < count; ++i) {
    auto lease = std::make_unique<ResultMemoryAllowance::Lease>(&allowance);
    if (i % 2) {
      ResultMemoryAllowance::PendingCopy pending;
      lease->add_pending(1024, pending);
      pending.start();
    } else {
      lease->add(1024);
    }
    leases.push_back(std::move(lease));
  }
  EXPECT_EQ(128u + count * 1024, allowance.active_bytes());
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&, worker] {
      for (size_t i = worker; i < count; i += 8) leases[i].reset();
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(128u, allowance.active_bytes());
  allowance.reclaim_idle();
  EXPECT_EQ(256u, allowance.reserved_bytes());
  pinned.retain_at_most(0);
  allowance.reclaim_idle();
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, AdmissionsAndReclaimCanRaceMixedRefunds) {
  ResultMemoryAllowance allowance;
  constexpr size_t count = 8;
  std::atomic<size_t> failures{0}, finished{0};
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < count; ++worker) {
    workers.emplace_back([&] {
      for (int i = 0; i < 2000; ++i) {
        try {
          ResultMemoryAllowance::Lease lease(&allowance);
          lease.add(512);
          {
            ResultMemoryAllowance::PendingCopy pending;
            lease.add_pending(1024, pending);
            pending.start();
          }
          lease.retain_at_most(1024);
          lease.add(128);
        } catch (const std::bad_alloc &) {
          ++failures;
        }
      }
      ++finished;
    });
  }
  while (finished.load() != count) {
    allowance.reclaim_idle();
    if (allowance.active_bytes() > count * 1536) ++failures;
    std::this_thread::yield();
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0u, failures.load());
  EXPECT_EQ(0u, allowance.active_bytes());
  allowance.reclaim_idle();
  EXPECT_EQ(0u, allowance.reserved_bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(ResultMemoryAllowanceTest, PressureRetiresOnlyTheLastOwnerBacking) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  ASSERT_TRUE(ailego::MemoryLimitPoolTestPeer::configure_rss(native));
  auto restore = ailego::ScopeGuard::Make([&] {
    native.set_memory_pressure(Pressure::kNormal);
    EXPECT_TRUE(native.configure_process_rss_limit(0));
  });
  ResultMemoryAllowance allowance;
  auto first = std::make_unique<ResultMemoryAllowance::Lease>(&allowance);
  auto second = std::make_unique<ResultMemoryAllowance::Lease>(&allowance);
  first->add(1024);
  {
    ResultMemoryAllowance::PendingCopy pending;
    second->add_pending(2048, pending);
    pending.start();
  }
  const auto reserved = allowance.reserved_bytes();
  native.set_memory_pressure(Pressure::kCritical);
  first.reset();
  EXPECT_EQ(2048u, allowance.active_bytes());
  EXPECT_EQ(reserved, allowance.reserved_bytes());
  second.reset();
  EXPECT_EQ(0u, allowance.reserved_bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

}  // namespace
}  // namespace zvec::detail

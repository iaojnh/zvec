// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/cache.h>
#include "db/common/rocksdb_block_cache.h"

namespace zvec {
namespace {

class DelayedReleaseCache : public rocksdb::CacheWrapper {
 public:
  DelayedReleaseCache()
      : CacheWrapper(rocksdb::NewLRUCache(64 << 10)),
        resume_release_(erase_completed_.get_future()) {}

  const char *Name() const override { return "DelayedReleaseCache"; }

  std::future<void> release_entered() { return release_entered_.get_future(); }

  bool Release(Handle *handle, bool erase_if_last_ref) override {
    release_entered_.set_value();
    // Give the owner a chance to purge while this handle is still pinned.
    // With synchronized close the purge follows the unpin instead.
    resume_release_.wait_for(std::chrono::milliseconds(250));
    return target_->Release(handle, erase_if_last_ref);
  }

  void EraseUnRefEntries() override {
    target_->EraseUnRefEntries();
    erase_completed_.set_value();
  }

 private:
  std::promise<void> release_entered_;
  std::promise<void> erase_completed_;
  std::future<void> resume_release_;
};

TEST(RocksdbBlockCache, LastContextCloseDoesNotStrandReleasedBlock) {
  auto target = std::make_shared<DelayedReleaseCache>();
  auto cache = std::make_shared<RocksdbBlockCache>(target);
  auto context = RocksdbBlockCache::AcquireContext(cache);
  ASSERT_TRUE(context);
  rocksdb::Cache::Handle *handle = nullptr;
  int value = 0;
  ASSERT_TRUE(cache->Insert("sst block", &value, &rocksdb::kNoopCacheItemHelper,
                            sizeof(value), &handle).ok());
  ASSERT_NE(handle, nullptr);
  ASSERT_GT(cache->GetPinnedUsage(), 0u);

  auto entered = target->release_entered();
  std::thread unpin([&] { cache->Release(handle); });
  entered.wait();
  context.reset();
  unpin.join();

  EXPECT_EQ(cache->GetUsage(), 0u);
}

}  // namespace
}  // namespace zvec

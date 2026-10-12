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

#include "utility/immutable_segment_view.h"
#include <filesystem>
#include <future>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "utility/utility_params.h"

namespace zvec::core {
TEST(ImmutableSegmentViewTest, BoundedReadsOwnTheirLifetimeUnderPressure) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(pool.used(), 0u);
  auto restore = ailego::ScopeGuard::Make([&]() {
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(previous), 0);
  });
  const std::string path = "immutable_segment_view_test.index";
  auto remove =
      ailego::ScopeGuard::Make([&]() { std::filesystem::remove(path); });
  std::vector<uint8_t> bytes(1024 * 1024 + 73);
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = i % 251;
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(dumper->create(path), 0);
  ASSERT_EQ(dumper->write(bytes.data(), bytes.size()), bytes.size());
  ASSERT_EQ(dumper->append("payload", bytes.size(), 0, 0), 0);
  ASSERT_EQ(dumper->close(), 0);
  // Also cover the bypass-only open: no cache page fits, but load must still
  // avoid materializing the complete segment on the heap.
  for (size_t mode_index = 0; mode_index < 3; ++mode_index) {
    const char *mode =
        mode_index == 0 ? "MMapFileReadStorage" : "BufferReadStorage";
    SCOPED_TRACE(mode);
    SCOPED_TRACE(mode_index);
    ASSERT_EQ(pool.init(mode_index == 2 ? 1 : 128 * 1024), 0);
    auto storage = IndexFactory::CreateStorage(mode);
    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE,
               BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_EQ(storage->init(params), 0);
    ASSERT_EQ(storage->open(path, false), 0);
    ImmutableSegmentView view;
    ASSERT_EQ(view.load(storage, "payload", 0, bytes.size()), 0);
    EXPECT_EQ(view.data() == nullptr, storage->vec_buffer_pool() != nullptr);
    EXPECT_LE(pool.used(), pool.capacity());
    if (mode_index == 2) {
      // Opening a view needs no payload copy. Returning an owned contiguous
      // range does need budget; bypass must not silently allocate outside it.
      IndexStorage::MemoryBlock guard;
      const void *data = nullptr;
      EXPECT_EQ(view.view(0, 8193, &guard, &data), IndexError_ReadData);
      EXPECT_EQ(nullptr, data);
      EXPECT_EQ(0u, pool.used());
      continue;
    }
    auto verify = [&view, &bytes](size_t start) {
      for (size_t offset = start; offset < bytes.size(); offset += 8191) {
        const size_t count = std::min<size_t>(8193, bytes.size() - offset);
        IndexStorage::MemoryBlock guard, second;
        const void *first = nullptr, *other = nullptr;
        if (view.view(offset, count, &guard, &first) != 0) return false;
        if (view.view(17, 4097, &second, &other) != 0) return false;
        if (std::memcmp(first, bytes.data() + offset, count) != 0) return false;
      }
      return true;
    };
    auto task = std::async(std::launch::async, verify, size_t{7});
    EXPECT_TRUE(verify(0));
    EXPECT_TRUE(task.get());
    EXPECT_NE(view.read(bytes.size(), bytes.data(), 1), 0);
    EXPECT_NE(view.read(std::numeric_limits<size_t>::max(), bytes.data(), 2),
              0);
    size_t copied = 0;
    ASSERT_EQ(view.copy_to([&](size_t offset, const void *data, size_t size) {
      EXPECT_LE(size, 64 * 1024u);
      EXPECT_EQ(std::memcmp(data, bytes.data() + offset, size), 0);
      copied += size;
      return 0;
    }),
              0);
    EXPECT_EQ(copied, bytes.size());
  }
}
}  // namespace zvec::core

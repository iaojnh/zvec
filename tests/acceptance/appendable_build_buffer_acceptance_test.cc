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

#include "../core/utility/appendable_build_buffer_test_helper.h"

namespace zvec::core {
namespace {
class AppendableBuildBufferAcceptanceTest
    : public AppendableBuildBufferTest {};

TEST_F(AppendableBuildBufferAcceptanceTest, ReportsWriteBudgetDenialAndRecovers) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  AppendableBuildBuffer buffer;
  ASSERT_EQ(0, buffer.init(1024 * 1024));
  ASSERT_TRUE(buffer.paged());

  auto &memory = ailego::MemoryLimitPool::get_instance();
  const auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(memory.configure_process_rss_limit(0));
  });
  ASSERT_TRUE(memory.configure_process_rss_limit(1));
  const size_t offset = buffer.size() - 1;
  const char marker = 'm';
  EXPECT_EQ(IndexError_NoMemory, buffer.write(offset, &marker, 1));

  ASSERT_TRUE(memory.configure_process_rss_limit(0));
  ASSERT_EQ(0, buffer.write(offset, &marker, 1));
  char read_back = 0;
  ASSERT_EQ(0, buffer.read(offset, &read_back, 1));
  EXPECT_EQ(marker, read_back);
}

}  // namespace
}  // namespace zvec::core

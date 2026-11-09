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

#pragma once

#include "utility/appendable_build_buffer.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <ailego/pattern/scope_guard.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class AppendableBuildBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
};

}  // namespace
}  // namespace zvec::core

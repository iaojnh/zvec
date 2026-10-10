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

#include "utility/visit_filter.h"
#include <memory>
#include <type_traits>
#include <gtest/gtest.h>

namespace zvec::core {
namespace {

class VisitFilterDispatchTest
    : public testing::TestWithParam<VisitFilter::Mode> {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, filter_.init(GetParam(), 1024, 256, 0.001f));
  }

  void TearDown() override {
    filter_.destroy();
  }

  VisitFilter filter_;
};

TEST_P(VisitFilterDispatchTest, ConcreteViewSharesStateWithLegacyAccessors) {
  filter_.set_visited(3);
  int calls = 0;
  EXPECT_TRUE(dispatch_visit_filter(filter_, [&](auto visit) {
    using View = decltype(visit);
    static_assert(sizeof(View) == sizeof(void *));
    static_assert(std::is_trivially_copyable_v<View>);
    static_assert(!std::is_default_constructible_v<View>);
    EXPECT_EQ(GetParam() == VisitFilter::BloomFilter,
              (std::is_same_v<View, VisitFilterView<VisitBloomFilter>>));
    EXPECT_EQ(GetParam() == VisitFilter::BitMap,
              (std::is_same_v<View, VisitFilterView<VisitBitMap>>));
    EXPECT_EQ(GetParam() == VisitFilter::ByteMap,
              (std::is_same_v<View, VisitFilterView<VisitByteMap>>));
    EXPECT_EQ(GetParam() == VisitFilter::EpochBitMap,
              (std::is_same_v<View, VisitFilterView<VisitEpochBitMap>>));
    ++calls;

    EXPECT_TRUE(visit.visited(3));
    visit.clear();
    EXPECT_FALSE(filter_.visited(3));
    visit.set_visited(7);
    EXPECT_TRUE(filter_.visited(7));

    auto copy = visit;
    copy.clear();
    EXPECT_FALSE(visit.visited(7));
    filter_.set_visited(9);
    EXPECT_TRUE(copy.visited(9));
  }));
  EXPECT_EQ(1, calls);
  // Destruction of the view does not release the owner's context.
  EXPECT_TRUE(filter_.visited(9));
}

TEST_P(VisitFilterDispatchTest,
       RepeatedDispatchAndClearPreserveEpochSemantics) {
  // Cross the ByteMap epoch rollover; other modes must also clear each time.
  for (int i = 0; i < 300; ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(dispatch_visit_filter(filter_, [&](auto visit) {
      visit.clear();
      EXPECT_FALSE(visit.visited(17));
      visit.set_visited(17);
      EXPECT_TRUE(visit.visited(17));
    }));
    EXPECT_TRUE(filter_.visited(17));
  }
}

TEST_P(VisitFilterDispatchTest, AcceptsMoveOnlyGenericCallback) {
  auto callback = [calls = std::make_unique<int>(0)](auto visit) mutable {
    EXPECT_EQ(0, (*calls)++);
    visit.set_visited(23);
  };
  EXPECT_TRUE(dispatch_visit_filter(filter_, std::move(callback)));
  EXPECT_TRUE(filter_.visited(23));
}

INSTANTIATE_TEST_SUITE_P(Modes, VisitFilterDispatchTest,
                         testing::Values(VisitFilter::BloomFilter,
                                         VisitFilter::BitMap,
                                         VisitFilter::ByteMap,
                                         VisitFilter::EpochBitMap));

TEST(VisitEpochBitMapTest, CompactExactStateSurvivesWordBoundariesAndGrowth) {
  VisitEpochBitMap::Context state;
  ASSERT_TRUE(VisitEpochBitMap::reset(&state, 1000000, 1000000));
  EXPECT_EQ(state.bits.size() * sizeof(uint64_t) + state.generations.size(),
            140625u);
  VisitFilter reference;
  ASSERT_EQ(reference.init(VisitFilter::ByteMap, 1000000, 1000000, 0), 0);
  for (size_t generation = 0; generation < 520; ++generation) {
    VisitEpochBitMap::clear(&state);
    reference.clear();
    for (uint32_t i = 0; i < 1000; ++i) {
      uint32_t id = (i * 7919 + generation * 61) % 1000000;
      EXPECT_EQ(VisitEpochBitMap::visited(&state, id), reference.visited(id));
      VisitEpochBitMap::set_visited(&state, id);
      reference.set_visited(id);
      EXPECT_TRUE(VisitEpochBitMap::visited(&state, id));
      EXPECT_EQ(VisitEpochBitMap::visited(&state, id ^ 1),
                reference.visited(id ^ 1));
    }
  }
  VisitEpochBitMap::clear(&state);
  for (uint32_t id : {0u, 63u, 64u, 999999u})
    VisitEpochBitMap::set_visited(&state, id);
  ASSERT_TRUE(VisitEpochBitMap::reset(&state, 1000100, 1000100));
  for (uint32_t id : {0u, 63u, 64u, 999999u})
    EXPECT_TRUE(VisitEpochBitMap::visited(&state, id));
  EXPECT_FALSE(VisitEpochBitMap::visited(&state, 1000099));
  VisitEpochBitMap::set_visited(&state, 1000099);
  EXPECT_TRUE(VisitEpochBitMap::visited(&state, 1000099));
}

TEST(VisitFilterDispatch, RejectsUninitializedFilterWithoutInvokingCallback) {
  VisitFilter filter;
  int calls = 0;
  EXPECT_FALSE(dispatch_visit_filter(filter, [&](auto) { ++calls; }));
  EXPECT_EQ(0, calls);
}

TEST(VisitFilterDispatch, RejectsUnsupportedModesWithoutInvokingCallback) {
  for (int mode : {0, -1, 42}) {
    SCOPED_TRACE(mode);
    VisitFilter filter;
    // Legacy init does not allocate a context for unsupported modes.
    filter.init(mode, 1024, 256, 0.001f);
    int calls = 0;
    EXPECT_FALSE(dispatch_visit_filter(filter, [&](auto) { ++calls; }));
    EXPECT_EQ(0, calls);
    filter.destroy();
  }
}

}  // namespace
}  // namespace zvec::core

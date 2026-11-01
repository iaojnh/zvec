// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <utility>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "flat/flat_utility.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace {
thread_local size_t observed_size = 0;
thread_local size_t observed_allocations = 0;

// Observe the candidate-sized allocations only, on the querying thread. The
// canary below verifies that the legacy initializer-list adapter is detected.
class CandidateAllocationScope {
 public:
  explicit CandidateAllocationScope(size_t bytes) {
    observed_allocations = 0;
    observed_size = bytes;
  }
  ~CandidateAllocationScope() {
    observed_size = 0;
  }
};
}  // namespace

void *operator new(size_t bytes) {
  if (observed_size != 0 && bytes == observed_size) ++observed_allocations;
  if (void *p = std::malloc(bytes ? bytes : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void *p) noexcept {
  std::free(p);
}
void operator delete(void *p, size_t) noexcept {
  std::free(p);
}

namespace zvec::core {
namespace {
constexpr size_t kDimension = 8;
constexpr size_t kCandidateCount = 8191;

TEST(FlatBorrowedKeysTest, AllocationObserverDetectsLegacyAdapterCopies) {
  const std::vector<uint64_t> keys(kCandidateCount, 1);
  size_t copies;
  {
    CandidateAllocationScope observe(keys.size() * sizeof(uint64_t));
    const std::vector<std::vector<uint64_t>> batch{keys};
    EXPECT_EQ(keys, batch[0]);
    EXPECT_NE(keys.data(), batch[0].data());
    copies = observed_allocations;
  }
  EXPECT_EQ(2u, copies);
}

void check_borrowed(IndexRunner &runner,
                    IndexRunner::Context::Pointer context) {
  ASSERT_NE(nullptr, context);
  const IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDimension);
  std::array<float, kDimension> query{};
  std::vector<uint64_t> keys(kCandidateCount);
  for (size_t i = 0; i < keys.size(); ++i) keys[i] = i % 64;
  const auto *original = keys.data();
  const std::vector<std::vector<uint64_t>> batch{keys};
  context->set_topk(4);
  ASSERT_EQ(
      0, runner.search_bf_by_p_keys_impl(query.data(), batch, qmeta, context));
  const auto expected = context->result();
  int ret;
  size_t copies;
  {
    CandidateAllocationScope observe(keys.size() * sizeof(uint64_t));
    ret = runner.search_bf_by_keys_impl(query.data(), keys, qmeta, context);
    copies = observed_allocations;
  }
  ASSERT_EQ(0, ret);
  EXPECT_EQ(0u, copies);
  ASSERT_EQ(expected.size(), context->result().size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i].key(), context->result()[i].key());
    EXPECT_FLOAT_EQ(expected[i].score(), context->result()[i].score());
  }
  EXPECT_EQ(original, keys.data());
  EXPECT_EQ(batch[0], keys);

  context->set_group_params(4, 2);
  context->set_group_by([](uint64_t key) { return std::to_string(key % 4); });
  ASSERT_EQ(
      0, runner.search_bf_by_p_keys_impl(query.data(), batch, qmeta, context));
  const auto expected_groups = context->group_result();
  {
    CandidateAllocationScope observe(keys.size() * sizeof(uint64_t));
    ret = runner.search_bf_by_keys_impl(query.data(), keys, qmeta, context);
    copies = observed_allocations;
  }
  ASSERT_EQ(0, ret);
  EXPECT_EQ(0u, copies);
  const auto &actual_groups = context->group_result();
  ASSERT_EQ(expected_groups.size(), actual_groups.size());
  for (size_t g = 0; g < expected_groups.size(); ++g) {
    EXPECT_EQ(expected_groups[g].group_id(), actual_groups[g].group_id());
    ASSERT_EQ(expected_groups[g].docs().size(), actual_groups[g].docs().size());
    for (size_t i = 0; i < expected_groups[g].docs().size(); ++i) {
      EXPECT_EQ(expected_groups[g].docs()[i].key(),
                actual_groups[g].docs()[i].key());
      EXPECT_FLOAT_EQ(expected_groups[g].docs()[i].score(),
                      actual_groups[g].docs()[i].score());
    }
  }
  EXPECT_EQ(batch[0], keys);
  const std::vector<uint64_t> empty;
  ASSERT_EQ(0,
            runner.search_bf_by_keys_impl(query.data(), empty, qmeta, context));
  EXPECT_TRUE(context->group_result().empty());
}

TEST(FlatBorrowedKeysTest, StreamerAndSearcherBorrowWithoutAdapterCopies) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "flat_borrowed_keys.index";
  const std::string dumped = "flat_borrowed_keys.dump";
  ASSERT_FALSE(std::filesystem::exists(path));
  ASSERT_FALSE(std::filesystem::exists(dumped));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
    std::filesystem::remove(path);
    std::filesystem::remove(dumped);
  });
  ASSERT_EQ(0, pool.init(8 * 1024 * 1024));
  IndexMeta meta(IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  auto streamer = IndexFactory::CreateStreamer("FlatStreamer");
  ailego::Params params;
  params.set(PARAM_FLAT_USE_CONTIGUOUS_MEMORY, false);
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto context = streamer->create_context();
  const IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDimension);
  std::array<float, kDimension> row{};
  for (size_t i = 0; i < 64; ++i) {
    row.fill(static_cast<float>(i));
    ASSERT_EQ(0, streamer->add_impl(i, row.data(), qmeta, context));
  }
  check_borrowed(*streamer, std::move(context));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(dumped));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  ASSERT_EQ(0, streamer->close());
  streamer.reset();
  ASSERT_EQ(0, storage->close());
  storage.reset();
  auto reader = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, reader->init({}));
  ASSERT_EQ(0, reader->open(dumped, false));
  auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
  ASSERT_EQ(0, searcher->init({}));
  ASSERT_EQ(0, searcher->load(reader, nullptr));
  check_borrowed(*searcher, searcher->create_context());
  ASSERT_EQ(0, searcher->unload());
  ASSERT_EQ(0, reader->close());
}
}  // namespace
}  // namespace zvec::core

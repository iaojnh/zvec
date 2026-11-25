// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <tuple>
#include <utility>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_streamer.h"

namespace {
thread_local bool inside_context = false;
thread_local bool fail_allocations = false;
thread_local bool fail_after_update_heap = false;
thread_local bool reached_update_heap = false;
thread_local bool injected = false;
thread_local void *context_object = nullptr;
thread_local size_t context_allocations = 0;
thread_local size_t context_deletions = 0;

constexpr uint32_t kTestM = 17;
constexpr size_t kUpdateHeapBytes =
    (static_cast<size_t>(
         zvec::core::HnswEntity::kDefaultL0MaxNeighborCntMultiplier * kTestM) +
     1) *
    sizeof(std::pair<zvec::core::node_id_t, zvec::core::dist_t>);
}  // namespace

void *operator new(std::size_t size) {
  if (fail_allocations) throw std::bad_alloc();
  if (inside_context && context_object) {
    if (reached_update_heap || size == kUpdateHeapBytes) {
      const bool fail = reached_update_heap || !fail_after_update_heap;
      reached_update_heap = true;
      if (fail) {
        injected = true;
        fail_allocations = true;
        throw std::bad_alloc();
      }
    }
  }
  void *ptr = std::malloc(size == 0 ? 1 : size);
  if (!ptr) throw std::bad_alloc();
  if (inside_context && size == sizeof(zvec::core::HnswContext)) {
    context_object = ptr;
    ++context_allocations;
  }
  return ptr;
}
void *operator new[](std::size_t size) {
  return ::operator new(size);
}
void operator delete(void *ptr) noexcept {
  if (ptr && ptr == context_object) {
    context_object = nullptr;
    ++context_deletions;
  }
  std::free(ptr);
}
void operator delete[](void *ptr) noexcept {
  ::operator delete(ptr);
}
void operator delete(void *ptr, std::size_t) noexcept {
  ::operator delete(ptr);
}
void operator delete[](void *ptr, std::size_t) noexcept {
  ::operator delete(ptr);
}

namespace zvec::core_interface {
namespace {
#if GTEST_HAS_DEATH_TEST
using StorageType = StorageOptions::StorageType;
namespace fs = std::filesystem;
thread_local bool armed = false;

class InitOomHnswStreamer : public core::HnswStreamer {
 public:
  Context::Pointer create_context() const override {
    if (armed) inside_context = true;
    return core::HnswStreamer::create_context();
  }
};

void CheckContextOwnership(StorageType type, bool legacy, bool later_failure) {
  ailego::LoggerBroker::Unregister();
  auto &memory = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, memory.init(size_t{256} << 20));
  const std::string directory =
      "hnsw_context_init_oom_" + std::to_string(static_cast<int>(type)) + "_" +
      std::to_string(legacy) + "_" + std::to_string(later_failure);
  fs::remove_all(directory);
  ASSERT_TRUE(fs::create_directory(directory));
  const std::string path = directory + "/target";
  auto param =
      HNSWIndexParamBuilder()
          .with_metric_type(MetricType::kL2sq)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(16)
          .with_m(kTestM)
          .with_quantizer_param(QuantizerParam(
              legacy ? QuantizerType::kInt8 : QuantizerType::kNone, legacy))
          .build();
  const std::array<float, 16> row{1.0F};
  auto warm = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, warm);
  ASSERT_EQ(0, warm->open(directory + "/warm", {type, true}));
  ASSERT_EQ(0, warm->add(VectorData{DenseVector{row.data()}}, 7));
  auto query = HNSWQueryParamBuilder().with_topk(1).with_ef_search(16).build();
  SearchResult warm_result;
  ASSERT_EQ(0, warm->search(VectorData{DenseVector{row.data()}}, query,
                            &warm_result));
  ASSERT_EQ(1u, warm_result.doc_list_.size());
  warm_result = SearchResult{};

  ailego::Factory<core::IndexStreamer>::Register<InitOomHnswStreamer> replace(
      "HnswStreamer");
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  for (int attempt = 0; attempt < 5; ++attempt) {
    SCOPED_TRACE(attempt);
    context_allocations = context_deletions = 0;
    reached_update_heap = injected = false;
    fail_after_update_heap = later_failure;
    armed = true;
    int result = 0;
    bool threw = false;
    try {
      result = index->open(path, {type, true});
    } catch (...) {
      threw = true;
    }
    fail_allocations = inside_context = armed = false;
    if (!injected || context_object) {
      std::fprintf(stderr,
                   "init_heap=%d injected=%d allocations=%zu deletions=%zu "
                   "live_context=%p result=%d\n",
                   reached_update_heap, injected, context_allocations,
                   context_deletions, context_object, result);
    }
    ASSERT_TRUE(reached_update_heap);
    ASSERT_TRUE(injected);
    ASSERT_EQ(1u, context_allocations);
    ASSERT_FALSE(threw);
    ASSERT_EQ(core::IndexError_NoMemory, result);
    ASSERT_EQ(nullptr, context_object)
        << "Context survived failed initialization";
    ASSERT_EQ(1u, context_deletions);
    ASSERT_FALSE(fs::exists(path));
    ASSERT_EQ(1u, std::distance(fs::directory_iterator(directory),
                                fs::directory_iterator{}));
  }
  ASSERT_EQ(0, index->open(path, {type, true}));
  ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 7));
  SearchResult result;
  ASSERT_EQ(0,
            index->search(VectorData{DenseVector{row.data()}}, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(7u, result.doc_list_[0].key());
  // Result vectors may own storage leases independently of the index.
  result = SearchResult{};
  ASSERT_EQ(0, index->close());
  ASSERT_EQ(0, warm->close());
  index.reset();
  warm.reset();
  EXPECT_EQ(nullptr, context_object);
  // Process-wide eviction queues and slab headers outlive individual indexes.
  const auto stats = memory.stats();
  EXPECT_EQ(0u, stats.page_used);
  EXPECT_EQ(0u, stats.external_used);
  EXPECT_EQ(stats.queue_metadata_used + stats.slab_header_bytes,
            stats.metadata_used);
  ailego::Factory<core::IndexStreamer>::Register<core::HnswStreamer> restore(
      "HnswStreamer");
  fs::remove_all(directory);
}

class HnswContextOomTest
    : public ::testing::TestWithParam<std::tuple<StorageType, bool, bool>> {};

TEST_P(HnswContextOomTest, InitFailureReleasesContextAndAllowsRetry) {
  const auto [type, legacy, later_failure] = GetParam();
  ASSERT_EXIT(
      {
        CheckContextOwnership(type, legacy, later_failure);
        std::fflush(stdout);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    BothBackendsAndQuantizationPipelines, HnswContextOomTest,
    ::testing::Combine(::testing::Values(StorageType::kMMAP,
                                         StorageType::kBufferPool),
                       ::testing::Bool(), ::testing::Bool()));
#endif
}  // namespace
}  // namespace zvec::core_interface

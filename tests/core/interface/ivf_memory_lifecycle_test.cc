// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <filesystem>
#include <future>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/ivf/ivf_params.h"
#include "algorithm/ivf/ivf_searcher_context.h"
#include "interface/indexes/holder_builder.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core_interface {
namespace {
class InspectableIVF : public IVFIndex {
 public:
  int initialize(const BaseIndexParam &param) {
    proxima_index_params_.set(core::PARAM_IVF_BUILDER_THREAD_COUNT, 2U);
    return init(param);
  }
  void replace_reformer(core::IndexReformer::Pointer reformer) {
    reformer_ = std::move(reformer);
  }
  std::weak_ptr<core::IndexStorage> storage_handle() const {
    return storage_;
  }
  core::IndexContext *query_context() {
    return acquire_context().get();
  }
  std::weak_ptr<core::IVFEntity> query_entity() {
    return dynamic_cast<core::IVFSearcherContext *>(query_context())->entity();
  }
};

class RetryReformer : public core::IndexReformer {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return unload();
  }
  int unload() override {
    ailego::BudgetedVector<char>().swap(partial);
    return 0;
  }
  int load(core::IndexStorage::Pointer) override {
    ++calls;
    partial.resize(1U << 20);
    return calls == 1 ? core::IndexError_NoMemory : 0;
  }
  int calls{0};
  ailego::BudgetedVector<char> partial;
};

class RejectConverter : public core::IndexConverter {
 public:
  int init(const core::IndexMeta &, const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int train(core::IndexHolder::Pointer) override {
    return core::IndexError_NoMemory;
  }
  int dump(const core::IndexDumper::Pointer &) override {
    return 0;
  }
  const Stats &stats() const override {
    return stats_;
  }
  const core::IndexMeta &meta() const override {
    return meta_;
  }

 private:
  Stats stats_;
  core::IndexMeta meta_;
};

class IVFMemoryLifecycleTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool.capacity();
    ASSERT_EQ(0, pool.init(64U << 20));
    std::filesystem::remove(path);
    param = IVFIndexParamBuilder()
                .with_dimension(256)
                .with_metric_type(MetricType::kL2sq)
                .with_data_type(DataType::DT_FP32)
                .with_n_list(8)
                .with_n_iters(1)
                .build();
    row.fill(0.25f);
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_));
    std::filesystem::remove(path);
  }
  std::shared_ptr<InspectableIVF> make_index() {
    auto index = std::make_shared<InspectableIVF>();
    EXPECT_EQ(0, index->initialize(*param));
    return index;
  }
  void build() {
    Index::Pointer index = make_index();
    ASSERT_EQ(
        0, index->open(path, {StorageOptions::StorageType::kBufferPool, true}));
    for (uint32_t i = 0; i < 64; ++i) {
      row[i % row.size()] += 0.1f;
      ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, i));
    }
    ASSERT_EQ(0, index->train());
    ASSERT_EQ(0, index->close());
  }
  int search(const Index::Pointer &index, const IVFQueryParam::Pointer &query) {
    SearchResult result;
    return index->search(VectorData{DenseVector{row.data()}}, query, &result);
  }
  ailego::PoolPayloadBudget &pool = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
  BaseIndexParam::Pointer param;
  std::array<float, 256> row;
  const std::string path = "ivf_memory_lifecycle.index";
};

TEST_F(IVFMemoryLifecycleTest, MmapInputRejectsGrowthAndPreservesAcceptedRows) {
  Index::Pointer index = make_index();
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  const auto before = pool.used();
  ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 16383));
  // Empty IDs need directory entries, not 16383 full vector payloads.
  EXPECT_LT(pool.used() - before, 1U << 20);
  const std::string accepted(reinterpret_cast<const char *>(row.data()),
                             sizeof(row));
  row.fill(0.75f);
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    EXPECT_EQ(core::IndexError_NoMemory,
              index->add(VectorData{DenseVector{row.data()}}, 16383));
    EXPECT_EQ(core::IndexError_NoMemory,
              index->add(VectorData{DenseVector{row.data()}}, 32767));
  }
  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(16383, &fetched));
  EXPECT_EQ(accepted, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  EXPECT_EQ(core::IndexError_NoExist, index->fetch(1, &fetched));
  ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 16383));
  ASSERT_EQ(0, index->fetch(16383, &fetched));
  EXPECT_NE(accepted, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  EXPECT_EQ(0, index->close());
}

TEST_F(IVFMemoryLifecycleTest, ConverterRejectionIsNotReportedAsSuccess) {
  std::vector<std::pair<uint64_t, std::string>> docs;
  docs.emplace_back(
      0, std::string(reinterpret_cast<const char *>(row.data()), sizeof(row)));
  core::IndexHolder::Pointer holder;
  EXPECT_EQ(core::IndexError_NoMemory,
            BuildMultiPassHolder(DataType::DT_FP32, row.size(), docs,
                                 std::make_shared<RejectConverter>(), &holder));
}

TEST_F(IVFMemoryLifecycleTest, FailedOpenReleasesPartialStateAndCanRetry) {
  build();
  auto impl = make_index();
  Index::Pointer index = impl;
  auto reformer = std::make_shared<RetryReformer>();
  impl->replace_reformer(reformer);
  const auto before = pool.used();
  EXPECT_EQ(core::IndexError_NoMemory,
            index->open(
                path, {StorageOptions::StorageType::kBufferPool, false, true}));
  EXPECT_EQ(0u, reformer->partial.capacity());
  EXPECT_EQ(before, pool.used());
  EXPECT_EQ(0, index->open(path, {StorageOptions::StorageType::kBufferPool,
                                  false, true}));
  EXPECT_EQ(2, reformer->calls);
  EXPECT_EQ(0, index->close());
  EXPECT_EQ(0u, reformer->partial.capacity());
}

TEST_F(IVFMemoryLifecycleTest, QueryReleasesFilterAndReusesContext) {
  build();
  auto impl = make_index();
  Index::Pointer index = impl;
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kBufferPool,
                                  false, true}));
  auto context = impl->query_context();
  std::weak_ptr<ailego::BudgetedVector<char>> weak;
  {
    auto payload = std::make_shared<ailego::BudgetedVector<char>>(1U << 20);
    weak = payload;
    auto query = std::make_shared<IVFQueryParam>();
    query->topk = 1;
    query->nprobe = 8;
    query->filter = std::make_shared<core::IndexFilter>();
    query->filter->set([payload](uint64_t) { return payload->empty(); });
    ASSERT_EQ(0, search(index, query));
  }
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(context, impl->query_context());
  EXPECT_EQ(0, index->close());
}

TEST_F(IVFMemoryLifecycleTest, CloseAndDestructionReleaseIdleWorkerReaders) {
  build();
  for (bool close_first : {true, false}) {
    auto impl = make_index();
    Index::Pointer index = impl;
    ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kBufferPool,
                                    false, true}));
    auto storage = impl->storage_handle();
    std::weak_ptr<core::IVFEntity> context;
    std::promise<int> queried;
    std::promise<void> release;
    auto done = queried.get_future();
    auto leave = release.get_future();
    std::thread worker([&] {
      auto query = std::make_shared<IVFQueryParam>();
      query->topk = 1;
      const int ret = search(index, query);
      context = impl->query_entity();
      queried.set_value(ret);
      leave.wait();
    });
    EXPECT_EQ(0, done.get());
    if (close_first) EXPECT_EQ(0, index->close());
    index.reset();
    impl.reset();
    EXPECT_TRUE(storage.expired());
    EXPECT_TRUE(context.expired());
    // Only weak-reference control blocks can remain until the worker exits.
    EXPECT_LT(pool.used(), 4096u);
    release.set_value();
    worker.join();
    context.reset();
    EXPECT_EQ(0u, pool.used());
  }
}

TEST_F(IVFMemoryLifecycleTest, SwitchingIndexesAndWorkerExitReleaseReaders) {
  build();
  auto first = make_index();
  auto second = make_index();
  Index::Pointer a = first, b = second;
  ASSERT_EQ(0, a->open(path, {StorageOptions::StorageType::kBufferPool, false,
                              true}));
  ASSERT_EQ(0, b->open(path, {StorageOptions::StorageType::kBufferPool, false,
                              true}));
  std::weak_ptr<core::IVFEntity> first_reader, second_reader;
  std::thread worker([&] {
    auto query = std::make_shared<IVFQueryParam>();
    query->topk = 1;
    EXPECT_EQ(0, search(a, query));
    first_reader = first->query_entity();
    EXPECT_FALSE(first_reader.expired());
    EXPECT_EQ(0, search(b, query));
    EXPECT_TRUE(first_reader.expired());
    second_reader = second->query_entity();
    EXPECT_FALSE(second_reader.expired());
  });
  worker.join();
  EXPECT_TRUE(second_reader.expired());
  EXPECT_EQ(0, a->close());
  EXPECT_EQ(0, b->close());
}

TEST_F(IVFMemoryLifecycleTest, LowMemoryQueryPreservesCauseAndCanRetry) {
  build();
  Index::Pointer index = make_index();
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kBufferPool,
                                  false, true}));
  auto query = std::make_shared<IVFQueryParam>();
  query->topk = 1;
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    EXPECT_EQ(core::IndexError_NoMemory, search(index, query));
  }
  EXPECT_EQ(0, search(index, query));
  EXPECT_EQ(0, index->close());
}
}  // namespace
}  // namespace zvec::core_interface

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <numeric>
#include <gtest/gtest.h>
#include "flat_sparse/flat_sparse_provider.h"
#include "flat_sparse/flat_sparse_search.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include <filesystem>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include <cstring>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "algorithm/flat/flat_utility.h"
#include "algorithm/flat_sparse/flat_sparse_streamer_entity.h"
#include "utility/sparse_utility.h"

namespace zvec::core {
namespace {
class SparseRows : public FlatSparseEntity {
 public:
  SparseRows() {
    std::vector<uint32_t> indices(16384);
    std::iota(indices.begin(), indices.end(), 1);
    std::vector<float> values(indices.size(), 1.0f);
    SparseUtility::TransSparseFormat(indices.size(), indices.data(),
                                     values.data(), sizeof(float), row_);
  }
  uint32_t doc_cnt() const override {
    return 2;
  }
  uint32_t total_sparse_count() const override {
    return 32768;
  }
  node_id_t get_id(uint64_t key) const override {
    return key < 2 ? key : kInvalidNodeId;
  }
  uint64_t get_key(node_id_t id) const override {
    if (fail_key == id) throw ReadError{IndexError_ReadData};
    return id;
  }
  size_t sparse_unit_size() const override {
    return sizeof(float);
  }
  int get_sparse_vector_ptr_by_id(node_id_t, const void **data,
                                  uint32_t *length) const override {
    *data = row_.data();
    *length = row_.size();
    return 0;
  }
  int get_sparse_vector_ptr_by_id(node_id_t id,
                                  IndexStorage::MemoryBlock &block,
                                  uint32_t *length) const override {
    if (id == fail_payload) return IndexError_ReadData;
    block = IndexStorage::MemoryBlock::MakeBorrowedView(
        const_cast<char *>(row_.data()));
    *length = row_.size();
    return 0;
  }
  float get_search_distance(std::string_view query,
                            node_id_t id) const override {
    seen_counts.push_back(*reinterpret_cast<const uint32_t *>(query.data()));
    return static_cast<float>(id);
  }
  mutable std::vector<uint32_t> seen_counts;
  uint32_t fail_payload{kInvalidNodeId};
  uint32_t fail_key{kInvalidNodeId};

 private:
  std::string row_;
};
class FlatSparseWorkspaceBudgetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(2 * 1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{};
};

TEST_F(FlatSparseWorkspaceBudgetTest,
       QueryConversionRejectsThenReusesCapacity) {
  SparseRows rows;
  auto *owner = static_cast<const FlatSparseSearcher *>(nullptr);
  IndexContext::Pointer context(new FlatSparseContext(owner));
  context->set_topk(1);
  const uint32_t counts[]{16384, 3};
  std::vector<uint32_t> indices(16387);
  std::iota(indices.begin(), indices.begin() + 16384, 1);
  std::iota(indices.begin() + 16384, indices.end(), 1);
  std::vector<float> values(indices.size(), 1.0f);
  IndexQueryMeta query_meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used() - 32768);
  EXPECT_EQ(IndexError_NoMemory,
            FlatSearch(counts, indices.data(), values.data(), false, {},
                       query_meta, 2, meta, context, &rows));
  EXPECT_TRUE(rows.seen_counts.empty());
  pressure.retain_at_most(0);
  for (bool grouped : {false, true}) {
    if (grouped) {
      context->set_group_params(1, 1);
      context->set_group_by([](uint64_t key) { return std::to_string(key); });
    }
    rows.seen_counts.clear();
    ASSERT_EQ(0, FlatSearch(counts, indices.data(), values.data(), false, {},
                            query_meta, 2, meta, context, &rows));
    EXPECT_EQ((std::vector<uint32_t>{16384, 16384, 3, 3}), rows.seen_counts);
    auto *flat = static_cast<FlatSparseContext *>(context.get());
    const auto capacity = flat->query_buffer().capacity();
    EXPECT_GE(capacity, 16384 * 6u);
    const auto *buffer = flat->query_buffer().data();
    ASSERT_EQ(0, FlatSearch(counts, indices.data(), values.data(), false, {},
                            query_meta, 2, meta, context, &rows));
    EXPECT_EQ(buffer, flat->query_buffer().data());
    EXPECT_EQ(capacity, flat->query_buffer().capacity());
    EXPECT_GE(pool_.used(), capacity);
  }
}

TEST_F(FlatSparseWorkspaceBudgetTest,
       IteratorFailureRefundsAndRetryOwnsBuffers) {
  auto rows = std::make_shared<SparseRows>();
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  FlatSparseIndexProvider<SparseRows> provider(rows, meta, "SparseRows");
  const auto before = pool_.used();
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used() - 70000);
  // Enough for indices, but not both expanded arrays. Failure is distinct
  // from EOF; destroying the failed iterator refunds any partial workspace.
  {
    auto failed = provider.create_iterator();
    ASSERT_NE(nullptr, failed);
    EXPECT_FALSE(failed->is_valid());
    EXPECT_EQ(IndexError_NoMemory, failed->status());
  }
  EXPECT_EQ(pool_.capacity() - 70000, pool_.used());
  pressure.retain_at_most(0);
  auto iterator = provider.create_iterator();
  ASSERT_NE(nullptr, iterator);
  ASSERT_TRUE(iterator->is_valid());
  EXPECT_EQ(16384u, iterator->sparse_count());
  EXPECT_GE(pool_.used(), before + 2 * 16384 * sizeof(uint32_t));
  const size_t retained = pool_.used();
  iterator->next();
  EXPECT_TRUE(iterator->is_valid());
  EXPECT_EQ(retained, pool_.used());
  iterator.reset();
  EXPECT_EQ(before, pool_.used());
}
}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {
TEST_F(FlatSparseWorkspaceBudgetTest,
       FetchedPayloadOwnsChargeAcrossCopyMoveAndReuse) {
  std::vector<uint32_t> indices(16384);
  std::iota(indices.begin(), indices.end(), 1);
  std::vector<float> values(indices.size(), 1.0f);
  ailego::BudgetedString encoded;
  SparseUtility::TransSparseFormat(indices.size(), indices.data(),
                                   values.data(), sizeof(float), encoded);
  const size_t before = pool_.used();
  IndexSparseDocument result;
  SparseUtility::ReverseSparseFormat(encoded.data(), result, sizeof(float));
  EXPECT_EQ(16384u, result.sparse_count());
  const size_t payload = pool_.used() - before;
  EXPECT_GE(payload, indices.size() * (sizeof(float) + sizeof(uint32_t)));
  IndexSparseDocument copy(result);
  EXPECT_EQ(before + 2 * payload, pool_.used());
  IndexSparseDocument moved(std::move(copy));
  EXPECT_EQ(before + 2 * payload, pool_.used());
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_THROW(IndexSparseDocument denied(result), std::bad_alloc);
  EXPECT_THROW(
      result.reserve_sparse_buffers(2 * indices.size() * sizeof(uint32_t),
                                    2 * values.size() * sizeof(float)),
      std::bad_alloc);
  EXPECT_EQ(16384u, result.sparse_count());
  EXPECT_EQ(indices.size() * sizeof(uint32_t), result.sparse_indices().size());
  pressure.retain_at_most(0);
  // Re-decoding the same record reuses capacity and does not append old data.
  SparseUtility::ReverseSparseFormat(encoded.data(), result, sizeof(float));
  EXPECT_EQ(before + 2 * payload, pool_.used());
  EXPECT_EQ(indices.size() * sizeof(uint32_t), result.sparse_indices().size());
  result = IndexSparseDocument();
  EXPECT_EQ(before + payload, pool_.used());
}
}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {
TEST_F(FlatSparseWorkspaceBudgetTest, IteratorReportsKeyAndPayloadFailures) {
  auto rows = std::make_shared<SparseRows>();
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  FlatSparseIndexProvider<SparseRows> provider(rows, meta, "SparseRows");
  for (bool key_error : {false, true}) {
    rows->fail_payload = key_error ? kInvalidNodeId : 1;
    rows->fail_key = key_error ? 1 : kInvalidNodeId;
    auto iterator = provider.create_iterator();
    ASSERT_TRUE(iterator->is_valid());
    EXPECT_EQ(0u, iterator->key());
    iterator->next();
    EXPECT_FALSE(iterator->is_valid());
    EXPECT_EQ(IndexError_ReadData, iterator->status());
    iterator->next();
    EXPECT_EQ(IndexError_ReadData, iterator->status());
  }
}
}  // namespace
}  // namespace zvec::core

#include <filesystem>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
namespace zvec::core {
namespace {
TEST_F(FlatSparseWorkspaceBudgetTest,
       BuilderRejectsPrematureEndAndConverterHandlesLargeRows) {
  // Dump buffers need more than the query-only fixture budget. The intended
  // failure below is the source read, after the first row was written.
  ASSERT_EQ(0, pool_.init(8 * 1024 * 1024));
  const std::string path = "flat_sparse_iterator_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  auto rows = std::make_shared<SparseRows>();
  rows->fail_payload = 1;
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  auto provider = std::make_shared<FlatSparseIndexProvider<SparseRows>>(
      rows, meta, "SparseRows");
  auto converter = IndexFactory::CreateConverter("HalfFloatSparseConverter");
  ASSERT_EQ(0, converter->init(meta, {}));
  ASSERT_EQ(0, converter->transform(provider));
  auto converted = converter->sparse_result();
  {
    auto iter = converted->create_iterator();
    ASSERT_TRUE(iter->is_valid());
    ASSERT_EQ(16384u, iter->sparse_count());
    const auto *data =
        static_cast<const ailego::Float16 *>(iter->sparse_data());
    EXPECT_FLOAT_EQ(1.0f, static_cast<float>(data[16383]));
    iter->next();
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(IndexError_ReadData, iter->status());
  }
  auto builder = IndexFactory::CreateBuilder("FlatSparseBuilder");
  ASSERT_EQ(0, builder->init(converter->meta(), {}));
  ASSERT_EQ(0, builder->train(converted));
  ASSERT_EQ(0, builder->build(converted));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path));
  EXPECT_EQ(IndexError_ReadData, builder->dump(dumper));
  EXPECT_EQ(0u, builder->stats().dumped_count());
  // A failed dump must not be finalized as a complete index.
  EXPECT_NE(0, dumper->close());
}
TEST_F(FlatSparseWorkspaceBudgetTest,
       PublicOutputsChargeLifetimeAndLegacyCannotGrow) {
  auto rows = std::make_shared<SparseRows>();
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  IndexSparseProvider::Pointer provider =
      std::make_shared<FlatSparseIndexProvider<SparseRows>>(rows, meta,
                                                            "SparseRows");
  const auto baseline = pool_.used();
  IndexSparseDocument result;
  std::string indices, values;
  uint32_t count = 7;
  ailego::MemoryReservation pressure;
  pressure.add(pool_.available());
  EXPECT_EQ(IndexError_NoMemory, provider->get_sparse_vector(0, result));
  EXPECT_EQ(IndexError_NoMemory,
            provider->get_sparse_vector(0, &count, &indices, &values));
  EXPECT_TRUE(indices.empty());
  EXPECT_TRUE(values.empty());
  EXPECT_EQ(7u, count);
  pressure.retain_at_most(0);
  ASSERT_EQ(0, provider->get_sparse_vector(0, result));
  ASSERT_EQ(16384u, result.sparse_count());
  EXPECT_GE(pool_.used() - baseline, 2 * 16384 * sizeof(uint32_t));
  const auto retained = pool_.used();
  {
    auto copy = result;
    EXPECT_GT(pool_.used(), retained);
    const auto copied = pool_.used();
    auto moved = std::move(copy);
    EXPECT_EQ(copied, pool_.used());
    EXPECT_EQ(result.sparse_values(), moved.sparse_values());
  }
  EXPECT_EQ(retained, pool_.used());
  // The old API can fill caller-supplied storage without allocating.
  indices.reserve(16384 * sizeof(uint32_t));
  values.reserve(16384 * sizeof(float));
  pressure.add(pool_.available());
  EXPECT_EQ(0, provider->get_sparse_vector(0, result));
  EXPECT_EQ(0, provider->get_sparse_vector(0, &count, &indices, &values));
  EXPECT_EQ(result.sparse_indices(), indices);
  EXPECT_EQ(result.sparse_values(), values);
  provider.reset();
  pressure.retain_at_most(0);
  EXPECT_EQ(retained, pool_.used());
  result = IndexSparseDocument{};
  EXPECT_EQ(baseline, pool_.used());
}

}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {
class SparseFlowContext : public IndexContext {
 public:
  void set_topk(uint32_t) override {}
  const IndexDocumentList &result() const override {
    return results.at(0);
  }
  const IndexDocumentList &result(size_t i) const override {
    return results.at(i);
  }
  IndexDocumentList *mutable_result(size_t i) override {
    return &results.at(i);
  }
  void reset() override {
    results.clear();
  }
  std::vector<IndexDocumentList> results;
};

// Observe the actual framework handoff, without introducing another index's
// scratch allocations or training variability into this conversion test.
class SparseFlowSearcher : public IndexSearcher {
 public:
  const IndexMeta &meta() const override {
    return meta_;
  }
  const ailego::Params &params() const override {
    return params_;
  }
  const Stats &stats() const override {
    return stats_;
  }
  int cleanup() override {
    return 0;
  }
  int unload() override {
    return 0;
  }
  int load(IndexStorage::Pointer storage, IndexMetric::Pointer) override {
    return IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  }
  Context::Pointer create_context() const override {
    return std::make_unique<SparseFlowContext>();
  }
  int search_impl(const uint32_t *counts, const uint32_t *, const void *query,
                  const IndexQueryMeta &meta, uint32_t count,
                  Context::Pointer &context) const override {
    ++batches;
    last_input = query;
    if (meta.data_type() != IndexMeta::DT_FP16 || meta.unit_size() != 2)
      return IndexError_Mismatch;
    auto &results = static_cast<SparseFlowContext *>(context.get())->results;
    results.resize(count);
    for (auto &result : results) result.clear();
    const auto *values = static_cast<const ailego::Float16 *>(query);
    for (uint32_t i = 0; i < count; ++i) {
      float sum = 0;
      for (uint32_t j = 0; j < counts[i]; ++j) sum += float(*values++);
      results[i].emplace_back(counts[i], sum);
    }
    return 0;
  }
  int search_bf_impl(const uint32_t *counts, const uint32_t *indices,
                     const void *query, const IndexQueryMeta &meta,
                     uint32_t count, Context::Pointer &ctx) const override {
    return search_impl(counts, indices, query, meta, count, ctx);
  }
  int search_impl(uint32_t count, const uint32_t *indices, const void *query,
                  const IndexQueryMeta &meta,
                  Context::Pointer &ctx) const override {
    return search_impl(&count, indices, query, meta, 1, ctx);
  }
  int search_bf_impl(uint32_t count, const uint32_t *indices, const void *query,
                     const IndexQueryMeta &meta,
                     Context::Pointer &ctx) const override {
    return search_impl(&count, indices, query, meta, 1, ctx);
  }
  mutable size_t batches{0};
  mutable const void *last_input{nullptr};

 private:
  IndexMeta meta_;
  ailego::Params params_;
  Stats stats_;
};

TEST(SparseFlowWorkspace, OneBatchNoRetainedCopyAndRetryAfterBudgetFailure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "sparse_flow_workspace.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
    std::filesystem::remove(path);
  });
  ASSERT_EQ(0, pool.init(0));
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP16);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  meta.set_reformer("HalfFloatSparseReformer", 0, ailego::Params{});
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexHelper::SerializeToDumper(meta, dumper.get()));
  ASSERT_EQ(0, dumper->close());
  auto searcher = std::make_shared<SparseFlowSearcher>();
  IndexSparseFlow flow;
  ASSERT_EQ(0, flow.set_searcher(searcher));
  ASSERT_EQ(0, flow.load(path));
  auto ctx = flow.create_context();
  ASSERT_NE(nullptr, ctx);
  std::vector<uint32_t> counts(2049);
  std::vector<float> values;
  std::vector<uint32_t> indices;
  for (size_t i = 0; i < counts.size(); ++i) {
    counts[i] = static_cast<uint32_t>(i % 4) * 16;
    for (uint32_t j = 0; j < counts[i]; ++j) {
      indices.push_back(j);
      values.push_back(float(i % 13) / 4);
    }
  }
  IndexQueryMeta query_meta(IndexMeta::DT_FP32);
  for (bool brute : {false, true}) {
    auto search = [&] {
      return brute ? flow.search_bf_impl(counts.data(), indices.data(),
                                         values.data(), query_meta,
                                         counts.size(), ctx)
                   : flow.search_impl(counts.data(), indices.data(),
                                      values.data(), query_meta, counts.size(),
                                      ctx);
    };
    ASSERT_EQ(0, pool.init(4096));
    const size_t before = searcher->batches;
    EXPECT_EQ(IndexError_NoMemory, search());
    EXPECT_EQ(before, searcher->batches);
    EXPECT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(1024 * 1024));
    ASSERT_EQ(0, search());
    EXPECT_EQ(before + 1, searcher->batches);
    EXPECT_GE(pool.used(), counts.size() * sizeof(IndexDocument));
    EXPECT_TRUE(ctx->features().empty());
    for (size_t i = 0; i < counts.size(); ++i) {
      ASSERT_EQ(1u, ctx->result(i).size());
      EXPECT_EQ(counts[i], ctx->result(i)[0].key());
    }
    ctx->searcher_context()->reset();
    EXPECT_EQ(0u, pool.used());
  }
  std::vector<ailego::Float16> half(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    half[i] = ailego::Float16(values[i]);
  ASSERT_EQ(0, pool.init(1024 * 1024));
  ASSERT_EQ(0, flow.search_impl(counts.data(), indices.data(), half.data(),
                                IndexQueryMeta(IndexMeta::DT_FP16),
                                counts.size(), ctx));
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  ASSERT_EQ(0, flow.search_impl(counts.data(), indices.data(), half.data(),
                                IndexQueryMeta(IndexMeta::DT_FP16),
                                counts.size(), ctx));
  EXPECT_EQ(half.data(), searcher->last_input);
  EXPECT_EQ(pool.capacity(), pool.used());
  pressure.retain_at_most(0);
  ctx->searcher_context()->reset();
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1024));
  ASSERT_EQ(
      0, flow.search_impl(16, indices.data(), values.data(), query_meta, ctx));
  ASSERT_EQ(0, flow.search_bf_impl(16, indices.data(), values.data(),
                                   query_meta, ctx));
  ctx->searcher_context()->reset();
  EXPECT_EQ(0u, pool.used());
}
}  // namespace
}  // namespace zvec::core

namespace zvec::core {
TEST(SparseQueryWorkspace, MatchesLegacyAndAliasesUnchangedInput) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  auto reformer = IndexFactory::CreateReformer("HalfFloatSparseReformer");
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init({}));
  const uint32_t counts[]{0, 10000, 3, 0, 7};
  std::vector<float> values(10010);
  std::vector<uint32_t> indices(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = float(int(i % 19) - 9) / 4;
    indices[i] = static_cast<uint32_t>(i);
  }
  IndexQueryMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  IndexQueryMeta converted, legacy_meta;
  std::string legacy;
  ASSERT_EQ(0, reformer->transform(counts, indices.data(), values.data(), meta,
                                   5, &legacy, &legacy_meta));
  IndexReformer::SparseQueryBuffer workspace;
  const void *output = nullptr;
  ASSERT_EQ(0, pool.init(1024));
  EXPECT_THROW(reformer->transform_sparse_batch(
                   counts, indices.data(), values.data(), meta, 5, &workspace,
                   &output, &converted),
               std::bad_alloc);
  EXPECT_TRUE(workspace.empty());
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(64 * 1024));
  ASSERT_EQ(0, reformer->transform_sparse_batch(
                   counts, indices.data(), values.data(), meta, 5, &workspace,
                   &output, &converted));
  EXPECT_EQ(workspace.data(), output);
  ASSERT_EQ(legacy.size(), workspace.size());
  EXPECT_EQ(0, std::memcmp(legacy.data(), output, legacy.size()));
  EXPECT_EQ(legacy_meta.data_type(), converted.data_type());
  EXPECT_GE(pool.used(), workspace.size());
  IndexReformer::SparseQueryBuffer().swap(workspace);
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1));
  ASSERT_EQ(0, reformer->transform_sparse_batch(
                   counts, indices.data(), legacy.data(), converted, 5,
                   &workspace, &output, &converted));
  EXPECT_EQ(legacy.data(), output);
  EXPECT_EQ(0u, pool.used());
  const uint32_t overflow[]{UINT32_MAX, 1};
  EXPECT_EQ(IndexError_OutOfRange,
            reformer->transform_sparse_batch(overflow, indices.data(),
                                             values.data(), meta, 2, &workspace,
                                             &output, &converted));
}
}  // namespace zvec::core

namespace zvec::core {
namespace {
TEST(FlatSparseMemoryTest, DirectIdsHolesUpdatesAndReopen) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  const size_t budget =
      2 * ailego::PoolPayloadBudget::storage_metadata_bytes(4096) +
      64 * ailego::kVectorPageSize;
  ASSERT_EQ(0, pool.init(budget));
  for (const auto *mode : {"MMapFileStorage", "BufferStorage"}) {
    const std::string path = std::string("flat_sparse_direct_") + mode;
    ASSERT_FALSE(std::filesystem::exists(path));
    AILEGO_DEFER([&] { std::filesystem::remove(path); });
    IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
    meta.set_metric("InnerProductSparse", 0, ailego::Params{});
    ailego::Params params;
    params.set(PARAM_FLAT_USE_ID_MAP, false);
    meta.set_streamer("FlatSparseStreamer", 0, params);
    const uint32_t indices[]{1, 4, 7};
    const float values[]{0.5f, 1.0f, 2.0f};
    std::string row;
    SparseUtility::TransSparseFormat(3, indices, values, sizeof(float), row);
    for (int round = 0; round < 2; ++round) {
      auto storage = IndexFactory::CreateStorage(mode);
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(path, true));
      IndexStreamer::Stats stats;
      FlatSparseStreamerEntity entity(stats);
      ASSERT_EQ(0, entity.open(storage, meta));
      if (!round) {
        for (uint32_t id = 3; id < 3001; ++id)
          ASSERT_EQ(0, entity.add_vector_with_id(id, row, 3));
      } else {
        EXPECT_EQ(3001u, stats.loaded_count());
      }
      EXPECT_EQ(kInvalidNodeId, entity.get_id(0));
      EXPECT_EQ(kInvalidNodeId, entity.get_id(uint64_t{1} << 32));
      EXPECT_EQ(kInvalidNodeId, entity.get_id(3001));
      EXPECT_EQ(3000u, entity.get_id(3000));
      ASSERT_EQ(0, entity.add_vector_with_id(1, row, 3));
      auto clone = entity.clone();
      EXPECT_EQ(1u, clone->get_id(1));
      EXPECT_EQ(3000u, clone->get_id(3000));
      std::string actual;
      EXPECT_EQ(IndexError_NoMemory,
                clone->get_sparse_vector_by_key(3000, &actual));
      {
        IndexStorage::MemoryBlock encoded;
        uint32_t stored_length = 0;
        ASSERT_EQ(0, clone->get_sparse_vector_ptr_by_key(3000, encoded,
                                                         &stored_length));
        // The stored representation includes alignment padding.
        actual.reserve(stored_length);
      }
      ASSERT_EQ(0, clone->get_sparse_vector_by_key(3000, &actual));
      EXPECT_EQ(row, actual.substr(0, row.size()));
      clone.reset();
      // Holes have zero payload and retain their sentinel key in dumps.
      IndexStorage::MemoryBlock hole;
      uint32_t hole_length = 123;
      ASSERT_EQ(0, entity.get_sparse_vector_ptr_by_id(0, hole, &hole_length));
      EXPECT_EQ(0u, hole_length);
      const std::string dump_path = path + ".dump";
      ASSERT_FALSE(std::filesystem::exists(dump_path));
      AILEGO_DEFER([&] { std::filesystem::remove(dump_path); });
      auto dumper = IndexFactory::CreateDumper("FileDumper");
      ASSERT_EQ(0, dumper->create(dump_path));
      ASSERT_EQ(0, entity.dump(dumper));
      ASSERT_EQ(0, dumper->close());
      auto dumped = IndexFactory::CreateStorage("MMapFileReadStorage");
      ASSERT_EQ(0, dumped->init({}));
      ASSERT_EQ(0, dumped->open(dump_path, false));
      auto keys = dumped->get(PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID);
      ASSERT_NE(nullptr, keys);
      uint64_t prefix[3]{};
      ASSERT_EQ(sizeof(prefix), keys->fetch(0, prefix, sizeof(prefix)));
      EXPECT_EQ(kInvalidKey, prefix[0]);
      EXPECT_EQ(1u, prefix[1]);
      EXPECT_EQ(kInvalidKey, prefix[2]);
      keys.reset();
      ASSERT_EQ(0, dumped->close());
      ASSERT_EQ(0, entity.flush(0));
      ASSERT_EQ(0, entity.close());
      ASSERT_EQ(0, storage->close());
    }
  }
}

TEST(FlatSparseMemoryTest, PagedDumpPreservesAllLegacySegments) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  auto holder =
      std::make_shared<MultiPassIndexSparseHolder<IndexMeta::DT_FP32>>();
  constexpr size_t count = 131075;
  for (size_t i = 0; i < count; ++i) {
    ailego::SparseVector<float> row;
    ailego::NumericalVector<uint32_t> indices(3);
    ailego::NumericalVector<float> values(3);
    for (size_t d = 0; d < 3; ++d) {
      indices[d] = static_cast<uint32_t>(1 + d * 3);
      values[d] = static_cast<float>(i + d);
    }
    row.add_sparses(indices, values);
    ASSERT_TRUE(holder->emplace((i * 17 % count) + 99, row));
  }
  std::vector<std::string> reference;
  // Two live writable stores (keys + offsets, or source + keys) need fixed
  // writeback metadata plus the explicitly enabled arbitrary-key map.
  // Allow up to 128 bytes per map entry, then 64 pages for the dump.
  const size_t budget =
      2 * ailego::PoolPayloadBudget::storage_metadata_bytes(4096) +
      64 * ailego::kVectorPageSize + count * 128;
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    ASSERT_EQ(0, pool.init(mode % 2 ? budget : 0));
    if (mode == 2) reference.clear();
    const std::string path = "flat_sparse_paged_dump_" + std::to_string(mode);
    const std::string stream_path = path + ".stream";
    ASSERT_FALSE(std::filesystem::exists(path));
    ASSERT_FALSE(std::filesystem::exists(stream_path));
    AILEGO_DEFER([&] {
      std::filesystem::remove(path);
      std::filesystem::remove(stream_path);
    });
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path));
    if (mode < 2) {
      auto builder = IndexFactory::CreateBuilder("FlatSparseBuilder");
      ASSERT_EQ(0, builder->init(meta, {}));
      ASSERT_EQ(0, builder->train(holder));
      ASSERT_EQ(0, builder->build(holder));
      ASSERT_EQ(0, builder->dump(dumper));
    } else {
      auto storage = IndexFactory::CreateStorage(mode == 2 ? "MMapFileStorage"
                                                           : "BufferStorage");
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(stream_path, true));
      auto streamer = IndexFactory::CreateStreamer("FlatSparseStreamer");
      ASSERT_EQ(0, streamer->init(meta, {}));
      ASSERT_EQ(0, streamer->open(storage));
      auto context = streamer->create_context();
      auto it = holder->create_iterator();
      for (; it->is_valid(); it->next())
        ASSERT_EQ(0, streamer->add_impl(it->key(), it->sparse_count(),
                                        it->sparse_indices(), it->sparse_data(),
                                        IndexQueryMeta(IndexMeta::MT_SPARSE,
                                                       IndexMeta::DT_FP32),
                                        context));
      ASSERT_EQ(0, streamer->dump(dumper));
      context.reset();
      ASSERT_EQ(0, streamer->close());
      ASSERT_EQ(0, storage->close());
    }
    ASSERT_EQ(0, dumper->close());
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, false));
    size_t index = 0;
    for (const auto &name : {PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID,
                             PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID,
                             PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID,
                             PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID}) {
      auto segment = storage->get(name);
      ASSERT_NE(nullptr, segment);
      std::string bytes(segment->data_size(), '\0');
      ASSERT_EQ(bytes.size(), segment->fetch(0, bytes.data(), bytes.size()));
      if (mode % 2 == 0)
        reference.push_back(bytes);
      else
        EXPECT_TRUE(reference[index] == bytes) << "segment: " << name;
      ++index;
    }
    ASSERT_EQ(0, storage->close());
    EXPECT_EQ(0u, pool.used());
  }
}
}  // namespace
}  // namespace zvec::core

#include "algorithm/flat_sparse/flat_sparse_searcher_entity.h"
#include "tests/core/algorithm/flat/flat_fault_storage.h"
namespace zvec::core {
namespace {
TEST(FlatSparseMemoryTest,
     OffsetCommitFailureRollsBackPayloadAndPreservesOldRow) {
  const std::string path = "flat_sparse_record_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  auto storage = std::make_shared<test::FaultStorage>(
      IndexFactory::CreateStorage("MMapFileStorage"));
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  IndexStreamer::Stats stats;
  FlatSparseStreamerEntity entity(stats);
  ASSERT_EQ(0, entity.open(storage, meta));
  const uint32_t indices[]{1, 4, 7};
  const float values[]{1, 2, 3};
  std::string row;
  SparseUtility::TransSparseFormat(3, indices, values, sizeof(float), row);
  ASSERT_EQ(0, entity.add_vector_with_id(0, row, 3));
  auto data = storage->get(PARAM_FLAT_SPARSE_DATA_SEG_ID_PREFIX + "0");
  ASSERT_NE(nullptr, data);
  const auto committed_size = data->data_size();
  storage->fault->segment = PARAM_FLAT_SPARSE_OFFSET_SEG_ID_PREFIX + "0";
  std::string replacement;
  const float changed[]{9.0f};
  SparseUtility::TransSparseFormat(1, indices, changed, sizeof(float),
                                   replacement);
  for (uint32_t id : {0u, 1u}) {
    storage->fault->write = true;
    EXPECT_EQ(IndexError_WriteData,
              entity.add_vector_with_id(id, replacement, 1));
    EXPECT_EQ(1u, entity.doc_cnt());
    EXPECT_EQ(committed_size, data->data_size());
    std::string actual;
    ASSERT_EQ(0, entity.get_sparse_vector_by_key(0, &actual));
    EXPECT_EQ(row, actual.substr(0, row.size()));
  }
  storage->fault->write = true;
  EXPECT_EQ(IndexError_WriteData, entity.add(99, row, 3));
  EXPECT_EQ(committed_size, data->data_size());
  ASSERT_EQ(0, entity.add(99, row, 3));
  EXPECT_EQ(2u, entity.doc_cnt());
  ASSERT_EQ(0, entity.flush(0));
  ASSERT_EQ(0, entity.close());
  ASSERT_EQ(0, storage->close());
}

TEST(FlatSparseMemoryTest, SearcherMetadataReadsAreScopedAndErrorsAreDistinct) {
  const std::string path = "flat_sparse_metadata_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  auto holder =
      std::make_shared<MultiPassIndexSparseHolder<IndexMeta::DT_FP32>>();
  ailego::SparseVector<float> row;
  ailego::NumericalVector<uint32_t> indices(1);
  indices[0] = 1;
  ailego::NumericalVector<float> values(1);
  values[0] = 1;
  row.add_sparses(indices, values);
  ASSERT_TRUE(holder->emplace(37, row));
  auto builder = IndexFactory::CreateBuilder("FlatSparseBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, builder->init(meta, {}));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  auto storage = std::make_shared<test::FaultStorage>(
      IndexFactory::CreateStorage("BufferReadStorage"));
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, false));
  FlatSparseSearcherEntity entity;
  ASSERT_EQ(0, entity.load(storage, meta));
  const size_t block_reads = storage->fault->block_reads;
  for (size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(37u, entity.get_key(0));
    EXPECT_EQ(0u, entity.get_id(37));
  }
  EXPECT_EQ(block_reads, storage->fault->block_reads);
  storage->fault->segment = PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID;
  storage->fault->read = true;
  EXPECT_THROW(entity.get_key(0), FlatSparseEntity::ReadError);
  storage->fault->read = true;
  EXPECT_THROW(entity.get_id(37), FlatSparseEntity::ReadError);
  EXPECT_EQ(kInvalidNodeId, entity.get_id(99));
  ASSERT_EQ(0, entity.unload());
  ASSERT_EQ(0, storage->close());
}
}  // namespace
}  // namespace zvec::core

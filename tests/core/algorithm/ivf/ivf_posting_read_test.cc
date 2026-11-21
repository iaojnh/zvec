// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "ivf_builder.h"
#include "ivf_searcher.h"
#include "ivf_streamer.h"

namespace zvec::core {
namespace {

class PostingSegment : public IndexStorage::Segment {
 public:
  explicit PostingSegment(std::string bytes) : bytes_(std::move(bytes)) {}
  size_t data_size() const override {
    return bytes_.size();
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return 0;
  }
  size_t capacity() const override {
    return bytes_.size();
  }
  size_t fetch(size_t offset, void *out, size_t size) const override {
    if (offset > bytes_.size() || size > bytes_.size() - offset) return 0;
    std::memcpy(out, bytes_.data() + offset, size);
    return size;
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &out,
              size_t size) override {
    if (offset > bytes_.size() || size > bytes_.size() - offset) return 0;
    out.reset(bytes_.data() + offset);
    return size;
  }
  size_t read_scatter(size_t offset, ScatterBlock &out, size_t size) override {
    out.reset();
    if (offset > read_limit || size > read_limit - offset) return 0;
    IndexStorage::MemoryBlock block;
    if (read(offset, block, size) != size) return 0;
    out.reset_contiguous(std::move(block), size);
    if (fragmented) {
      const size_t first = 17;  // Split a vector and a CPU cache line.
      out.spans = {{base_data() + offset, first},
                   {base_data() + offset + first, size - first}};
    }
    auto active = active_leases;
    ++*active;
    out.lease =
        std::shared_ptr<void>(active.get(), [active](void *) { --*active; });
    return fail_read ? size - 1 : size;
  }
  const uint8_t *base_data() const override {
    return reinterpret_cast<const uint8_t *>(bytes_.data());
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<PostingSegment>(bytes_);
  }

  size_t read_limit{0};
  bool fragmented{false};
  bool fail_read{false};
  std::shared_ptr<size_t> active_leases{std::make_shared<size_t>(0)};

 private:
  std::string bytes_;
};

class PostingEntity : public IVFEntity {
 public:
  int prepare() {
    meta_.set_meta(IndexMeta::DT_FP32, 17);
    meta_.set_major_order(IndexMeta::MO_ROW);
    meta_.set_metric("SquaredEuclidean", 0, ailego::Params());
    auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
    if (!metric) return IndexError_NoExist;
    const int ret = metric->init(meta_, ailego::Params());
    if (ret != 0) return ret;
    calculator_ = std::make_shared<IVFDistanceCalculator>(meta_, metric, 32);
    header_.block_vector_count = 32;
    header_.block_size = 32 * meta_.element_size();
    header_.inverted_list_count = 2;
    header_.total_vector_count = 66;
    list_bytes = header_.block_size + ailego_align(meta_.element_size(), 32);
    header_.inverted_body_size = 2 * list_bytes;
    std::string bytes(header_.inverted_body_size, '\0');
    InvertedListMeta lists[2]{};
    uint64_t keys[66];
    for (size_t list = 0; list < 2; ++list) {
      lists[list].offset = list * list_bytes;
      lists[list].vector_count = 33;
      lists[list].block_count = 2;
      lists[list].id_offset = list * 33;
      for (size_t row = 0; row < 33; ++row) {
        const size_t id = list * 33 + row;
        keys[id] = id;
        const std::vector<float> vector(17, id);
        std::memcpy(
            bytes.data() + list * list_bytes + row * meta_.element_size(),
            vector.data(), meta_.element_size());
      }
    }
    postings = std::make_shared<PostingSegment>(std::move(bytes));
    inverted_ = postings;
    inverted_meta_ = std::make_shared<PostingSegment>(
        std::string(reinterpret_cast<const char *>(lists), sizeof(lists)));
    keys_ = std::make_shared<PostingSegment>(
        std::string(reinterpret_cast<const char *>(keys), sizeof(keys)));
    return 0;
  }
  const float *distance_data() const {
    return scatter_distances_.data();
  }
  size_t span_capacity() const {
    return scatter_data_.spans.capacity();
  }
  bool scatter_empty() const {
    return scatter_data_.spans.empty() && !scatter_data_.lease &&
           !scatter_data_.fallback.data();
  }
  std::shared_ptr<PostingSegment> postings;
  size_t list_bytes{0};
};

TEST(IVFPostingRead, AdjacentListBoundsAndScratchLifetimes) {
  PostingEntity entity;
  ASSERT_EQ(0, entity.prepare());
  const std::vector<float> query(17, 65.25f);
  const float *distances = nullptr;
  for (bool fragmented : {false, true}) {
    entity.postings->fragmented = fragmented;
    for (bool filtered : {false, true}) {
      for (size_t list : {0U, 1U}) {
        entity.postings->read_limit = (list + 1) * entity.list_bytes;
        IndexDocumentHeap heap;
        heap.limit(1);
        IndexContext::Stats stats;
        uint32_t scanned = 0;
        if (filtered) {
          IndexFilter filter;
          filter.set([](uint64_t key) { return key % 2 == 0; });
          ASSERT_EQ(0, entity.search(list, query.data(), filter, &scanned,
                                     &heap, &stats));
        } else {
          ASSERT_EQ(0,
                    entity.search(list, query.data(), &scanned, &heap, &stats));
        }
        ASSERT_EQ(1U, heap.size());
        EXPECT_EQ(list == 0 ? (filtered ? 31U : 32U) : 65U, heap[0].key());
        EXPECT_EQ(33U, scanned);
        EXPECT_EQ(0U, *entity.postings->active_leases);
        EXPECT_TRUE(entity.scatter_empty());
        EXPECT_GE(entity.span_capacity(), fragmented ? 2U : 1U);
        if (distances) EXPECT_EQ(distances, entity.distance_data());
        distances = entity.distance_data();
      }
    }
  }
  entity.postings->fail_read = true;
  for (bool filtered : {false, true}) {
    IndexDocumentHeap heap;
    heap.limit(1);
    IndexContext::Stats stats;
    uint32_t scanned = 0;
    IndexFilter filter;
    filter.set([](uint64_t key) { return key % 2 == 0; });
    const int ret =
        filtered
            ? entity.search(0, query.data(), filter, &scanned, &heap, &stats)
            : entity.search(0, query.data(), &scanned, &heap, &stats);
    EXPECT_EQ(IndexError_ReadData, ret);
    EXPECT_EQ(0U, *entity.postings->active_leases);
    EXPECT_TRUE(entity.scatter_empty());
  }
}

TEST(IVFPostingRead, PlainRowTailsPreserveFilteredAndUnfilteredResults) {
  for (uint32_t dimension : {17U, 33U}) {
    for (uint32_t count : {1U, 33U, 65U, 321U}) {
      SCOPED_TRACE(dimension);
      SCOPED_TRACE(count);
      const std::string path = "ivf_posting_read_tail.index";
      auto remove =
          ailego::ScopeGuard::Make([&] { ailego::File::RemovePath(path); });
      IndexMeta meta;
      meta.set_meta(IndexMeta::DT_FP32, dimension);
      meta.set_major_order(IndexMeta::MO_ROW);
      meta.set_metric("SquaredEuclidean", 0, ailego::Params());
      auto holder =
          std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dimension);
      for (uint32_t i = 0; i < count; ++i) {
        ailego::NumericalVector<float> vector(dimension);
        std::fill(vector.begin(), vector.end(), i / 8.0f);
        holder->emplace(i, vector);
      }
      ailego::Params build_params;
      build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
      build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
      IVFBuilder builder;
      ASSERT_EQ(0, builder.init(meta, build_params));
      ASSERT_EQ(0, builder.train(IndexThreads::Pointer(), holder));
      ASSERT_EQ(0, builder.build(IndexThreads::Pointer(), holder));
      auto dumper = IndexFactory::CreateDumper("FileDumper");
      ASSERT_NE(nullptr, dumper);
      ASSERT_EQ(0, dumper->create(path));
      ASSERT_EQ(0, builder.dump(dumper));
      ASSERT_EQ(0, dumper->close());
      ailego::Params search_params;
      search_params.set(PARAM_IVF_SEARCHER_NPROBE, 1U);
      search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
      search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0U);
      const float value = (count - 1) / 8.0f + 0.03125f;
      const std::vector<float> query(dimension, value);
      const IndexQueryMeta query_meta(IndexMeta::DT_FP32, dimension);
      const auto check = [&](auto *searcher) {
        auto context = searcher->create_context();
        ASSERT_NE(nullptr, context);
        context->set_topk(3);
        for (bool filtered : {false, true}) {
          SCOPED_TRACE(filtered);
          if (filtered) {
            context->set_filter([](uint64_t id) { return id % 2 == 0; });
          } else {
            context->reset_filter();
          }
          std::vector<std::pair<float, uint64_t>> expected;
          for (uint32_t i = 0; i < count; ++i) {
            if (filtered && i % 2 == 0) continue;
            const float delta = value - i / 8.0f;
            expected.emplace_back(delta * delta * dimension, i);
          }
          std::sort(expected.begin(), expected.end());
          expected.resize(std::min<size_t>(3, expected.size()));
          ASSERT_EQ(0,
                    searcher->search_impl(query.data(), query_meta, context));
          const auto &result = context->result(0);
          ASSERT_EQ(expected.size(), result.size());
          for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(expected[i].second, result[i].key());
            EXPECT_NEAR(expected[i].first, result[i].score(),
                        1e-3f * std::max(1.0f, expected[i].first));
          }
        }
      };
      for (const char *storage_name :
           {"FileReadStorage", "MMapFileReadStorage"}) {
        SCOPED_TRACE(storage_name);
        auto storage = IndexFactory::CreateStorage(storage_name);
        ASSERT_NE(nullptr, storage);
        ASSERT_EQ(0, storage->init(ailego::Params()));
        ASSERT_EQ(0, storage->open(path, false));
        IVFSearcher searcher;
        ASSERT_EQ(0, searcher.init(search_params));
        ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
        check(&searcher);
        ASSERT_EQ(0, searcher.unload());
        IVFStreamer streamer;
        ASSERT_EQ(0, streamer.init(meta, search_params));
        ASSERT_EQ(0, streamer.open(storage));
        check(&streamer);
        ASSERT_EQ(0, streamer.close());
      }
    }
  }
}

}  // namespace
}  // namespace zvec::core

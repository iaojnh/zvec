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
#include <algorithm>
#include <filesystem>
#include <future>
#include <numeric>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "ailego/pattern/defer.h"
#include "flat/flat_searcher.h"
#include "utility/utility_params.h"

namespace zvec::core {
namespace {
class WatchedSegment : public IndexStorage::Segment {
 public:
  explicit WatchedSegment(Pointer source, std::shared_ptr<size_t> bytes =
                                              std::make_shared<size_t>(0))
      : bytes_read(std::move(bytes)), source_(std::move(source)) {}
  size_t data_size() const override {
    return source_->data_size();
  }
  uint32_t data_crc() const override {
    return source_->data_crc();
  }
  size_t padding_size() const override {
    return source_->padding_size();
  }
  size_t capacity() const override {
    return source_->capacity();
  }
  size_t fetch(size_t offset, void *out, size_t size) const override {
    max_read = std::max(max_read, size);
    *bytes_read += size;
    return fail ? 0 : source_->fetch(offset, out, size);
  }
  size_t read(size_t offset, const void **out, size_t size) override {
    max_read = std::max(max_read, size);
    *bytes_read += size;
    return fail ? 0 : source_->read(offset, out, size);
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &out,
              size_t size) override {
    max_read = std::max(max_read, size);
    *bytes_read += size;
    return fail ? 0 : source_->read(offset, out, size);
  }
  size_t write(size_t offset, const void *data, size_t size) override {
    return source_->write(offset, data, size);
  }
  size_t resize(size_t size) override {
    return source_->resize(size);
  }
  void update_data_crc(uint32_t crc) override {
    source_->update_data_crc(crc);
  }
  Pointer clone() override {
    return std::make_shared<WatchedSegment>(source_->clone(), bytes_read);
  }
  bool fail{false};
  mutable size_t max_read{0};
  std::shared_ptr<size_t> bytes_read;

 private:
  Pointer source_;
};

class WatchedStorage : public IndexStorage {
 public:
  explicit WatchedStorage(Pointer source) : source_(std::move(source)) {}
  int init(const ailego::Params &p) override {
    return source_->init(p);
  }
  int cleanup() override {
    return source_->cleanup();
  }
  int open(const std::string &p, bool create) override {
    return source_->open(p, create);
  }
  int flush() override {
    return source_->flush();
  }
  int close() override {
    return source_->close();
  }
  int append(const std::string &id, size_t size) override {
    return source_->append(id, size);
  }
  void refresh(uint64_t point) override {
    source_->refresh(point);
  }
  uint64_t check_point() const override {
    return source_->check_point();
  }
  Segment::Pointer get(const std::string &id, int level = -1) override {
    auto segment = source_->get(id, level);
    if (!segment) return nullptr;
    auto &watch = watched[id];
    if (!watch) watch = std::make_shared<WatchedSegment>(std::move(segment));
    return watch;
  }
  bool has(const std::string &id) const override {
    return source_->has(id);
  }
  uint32_t magic() const override {
    return source_->magic();
  }
  MemoryBlock::MemoryBlockType memory_block_type() const override {
    return source_->memory_block_type();
  }
  std::shared_ptr<ailego::VecBufferPool> vec_buffer_pool() const override {
    return source_->vec_buffer_pool();
  }
  std::map<std::string, std::shared_ptr<WatchedSegment>> watched;

 private:
  Pointer source_;
};

TEST(FlatPagedMetadataTest, LazyKeysMappingQueriesAndReadErrors) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  const std::string path = "flat_paged_metadata_test.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
    std::filesystem::remove(path);
  });
  constexpr size_t count = 1027, dimension = 8;
  for (const auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    ASSERT_EQ(0, pool.init(0));
    IndexMeta meta(IndexMeta::DT_FP32, dimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    meta.set_major_order(order);
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dimension);
    // Shuffled keys include one duplicate; the last row must win on fetch.
    std::vector<uint64_t> keys(count);
    for (size_t i = 0; i < count; ++i) {
      keys[i] = (i * 17 % count) * 3 + 7;
      if (i == count - 1) keys[i] = keys[0];
      ailego::NumericalVector<float> row(dimension);
      for (size_t d = 0; d < dimension; ++d) row[d] = static_cast<float>(i);
      ASSERT_TRUE(holder->emplace(keys[i], std::move(row)));
    }
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, builder->init(meta, {}));
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
    ASSERT_EQ(0, dumper->close());
    builder.reset();
    holder.reset();
    for (size_t mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(mode);
      SCOPED_TRACE(static_cast<int>(order));
      ASSERT_EQ(0, pool.init(mode == 2 ? 512 : 2 * 1024 * 1024));
      auto source = IndexFactory::CreateStorage(mode ? "BufferReadStorage"
                                                     : "MMapFileReadStorage");
      ailego::Params params;
      params.set(BUFFER_READ_STORAGE_WARMUP_MODE,
                 BUFFER_READ_STORAGE_WARMUP_NONE);
      ASSERT_EQ(0, source->init(params));
      ASSERT_EQ(0, source->open(path, false));
      auto storage = std::make_shared<WatchedStorage>(source);
      auto searcher_owner = IndexFactory::CreateSearcher("FlatSearcher");
      ASSERT_NE(nullptr, searcher_owner);
      auto &searcher = *searcher_owner;
      auto *lookup = dynamic_cast<FlatSearcher<32> *>(searcher_owner.get());
      ASSERT_NE(nullptr, lookup);
      ASSERT_EQ(0, searcher.init({}));
      ASSERT_EQ(0, searcher.load(storage, nullptr));
      auto key_segment = storage->watched.at(FLAT_SEGMENT_KEYS_SEG_ID);
      if (mode) {
        EXPECT_EQ(0u, key_segment->max_read);
        EXPECT_EQ(0u,
                  storage->watched.at(FLAT_SEGMENT_MAPPING_SEG_ID)->max_read);
      }
      auto provider = searcher.create_provider();
      ASSERT_NE(nullptr, provider);
      for (size_t i : {0u, 17u, 900u}) {
        EXPECT_EQ(i == 0 ? count - 1 : i, lookup->get_id(keys[i]));
        const auto *row =
            static_cast<const float *>(provider->get_vector(keys[i]));
        ASSERT_NE(nullptr, row);
        EXPECT_FLOAT_EQ(static_cast<float>(i == 0 ? count - 1 : i), row[0]);
      }
      EXPECT_EQ(nullptr, provider->get_vector(1));
      auto iter = provider->create_iterator();
      size_t visited = 0;
      for (; iter->is_valid(); iter->next(), ++visited) {
        ASSERT_LT(visited, count);
        EXPECT_EQ(keys[visited], iter->key());
      }
      EXPECT_EQ(count, visited);
      EXPECT_EQ(0, iter->status());
      auto context = searcher.create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(1);
      const float query[dimension] = {17, 17, 17, 17, 17, 17, 17, 17};
      IndexQueryMeta qmeta(IndexMeta::DT_FP32, dimension);
      ASSERT_EQ(0, searcher.search_impl(query, qmeta, context));
      ASSERT_EQ(1u, context->result().size());
      EXPECT_EQ(keys[17], context->result()[0].key());
      if (mode != 2) {
        // More than one transpose tile, including a partial SIMD tail. Every
        // feature byte must still be read only once, not once per query tile.
        constexpr size_t batch_count = 2053;
        std::vector<float> queries(batch_count * dimension);
        for (size_t q = 0; q < batch_count; ++q)
          std::fill_n(queries.data() + q * dimension, dimension,
                      float(q % count));
        auto feature_segment =
            storage->watched.at(FLAT_SEGMENT_FEATURES_SEG_ID);
        auto single = searcher.create_context();
        ASSERT_NE(nullptr, single);
        single->set_topk(1);
        for (bool filtered : {false, true}) {
          if (filtered) {
            context->set_filter([](uint64_t key) { return key % 2 == 0; });
            single->set_filter([](uint64_t key) { return key % 2 == 0; });
          }
          *feature_segment->bytes_read = 0;
          ASSERT_EQ(0, searcher.search_impl(queries.data(), qmeta, batch_count,
                                            context));
          EXPECT_EQ(feature_segment->data_size(), *feature_segment->bytes_read);
          for (size_t q = 0; q < batch_count; ++q) {
            const auto &result = context->result(q);
            ASSERT_EQ(1u, result.size());
            if (!filtered) {
              EXPECT_EQ(keys[q % count], result[0].key());
            } else if (q % 127 == 0) {
              ASSERT_EQ(0, searcher.search_impl(queries.data() + q * dimension,
                                                qmeta, single));
              ASSERT_EQ(1u, single->result().size());
              EXPECT_EQ(single->result()[0].key(), result[0].key());
              EXPECT_FLOAT_EQ(single->result()[0].score(), result[0].score());
            }
          }
        }
        context->reset_filter();
        single.reset();
      }
      // Reusing a batch context for a single query must not expose old heaps.
      ASSERT_EQ(0, searcher.search_impl(query, qmeta, context));
      ASSERT_EQ(1u, context->result().size());
      EXPECT_EQ(keys[17], context->result()[0].key());

      if (mode) {
        EXPECT_LE(key_segment->max_read, sizeof(uint64_t));
        EXPECT_LE(storage->watched.at(FLAT_SEGMENT_MAPPING_SEG_ID)->max_read,
                  sizeof(uint32_t));
        key_segment->fail = true;
        EXPECT_EQ(IndexError_ReadData,
                  searcher.search_impl(query, qmeta, context));
        auto bad = provider->create_iterator();
        bad->key();
        EXPECT_EQ(IndexError_ReadData, bad->status());
        EXPECT_EQ(nullptr, provider->get_vector(keys[17]));
        key_segment->fail = false;
        auto mapping = storage->watched.at(FLAT_SEGMENT_MAPPING_SEG_ID);
        mapping->fail = true;
        EXPECT_EQ(IndexError_ReadData,
                  searcher.search_bf_by_p_keys_impl(query, {{keys[17]}}, qmeta,
                                                    1, context));
        mapping->fail = false;
      }
      EXPECT_LE(pool.used(), pool.capacity());
      iter.reset();
      provider.reset();
      context.reset();
      ASSERT_EQ(0, searcher.unload());
      EXPECT_EQ(nullptr, searcher.create_provider());
      // Reload must not reuse a stale mapping or hash.
      ASSERT_EQ(0, searcher.load(storage, nullptr));
      EXPECT_EQ(17u, lookup->get_id(keys[17]));
    }
  }
}
}  // namespace
}  // namespace zvec::core

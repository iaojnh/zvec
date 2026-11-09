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

#include <cstring>
#include <map>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "ivf_dumper.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include <utility>
#include <vector>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_memory.h>
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/cluster/stratified_cluster_trainer.h"
#include "ivf_builder.h"
#include "ivf_searcher.h"
#include <filesystem>
#include <limits>
#include <string>
#include "algorithm/ivf/ivf_builder.h"

namespace zvec::core {
namespace {

class RecordingDumper : public IndexDumper {
 public:
  struct Segment {
    std::string bytes;
    size_t size;
    size_t padding;
    uint32_t crc;
    size_t max_write;
  };
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int create(const std::string &) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  int append(const std::string &id, size_t size, size_t padding,
             uint32_t crc) override {
    if (id == fail_append) return IndexError_WriteData;
    EXPECT_EQ(size + padding, pending.size());
    if (crc) EXPECT_EQ(crc, ailego::Crc32c::Hash(pending.data(), size));
    segments.emplace(
        id, Segment{std::move(pending), size, padding, crc, max_write});
    pending.clear();
    max_write = 0;
    return 0;
  }
  size_t write(const void *data, size_t size) override {
    if (segments.size() == fail_write_after_segments) return 0;
    if (size) pending.append(static_cast<const char *>(data), size);
    max_write = std::max(max_write, size);
    return size;
  }
  std::map<std::string, Segment> segments;
  std::string pending;
  std::string fail_append;
  size_t max_write{0};
  size_t fail_write_after_segments{std::numeric_limits<size_t>::max()};
};

class IVFDumperStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    previous_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    budget_ =
        2 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
        64 * 1024;
    ASSERT_EQ(0, pool.init(0));
  }
  void TearDown() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_));
  }
  static uint64_t key(size_t id, size_t count, bool duplicates) {
    return duplicates && id % 4 == 0 ? 42
                                     : (uint64_t{1} << 40) + (count - id) * 19;
  }
  int dump(const std::shared_ptr<RecordingDumper> &output, size_t count,
           bool column, bool blocks, bool duplicates = false) {
    IndexMeta meta(IndexMeta::DT_FP32, 8);
    meta.set_major_order(column ? IndexMeta::MO_COLUMN : IndexMeta::MO_ROW);
    IVFDumper dumper(meta, output, 3, 32);
    int ret =
        dumper.reserve_keys(count + 7);  // Filtering may leave unused capacity.
    if (ret != 0) return ret;
    const size_t split = count > 7 ? count - 7 : count;
    size_t id = 0;
    for (uint32_t list : {0u, 2u}) {
      const size_t end = list == 0 ? split : count;
      while (id < end) {
        if (blocks && end - id >= 32) {
          float rows[32 * 8];
          uint64_t keys[32];
          for (size_t i = 0; i < 32; ++i) {
            keys[i] = key(id + i, count, duplicates);
            for (size_t d = 0; d < 8; ++d)
              rows[column ? d * 32 + i : i * 8 + d] = float(id + i + d);
          }
          ret = dumper.dump_inverted_block(list, keys, rows, 32, column);
          id += 32;
        } else {
          float row[8];
          for (size_t d = 0; d < 8; ++d) row[d] = float(id + d);
          ret = dumper.dump_inverted_vector(list, key(id, count, duplicates),
                                            row);
          ++id;
        }
        if (ret != 0) return ret;
      }
    }
    return dumper.dump_inverted_vector_finished();
  }
  size_t previous_{0};
  size_t budget_{0};
};

TEST_F(IVFDumperStorageTest, PagedKeysAndStreamingOffsetsMatchLegacySegments) {
  for (bool column : {false, true}) {
    for (bool blocks : {false, true}) {
      for (size_t count : {size_t{0}, size_t{35}, size_t{16389}}) {
        SCOPED_TRACE(::testing::Message()
                     << column << ':' << blocks << ':' << count);
        auto &pool = ailego::PoolPayloadBudget::get_instance();
        ASSERT_EQ(0, pool.init(0));
        auto reference = std::make_shared<RecordingDumper>();
        ASSERT_EQ(0, dump(reference, count, column, blocks));
        ASSERT_EQ(0, pool.init(budget_));
        auto actual = std::make_shared<RecordingDumper>();
        ASSERT_EQ(0, dump(actual, count, column, blocks));
        ASSERT_EQ(reference->segments.size(), actual->segments.size());
        for (const auto &entry : reference->segments) {
          SCOPED_TRACE(entry.first);
          ASSERT_EQ(1u, actual->segments.count(entry.first));
          const auto &value = actual->segments.at(entry.first);
          EXPECT_EQ(entry.second.bytes, value.bytes);
          EXPECT_EQ(entry.second.crc, value.crc);
          EXPECT_EQ(entry.second.size, value.size);
          EXPECT_EQ(entry.second.padding, value.padding);
        }
        EXPECT_LE(actual->segments.at(IVF_OFFSETS_SEG_ID).max_write,
                  32u * 1024);
        EXPECT_LE(actual->segments.at(IVF_KEYS_SEG_ID).max_write, 32u * 1024);
        EXPECT_LE(actual->segments.at(IVF_MAPPING_SEG_ID).max_write,
                  16u * 1024);
        // Decode offsets independently, including column-major blocks and
        // row-major tails on both sides of the empty middle list.
        const auto &offsets = actual->segments.at(IVF_OFFSETS_SEG_ID);
        const auto &body = actual->segments.at(IVF_INVERTED_BODY_SEG_ID);
        for (size_t i = 0; i < count; ++i) {
          InvertedVecLocation location(0, false);
          std::memcpy(&location, offsets.bytes.data() + i * sizeof(location),
                      sizeof(location));
          for (size_t d = 0; d < 8; ++d) {
            const size_t offset =
                location.offset +
                d * sizeof(float) * (location.column_major ? 32 : 1);
            ASSERT_LE(offset + sizeof(float), body.size);
            float value;
            std::memcpy(&value, body.bytes.data() + offset, sizeof(value));
            ASSERT_FLOAT_EQ(value, float(i + d));
          }
        }
        EXPECT_EQ(0u, pool.used());
      }
    }
  }
}

TEST_F(IVFDumperStorageTest, DuplicateKeysKeepEveryOriginalOrdinal) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(budget_));
  constexpr size_t count = 8197;
  auto output = std::make_shared<RecordingDumper>();
  ASSERT_EQ(0, dump(output, count, true, true, true));
  const auto &keys = output->segments.at(IVF_KEYS_SEG_ID);
  const auto &mapping = output->segments.at(IVF_MAPPING_SEG_ID);
  std::vector<bool> seen(count, false);
  uint64_t previous = 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t id;
    std::memcpy(&id, mapping.bytes.data() + i * sizeof(id), sizeof(id));
    ASSERT_LT(id, count);
    EXPECT_FALSE(seen[id]);
    seen[id] = true;
    uint64_t value;
    std::memcpy(&value, keys.bytes.data() + id * sizeof(value), sizeof(value));
    EXPECT_EQ(key(id, count, true), value);
    EXPECT_GE(value, previous);
    previous = value;
  }
  EXPECT_EQ(0u, pool.used());
}

TEST_F(IVFDumperStorageTest, BudgetAndOutputFailuresDoNotLeakOrFallback) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(1));
  auto output = std::make_shared<RecordingDumper>();
  // Metadata and the first block now require admission in the constructor.
  EXPECT_THROW(dump(output, 8197, false, false), std::bad_alloc);
  EXPECT_TRUE(output->segments.empty());
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(budget_));
  for (size_t fail_after : {3u, 4u, 5u}) {
    output = std::make_shared<RecordingDumper>();
    output->fail_write_after_segments = fail_after;
    EXPECT_EQ(IndexError_WriteData, dump(output, 8197, false, true));
    EXPECT_EQ(0u, pool.used());
  }
  for (const auto &segment :
       {IVF_KEYS_SEG_ID, IVF_MAPPING_SEG_ID, IVF_OFFSETS_SEG_ID}) {
    output = std::make_shared<RecordingDumper>();
    output->fail_append = segment;
    EXPECT_EQ(IndexError_WriteData, dump(output, 8197, false, true));
    EXPECT_EQ(0u, pool.used());
  }
}

}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {

class CountingCentroidTrainer : public StratifiedClusterTrainer {
 public:
  IndexBundle::Pointer indexes() const override {
    ++serialize_calls;
    return StratifiedClusterTrainer::indexes();
  }
  mutable size_t serialize_calls{0};
};

// Deliberately exposes only the public trainer interface, like an existing
// third-party trainer. Its output must remain usable without CentroidTrainer.
class LegacyTrainer : public IndexTrainer {
 public:
  explicit LegacyTrainer(IndexTrainer::Pointer trainer)
      : trainer_(std::move(trainer)) {}
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    return trainer_->init(meta, params);
  }
  int cleanup() override {
    return trainer_->cleanup();
  }
  int train(IndexThreads::Pointer threads,
            IndexHolder::Pointer holder) override {
    return trainer_->train(std::move(threads), std::move(holder));
  }
  int load(IndexStorage::Pointer storage) override {
    return trainer_->load(std::move(storage));
  }
  int dump(const IndexDumper::Pointer &dumper) override {
    return trainer_->dump(dumper);
  }
  const IndexMeta &meta() const override {
    return trainer_->meta();
  }
  const Stats &stats() const override {
    return trainer_->stats();
  }
  IndexBundle::Pointer indexes() const override {
    return trainer_->indexes();
  }

 private:
  IndexTrainer::Pointer trainer_;
};

class InvalidCentroidTrainer : public LegacyTrainer, public CentroidTrainer {
 public:
  using LegacyTrainer::LegacyTrainer;
  const IndexCluster::CentroidList &centroids() const override {
    return tree;
  }
  IndexCluster::CentroidList tree;
};

class IVFCentroidTrainerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(0));
  }
  void TearDown() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(capacity_));
  }

  using Results = std::vector<std::pair<uint64_t, float>>;

  void finish_and_query(IVFBuilder &builder, const IndexHolder::Pointer &holder,
                        const std::string &path, Results *results) {
    AILEGO_DEFER([&] { IndexMemory::Instance()->remove(path); });
    ASSERT_EQ(0, builder.build(threads_, holder));
    auto dumper = IndexFactory::CreateDumper("MemoryDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());
    EXPECT_EQ(holder->count(), builder.stats().dumped_count());
    ASSERT_EQ(0, builder.cleanup());

    auto storage = IndexFactory::CreateStorage("MemoryReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    IVFSearcher searcher;
    ailego::Params params;
    params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
    params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
    ASSERT_EQ(0, searcher.init(params));
    ASSERT_EQ(0, searcher.load(storage, nullptr));
    auto context = searcher.create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(10);
    IndexQueryMeta meta(holder->data_type(), holder->dimension());
    auto it = holder->create_iterator();
    for (size_t i = 0; i < 8; ++i, it->next()) {
      ASSERT_TRUE(it->is_valid());
      ASSERT_EQ(0, searcher.search_impl(it->data(), meta, context));
      const auto &docs = context->result(0);
      ASSERT_EQ(10u, docs.size());
      EXPECT_EQ(it->key(), docs.front().key());
      EXPECT_FLOAT_EQ(0.0f, docs.front().score());
      for (const auto &doc : docs) {
        results->emplace_back(doc.key(), doc.score());
      }
    }
  }

  template <IndexMeta::DataType Type, typename Scalar>
  void check_handoff(bool two_levels, bool quantize = false) {
    constexpr size_t dimension = 8;
    IndexMeta meta(Type, dimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    auto holder = std::make_shared<MultiPassIndexHolder<Type>>(dimension);
    for (size_t i = 0; i < 128; ++i) {
      ailego::NumericalVector<Scalar> row(dimension);
      for (size_t d = 0; d < dimension; ++d) {
        row[d] = static_cast<Scalar>(static_cast<float>((i + d * 7) % 128));
      }
      holder->emplace(1000 + i, std::move(row));
    }
    auto counting = std::make_shared<CountingCentroidTrainer>();
    IndexTrainer::Pointer trainer = counting;
    std::weak_ptr<IndexTrainer> lifetime = trainer;
    ailego::Params params;
    const char *count = two_levels ? "2*2" : "4";
    params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, count);
    ailego::Params level;
    level.set(OPTKMEANS_CLUSTER_MAX_ITERATIONS, 2);
    params.set(STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + "1", level);
    params.set(STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + "2", level);
    ASSERT_EQ(0, trainer->init(meta, params));
    ASSERT_EQ(0, trainer->train(threads_, holder));
    ASSERT_FALSE(counting->centroids().empty());
    if (two_levels) {
      ASSERT_FALSE(counting->centroids().front().subitems().empty());
    }
    const void *original_data = counting->centroids().front().feature();
    IndexBundle::Pointer before;
    ASSERT_EQ(0, IndexCluster::Serialize(meta, counting->centroids(), &before));

    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, count);
    if (quantize) {
      params.set(PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_CLASS,
                 "HalfFloatConverter");
    }
    IVFBuilder borrowed, legacy;
    ASSERT_EQ(0, borrowed.init(meta, params));
    ASSERT_EQ(0, borrowed.train(trainer));
    EXPECT_EQ(0u, counting->serialize_calls);
    EXPECT_EQ(original_data, counting->centroids().front().feature());
    ASSERT_EQ(0, legacy.init(meta, params));
    ASSERT_EQ(0, legacy.train(std::make_shared<LegacyTrainer>(trainer)));
    EXPECT_EQ(1u, counting->serialize_calls);
    EXPECT_EQ(legacy.centroid_index()->centroids_count(),
              borrowed.centroid_index()->centroids_count());

    IndexBundle::Pointer after;
    ASSERT_EQ(0, IndexCluster::Serialize(meta, counting->centroids(), &after));
    for (const char *name : {"IndexCluster.Centroids.Features",
                             "IndexCluster.Centroids.Indexes"}) {
      const auto old_data = before->get(name), new_data = after->get(name);
      ASSERT_EQ(old_data.size(), new_data.size());
      EXPECT_EQ(0, std::memcmp(old_data.buffer(), new_data.buffer(),
                               old_data.size()));
    }
    before.reset();
    after.reset();
    ASSERT_EQ(0, trainer->cleanup());
    EXPECT_TRUE(counting->centroids().empty());
    counting.reset();
    trainer.reset();
    ASSERT_TRUE(lifetime.expired());
    // Both coarse indexes must own their output, not the trainer or its tree.
    Results actual, expected;
    ASSERT_NO_FATAL_FAILURE(finish_and_query(
        borrowed, holder, "centroid_handoff_borrowed", &actual));
    ASSERT_NO_FATAL_FAILURE(
        finish_and_query(legacy, holder, "centroid_handoff_legacy", &expected));
    EXPECT_EQ(expected, actual);
  }

  size_t capacity_{0};
  IndexThreads::Pointer threads_ =
      std::make_shared<SingleQueueIndexThreads>(1, false);
};

TEST_F(IVFCentroidTrainerTest, BorrowedAndLegacyTreesMatchAfterTrainerRelease) {
  for (bool two_levels : {false, true}) {
    SCOPED_TRACE(two_levels);
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_FP32, float>(two_levels)));
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_FP16, ailego::Float16>(two_levels)));
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_INT8, int8_t>(two_levels)));
  }
}

TEST_F(IVFCentroidTrainerTest, QuantizedCoarseIndexAlsoOutlivesTrainer) {
  for (bool two_levels : {false, true}) {
    SCOPED_TRACE(two_levels);
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_FP32, float>(two_levels, true)));
  }
}

TEST_F(IVFCentroidTrainerTest, BorrowedHandoffWithSharedPool) {
  // Allow the source, partition and training scratch stores' fixed metadata,
  // plus a small page budget; do not assume platform-independent struct sizes.
  const size_t budget =
      3 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
      256 * 1024;
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(budget));
  ASSERT_NO_FATAL_FAILURE(
      (check_handoff<IndexMeta::DT_FP32, float>(true, true)));
}

TEST_F(IVFCentroidTrainerTest, FactoryTrainerExposesOptionalCapability) {
  auto trainer = IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ASSERT_NE(nullptr, trainer);
  auto *source = dynamic_cast<CentroidTrainer *>(trainer.get());
  ASSERT_NE(nullptr, source);
  EXPECT_TRUE(source->centroids().empty());
}

TEST_F(IVFCentroidTrainerTest,
       InvalidTreesFailWithoutSerializationOrPublication) {
  auto counting = std::make_shared<CountingCentroidTrainer>();
  IndexTrainer::Pointer trainer = counting;
  IndexMeta meta(IndexMeta::DT_FP32, 8);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, "1");
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  ASSERT_EQ(0, trainer->init(meta, params));
  auto invalid = std::make_shared<InvalidCentroidTrainer>(trainer);
  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params));
  EXPECT_EQ(IndexError_InvalidFormat, builder.train(invalid));
  EXPECT_EQ(nullptr, builder.centroid_index());
  float row[8]{};
  invalid->tree.emplace_back(row, sizeof(float));
  EXPECT_EQ(IndexError_Mismatch, builder.train(invalid));
  invalid->tree.front().set_feature(row, sizeof(row));
  invalid->tree.front().mutable_subitems()->emplace_back(row, sizeof(float));
  EXPECT_EQ(IndexError_Mismatch, builder.train(invalid));
  EXPECT_EQ(nullptr, builder.centroid_index());
  EXPECT_EQ(0u, counting->serialize_calls);
  invalid->tree.front().mutable_subitems()->front().set_feature(row,
                                                                sizeof(row));
  EXPECT_EQ(0, builder.train(invalid));
}

TEST_F(IVFCentroidTrainerTest, LegacyEmptyBundleStillFails) {
  IndexTrainer::Pointer trainer = std::make_shared<CountingCentroidTrainer>();
  IndexMeta meta(IndexMeta::DT_FP32, 8);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, "1");
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  ASSERT_EQ(0, trainer->init(meta, params));
  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params));
  EXPECT_NE(0, builder.train(std::make_shared<LegacyTrainer>(trainer)));
  EXPECT_EQ(nullptr, builder.centroid_index());
}

}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {

class IVFBufferedHolderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    const size_t metadata =
        ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
    ASSERT_EQ(pool.init(metadata + 128 * 1024), 0);
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }
  void TearDown() override {
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
    EXPECT_TRUE(std::filesystem::remove(directory_));
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(capacity_), 0);
  }
  const std::filesystem::path directory_{"ivf_buffered_holder_data"};
  size_t capacity_{0};
};

TEST_F(IVFBufferedHolderTest, IndependentIteratorsRemainValidUnderPressure) {
  for (auto type : {IndexMeta::DT_FP32, IndexMeta::DT_FP16}) {
    IndexMeta meta;
    meta.set_meta(type, 512);
    IVFBuilder::RandomAccessIndexHolder holder(meta);
    constexpr size_t kRows = 1024;
    ASSERT_EQ(holder.enable_buffered_storage((directory_ / "vectors").string(),
                                             kRows),
              0);
    holder.reserve(kRows);
    ailego::BudgetedString bytes(meta.element_size(), 0);
    for (size_t id = 0; id < kRows; ++id) {
      std::fill(bytes.begin(), bytes.end(), static_cast<char>(id % 113));
      ASSERT_EQ(holder.emplace(id + 100, bytes.data()), 0);
    }
    ASSERT_EQ(holder.flush(), 0);
    auto first = holder.create_iterator();
    auto second = holder.create_iterator();
    const void *saved = first->data();
    ASSERT_NE(saved, nullptr);
    const std::string first_value(static_cast<const char *>(saved),
                                  bytes.size());
    for (size_t id = 0; id < kRows; ++id) {
      ASSERT_TRUE(second->is_valid());
      ASSERT_EQ(second->key(), id + 100);
      const auto *actual = static_cast<const char *>(second->data());
      ASSERT_NE(actual, nullptr);
      EXPECT_EQ(second->status(), 0);
      EXPECT_EQ(std::string(actual, bytes.size()),
                std::string(bytes.size(), static_cast<char>(id % 113)));
      second->next();
    }
    EXPECT_FALSE(second->is_valid());
    EXPECT_EQ(second->status(), 0);
    EXPECT_EQ(std::string(static_cast<const char *>(saved), bytes.size()),
              first_value);
    EXPECT_LE(ailego::PoolPayloadBudget::get_instance().used(),
              ailego::PoolPayloadBudget::get_instance().capacity());
    const void *invalid = nullptr;
    EXPECT_EQ(holder.read_element(kRows, &bytes, &invalid),
              IndexError_OutOfRange);
  }
}

TEST_F(IVFBufferedHolderTest, RejectsOverflowWithoutCreatingScratch) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, 512);
  IVFBuilder::RandomAccessIndexHolder holder(meta);
  EXPECT_EQ(holder.enable_buffered_storage((directory_ / "overflow").string(),
                                           std::numeric_limits<size_t>::max()),
            IndexError_InvalidArgument);
  EXPECT_TRUE(std::filesystem::is_empty(directory_));
}

}  // namespace
}  // namespace zvec::core

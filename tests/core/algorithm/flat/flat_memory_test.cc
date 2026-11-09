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

#include <array>
#include <filesystem>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "flat/flat_builder.h"
#include "flat/flat_streamer.h"
#include "tests/core/algorithm/graph_dump_test_helper.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include <cstdlib>
#include <new>
#include <utility>
#include "flat/flat_utility.h"
#include <algorithm>
#include <future>
#include <numeric>
#include <zvec/core/framework/index_factory.h>
#include "flat/flat_searcher.h"
#include "utility/utility_params.h"

namespace zvec::core {
namespace {
class FlatCapture : public testing::GraphDumpCapture {
 public:
  int append(const std::string &id, size_t size, size_t pad,
             uint32_t crc) override {
    EXPECT_EQ(size + pad, pending.size());
    EXPECT_EQ(std::string(pad, '\0'), pending.substr(size));
    if (crc) EXPECT_EQ(ailego::Crc32c::Hash(pending.data(), size), crc);
    segments[id] = pending.substr(0, size);
    pending.clear();
    return fail_append ? IndexError_WriteData : 0;
  }
  bool fail_append{false};
};

class OnePassRows : public IndexHolder {
 public:
  explicit OnePassRows(size_t n) : n_(n) {}
  size_t count() const override {
    return SIZE_MAX;
  }
  size_t dimension() const override {
    return 8;
  }
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DT_FP32;
  }
  size_t element_size() const override {
    return 8 * sizeof(float);
  }
  bool multipass() const override {
    return false;
  }
  struct Cursor : Iterator {
    explicit Cursor(OnePassRows &owner) : owner_(owner) {}
    const void *data() const override {
      row_.fill(static_cast<float>(position_));
      return owner_.fail_data && position_ == 7 ? nullptr : row_.data();
    }
    bool is_valid() const override {
      return position_ < owner_.n_;
    }
    uint64_t key() const override {
      return (position_ * 17 % 1031) + 1;
    }
    void next() override {
      ++position_;
    }
    int status() const override {
      return owner_.fail_end && position_ == owner_.n_ ? IndexError_ReadData
                                                       : 0;
    }
    OnePassRows &owner_;
    size_t position_{0};
    mutable std::array<float, 8> row_{};
  };
  Iterator::Pointer create_iterator() override {
    ++passes;
    return passes == 1 ? std::make_unique<Cursor>(*this) : nullptr;
  }
  bool fail_data{false}, fail_end{false};
  size_t passes{0};

 private:
  size_t n_;
};

class FlatMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    old_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(0));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_));
  }
  static IndexMeta meta() {
    IndexMeta result(IndexMeta::DT_FP32, 8);
    result.set_metric("SquaredEuclidean", 0, ailego::Params{});
    return result;
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t old_{0};
};

TEST_F(FlatMemoryTest, DumpUnknownCountRowAndColumnMatchWithoutCorpusCopies) {
  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    std::shared_ptr<FlatCapture> reference;
    for (bool buffered : {false, true}) {
      ASSERT_EQ(0, pool_.init(buffered ? 4 * 1024 * 1024 : 0));
      auto metadata = meta();
      metadata.set_major_order(order);
      auto input = std::make_shared<OnePassRows>(70003);
      auto builder = IndexFactory::CreateBuilder("FlatBuilder");
      auto output = std::make_shared<FlatCapture>();
      ASSERT_EQ(0, builder->init(metadata, {}));
      ASSERT_EQ(0, builder->build(nullptr, input));
      ASSERT_EQ(0, builder->dump(output));
      EXPECT_EQ(1u, input->passes);
      EXPECT_EQ(70003u, builder->stats().dumped_count());
      EXPECT_LE(output->max_write, 8192u);
      auto keys = output->records<uint64_t>(FLAT_SEGMENT_KEYS_SEG_ID);
      auto mapping = output->records<uint32_t>(FLAT_SEGMENT_MAPPING_SEG_ID);
      ASSERT_EQ(keys.size(), mapping.size());
      for (size_t i = 1; i < mapping.size(); ++i) {
        EXPECT_LE(keys[mapping[i - 1]], keys[mapping[i]]);
        if (keys[mapping[i - 1]] == keys[mapping[i]])
          EXPECT_LT(mapping[i - 1], mapping[i]);
      }
      if (reference)
        EXPECT_EQ(reference->segments, output->segments);
      else
        reference = output;
    }
  }
}

TEST_F(FlatMemoryTest, EmptyAndSingleRowDumpKeepLegacySegments) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  for (size_t count : {0u, 1u}) {
    auto input = std::make_shared<OnePassRows>(count);
    auto output = std::make_shared<FlatCapture>();
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    ASSERT_EQ(0, builder->init(meta(), {}));
    ASSERT_EQ(0, builder->build(nullptr, input));
    ASSERT_EQ(0, builder->dump(output));
    EXPECT_EQ(count * sizeof(uint64_t),
              output->segments.at(FLAT_SEGMENT_KEYS_SEG_ID).size());
    EXPECT_EQ(count * sizeof(uint32_t),
              output->segments.at(FLAT_SEGMENT_MAPPING_SEG_ID).size());
    EXPECT_EQ(1u, input->passes);
  }
}

TEST_F(FlatMemoryTest, DumpReportsInputAndOutputFailures) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  for (int failure = 0; failure != 4; ++failure) {
    auto input = std::make_shared<OnePassRows>(40);
    input->fail_data = failure == 0;
    input->fail_end = failure == 1;
    auto output = std::make_shared<FlatCapture>();
    output->fail_write = failure == 2;
    output->fail_append = failure == 3;
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    ASSERT_EQ(0, builder->init(meta(), {}));
    ASSERT_EQ(0, builder->build(nullptr, input));
    EXPECT_EQ(failure < 2 ? IndexError_ReadData : IndexError_WriteData,
              builder->dump(output));
  }
}

TEST_F(FlatMemoryTest, ContinuousIdsHolesReopenAndChangedSegmentSize) {
  for (const auto *mode : {"MMapFileStorage", "BufferStorage"}) {
    ASSERT_EQ(0, pool_.init(16 * 1024 * 1024));
    const std::string path = std::string("flat_direct_ids_") + mode;
    ASSERT_FALSE(std::filesystem::exists(path));
    AILEGO_DEFER([&] { std::filesystem::remove(path); });
    for (uint32_t round = 0; round < 2; ++round) {
      auto storage = IndexFactory::CreateStorage(mode);
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(path, true));
      IndexStreamer::Stats stats;
      FlatStreamerEntity entity(stats);
      *entity.mutable_meta() = meta();
      entity.set_use_key_info_map(false);
      entity.set_block_vector_count(32);
      entity.set_linear_list_count(1);
      entity.set_segment_size(round ? 65536 : 4096);
      ASSERT_EQ(0, entity.open(storage, meta()));
      std::array<float, 8> row{};
      const uint32_t begin = round ? 231 : 3;
      const uint32_t end = round ? 631 : 231;
      for (uint32_t id = begin; id < end; ++id) {
        row.fill(static_cast<float>(id));
        ASSERT_EQ(0, entity.add_vector_with_id(id, row.data(), sizeof(row)));
      }
      // A hole is a persisted invalid key, not a synthesized key equal to ID.
      if (!round) EXPECT_EQ(kInvalidKey, entity.key(1));
      row.fill(1.0f);
      ASSERT_EQ(0, entity.add_vector_with_id(1, row.data(), sizeof(row)));
      for (uint32_t id : {1u, 3u, 123u, end - 1}) {
        EXPECT_EQ(id, entity.key(id));
        IndexStorage::MemoryBlock block;
        ASSERT_EQ(0, entity.get_vector_by_key(id, block));
        EXPECT_FLOAT_EQ(static_cast<float>(id),
                        static_cast<const float *>(block.data())[0]);
      }
      EXPECT_EQ(nullptr, entity.get_vector_by_key(uint64_t{1} << 32));
      const uint64_t ids[]{1, 3, 123, end - 1};
      std::vector<IndexStorage::MemoryBlock> external;
      EXPECT_EQ(IndexError_NoMemory,
                entity.get_vectors_by_key(ids, 4, external));
      EXPECT_EQ(0u, external.capacity());
      external.reserve(4);
      ASSERT_EQ(0, entity.get_vectors_by_key(ids, 4, external));
      external.clear();
      IndexProvider::VectorBlocks blocks;
      ASSERT_EQ(0, entity.get_vectors_by_key(ids, 4, blocks));
      blocks.clear();
      auto iterator = entity.creater_iterator();
      ASSERT_NE(nullptr, iterator);
      // Appending must not extend an already-created iterator.
      row.fill(static_cast<float>(end));
      ASSERT_EQ(0, entity.add_vector_with_id(end, row.data(), sizeof(row)));
      ASSERT_EQ(0, entity.flush(0));
      ASSERT_EQ(0, entity.close());
      size_t visited = 0;
      for (; iterator->is_valid(); iterator->next(), ++visited) {
        EXPECT_FLOAT_EQ(
            visited == 0 || visited == 2 ? 0.0f : static_cast<float>(visited),
            static_cast<const float *>(iterator->data())[0]);
      }
      EXPECT_EQ(end, visited);
      EXPECT_EQ(0, iterator->status());
      iterator.reset();
      ASSERT_EQ(0, storage->close());
    }
    std::filesystem::remove(path);
  }
}

TEST_F(FlatMemoryTest, ArbitraryKeysKeepLookupAndIteratorCompatible) {
  const std::string path = "flat_arbitrary_keys.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  for (int round = 0; round < 2; ++round) {
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, round == 0));
    IndexStreamer::Stats stats;
    FlatStreamerEntity entity(stats);
    *entity.mutable_meta() = meta();
    entity.set_block_vector_count(32);
    entity.set_linear_list_count(1);
    entity.set_segment_size(4096);
    ASSERT_EQ(0, entity.open(storage, meta()));
    std::array<float, 8> row{};
    if (!round)
      for (uint64_t id = 0; id < 199; ++id) {
        row.fill(static_cast<float>(id));
        ASSERT_EQ(
            0, entity.add((uint64_t{1} << 40) + id, row.data(), sizeof(row)));
      }
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, entity.get_vector_by_key((uint64_t{1} << 40) + 198, block));
    EXPECT_FLOAT_EQ(198.0f, static_cast<const float *>(block.data())[0]);
    block.reset(nullptr);
    auto iterator = entity.creater_iterator();
    ASSERT_NE(nullptr, iterator);
    size_t visited = 0;
    for (; iterator->is_valid(); iterator->next(), ++visited)
      EXPECT_EQ((uint64_t{1} << 40) + visited, iterator->key());
    EXPECT_EQ(199u, visited);
    ASSERT_EQ(0, entity.flush(0));
    iterator.reset();
    ASSERT_EQ(0, entity.close());
    ASSERT_EQ(0, storage->close());
  }
  std::filesystem::remove(path);
}
}  // namespace
}  // namespace zvec::core

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
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    // MSVC's large-vector path asks operator new for alignment padding too.
    // Observe the actual request instead of missing every legacy copy.
    if (bytes >= std::_Big_allocation_threshold)
      observed_size += std::_Non_user_size;
#endif
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
  {
    auto provider = streamer->create_provider();
    ASSERT_NE(nullptr, provider);
    {
      ailego::MemoryReservation pressure;
      pressure.add(pool.capacity() - pool.used());
      EXPECT_THROW(provider->create_iterator(), std::bad_alloc);
    }
    auto iterator = provider->create_iterator();
    ASSERT_NE(nullptr, iterator);
    size_t count = 0;
    for (; iterator->is_valid(); iterator->next()) ++count;
    EXPECT_EQ(64u, count);
    EXPECT_EQ(0, iterator->status());
  }
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
  auto &pool = ailego::PoolPayloadBudget::get_instance();
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
      // The bypass path still needs charged metadata reads/query scratch.
      // A copied query tile and aligned input scratch need at least eight
      // physical pages when the host uses 16 KiB pages.
      const size_t query_budget =
          std::max<size_t>(64 * 1024, 8 * pool.page_buffer_size());
      ASSERT_EQ(0, pool.init(mode == 2 ? query_budget : 2 * 1024 * 1024));
      size_t admission_pressure = 0;
      AILEGO_DEFER([&] {
        if (admission_pressure != 0) pool.release_external(admission_pressure);
      });
      if (mode == 2) {
        // Leave room to decode the small container directory, but not to
        // admit even a compact page table. Release this pressure after open
        // so the bounded query/scratch allowance remains available.
        const size_t directory_allowance = pool.page_buffer_size();
        ASSERT_GT(pool.available(), directory_allowance);
        const size_t requested = pool.available() - directory_allowance;
        ASSERT_TRUE(pool.try_charge_external(requested));
        admission_pressure = requested;
        ASSERT_LT(pool.available(),
                  ailego::VecBufferPool::metadata_bytes_for_page_count(1));
      }
      auto source = IndexFactory::CreateStorage(mode ? "BufferReadStorage"
                                                     : "MMapFileReadStorage");
      ailego::Params params;
      params.set(BUFFER_READ_STORAGE_WARMUP_MODE,
                 BUFFER_READ_STORAGE_WARMUP_NONE);
      ASSERT_EQ(0, source->init(params));
      ASSERT_EQ(0, source->open(path, false));
      if (mode != 0) {
        ASSERT_NE(nullptr, source->vec_buffer_pool());
        ASSERT_EQ(mode == 1, source->vec_buffer_pool()->cache_enabled());
      }
      if (admission_pressure != 0) {
        pool.release_external(admission_pressure);
        admission_pressure = 0;
      }
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
      const size_t steady = pool.used();
      for (size_t repeat = 0; repeat < 1000; ++repeat) {
        ASSERT_NE(nullptr, provider->get_vector(keys[17]));
      }
      EXPECT_EQ(steady, pool.used());
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

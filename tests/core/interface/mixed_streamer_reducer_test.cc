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

#include "mixed_reducer/mixed_streamer_reducer.h"
#include <array>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_error.h>
#include "mixed_reducer/mixed_reducer_params.h"

namespace zvec {
namespace core {
namespace {

class TestProvider : public IndexProvider {
 public:
  enum class IteratorMode { kValid, kMissing, kReadFailure };

  TestProvider(const IndexMeta &meta, size_t count, IteratorMode mode)
      : TestProvider(meta, count, mode, {1.0F, 2.0F}) {}

  TestProvider(const IndexMeta &meta, size_t count, IteratorMode mode,
               std::array<float, 2> values)
      : meta_(meta), mode_(mode), values_(values), keys_(count) {
    std::iota(keys_.begin(), keys_.end(), uint64_t{0});
  }

  TestProvider(const IndexMeta &meta, std::vector<uint64_t> keys,
               IteratorMode mode, std::array<float, 2> values)
      : meta_(meta), mode_(mode), values_(values), keys_(std::move(keys)) {}

  class Iterator : public IndexHolder::Iterator {
   public:
    Iterator(std::vector<uint64_t> keys, bool read_failure,
             std::array<float, 2> values)
        : keys_(std::move(keys)),
          read_failure_(read_failure),
          vector_{values[0], values[1]} {}

    const void *data() const override {
      return read_failure_ ? nullptr : vector_;
    }

    bool is_valid() const override {
      return index_ < keys_.size();
    }

    uint64_t key() const override {
      return keys_[index_];
    }

    void next() override {
      ++index_;
    }

   private:
    std::vector<uint64_t> keys_;
    bool read_failure_{false};
    size_t index_{0};
    float vector_[2]{1.0F, 2.0F};
  };

  size_t count() const override {
    return keys_.size();
  }

  size_t dimension() const override {
    return meta_.dimension();
  }

  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

  size_t element_size() const override {
    return meta_.element_size();
  }

  IndexHolder::Iterator::Pointer create_iterator() override {
    if (mode_ == IteratorMode::kMissing) {
      return nullptr;
    }
    return IndexHolder::Iterator::Pointer(
        new Iterator(keys_, mode_ == IteratorMode::kReadFailure, values_));
  }

  const void *get_vector(uint64_t) const override {
    return nullptr;
  }

  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  IndexMeta meta_;
  IteratorMode mode_{IteratorMode::kValid};
  std::array<float, 2> values_{};
  std::vector<uint64_t> keys_;
  std::string owner_class_{"MixedStreamerReducerTest"};
};

class TestStreamer : public IndexStreamer {
 public:
  TestStreamer(const IndexMeta &meta, IndexProvider::Pointer provider,
               IndexSparseProvider::Pointer sparse_provider =
                   IndexSparseProvider::Pointer())
      : meta_(meta),
        provider_(std::move(provider)),
        sparse_provider_(std::move(sparse_provider)) {}

  int open(IndexStorage::Pointer) override {
    return 0;
  }

  int flush(uint64_t) override {
    return 0;
  }

  int close() override {
    return 0;
  }

  const IndexMeta &meta() const override {
    return meta_;
  }

  const Stats &stats() const override {
    return stats_;
  }

  int cleanup() override {
    ++cleanup_count_;
    return 0;
  }

  int dump(const IndexDumper::Pointer &) override {
    return dump_result_;
  }

  int add_with_id_impl(uint32_t id, const void *query,
                       const IndexQueryMeta &qmeta,
                       Context::Pointer &) override {
    if (query == nullptr || qmeta.data_type() != IndexMeta::DataType::DT_FP32 ||
        qmeta.dimension() != 2) {
      return IndexError_Mismatch;
    }
    if (added_vectors_.size() <= id) {
      added_vectors_.resize(id + 1);
    }
    std::memcpy(added_vectors_[id].data(), query, sizeof(float) * 2);
    return 0;
  }

  Provider::Pointer create_provider() const override {
    return provider_;
  }

  SparseProvider::Pointer create_sparse_provider() const override {
    return sparse_provider_;
  }

  void set_dump_result(int result) {
    dump_result_ = result;
  }

  const std::vector<std::array<float, 2>> &added_vectors() const {
    return added_vectors_;
  }

  size_t cleanup_count() const {
    return cleanup_count_;
  }

 private:
  IndexMeta meta_;
  IndexProvider::Pointer provider_;
  IndexSparseProvider::Pointer sparse_provider_;
  Stats stats_;
  int dump_result_{0};
  std::vector<std::array<float, 2>> added_vectors_;
  size_t cleanup_count_{0};
};

class OffsetReformer final : public IndexReformer {
 public:
  OffsetReformer(float revert_offset, float convert_offset)
      : revert_offset_(revert_offset), convert_offset_(convert_offset) {}

  int init(const ailego::Params &) override {
    return 0;
  }

  int cleanup() override {
    return 0;
  }

  int load(IndexStorage::Pointer) override {
    return 0;
  }

  int unload() override {
    return 0;
  }

  int revert(const void *input, const IndexQueryMeta &,
             std::string *output) const override {
    return transform_with_offset(input, revert_offset_, output, nullptr);
  }

  int convert(const void *input, const IndexQueryMeta &, std::string *output,
              IndexQueryMeta *output_meta) const override {
    return transform_with_offset(input, convert_offset_, output, output_meta);
  }

 private:
  static int transform_with_offset(const void *input, float offset,
                                   std::string *output,
                                   IndexQueryMeta *output_meta) {
    if (input == nullptr || output == nullptr) {
      return IndexError_InvalidArgument;
    }
    std::array<float, 2> values{};
    std::memcpy(values.data(), input, sizeof(values));
    values[0] += offset;
    values[1] += offset;
    output->assign(reinterpret_cast<const char *>(values.data()),
                   sizeof(values));
    if (output_meta != nullptr) {
      *output_meta = IndexQueryMeta(IndexMeta::DataType::DT_FP32, 2);
    }
    return 0;
  }

  float revert_offset_{0};
  float convert_offset_{0};
};

class TestDumper final : public IndexDumper {
 public:
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

  int append(const std::string &, size_t, size_t, uint32_t) override {
    return 0;
  }

  size_t write(const void *, size_t length) override {
    return length;
  }

  uint32_t magic() const override {
    return 0;
  }
};

class TestSparseProvider : public IndexSparseProvider {
 public:
  TestSparseProvider(size_t count, bool read_failure)
      : count_(count), read_failure_(read_failure) {}

  class Iterator : public IndexSparseHolder::Iterator {
   public:
    Iterator(size_t count, bool read_failure)
        : count_(count), read_failure_(read_failure) {}

    bool is_valid() const override {
      return index_ < count_;
    }

    uint64_t key() const override {
      return index_;
    }

    uint32_t sparse_count() const override {
      return 1;
    }

    const uint32_t *sparse_indices() const override {
      return read_failure_ ? nullptr : &sparse_index_;
    }

    const void *sparse_data() const override {
      return read_failure_ ? nullptr : &sparse_value_;
    }

    void next() override {
      ++index_;
    }

   private:
    size_t count_{0};
    bool read_failure_{false};
    size_t index_{0};
    uint32_t sparse_index_{1};
    float sparse_value_{2.0F};
  };

  size_t count() const override {
    return count_;
  }

  IndexMeta::DataType data_type() const override {
    return IndexMeta::DataType::DT_FP32;
  }

  IndexSparseHolder::Iterator::Pointer create_iterator() override {
    return IndexSparseHolder::Iterator::Pointer(
        new Iterator(count_, read_failure_));
  }

  size_t total_sparse_count() const override {
    return count_;
  }

  int get_sparse_vector(uint64_t, uint32_t *, std::string *,
                        std::string *) const override {
    return IndexError_NotImplemented;
  }

  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  size_t count_{0};
  bool read_failure_{false};
  std::string owner_class_{"MixedStreamerReducerSparseTest"};
};

IndexMeta MakeDenseMeta() {
  IndexMeta meta;
  meta.set_meta_type(IndexMeta::MetaType::MT_DENSE);
  meta.set_meta(IndexMeta::DataType::DT_FP32, 2);
  return meta;
}

IndexMeta MakeSparseMeta() {
  return IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                   IndexMeta::DataType::DT_FP32);
}

int RunReduce(const IndexProvider::Pointer &target_provider,
              const IndexProvider::Pointer &source_provider) {
  const IndexMeta meta = MakeDenseMeta();
  auto target = std::make_shared<TestStreamer>(meta, target_provider);
  auto source = std::make_shared<TestStreamer>(meta, source_provider);

  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1U);
  if (reducer.init(params) != 0 ||
      reducer.set_target_streamer_wiht_info(
          nullptr, target, nullptr, nullptr,
          IndexQueryMeta(IndexMeta::MetaType::MT_DENSE,
                         IndexMeta::DataType::DT_FP32, 2)) != 0 ||
      reducer.feed_streamer_with_reformer(source, nullptr) != 0) {
    return IndexError_Runtime;
  }

  ailego::ThreadPool thread_pool(1, false);
  reducer.set_thread_pool(&thread_pool);
  return reducer.reduce(IndexFilter());
}

int RunSparseReduce(const IndexSparseProvider::Pointer &target_provider,
                    const IndexSparseProvider::Pointer &source_provider) {
  const IndexMeta meta = MakeSparseMeta();
  auto target = std::make_shared<TestStreamer>(meta, IndexProvider::Pointer(),
                                               target_provider);
  auto source = std::make_shared<TestStreamer>(meta, IndexProvider::Pointer(),
                                               source_provider);

  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1U);
  if (reducer.init(params) != 0 ||
      reducer.set_target_streamer_wiht_info(
          nullptr, target, nullptr, nullptr,
          IndexQueryMeta(IndexMeta::MetaType::MT_SPARSE,
                         IndexMeta::DataType::DT_FP32)) != 0 ||
      reducer.feed_streamer_with_reformer(source, nullptr) != 0) {
    return IndexError_Runtime;
  }

  ailego::ThreadPool thread_pool(1, false);
  reducer.set_thread_pool(&thread_pool);
  return reducer.reduce(IndexFilter());
}

TEST(MixedStreamerReducer, RejectsMissingTargetProvider) {
  const IndexMeta meta = MakeDenseMeta();
  auto source = std::make_shared<TestProvider>(
      meta, 1, TestProvider::IteratorMode::kValid);
  EXPECT_EQ(IndexError_Runtime, RunReduce(nullptr, source));
}

TEST(MixedStreamerReducer, RejectsMissingSourceProvider) {
  const IndexMeta meta = MakeDenseMeta();
  auto target = std::make_shared<TestProvider>(
      meta, 0, TestProvider::IteratorMode::kValid);
  EXPECT_EQ(IndexError_Runtime, RunReduce(target, nullptr));
}

TEST(MixedStreamerReducer, RejectsMissingSourceIterator) {
  const IndexMeta meta = MakeDenseMeta();
  auto target = std::make_shared<TestProvider>(
      meta, 0, TestProvider::IteratorMode::kValid);
  auto source = std::make_shared<TestProvider>(
      meta, 1, TestProvider::IteratorMode::kMissing);
  EXPECT_EQ(IndexError_Runtime, RunReduce(target, source));
}

TEST(MixedStreamerReducer, RejectsUnreadableSourceVector) {
  const IndexMeta meta = MakeDenseMeta();
  auto target = std::make_shared<TestProvider>(
      meta, 0, TestProvider::IteratorMode::kValid);
  auto source = std::make_shared<TestProvider>(
      meta, 1, TestProvider::IteratorMode::kReadFailure);
  EXPECT_EQ(IndexError_Runtime, RunReduce(target, source));
}

TEST(MixedStreamerReducer, RejectsUnreadableSourceSparseVector) {
  auto target = std::make_shared<TestSparseProvider>(0, false);
  auto source = std::make_shared<TestSparseProvider>(1, true);
  EXPECT_EQ(IndexError_Runtime, RunSparseReduce(target, source));
}

TEST(MixedStreamerReducer, RejectsFirstSourceVectorLayoutMismatch) {
  const IndexMeta target_meta = MakeDenseMeta();
  IndexMeta source_meta;
  source_meta.set_meta_type(IndexMeta::MetaType::MT_DENSE);
  source_meta.set_meta(IndexMeta::DataType::DT_FP32, 1);

  auto target = std::make_shared<TestStreamer>(
      target_meta, std::make_shared<TestProvider>(
                       target_meta, 0, TestProvider::IteratorMode::kValid));
  auto source = std::make_shared<TestStreamer>(
      source_meta, std::make_shared<TestProvider>(
                       source_meta, 1, TestProvider::IteratorMode::kValid));

  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1U);
  ASSERT_EQ(0, reducer.init(params));
  ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(
                   nullptr, target, nullptr, nullptr,
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, 2)));
  EXPECT_EQ(IndexError_InvalidArgument,
            reducer.feed_streamer_with_reformer(source, nullptr));
}

TEST(MixedStreamerReducer, ConvertsEachSourceWithItsOwnReformer) {
  IndexMeta target_meta = MakeDenseMeta();
  target_meta.set_reformer("target", 0, ailego::Params());
  IndexMeta matching_source_meta = target_meta;
  IndexMeta different_source_meta = MakeDenseMeta();
  different_source_meta.set_reformer("source", 0, ailego::Params());

  auto target = std::make_shared<TestStreamer>(
      target_meta, std::make_shared<TestProvider>(
                       target_meta, 0, TestProvider::IteratorMode::kValid));
  auto matching_source = std::make_shared<TestStreamer>(
      matching_source_meta,
      std::make_shared<TestProvider>(matching_source_meta, 1,
                                     TestProvider::IteratorMode::kValid,
                                     std::array<float, 2>{1.0F, 2.0F}));
  auto different_source = std::make_shared<TestStreamer>(
      different_source_meta,
      std::make_shared<TestProvider>(different_source_meta, 1,
                                     TestProvider::IteratorMode::kValid,
                                     std::array<float, 2>{3.0F, 4.0F}));
  auto target_reformer = std::make_shared<OffsetReformer>(0.0F, 10.0F);
  auto source_reformer = std::make_shared<OffsetReformer>(2.0F, 0.0F);

  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1U);
  ASSERT_EQ(0, reducer.init(params));
  ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(
                   nullptr, target, nullptr, target_reformer,
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, 2)));
  ASSERT_EQ(0, reducer.feed_streamer_with_reformer(matching_source, nullptr));
  ASSERT_EQ(0, reducer.feed_streamer_with_reformer(different_source,
                                                   source_reformer));

  ailego::ThreadPool thread_pool(1, false);
  reducer.set_thread_pool(&thread_pool);
  ASSERT_EQ(0, reducer.reduce(IndexFilter()));

  const auto &added = target->added_vectors();
  ASSERT_EQ(2U, added.size());
  EXPECT_FLOAT_EQ(1.0F, added[0][0]);
  EXPECT_FLOAT_EQ(2.0F, added[0][1]);
  EXPECT_FLOAT_EQ(15.0F, added[1][0]);
  EXPECT_FLOAT_EQ(16.0F, added[1][1]);

  target->set_dump_result(IndexError_WriteData);
  EXPECT_EQ(IndexError_WriteData, reducer.dump(std::make_shared<TestDumper>()));
  EXPECT_EQ(0, reducer.cleanup());
  EXPECT_EQ(1U, target->cleanup_count());
}

TEST(MixedStreamerReducer, FilterOffsetsUseSourceKeySpan) {
  const IndexMeta meta = MakeDenseMeta();
  auto target = std::make_shared<TestStreamer>(
      meta, std::make_shared<TestProvider>(meta, 0,
                                           TestProvider::IteratorMode::kValid));
  auto first_source = std::make_shared<TestStreamer>(
      meta, std::make_shared<TestProvider>(meta, std::vector<uint64_t>{0, 2},
                                           TestProvider::IteratorMode::kValid,
                                           std::array<float, 2>{1.0F, 2.0F}));
  auto second_source = std::make_shared<TestStreamer>(
      meta, std::make_shared<TestProvider>(meta, std::vector<uint64_t>{0},
                                           TestProvider::IteratorMode::kValid,
                                           std::array<float, 2>{3.0F, 4.0F}));

  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1U);
  ASSERT_EQ(0, reducer.init(params));
  ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(
                   nullptr, target, nullptr, nullptr,
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, 2)));
  ASSERT_EQ(0, reducer.feed_streamer_with_reformer(first_source, nullptr));
  ASSERT_EQ(0, reducer.feed_streamer_with_reformer(second_source, nullptr));

  std::vector<uint64_t> filtered_ids;
  IndexFilter filter;
  filter.set([&filtered_ids](uint64_t id) {
    filtered_ids.push_back(id);
    return id == 3;
  });
  ailego::ThreadPool thread_pool(1, false);
  reducer.set_thread_pool(&thread_pool);
  ASSERT_EQ(0, reducer.reduce(filter));

  EXPECT_EQ((std::vector<uint64_t>{0, 2, 3}), filtered_ids);
  const auto &added = target->added_vectors();
  ASSERT_EQ(2U, added.size());
  EXPECT_FLOAT_EQ(1.0F, added[0][0]);
  EXPECT_FLOAT_EQ(1.0F, added[1][0]);
}

TEST(MixedStreamerReducer, AppendsAfterGappedTargetKeySpan) {
  const IndexMeta meta = MakeDenseMeta();
  auto target = std::make_shared<TestStreamer>(
      meta, std::make_shared<TestProvider>(meta, std::vector<uint64_t>{0, 2},
                                           TestProvider::IteratorMode::kValid,
                                           std::array<float, 2>{1.0F, 2.0F}));
  auto source = std::make_shared<TestStreamer>(
      meta, std::make_shared<TestProvider>(meta, std::vector<uint64_t>{0},
                                           TestProvider::IteratorMode::kValid,
                                           std::array<float, 2>{3.0F, 4.0F}));

  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1U);
  ASSERT_EQ(0, reducer.init(params));
  ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(
                   nullptr, target, nullptr, nullptr,
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, 2)));
  ASSERT_EQ(0, reducer.feed_streamer_with_reformer(source, nullptr));

  ailego::ThreadPool thread_pool(1, false);
  reducer.set_thread_pool(&thread_pool);
  ASSERT_EQ(0, reducer.reduce(IndexFilter()));

  const auto &added = target->added_vectors();
  ASSERT_EQ(4U, added.size());
  EXPECT_FLOAT_EQ(3.0F, added[3][0]);
  EXPECT_FLOAT_EQ(4.0F, added[3][1]);
}

TEST(MixedStreamerReducer, CleanupWithoutTargetIsSafe) {
  MixedStreamerReducer reducer;
  ailego::Params params;
  params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1U);
  ASSERT_EQ(0, reducer.init(params));
  EXPECT_EQ(0, reducer.cleanup());
}

}  // namespace
}  // namespace core
}  // namespace zvec

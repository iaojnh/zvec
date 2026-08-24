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
#include <memory>
#include <string>
#include <utility>
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
      : meta_(meta), count_(count), mode_(mode) {}

  class Iterator : public IndexHolder::Iterator {
   public:
    Iterator(size_t count, bool read_failure)
        : count_(count), read_failure_(read_failure) {}

    const void *data() const override {
      return read_failure_ ? nullptr : vector_;
    }

    bool is_valid() const override {
      return index_ < count_;
    }

    uint64_t key() const override {
      return index_;
    }

    void next() override {
      ++index_;
    }

   private:
    size_t count_{0};
    bool read_failure_{false};
    size_t index_{0};
    float vector_[2]{1.0F, 2.0F};
  };

  size_t count() const override {
    return count_;
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
        new Iterator(count_, mode_ == IteratorMode::kReadFailure));
  }

  const void *get_vector(uint64_t) const override {
    return nullptr;
  }

  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  IndexMeta meta_;
  size_t count_{0};
  IteratorMode mode_{IteratorMode::kValid};
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
    return 0;
  }

  Provider::Pointer create_provider() const override {
    return provider_;
  }

  SparseProvider::Pointer create_sparse_provider() const override {
    return sparse_provider_;
  }

 private:
  IndexMeta meta_;
  IndexProvider::Pointer provider_;
  IndexSparseProvider::Pointer sparse_provider_;
  Stats stats_;
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

}  // namespace
}  // namespace core
}  // namespace zvec

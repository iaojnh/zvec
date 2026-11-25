// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <rabitqlib/utils/cpu_features.hpp>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_streamer.h"
#include "algorithm/hnsw_rabitq/rabitq_converter.h"
#include "algorithm/hnsw_rabitq/rabitq_reformer.h"

namespace {
thread_local bool fail_allocations = false;
thread_local long allocations_until_failure = -1;
}  // namespace

void *operator new(std::size_t size) {
  if (fail_allocations || allocations_until_failure == 0)
    throw std::bad_alloc();
  if (allocations_until_failure > 0) --allocations_until_failure;
  if (void *ptr = std::malloc(size == 0 ? 1 : size)) return ptr;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
  return ::operator new(size);
}
void operator delete(void *ptr) noexcept {
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

class OomOpenHnswStreamer : public core::HnswStreamer {
 public:
  using core::HnswStreamer::open;
  explicit OomOpenHnswStreamer(bool *armed, bool fail_context)
      : armed_(armed), fail_context_(fail_context) {}

  int open(core::IndexStorage::Pointer storage, bool read_only) override {
    if (*armed_ && !fail_context_) {
      *armed_ = false;
      // Fail the real streamer's entity allocation and every allocation
      // during the ensuing rollback, including storage destruction.
      fail_allocations = true;
    }
    return core::HnswStreamer::open(std::move(storage), read_only);
  }

  int create_context_with_status(Context::Pointer &context) const override {
    if (*armed_ && fail_context_) {
      *armed_ = false;
      fail_allocations = true;
    }
    return core::HnswStreamer::create_context_with_status(context);
  }

 private:
  bool *armed_;
  bool fail_context_;
};

void CheckOpenCleanup(StorageType type, bool create_new, bool read_only,
                      bool fail_context = false, bool legacy = false) {
  ailego::LoggerBroker::Unregister();
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(size_t{256} << 20));
  const auto directory =
      "index_oom_cleanup_" + std::to_string(static_cast<int>(type)) + "_" +
      std::to_string(create_new) + "_" + std::to_string(read_only);
  fs::remove_all(directory);
  ASSERT_TRUE(fs::create_directory(directory));
  const std::string path = directory + "/index";
  auto param =
      HNSWIndexParamBuilder()
          .with_metric_type(MetricType::kL2sq)
          .with_data_type(DataType::DT_FP32)
          .with_dimension(16)
          .with_quantizer_param(QuantizerParam(
              legacy ? QuantizerType::kInt8 : QuantizerType::kNone, legacy))
          .build();
  const std::array<float, 16> row{1.0F};
  std::vector<char> original_file;
  ailego::File::Identity original_identity;
  if (!create_new) {
    auto writer = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, writer);
    ASSERT_EQ(0, writer->open(path, {type, true}));
    ASSERT_EQ(0, writer->add(VectorData{DenseVector{row.data()}}, 7));
    ASSERT_EQ(0, writer->close());
    ailego::File file;
    ASSERT_TRUE(file.open(path, true));
    original_file.resize(file.size());
    ASSERT_EQ(original_file.size(),
              file.read(original_file.data(), original_file.size()));
    original_identity = file.identity();
  }
  bool armed = false;
  ailego::Factory<core::IndexStreamer>::Register<OomOpenHnswStreamer> replace(
      "HnswStreamer", &armed, fail_context);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  for (int attempt = 0; attempt < 5; ++attempt) {
    SCOPED_TRACE(attempt);
    armed = true;
    bool threw = false;
    int result = 0;
    try {
      result = index->open(path, {type, create_new, read_only});
    } catch (...) {
      threw = true;
    }
    fail_allocations = false;
    ASSERT_FALSE(armed);
    ASSERT_FALSE(threw);
    ASSERT_EQ(core::IndexError_NoMemory, result);
    ASSERT_EQ(!create_new, ailego::File::IsExist(path));
    if (create_new) {
      ASSERT_TRUE(fs::is_empty(directory))
          << "Private staging files survived failed creation";
    } else {
      ailego::File file;
      ASSERT_TRUE(file.open(path, true));
      ASSERT_TRUE(original_identity == file.identity());
      ASSERT_EQ(original_file.size(), file.size());
      std::vector<char> content(original_file.size());
      ASSERT_EQ(content.size(), file.read(content.data(), content.size()));
      ASSERT_EQ(original_file, content);
    }
  }
  ASSERT_EQ(0, index->open(path, {type, create_new, read_only}));
  if (create_new) {
    ASSERT_EQ(0, index->add(VectorData{DenseVector{row.data()}}, 7));
  }
  auto query = HNSWQueryParamBuilder().with_topk(1).with_ef_search(16).build();
  SearchResult search;
  ASSERT_EQ(0,
            index->search(VectorData{DenseVector{row.data()}}, query, &search));
  ASSERT_EQ(1u, search.doc_list_.size());
  EXPECT_EQ(7u, search.doc_list_[0].key());
  EXPECT_TRUE(std::isfinite(search.doc_list_[0].score()));
  ASSERT_EQ(0, index->close());
  index.reset();
  ailego::Factory<core::IndexStreamer>::Register<core::HnswStreamer> restore(
      "HnswStreamer");
  ailego::File::Delete(path);
  ASSERT_TRUE(fs::is_empty(directory));
  ASSERT_TRUE(fs::remove(directory));
}

class IndexOomCleanupTest
    : public ::testing::TestWithParam<std::tuple<StorageType, int>> {};

TEST_P(IndexOomCleanupTest, PersistentFailureReturnsNoMemoryAndAllowsRetry) {
  const auto [type, mode] = GetParam();
  ASSERT_EXIT(
      {
        CheckOpenCleanup(type, mode == 0, mode == 2);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    MmapAndBufferPool, IndexOomCleanupTest,
    ::testing::Combine(::testing::Values(StorageType::kMMAP,
                                         StorageType::kBufferPool),
                       ::testing::Values(0, 1, 2)));

class IndexContextOomCleanupTest
    : public ::testing::TestWithParam<std::tuple<StorageType, int, bool>> {};

TEST_P(IndexContextOomCleanupTest, FailureAfterStreamerOpenAllowsRetry) {
  const auto [type, mode, legacy] = GetParam();
  ASSERT_EXIT(
      {
        CheckOpenCleanup(type, mode == 0, mode == 2, true, legacy);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    TurboAndLegacyMmapAndBufferPool, IndexContextOomCleanupTest,
    ::testing::Combine(::testing::Values(StorageType::kMMAP,
                                         StorageType::kBufferPool),
                       ::testing::Values(0, 1, 2), ::testing::Bool()));

void CheckStreamerAbort(const char *name, StorageType type) {
  ailego::LoggerBroker::Unregister();
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(size_t{256} << 20));
  const std::string directory = std::string("streamer_abort_") + name +
                                std::to_string(static_cast<int>(type));
  fs::remove_all(directory);
  ASSERT_TRUE(fs::create_directory(directory));
  auto storage = core::IndexFactory::CreateStorage(
      type == StorageType::kMMAP ? "MMapFileStorage" : "BufferStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  ASSERT_EQ(
      0, storage->open(directory + "/index", core::IndexStorage::OpenOptions{
                                                 true, false, nullptr, true}));
  const bool sparse = std::string(name).find("Sparse") != std::string::npos;
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 128);
  if (sparse) {
    meta =
        core::IndexMeta(core::IndexMeta::MT_SPARSE, core::IndexMeta::DT_FP32);
    meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  }
  auto streamer = core::IndexFactory::CreateStreamer(name);
  ailego::Params params;
  if (std::string(name) == "HnswRabitqStreamer") {
    auto holder = std::make_shared<
        core::MultiPassIndexProvider<core::IndexMeta::DT_FP32>>(128);
    for (size_t i = 0; i < 128; ++i) {
      ailego::NumericalVector<float> row(128);
      for (size_t j = 0; j < row.size(); ++j) {
        row[j] = static_cast<float>((i + 1) * (j + 3) % 31) / 31.0F;
      }
      ASSERT_TRUE(holder->emplace(i, row));
    }
    core::RabitqConverter converter;
    ASSERT_EQ(0, converter.init(meta, ailego::Params{}));
    ASSERT_EQ(0, converter.train(holder));
    core::IndexReformer::Pointer reformer;
    ASSERT_EQ(0, converter.to_reformer(&reformer));
    auto trained = std::dynamic_pointer_cast<core::RabitqReformer>(reformer);
    ASSERT_NE(nullptr, trained);
    ASSERT_EQ(0, trained->dump(storage));
    params.set("proxima.hnsw_rabitq.general.dimension", 128U);
  }
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));
  int result = -1;
  bool threw = false;
  fail_allocations = true;
  try {
    result = streamer->abort_open();
  } catch (...) {
    threw = true;
  }
  streamer.reset();
  storage.reset();
  fail_allocations = false;
  ASSERT_FALSE(threw);
  ASSERT_EQ(0, result);
  ASSERT_TRUE(fs::is_empty(directory));
  ASSERT_TRUE(fs::remove(directory));
}

class StreamerAbortOomTest
    : public ::testing::TestWithParam<std::tuple<const char *, StorageType>> {};

TEST_P(StreamerAbortOomTest, OpenedStreamerReleasesStorageWithoutAllocating) {
  const auto [name, type] = GetParam();
  if (std::string(name) == "HnswRabitqStreamer" &&
      !rabitqlib::cpu::has_avx512_core() && !rabitqlib::cpu::has_avx2() &&
      !rabitqlib::cpu::has_neon()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA, AVX512F/BW/DQ, or NEON";
  }
  ASSERT_EXIT(
      {
        CheckStreamerAbort(name, type);
        std::fflush(stdout);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    PersistenceStreamersAndBothBackends, StreamerAbortOomTest,
    ::testing::Combine(::testing::Values("HnswStreamer", "HnswSparseStreamer",
                                         "HnswRabitqStreamer", "VamanaStreamer",
                                         "FlatStreamer", "FlatSparseStreamer"),
                       ::testing::Values(StorageType::kMMAP,
                                         StorageType::kBufferPool)));

enum class StagedCleanupMode { kEmpty, kPrivate, kPublished, kConflict };

void CheckStagedCleanup(StagedCleanupMode mode) {
  const auto directory = fs::u8path(std::string(u8"staged_oom_cleanup_目录_") +
                                    std::to_string(static_cast<int>(mode)));
  fs::remove_all(directory);
  ASSERT_TRUE(fs::create_directory(directory));
  const auto target = (directory / fs::u8path(u8"索引.zvec")).u8string();
  auto staged = std::make_unique<ailego::StagedPath>();
  ASSERT_TRUE(staged->prepare(target));
  const auto private_directory = fs::u8path(staged->path()).parent_path();
  ailego::File file;
  if (mode != StagedCleanupMode::kEmpty) {
    ASSERT_TRUE(file.create(staged->path(), 1));
    ASSERT_EQ(1u, file.write("d", 1));
    file.close();
  }
  if (mode == StagedCleanupMode::kConflict) {
    ASSERT_TRUE(file.create(target, 1));
    ASSERT_EQ(1u, file.write("w", 1));
    file.close();
    ASSERT_FALSE(staged->publish());
  } else if (mode == StagedCleanupMode::kPublished) {
    ASSERT_TRUE(staged->publish());
  }
  fail_allocations = true;
  staged.reset();
  fail_allocations = false;
  ASSERT_FALSE(fs::exists(private_directory));
  const bool published = mode == StagedCleanupMode::kPublished ||
                         mode == StagedCleanupMode::kConflict;
  ASSERT_EQ(published, ailego::File::IsExist(target));
  if (published) {
    ASSERT_TRUE(file.open(target, true));
    char value = 0;
    ASSERT_EQ(1u, file.read(&value, 1));
    EXPECT_EQ(mode == StagedCleanupMode::kPublished ? 'd' : 'w', value);
    file.close();
    ASSERT_TRUE(ailego::File::Delete(target));
  }
  ASSERT_TRUE(fs::is_empty(directory));
  ASSERT_TRUE(fs::remove(directory));
}

class StagedPathOomCleanupTest
    : public ::testing::TestWithParam<StagedCleanupMode> {};

TEST_P(StagedPathOomCleanupTest, CleanupPreservesPublishedOwner) {
  if (!ailego::StagedPath::supported()) GTEST_SKIP();
  const auto mode = GetParam();
  ASSERT_EXIT(
      {
        CheckStagedCleanup(mode);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(EmptyPrivatePublishedAndConflict,
                         StagedPathOomCleanupTest,
                         ::testing::Values(StagedCleanupMode::kEmpty,
                                           StagedCleanupMode::kPrivate,
                                           StagedCleanupMode::kPublished,
                                           StagedCleanupMode::kConflict));

void CheckPrepareCleanup() {
  const std::string directory = "staged_prepare_oom_cleanup";
  fs::remove_all(directory);
  ASSERT_TRUE(fs::create_directory(directory));
  const std::string target = directory + "/index";
  for (long point = 0; point < 128; ++point) {
    SCOPED_TRACE(point);
    auto staged = std::make_unique<ailego::StagedPath>();
    allocations_until_failure = point;
    try {
      (void)staged->prepare(target);
    } catch (const std::bad_alloc &) {
    }
    fail_allocations = true;
    staged.reset();
    fail_allocations = false;
    allocations_until_failure = -1;
    ASSERT_TRUE(fs::is_empty(directory));
  }
  ASSERT_TRUE(fs::remove(directory));
}

TEST(IndexOomCleanupDeathTest,
     StagedPreparationFailureRetainsCleanupOwnership) {
  if (!ailego::StagedPath::supported()) GTEST_SKIP();
  ASSERT_EXIT(
      {
        CheckPrepareCleanup();
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void CheckDirtyPoolCleanup(bool explicit_flush) {
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(size_t{256} << 20));
  // Exercise destructor diagnostics as well as writeback while allocation
  // remains unavailable. A diagnostic must not prevent resource cleanup.
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);
  const std::string path = "pool_oom_cleanup.index";
  constexpr size_t kPages = 130;
  ailego::File file;
  ASSERT_TRUE(file.create(path, kPages * ailego::kVectorPageSize));
  file.close();
  auto pool = std::make_unique<ailego::VecBufferPool>(path, true);
  ASSERT_EQ(0, pool->init());
  for (size_t i = 0; i < kPages; ++i) {
    const char value = static_cast<char>(i + 1);
    ASSERT_EQ(0, pool->write_range(i * ailego::kVectorPageSize, 1, &value));
  }
  int result = 0;
  bool threw = false;
  fail_allocations = true;
  try {
    if (explicit_flush) result = pool->flush_all();
  } catch (...) {
    threw = true;
  }
  pool.reset();
  fail_allocations = false;
  ASSERT_FALSE(threw);
  ASSERT_EQ(0, result);
  ASSERT_TRUE(file.open(path, true));
  for (size_t i = 0; i < kPages; ++i) {
    char value = 0;
    ASSERT_EQ(1u, file.read(i * ailego::kVectorPageSize, &value, 1));
    ASSERT_EQ(static_cast<char>(i + 1), value) << i;
  }
  file.close();
  ailego::File::Delete(path);
}

TEST(IndexOomCleanupDeathTest, DirtyFlushAndDestructionPreserveData) {
  ASSERT_EXIT(
      {
        CheckDirtyPoolCleanup(true);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(IndexOomCleanupDeathTest, DirtyDestructionPreservesData) {
  ASSERT_EXIT(
      {
        CheckDirtyPoolCleanup(false);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}
#endif
}  // namespace
}  // namespace zvec::core_interface

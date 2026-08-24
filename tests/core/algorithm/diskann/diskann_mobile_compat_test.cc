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

#if defined(_WIN32) || defined(_WIN64)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_builder.h"
#include "diskann_file_reader.h"
#include "diskann_pq_trainer.h"
#include "diskann_searcher_entity.h"
#include "diskann_util.h"

namespace zvec::core {
namespace {

class TemporaryFile {
 public:
  TemporaryFile() {
#if defined(_WIN32) || defined(_WIN64)
    char temp_directory[MAX_PATH]{};
    char temp_file[MAX_PATH]{};
    if (::GetTempPathA(MAX_PATH, temp_directory) != 0 &&
        ::GetTempFileNameA(temp_directory, "zvc", 0, temp_file) != 0) {
      path_ = temp_file;
      fd_ = ::_open(path_.c_str(), _O_BINARY | _O_RDWR);
    }
#else
    char path[] = "DiskAnnMobileCompatTest.XXXXXX";
    fd_ = ::mkstemp(path);
    path_ = path;
#endif
  }

  ~TemporaryFile() {
    if (fd_ >= 0) {
      close_descriptor(fd_);
    }
    remove_file(path_.c_str());
  }

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

  int fd() const {
    return fd_;
  }

  const char *path() const {
    return path_.c_str();
  }

  void close() {
    if (fd_ >= 0) {
      close_descriptor(fd_);
      fd_ = -1;
    }
  }

  void release_descriptor_and_unlink() {
    close();
    remove_file(path_.c_str());
  }

 private:
  static int close_descriptor(int fd) {
#if defined(_WIN32) || defined(_WIN64)
    return ::_close(fd);
#else
    return ::close(fd);
#endif
  }

  static int remove_file(const char *path) {
#if defined(_WIN32) || defined(_WIN64)
    return ::_unlink(path);
#else
    return ::unlink(path);
#endif
  }

  std::string path_;
  int fd_{-1};
};

int64_t WriteAt(int fd, const void *data, size_t length, uint64_t offset) {
#if defined(_WIN32) || defined(_WIN64)
  if (::_lseeki64(fd, static_cast<__int64>(offset), SEEK_SET) < 0 ||
      length >
          static_cast<size_t>((std::numeric_limits<unsigned int>::max)())) {
    return -1;
  }
  return ::_write(fd, data, static_cast<unsigned int>(length));
#else
  return ::pwrite(fd, data, length, static_cast<off_t>(offset));
#endif
}

class VectorSegment final : public IndexStorage::Segment {
 public:
  explicit VectorSegment(std::vector<uint8_t> data) : data_(std::move(data)) {}

  size_t data_size() const override {
    return data_.size();
  }

  uint32_t data_crc() const override {
    return 0;
  }

  size_t padding_size() const override {
    return 0;
  }

  size_t capacity() const override {
    return data_.size();
  }

  size_t fetch(size_t offset, void *buffer, size_t length) const override {
    if (offset > data_.size()) {
      return 0;
    }
    const size_t read_size = std::min(length, data_.size() - offset);
    if (read_size != 0) {
      std::memcpy(buffer, data_.data() + offset, read_size);
    }
    return read_size;
  }

  size_t read(size_t offset, const void **data, size_t length) override {
    if (offset > data_.size()) {
      *data = nullptr;
      return 0;
    }
    const size_t read_size = std::min(length, data_.size() - offset);
    *data = read_size == 0 ? nullptr : data_.data() + offset;
    return read_size;
  }

  size_t read(size_t offset, IndexStorage::MemoryBlock &data,
              size_t length) override {
    const void *read_data = nullptr;
    const size_t read_size = read(offset, &read_data, length);
    data.reset(const_cast<void *>(read_data));
    return read_size;
  }

  size_t write(size_t, const void *, size_t) override {
    return 0;
  }

  size_t resize(size_t) override {
    return 0;
  }

  void update_data_crc(uint32_t) override {}

  Pointer clone() override {
    return std::make_shared<VectorSegment>(data_);
  }

 private:
  std::vector<uint8_t> data_;
};

class VectorStorage final : public IndexStorage {
 public:
  void add(const std::string &id, std::vector<uint8_t> data) {
    segments_[id] = std::make_shared<VectorSegment>(std::move(data));
  }

  int init(const ailego::Params &) override {
    return 0;
  }

  int cleanup() override {
    return 0;
  }

  int open(const std::string &, bool) override {
    return 0;
  }

  int flush() override {
    return 0;
  }

  int close() override {
    return 0;
  }

  int append(const std::string &, size_t) override {
    return IndexError_NotImplemented;
  }

  void refresh(uint64_t) override {}

  uint64_t check_point() const override {
    return 0;
  }

  Segment::Pointer get(const std::string &id, int = -1) override {
    const auto it = segments_.find(id);
    return it == segments_.end() ? nullptr : it->second;
  }

  bool has(const std::string &id) const override {
    return segments_.find(id) != segments_.end();
  }

  uint32_t magic() const override {
    return 0;
  }

 private:
  std::map<std::string, Segment::Pointer> segments_;
};

template <typename T>
void AppendBytes(std::vector<uint8_t> *bytes, const T &value) {
  const auto *begin = reinterpret_cast<const uint8_t *>(&value);
  bytes->insert(bytes->end(), begin, begin + sizeof(value));
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

std::shared_ptr<VectorStorage> MakeMinimalEntityStorage(
    const std::vector<uint32_t> &chunk_offsets,
    const std::vector<diskann_id_t> &key_mapping,
    const std::vector<diskann_key_t> &keys = {42},
    uint64_t declared_pivot_size = 2 * sizeof(float) *
                                   PQTable::kPQCentroidNum) {
  constexpr uint32_t kDimension = 2;
  constexpr uint64_t kChunkCount = 2;
  const uint64_t document_count = keys.size();

  auto storage = std::make_shared<VectorStorage>();

  DiskAnnMetaHeader header;
  header.doc_cnt = document_count;
  header.ndims = kDimension;
  storage->add(DiskAnnEntity::kDiskAnnMetaSegmentId,
               ToBytes(std::vector<DiskAnnMetaHeader>{header}));

  DiskAnnPqMeta pq_meta;
  pq_meta.full_pivot_data_size = declared_pivot_size;
  pq_meta.centroid_data_size = kDimension * sizeof(float);
  // Legacy indexes leave this field at zero, so the valid case deliberately
  // exercises that compatibility path.
  pq_meta.chunk_offsets_size = 0;
  pq_meta.chunk_num = kChunkCount;
  std::vector<uint8_t> pq_meta_data;
  AppendBytes(&pq_meta_data, pq_meta);
  pq_meta_data.resize(pq_meta_data.size() +
                          kDimension * sizeof(float) * PQTable::kPQCentroidNum +
                          kDimension * sizeof(float),
                      0);
  const std::vector<uint8_t> chunk_offset_bytes = ToBytes(chunk_offsets);
  pq_meta_data.insert(pq_meta_data.end(), chunk_offset_bytes.begin(),
                      chunk_offset_bytes.end());
  storage->add(DiskAnnEntity::kDiskAnnPqMetaSegmentId, std::move(pq_meta_data));
  storage->add(DiskAnnEntity::kDiskAnnPqDataSegmentId,
               std::vector<uint8_t>(document_count * kChunkCount, 0));

  storage->add(DiskAnnEntity::kDiskAnnKeySegmentId, ToBytes(keys));
  storage->add(DiskAnnEntity::kDiskAnnKeyMappingSegmentId,
               ToBytes(key_mapping));
  storage->add(DiskAnnEntity::kDiskAnnEntryPointSegmentId,
               ToBytes(std::vector<uint32_t>{0}));
  storage->add(DiskAnnEntity::kDiskAnnVectorSegmentId, {});
  return storage;
}

TEST(DiskAnnMobileCompatTest, AlignedAllocationSupportsUnroundedSize) {
  constexpr size_t kSize = 400;
  constexpr size_t kAlignment = 256;

  void *buffer = nullptr;
  DiskAnnUtil::alloc_aligned(&buffer, kSize, kAlignment);

  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer) % kAlignment, 0u);
  std::memset(buffer, 0xa5, kSize);
  DiskAnnUtil::free_aligned(buffer);
}

template <typename T>
void ExpectExactPqPivotCopy(IndexMeta::DataType data_type) {
  constexpr uint32_t kDimension = 4;
  constexpr uint32_t kCenterCount = 2;
  constexpr uint32_t kChunkCount = 2;
  const std::vector<uint32_t> chunk_dims{2, 2};
  const std::vector<uint32_t> chunk_offsets{0, 2, 4};
  const std::array<std::array<float, 2>, 4> values{{
      {{1.0F, 2.0F}},
      {{5.0F, 6.0F}},
      {{3.0F, 4.0F}},
      {{7.0F, 8.0F}},
  }};

  IndexCluster::CentroidList centroids(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const std::array<T, 2> feature{{T(values[i][0]), T(values[i][1])}};
    centroids[i].set_feature(feature.data(), sizeof(feature));
  }

  IndexMeta meta(data_type, kDimension);
  std::vector<uint8_t> pivots;
  ASSERT_EQ(DiskAnnPqTrainer::convert_pivot_data<T>(
                meta, kCenterCount, kChunkCount, chunk_dims, chunk_offsets,
                centroids, pivots),
            0);
  ASSERT_EQ(pivots.size(), kCenterCount * meta.element_size());

  std::array<T, kCenterCount * kDimension> actual{};
  std::memcpy(actual.data(), pivots.data(), pivots.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(actual[i]), static_cast<float>(i + 1));
  }
}

TEST(DiskAnnMobileCompatTest, PqPivotConversionCopiesExactChunkWidths) {
  ExpectExactPqPivotCopy<float>(IndexMeta::DataType::DT_FP32);
  ExpectExactPqPivotCopy<ailego::Float16>(IndexMeta::DataType::DT_FP16);
}

TEST(DiskAnnMobileCompatTest, MinimalEntityUsesValidatedTypedKeyStorage) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 2);
  // Two uint64_t keys and two uint32_t mapping entries both fit in libc++'s
  // small-string storage. This specifically exercises the layout that was
  // unsafe when these buffers were strings cast to typed pointers.
  auto storage = MakeMinimalEntityStorage({0, 1, 2}, {1, 0}, {84, 42});

  DiskAnnSearcherEntity entity;
  ASSERT_EQ(entity.load(meta, storage), 0);
  EXPECT_EQ(entity.get_id(42), 1u);
  EXPECT_EQ(entity.get_id(84), 0u);
  EXPECT_EQ(entity.get_id(41), kInvalidId);
  EXPECT_EQ(entity.get_key(0), 84u);
  EXPECT_EQ(entity.get_key(1), 42u);
  EXPECT_EQ(entity.get_key(2), kInvalidKey);

  const auto cloned = entity.clone();
  ASSERT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->get_id(42), 1u);
  EXPECT_EQ(cloned->get_key(0), 84u);
}

TEST(DiskAnnMobileCompatTest, EntityAllowsMultipleInvalidKeySlots) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 2);
  auto storage = MakeMinimalEntityStorage({0, 1, 2}, {0, 1, 2},
                                          {42, kInvalidKey, kInvalidKey});

  DiskAnnSearcherEntity entity;
  ASSERT_EQ(entity.load(meta, storage), 0);
  EXPECT_EQ(entity.get_id(42), 0u);
  EXPECT_EQ(entity.get_id(kInvalidKey), kInvalidId);
  EXPECT_EQ(entity.get_key(1), kInvalidKey);
  EXPECT_EQ(entity.get_key(2), kInvalidKey);
}

TEST(DiskAnnMobileCompatTest, EntityRejectsMalformedMetadata) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 2);

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage(
        {0, 1, 2}, {0}, {42}, 2 * sizeof(float) * PQTable::kPQCentroidNum - 1);
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage(
        {0, 1, 2}, {0}, {42}, std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage({0, 2, 2}, {0});
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage(
        {0, 1, 2}, {std::numeric_limits<diskann_id_t>::max()});
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage({0, 1, 2}, {0, 0}, {42, 84});
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage({0, 1, 2}, {0});
    DiskAnnMetaHeader header;
    header.doc_cnt = 1;
    header.ndims = 3;
    storage->add(DiskAnnEntity::kDiskAnnMetaSegmentId,
                 ToBytes(std::vector<DiskAnnMetaHeader>{header}));
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage({0, 1, 2}, {0});
    storage->add(DiskAnnEntity::kDiskAnnEntryPointSegmentId,
                 ToBytes(std::vector<uint32_t>{1, 1}));
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage({0, 1, 2}, {0});
    storage->add(DiskAnnEntity::kDiskAnnEntryPointSegmentId,
                 ToBytes(std::vector<uint32_t>{2, 0, 0}));
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }

  {
    DiskAnnSearcherEntity entity;
    auto storage = MakeMinimalEntityStorage({0, 1, 2}, {1, 0}, {42, 84});
    EXPECT_EQ(entity.load(meta, storage), IndexError_InvalidFormat);
  }
}

TEST(DiskAnnMobileCompatTest, PortableReaderReadsAlignedBatch) {
  constexpr size_t kBlockSize = 4096;
  constexpr size_t kBlockCount = 2;
  constexpr size_t kDataSize = kBlockSize * kBlockCount;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> expected(kDataSize);
  std::fill(expected.begin(), expected.begin() + kBlockSize, 0x3c);
  std::fill(expected.begin() + kBlockSize, expected.end(), 0xc3);
  ASSERT_EQ(WriteAt(file.fd(), expected.data(), expected.size(), 0),
            static_cast<int64_t>(expected.size()));
  file.close();

  void *output = nullptr;
  DiskAnnUtil::alloc_aligned(&output, kDataSize, kBlockSize);
  ASSERT_NE(output, nullptr);
  std::memset(output, 0, kDataSize);

  PlatformAlignedFileReader reader;
  reader.open(file.path());
  IOContext context{};
  std::vector<AlignedRead> requests;
  requests.emplace_back(0, kBlockSize, output);
  requests.emplace_back(kBlockSize, kBlockSize,
                        static_cast<uint8_t *>(output) + kBlockSize);

  EXPECT_EQ(reader.read(requests, context), 0);
  EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

  reader.close();
  DiskAnnUtil::free_aligned(output);
}

TEST(DiskAnnMobileCompatTest, PortableReaderRejectsShortRead) {
  constexpr size_t kBlockSize = 4096;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> expected(kBlockSize, 0x5a);
  ASSERT_EQ(WriteAt(file.fd(), expected.data(), expected.size(), 0),
            static_cast<int64_t>(expected.size()));
  file.close();

  void *output = nullptr;
  DiskAnnUtil::alloc_aligned(&output, kBlockSize * 2, kBlockSize);
  ASSERT_NE(output, nullptr);

  PlatformAlignedFileReader reader;
  reader.open(file.path());
  IOContext context{};
  std::vector<AlignedRead> requests;
  requests.emplace_back(0, kBlockSize * 2, output);

  EXPECT_NE(reader.read(requests, context), 0);

  requests.clear();
  requests.emplace_back(0, kBlockSize, output);
  EXPECT_EQ(reader.read(requests, context), 0);
  EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

  reader.close();
  DiskAnnUtil::free_aligned(output);
}

TEST(DiskAnnMobileCompatTest, PortableReaderRecoversAfterOpenFailure) {
  constexpr size_t kBlockSize = 4096;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> expected(kBlockSize, 0x6b);
  ASSERT_EQ(WriteAt(file.fd(), expected.data(), expected.size(), 0),
            static_cast<int64_t>(expected.size()));
  file.close();

  void *output = nullptr;
  DiskAnnUtil::alloc_aligned(&output, kBlockSize, kBlockSize);
  ASSERT_NE(output, nullptr);

  PlatformAlignedFileReader reader;
  reader.open("DiskAnnMobileCompatTest.missing");
  IOContext context{};
  std::vector<AlignedRead> requests;
  requests.emplace_back(0, kBlockSize, output);
  EXPECT_NE(reader.read(requests, context), 0);

  reader.open(file.path());
  EXPECT_EQ(reader.read(requests, context), 0);
  EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

  reader.close();
  DiskAnnUtil::free_aligned(output);
}

TEST(DiskAnnMobileCompatTest, PortableReaderSupportsConcurrentReads) {
  constexpr size_t kBlockSize = 4096;
  constexpr size_t kThreadCount = 4;
  constexpr size_t kDataSize = kBlockSize * kThreadCount;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> expected(kDataSize);
  for (size_t i = 0; i < kThreadCount; ++i) {
    std::fill(expected.begin() + i * kBlockSize,
              expected.begin() + (i + 1) * kBlockSize,
              static_cast<uint8_t>(i + 1));
  }
  ASSERT_EQ(WriteAt(file.fd(), expected.data(), expected.size(), 0),
            static_cast<int64_t>(expected.size()));
  file.close();

  std::array<void *, kThreadCount> outputs{};
  for (void *&output : outputs) {
    DiskAnnUtil::alloc_aligned(&output, kBlockSize, kBlockSize);
    ASSERT_NE(output, nullptr);
  }

  PlatformAlignedFileReader reader;
  reader.open(file.path());
  std::array<int, kThreadCount> statuses{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      IOContext context{};
      std::vector<AlignedRead> requests;
      requests.emplace_back(i * kBlockSize, kBlockSize, outputs[i]);
      statuses[i] = reader.read(requests, context);
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  for (size_t i = 0; i < kThreadCount; ++i) {
    EXPECT_EQ(statuses[i], 0);
    EXPECT_EQ(
        std::memcmp(outputs[i], expected.data() + i * kBlockSize, kBlockSize),
        0);
    DiskAnnUtil::free_aligned(outputs[i]);
  }
  reader.close();
}

TEST(DiskAnnMobileCompatTest, BuildDumpLoadAndSearch) {
  constexpr size_t kDimension = 10;
  constexpr size_t kDocCount = 64;
  constexpr uint64_t kExpectedKey = 12;

  TemporaryFile index_file;
  ASSERT_GE(index_file.fd(), 0);
  index_file.release_descriptor_and_unlink();

  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  for (size_t i = 0; i < kDocCount; ++i) {
    ailego::NumericalVector<float> vector(kDimension, static_cast<float>(i));
    ASSERT_TRUE(holder->emplace(i, vector));
  }

  ailego::Params build_params;
  build_params.set("zvec.diskann.builder.max_degree", 16);
  build_params.set("zvec.diskann.builder.list_size", 32);
  build_params.set("zvec.diskann.builder.max_pq_chunk_num", 2);
  build_params.set("zvec.diskann.builder.threads", 2);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);
  ASSERT_EQ(builder->init(meta, build_params), 0);
  ASSERT_EQ(builder->train(holder), 0);
  ASSERT_EQ(builder->build(holder), 0);

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(dumper->create(index_file.path()), 0);
  ASSERT_EQ(builder->dump(dumper), 0);
  ASSERT_EQ(dumper->close(), 0);

  std::ifstream snapshot_input(index_file.path(),
                               std::ios::binary | std::ios::ate);
  ASSERT_TRUE(snapshot_input.is_open());
  const std::streamsize snapshot_size = snapshot_input.tellg();
  ASSERT_GT(snapshot_size, 4096);
  snapshot_input.seekg(0);
  std::vector<uint8_t> snapshot(static_cast<size_t>(snapshot_size));
  ASSERT_TRUE(snapshot_input.read(reinterpret_cast<char *>(snapshot.data()),
                                  snapshot_size));
  snapshot_input.close();

  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 64);
  ASSERT_EQ(searcher->init(search_params), 0);

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(storage->open(index_file.path(), false), 0);
  ASSERT_EQ(searcher->load(storage, IndexMetric::Pointer()), 0);

  auto context = searcher->create_context();
  ASSERT_NE(context, nullptr);
  context->set_topk(5);

  ailego::NumericalVector<float> query(kDimension, 12.1f);
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDimension);
  ASSERT_EQ(searcher->search_impl(query.data(), query_meta, context), 0);

  const auto &result = context->result();
  ASSERT_FALSE(result.empty());
  EXPECT_NE(
      std::find_if(result.begin(), result.end(),
                   [](const auto &item) { return item.key() == kExpectedKey; }),
      result.end());

  IndexStreamer::Pointer first_streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(first_streamer, nullptr);
  ASSERT_EQ(first_streamer->init(meta, search_params), 0);
  auto first_streamer_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(first_streamer_storage, nullptr);
  ASSERT_EQ(first_streamer_storage->open(index_file.path(), false), 0);
  ASSERT_EQ(first_streamer->open(first_streamer_storage), 0);

  IndexStreamer::Pointer second_streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(second_streamer, nullptr);
  ASSERT_EQ(second_streamer->init(meta, search_params), 0);
  auto second_streamer_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(second_streamer_storage, nullptr);
  ASSERT_EQ(second_streamer_storage->open(index_file.path(), false), 0);
  ASSERT_EQ(second_streamer->open(second_streamer_storage), 0);

  auto switching_context = first_streamer->create_context();
  ASSERT_NE(switching_context, nullptr);
  switching_context->set_topk(5);
  switching_context->set_filter(
      [](uint64_t key) { return key != kExpectedKey; });
  ASSERT_EQ(
      second_streamer->search_impl(query.data(), query_meta, switching_context),
      0);
  ASSERT_EQ(switching_context->result().size(), 1u);
  EXPECT_EQ(switching_context->result().front().key(), kExpectedKey);

  switching_context.reset();
  ASSERT_EQ(first_streamer->close(), 0);
  ASSERT_EQ(second_streamer->close(), 0);
  first_streamer.reset();
  second_streamer.reset();
  context.reset();
  searcher.reset();
  storage.reset();

  ASSERT_NO_THROW(
      std::filesystem::resize_file(index_file.path(), snapshot.size() - 4096));
  searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(searcher->init(search_params), 0);
  storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  int corrupt_open_result = storage->open(index_file.path(), false);
  bool corrupt_index_rejected = corrupt_open_result != 0;
  if (corrupt_open_result == 0) {
    corrupt_index_rejected =
        searcher->load(storage, IndexMetric::Pointer()) != 0;
  }
  EXPECT_TRUE(corrupt_index_rejected);

  searcher.reset();
  storage.reset();
  std::ofstream restore_output(index_file.path(),
                               std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(restore_output.is_open());
  restore_output.write(reinterpret_cast<const char *>(snapshot.data()),
                       static_cast<std::streamsize>(snapshot.size()));
  restore_output.flush();
  ASSERT_TRUE(restore_output.good());
  restore_output.close();

  searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(searcher->init(search_params), 0);
  storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(storage->open(index_file.path(), false), 0);
  ASSERT_EQ(searcher->load(storage, IndexMetric::Pointer()), 0);
  context = searcher->create_context();
  ASSERT_NE(context, nullptr);
  context->set_topk(5);
  ASSERT_EQ(searcher->search_impl(query.data(), query_meta, context), 0);
  EXPECT_NE(
      std::find_if(context->result().begin(), context->result().end(),
                   [](const auto &item) { return item.key() == kExpectedKey; }),
      context->result().end());
}

}  // namespace
}  // namespace zvec::core

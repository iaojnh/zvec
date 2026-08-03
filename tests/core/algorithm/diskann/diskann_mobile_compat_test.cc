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

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_builder.h"
#include "diskann_file_reader.h"
#include "diskann_util.h"

namespace zvec::core {
namespace {

class TemporaryFile {
 public:
  TemporaryFile() : fd_(::mkstemp(path_)) {}

  ~TemporaryFile() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    ::unlink(path_);
  }

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

  int fd() const {
    return fd_;
  }

  const char *path() const {
    return path_;
  }

  void release_descriptor_and_unlink() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    ::unlink(path_);
  }

 private:
  char path_[64] = "DiskAnnMobileCompatTest.XXXXXX";
  int fd_{-1};
};

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

TEST(DiskAnnMobileCompatTest, PortableReaderReadsAlignedBatch) {
  constexpr size_t kBlockSize = 4096;
  constexpr size_t kBlockCount = 2;
  constexpr size_t kDataSize = kBlockSize * kBlockCount;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> expected(kDataSize);
  std::fill(expected.begin(), expected.begin() + kBlockSize, 0x3c);
  std::fill(expected.begin() + kBlockSize, expected.end(), 0xc3);
  ASSERT_EQ(::pwrite(file.fd(), expected.data(), expected.size(), 0),
            static_cast<ssize_t>(expected.size()));

  void *output = nullptr;
  DiskAnnUtil::alloc_aligned(&output, kDataSize, kBlockSize);
  ASSERT_NE(output, nullptr);
  std::memset(output, 0, kDataSize);

  LinuxAlignedFileReader reader;
  reader.open(file.path());
  IOContext context{0};
  std::vector<AlignedRead> requests;
  requests.emplace_back(0, kBlockSize, output);
  requests.emplace_back(kBlockSize, kBlockSize,
                        static_cast<uint8_t *>(output) + kBlockSize);

  EXPECT_EQ(reader.read(requests, context), 0);
  EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

  reader.close();
  DiskAnnUtil::free_aligned(output);
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
}

}  // namespace
}  // namespace zvec::core

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
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_unpacker.h>
#include "algorithm/streamer_layout.h"

using namespace zvec;
using namespace zvec::core;

TEST(PortableFormat, CanonicalBytesAndAlignmentOverflow) {
  DiskUInt64 value = 0x0123456789abcdefULL;
  const std::array<uint8_t, 8> expected{0xef, 0xcd, 0xab, 0x89,
                                        0x67, 0x45, 0x23, 0x01};
  EXPECT_EQ(0, std::memcmp(&value, expected.data(), expected.size()));
  EXPECT_EQ(0x0123456789abcdefULL, static_cast<uint64_t>(value));
  uint64_t aligned = 0;
  ASSERT_TRUE(AlignDiskSize(4097, &aligned));
  EXPECT_EQ(8192u, aligned);
  ASSERT_TRUE(AlignDiskSize(UINT64_MAX - 4095, &aligned));
  EXPECT_EQ(UINT64_MAX - 4095, aligned);
  EXPECT_FALSE(AlignDiskSize(UINT64_MAX - 4094, &aligned));
  EXPECT_FALSE(AlignDiskSize(UINT64_MAX, &aligned));

  IndexFormat::MetaHeader header{};
  IndexFormat::SetupMetaHeader(&header, 8064, 8192);
  const auto *bytes = reinterpret_cast<const uint8_t *>(&header);
  EXPECT_EQ(3, bytes[6]);
  EXPECT_EQ(0, bytes[7]);
  EXPECT_EQ(0, bytes[16]);
  EXPECT_EQ(16, bytes[17]);  // 4096, little endian
  EXPECT_EQ(0, std::memcmp(bytes + 40, "ZVECIDX3", 8));
  EXPECT_TRUE(IndexFormat::Supported(header));
}

TEST(PortableFormat, MappingAllocates4KiBAndReopensDirectoryChains) {
  const std::string path = "portable_mapping.index";
  IndexMapping mapping;
  ASSERT_EQ(0, mapping.create(path, 256));
  EXPECT_EQ(8192u, mapping.index_size());
  for (size_t i = 0; i < 150; ++i) {
    const std::string id = "segment-" + std::to_string(i);
    ASSERT_EQ(0, mapping.append(id, 4097));
    auto *segment = mapping.map(id, false, false);
    ASSERT_NE(nullptr, segment);
    EXPECT_EQ(8192u,
              segment->meta()->data_size + segment->meta()->padding_size);
    EXPECT_EQ(0u, segment->meta()->data_index % kDiskAlignment);
    static_cast<uint8_t *>(segment->data())[0] = i;
    segment->meta()->data_size = 1;
    segment->meta()->padding_size = 8191;
    segment->set_dirty();
  }
  const size_t size = mapping.index_size();
  EXPECT_NE(0, mapping.append("overflow", SIZE_MAX));
  EXPECT_EQ(size, mapping.index_size());
  mapping.refresh(7);
  ASSERT_EQ(0, mapping.flush());
  mapping.close();
  ASSERT_EQ(0, mapping.open(path, false, false));
  for (size_t i = 0; i < 150; ++i) {
    auto *segment = mapping.map("segment-" + std::to_string(i), false, false);
    ASSERT_NE(nullptr, segment);
    EXPECT_EQ(i, static_cast<const uint8_t *>(segment->data())[0]);
  }
  mapping.close();
  EXPECT_TRUE(ailego::File::Delete(path));
}

TEST(PortableFormat, MMapAndBufferStorageShare4KiBExtents) {
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(64 * 1024 * 1024));
  const std::string path = "portable_backends.index";
  IndexMapping initial;
  ASSERT_EQ(0, initial.create(path, 256));
  initial.close();
  size_t count = 0;
  for (const char *backend :
       {"MMapFileStorage", "BufferStorage", "MMapFileStorage"}) {
    SCOPED_TRACE(backend);
    auto storage = IndexFactory::CreateStorage(backend);
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params{}));
    ASSERT_EQ(0, storage->open(path, true));
    const size_t end = count + 70;
    for (; count < end; ++count) {
      const std::string id = "part-" + std::to_string(count);
      ASSERT_EQ(0, storage->append(id, 4096));
      auto segment = storage->get(id);
      ASSERT_NE(nullptr, segment);
      ASSERT_EQ(4096u, segment->capacity());
      const std::string bytes(4096, static_cast<char>(count));
      ASSERT_EQ(bytes.size(), segment->write(0, bytes.data(), bytes.size()));
    }
    for (size_t i = 0; i < count; ++i) {
      auto segment = storage->get("part-" + std::to_string(i));
      ASSERT_NE(nullptr, segment);
      std::string bytes(4096, '\0');
      ASSERT_EQ(bytes.size(), segment->fetch(0, bytes.data(), bytes.size()));
      EXPECT_EQ(std::string(4096, static_cast<char>(i)), bytes);
    }
    storage->refresh(0);
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
    ailego::File file;
    ASSERT_TRUE(file.open(path, true));
    EXPECT_EQ(0u, file.size() % kDiskAlignment);
  }
  EXPECT_TRUE(ailego::File::Delete(path));
}

TEST(PortableFormat, RejectsOldVersionAndUnknownRequiredFeatures) {
  const std::string path = "portable_rejected.index";
  IndexMapping mapping;
  ASSERT_EQ(0, mapping.create(path, 256));
  mapping.close();
  for (bool old_version : {true, false}) {
    ailego::File file;
    ASSERT_TRUE(file.open(path, false));
    const auto original_size = file.size();
    IndexFormat::MetaHeader header{};
    ASSERT_EQ(sizeof(header), file.read(0, &header, sizeof(header)));
    header.version = old_version ? 2 : 3;
    header.required_features = old_version ? 1 : 3;
    IndexFormat::UpdateMetaHeader(&header);
    ASSERT_EQ(sizeof(header), file.write(0, &header, sizeof(header)));
    file.close();
    EXPECT_EQ(IndexError_Unsupported, mapping.open(path, false, false));
    mapping.close();
    ASSERT_TRUE(file.open(path, true));
    EXPECT_EQ(original_size, file.size());
    IndexFormat::MetaHeader after{};
    ASSERT_EQ(sizeof(after), file.read(0, &after, sizeof(after)));
    EXPECT_EQ(0, std::memcmp(&header, &after, sizeof(after)));
  }
  EXPECT_TRUE(ailego::File::Delete(path));
}

TEST(PortableFormat, DumpedSegmentsHaveCanonicalOffsets) {
  const std::string path = "portable_dump.index";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->create(path));
  for (size_t size : {1u, 4096u, 4097u}) {
    std::string payload(size, 'x');
    ASSERT_EQ(size, dumper->write(payload.data(), size));
    ASSERT_EQ(0, dumper->append(std::to_string(size), size, 0, 0));
  }
  ASSERT_EQ(0, dumper->close());
  ailego::File file;
  ASSERT_TRUE(file.open(path, true));
  std::string bytes(file.size(), '\0');
  ASSERT_EQ(bytes.size(), file.read(0, bytes.data(), bytes.size()));
  IndexUnpacker unpacker;
  ASSERT_TRUE(unpacker.unpack(
      [&bytes](size_t offset, const void **data, size_t len) -> size_t {
        if (offset > bytes.size() || len > bytes.size() - offset) return 0;
        *data = bytes.data() + offset;
        return len;
      },
      bytes.size(), true));
  for (const auto &entry : unpacker.segments()) {
    EXPECT_EQ(0u, entry.second.data_offset() % kDiskAlignment);
    EXPECT_EQ(0u, (entry.second.data_size() + entry.second.padding_size()) %
                      kDiskAlignment);
  }
  file.close();
  EXPECT_TRUE(ailego::File::Delete(path));
}

TEST(PortableFormat, RejectingOldFileDoesNotExtendIt) {
  const std::string path = "portable_old_partial.index";
  const std::string original(217, 'x');
  ailego::File file;
  ASSERT_TRUE(file.create(path, original.size()));
  ASSERT_EQ(original.size(), file.write(original.data(), original.size()));
  file.close();
  auto storage = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  EXPECT_EQ(IndexError_Unsupported, storage->open(path, true));
  ASSERT_EQ(0, storage->close());
  ASSERT_TRUE(file.open(path, true));
  ASSERT_EQ(original.size(), file.size());
  std::string after(original.size(), '\0');
  ASSERT_EQ(after.size(), file.read(0, after.data(), after.size()));
  EXPECT_EQ(original, after);
  file.close();
  EXPECT_TRUE(ailego::File::Delete(path));
}

TEST(PortableFormat, GraphLayoutIsFrozenAndChecksummed) {
  const std::string path = "portable_layout.index";
  for (auto algorithm : {StreamerLayout::Hnsw, StreamerLayout::HnswSparse,
                         StreamerLayout::Vamana}) {
    SCOPED_TRACE(algorithm);
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params{}));
    ASSERT_EQ(0, storage->open(path, true));
    const bool upper = algorithm != StreamerLayout::Vamana;
    StreamerLayout created{};
    ASSERT_EQ(
        0, StreamerLayout::Bind(storage, algorithm, 4096, 352, upper ? 64 : 0,
                                upper ? 10 : 0, 15, &created));
    EXPECT_EQ(16u, static_cast<uint32_t>(created.node_count));
    EXPECT_EQ(8192u, static_cast<uint32_t>(created.node_chunk_size));
    storage->refresh(0);
    ASSERT_EQ(0, storage->flush());
    ASSERT_EQ(0, storage->close());
    ASSERT_EQ(0, storage->open(path, false));
    StreamerLayout loaded{};
    ASSERT_EQ(
        0, StreamerLayout::Bind(storage, algorithm, 16384, 352, upper ? 64 : 0,
                                upper ? 10 : 0, 15, &loaded));
    EXPECT_EQ(0, std::memcmp(&created, &loaded, sizeof(created)));
    auto segment = storage->get(StreamerLayout::kSegment);
    ASSERT_NE(nullptr, segment);
    loaded.node_count = 8;
    ASSERT_EQ(sizeof(loaded), segment->write(0, &loaded, sizeof(loaded)));
    EXPECT_EQ(
        IndexError_Mismatch,
        StreamerLayout::Bind(storage, algorithm, 4096, 352, upper ? 64 : 0,
                             upper ? 10 : 0, 15, &loaded));
    ASSERT_EQ(0, storage->close());
    EXPECT_TRUE(ailego::File::Delete(path));
  }
}

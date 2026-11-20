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
#include <string>
#include <vector>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_memory.h>
#include <zvec/core/framework/index_packer.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec {
namespace core {
namespace {

std::string Pack(const std::string &id, const std::string &data) {
  std::string bytes;
  auto write = [&](const void *buf, size_t size) {
    bytes.append(static_cast<const char *>(buf), size);
    return size;
  };
  IndexPacker packer;
  EXPECT_TRUE(packer.setup(write));
  EXPECT_EQ(data.size(), packer.pack(write, data.data(), data.size()));
  std::vector<IndexPacker::SegmentMeta> table{{id, data.size(), 0, 0}};
  EXPECT_TRUE(packer.finish(write, table));
  return bytes;
}

class IndexMetadataOpenTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0,
              ailego::PoolPayloadBudget::get_instance().init(16 * 1024 * 1024));
  }

  void TearDown() override {
    for (const auto *path : {kGood, kBad}) {
      ailego::File::Delete(path);
      IndexMemory::Instance()->remove(path);
    }
    EXPECT_EQ(0u, pool_.external_used());
  }

  void Write(const char *path, const std::string &data) {
    ailego::File file;
    ASSERT_TRUE(file.create(path, 0));
    ASSERT_EQ(data.size(), file.write(data.data(), data.size()));
    auto rope = IndexMemory::Instance()->create(path);
    ASSERT_EQ(data.size(), rope->append(0).append(data.data(), data.size()));
  }

  static constexpr const char *kGood = "index_metadata_open_good";
  static constexpr const char *kBad = "index_metadata_open_bad";
  ailego::MemoryLimitPool &pool_ = ailego::MemoryLimitPool::get_instance();
};

TEST_F(IndexMetadataOpenTest, FailedReopenKeepsNestedReadersAndMapping) {
  const auto child = Pack("data", "nested");
  const auto outer = Pack("child", child);
  auto corrupt = outer;
  corrupt[corrupt.size() - sizeof(IndexFormat::MetaFooter)] ^= 1;
  Write(kGood, outer);
  Write(kBad, corrupt);

  for (const auto *name : {"FileReadStorage", "MMapFileReadStorage",
                           "MemoryReadStorage", "BufferReadStorage"}) {
    SCOPED_TRACE(name);
    auto storage = IndexFactory::CreateStorage(name);
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params{}));
    ASSERT_EQ(0, storage->open(kGood, false));
    const auto magic = storage->magic();
    auto parent = storage->get("child");
    ASSERT_NE(nullptr, parent);
    IndexSegmentStorage nested(parent);
    const auto cold_pages = pool_.stats().page_used;
    ASSERT_EQ(0, nested.open("", false));
    EXPECT_EQ(cold_pages, pool_.stats().page_used);
    ASSERT_TRUE(nested.has("data"));

    IndexStorage::MemoryBlock active;
    ASSERT_EQ(64u, parent->read(0, active, 64));
    const auto warm_pages = pool_.stats().page_used;
    ASSERT_EQ(0, nested.open("", false));
    EXPECT_EQ(warm_pages, pool_.stats().page_used);
    EXPECT_EQ(IndexError_InvalidChecksum, storage->open(kBad, false));
    EXPECT_EQ(magic, storage->magic());
    EXPECT_TRUE(storage->has("child"));
    EXPECT_EQ(0, std::memcmp(child.data(), active.data(), 64));

    auto result = nested.get("data", -1);
    ASSERT_NE(nullptr, result);
    IndexStorage::MemoryBlock contents;
    ASSERT_EQ(6u, result->read(0, contents, 6));
    EXPECT_EQ(0, std::memcmp("nested", contents.data(), 6));
    ASSERT_EQ(0, storage->open(kGood, false));
    EXPECT_EQ(0, std::memcmp(child.data(), active.data(), 64));
  }
}

TEST_F(IndexMetadataOpenTest, BudgetRefusalPreservesStateAndAllowsRetry) {
  Write(kGood, Pack("child", Pack("data", "nested")));
  Write(kBad, Pack(std::string(8192, 'x'), "candidate"));

  auto storage = IndexFactory::CreateStorage("BufferReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params{}));
  ASSERT_EQ(0, storage->open(kGood, false));
  auto parent = storage->get("child");
  ASSERT_NE(nullptr, parent);
  IndexStorage::MemoryBlock active;
  ASSERT_EQ(64u, parent->read(0, active, 64));
  IndexSegmentStorage nested(parent);
  ASSERT_EQ(0, nested.open("", false));

  {
    const size_t available = pool_.available();
    ASSERT_GT(available, 1024u);
    const size_t charge = available - 1024;
    ASSERT_TRUE(pool_.try_charge_workspace(charge));
    AILEGO_DEFER([&] { pool_.release_external(charge); });
    const auto before = pool_.external_used();

    EXPECT_EQ(IndexError_NoMemory, storage->open(kBad, false));
    EXPECT_EQ(before, pool_.external_used());
    EXPECT_TRUE(storage->has("child"));
    EXPECT_EQ(0, nested.open("", false));
    EXPECT_EQ(before, pool_.external_used());
    const auto fill = pool_.available();
    ASSERT_TRUE(pool_.try_charge_workspace(fill));
    AILEGO_DEFER([&] { pool_.release_external(fill); });
    EXPECT_EQ(IndexError_NoMemory, nested.open("", false));
    EXPECT_TRUE(nested.has("data"));
    EXPECT_EQ(before + fill, pool_.external_used());
  }

  EXPECT_EQ(0u, pool_.external_used());
  EXPECT_EQ(0, storage->open(kBad, false));
  EXPECT_TRUE(storage->has(std::string(8192, 'x')));
}

TEST_F(IndexMetadataOpenTest, FileAndMmapOpenWithSharedBudgetDisabled) {
  ASSERT_EQ(0, pool_.init(0));
  Write(kGood, Pack("child", Pack("data", "nested")));
  for (const auto *name : {"FileReadStorage", "MMapFileReadStorage",
                           "MemoryReadStorage", "BufferReadStorage"}) {
    SCOPED_TRACE(name);
    auto storage = IndexFactory::CreateStorage(name);
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params{}));
    ASSERT_EQ(0, storage->open(kGood, false));
    ASSERT_TRUE(storage->has("child"));
    IndexSegmentStorage nested(storage->get("child"));
    EXPECT_EQ(0, nested.open("", false));
    EXPECT_TRUE(nested.has("data"));
  }
}

TEST_F(IndexMetadataOpenTest,
       ChecksumReadsReuseOneBufferAndShortReadRollsBack) {
  const std::string payload(4096 * 16, 'a');
  const auto bytes = Pack("data", payload);
  IndexUnpacker unpacker;
  const void *crc_buffer = nullptr;
  size_t crc_reads = 0;
  auto fetch = [&](size_t offset, void *buf, size_t len) {
    if (len == 4096) {
      if (!crc_buffer) {
        crc_buffer = buf;
      }
      EXPECT_EQ(crc_buffer, buf);
      EXPECT_EQ(4096u, pool_.external_used());
      ++crc_reads;
    }
    std::memcpy(buf, bytes.data() + offset, len);
    return len;
  };

  ASSERT_EQ(0, unpacker.unpack_fetch(fetch, bytes.size(), true));
  EXPECT_EQ(16u, crc_reads);
  EXPECT_EQ(0u, pool_.external_used());
  const auto magic = unpacker.magic();
  auto short_fetch = [&](size_t offset, void *buf, size_t len) {
    return len == 4096 ? len - 1 : fetch(offset, buf, len);
  };
  EXPECT_EQ(IndexError_ReadData,
            unpacker.unpack_fetch(short_fetch, bytes.size(), true));
  EXPECT_EQ(magic, unpacker.magic());
  EXPECT_EQ(1u, unpacker.segments().count("data"));
  EXPECT_EQ(0u, pool_.external_used());
  auto corrupt = bytes;
  corrupt[sizeof(IndexFormat::MetaHeader)] ^= 1;
  EXPECT_EQ(IndexError_InvalidChecksum,
            unpacker.unpack_view(corrupt.data(), corrupt.size(), true));
  EXPECT_EQ(0, unpacker.unpack_view(bytes.data(), bytes.size(), true));
}

TEST_F(IndexMetadataOpenTest, MutableReadersRetainStateAfterInvalidChainLink) {
  for (auto path : {kGood, kBad}) {
    IndexMapping mapping;
    ASSERT_EQ(0, mapping.create(path, 4096));
    ASSERT_EQ(0, mapping.append("data", 16));
  }
  {
    ailego::File file;
    ASSERT_TRUE(file.open(kBad, false, false));
    IndexFormat::MetaHeader header;
    ASSERT_EQ(sizeof(header), file.read(&header, sizeof(header)));
    IndexFormat::MetaFooter footer;
    ASSERT_EQ(sizeof(footer),
              file.read(header.meta_footer_offset, &footer, sizeof(footer)));
    footer.next_meta_header_offset = 1;
    IndexFormat::UpdateMetaFooter(&footer, 0);
    ASSERT_EQ(sizeof(footer),
              file.write(header.meta_footer_offset, &footer, sizeof(footer)));
  }

  for (const auto *name : {"BufferStorage", "MMapFileStorage"}) {
    SCOPED_TRACE(name);
    auto storage = IndexFactory::CreateStorage(name);
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params{}));
    ASSERT_EQ(0, storage->open(kGood, false));
    const auto magic = storage->magic();
    const auto charge = pool_.external_used();
    EXPECT_EQ(IndexError_InvalidLength, storage->open(kBad, false));
    EXPECT_EQ(magic, storage->magic());
    EXPECT_TRUE(storage->has("data"));
    EXPECT_EQ(charge, pool_.external_used());
    EXPECT_EQ(0, storage->open(kGood, false));
    EXPECT_TRUE(storage->has("data"));
  }
}

}  // namespace
}  // namespace core
}  // namespace zvec

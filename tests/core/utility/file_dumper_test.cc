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
#include <iostream>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#include <gtest/gtest.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_helper.h"
#include "zvec/core/framework/index_packer.h"
#include "zvec/core/framework/index_segment_storage.h"

using namespace zvec;
using namespace zvec::core;

TEST(FileDumper, General) {
  std::string file_path = "file_dumper_test_file";

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);

  IndexMeta meta1;
  meta1.set_trainer("index_trainer", 0, ailego::Params());
  ASSERT_EQ(0, dumper->create(file_path));
  EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper.get()));

  for (size_t i = 0; i < 10; ++i) {
    std::string hello = "Hello world!!! #" + std::to_string(i);
    EXPECT_EQ(hello.size(), dumper->write(hello.data(), hello.size()));
    EXPECT_EQ(0, dumper->append(std::to_string(i), hello.size(), 0, 0));
  }
  ASSERT_EQ(0, dumper->close());

  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(container);

  ailego::Params params;
  params.set("proxima.mmap_file.container.memory_locking", true);
  params.set("proxima.mmap_file.container.memory_warmup", true);
  params.set("proxima.mmap_file.container.checksum_validation", true);
  ASSERT_EQ(0, container->init(params));

  IndexMeta meta2;
  EXPECT_EQ("", meta2.trainer_name());
  ASSERT_EQ(0, container->open(file_path, false));
  EXPECT_EQ(0, IndexHelper::DeserializeFromStorage(container.get(), &meta2));
  EXPECT_EQ("index_trainer", meta2.trainer_name());

  for (size_t i = 0; i < 10; ++i) {
    auto seg = container->get(std::to_string(i));
    IndexStorage::MemoryBlock data;
    EXPECT_EQ(seg->data_size(), seg->read(0, data, seg->data_size()));

    std::string hello = "Hello world!!! #" + std::to_string(i);
    EXPECT_EQ(hello, std::string((const char *)data.data(), seg->data_size()));
  }
}

TEST(IndexSegmentDumper, General) {
  std::string file_path = "index_segment_dumper_test_file";

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);
  ASSERT_EQ(0, dumper->create(file_path));

  {
    IndexDumper::Pointer dumper2 =
        std::make_shared<IndexSegmentDumper>(dumper, "AAAAA");

    IndexMeta meta1;
    meta1.set_trainer("index_trainer", 0, ailego::Params());
    EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper2.get()));

    for (size_t i = 0; i < 10; ++i) {
      std::string hello = "A: Hello world!!! #" + std::to_string(i);
      EXPECT_EQ(hello.size(), dumper2->write(hello.data(), hello.size()));
      EXPECT_EQ(0, dumper2->append(std::to_string(i), hello.size(), 0, 0));
    }
  }

  {
    IndexDumper::Pointer dumper2 =
        std::make_shared<IndexSegmentDumper>(dumper, "BBBBB");

    IndexMeta meta1;
    meta1.set_builder("index_builder", 0, ailego::Params());
    EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper2.get()));

    for (size_t i = 100; i < 110; ++i) {
      std::string hello = "B: Hello world!!! #" + std::to_string(i);
      EXPECT_EQ(hello.size(), dumper2->write(hello.data(), hello.size()));
      EXPECT_EQ(0, dumper2->append(std::to_string(i), hello.size(), 0, 0));
    }
  }

  {
    IndexDumper::Pointer dumper2 =
        std::make_shared<IndexSegmentDumper>(dumper, "CCCCC");

    IndexMeta meta1;
    meta1.set_converter("index_converter", 0, ailego::Params());
    EXPECT_EQ(0, IndexHelper::SerializeToDumper(meta1, dumper2.get()));

    for (size_t i = 1000; i < 1010; ++i) {
      std::string hello = "C: Hello world!!! #" + std::to_string(i);
      EXPECT_EQ(hello.size(), dumper2->write(hello.data(), hello.size()));
      EXPECT_EQ(0, dumper2->append(std::to_string(i), hello.size(), 0, 0));
    }
  }
  ASSERT_EQ(0, dumper->close());

  ///// Read data with container

  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(container);

  ailego::Params params;
  params.set("proxima.mmap_file.container.memory_locking", true);
  params.set("proxima.mmap_file.container.memory_warmup", true);
  params.set("proxima.mmap_file.container.checksum_validation", true);
  ASSERT_EQ(0, container->init(params));
  ASSERT_EQ(0, container->open(file_path, false));

  // Exercise nested unpacking with independently owned, file-backed reads.
  auto file_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(file_storage);
  ASSERT_EQ(0, file_storage->init(ailego::Params()));
  ASSERT_EQ(0, file_storage->open(file_path, false));
  auto parent = file_storage->get("AAAAA", 1);
  ASSERT_TRUE(parent);
  IndexSegmentStorage nested(parent);
  ASSERT_EQ(0, nested.open("", false));
  auto segment = nested.get("3", 0);
  ASSERT_TRUE(segment);
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(segment->data_size(),
            segment->read(0, block, segment->data_size()));
  EXPECT_EQ("A: Hello world!!! #3",
            std::string(static_cast<const char *>(block.data()),
                        segment->data_size()));

  const auto last = segment->capacity() - 1;
  char tail[2] = {};
  EXPECT_EQ(1u, segment->fetch(last, tail, sizeof(tail)));
  EXPECT_EQ('3', tail[0]);
  EXPECT_EQ(1u, segment->read(last, block, sizeof(tail)));
  EXPECT_EQ(0u, segment->read(last + 1, block, 1));
  EXPECT_EQ(nullptr, block.data());
  IndexStorage::SegmentData ranges[] = {{last, 1}, {last, 2}};
  EXPECT_FALSE(segment->read(ranges, 2));
  EXPECT_EQ(last, ranges[0].offset);
  EXPECT_EQ(nullptr, ranges[0].data);
  ranges[1] = {0, 2};
  ASSERT_TRUE(segment->read(ranges, 2));
  EXPECT_EQ("A:", std::string(static_cast<const char *>(ranges[1].data), 2));
  EXPECT_EQ(last, ranges[0].offset);
  EXPECT_EQ(0u, ranges[1].offset);

  // File handle duplication can fail independently of opening the directory.
  struct Parent : IndexSegmentStorage::Segment {
    explicit Parent(const IndexStorage::Segment::Pointer &source)
        : IndexSegmentStorage::Segment(source, 0, source->data_size(), 0, 0) {}
    IndexStorage::Segment::Pointer clone() override {
      return refuse_clone ? nullptr : shared_from_this();
    }
    bool read(IndexStorage::SegmentData *, size_t) override {
      throw std::bad_alloc();
    }
    bool refuse_clone{false};
  };
  auto controlled = std::make_shared<Parent>(parent);
  IndexSegmentStorage controlled_storage(controlled);
  ASSERT_EQ(0, controlled_storage.open("", false));
  auto child = controlled_storage.get("3", 0);
  ASSERT_TRUE(child);
  EXPECT_THROW(child->read(ranges, 2), std::bad_alloc);
  EXPECT_EQ(last, ranges[0].offset);
  EXPECT_EQ(0u, ranges[1].offset);
  controlled->refuse_clone = true;
  EXPECT_FALSE(child->clone());
  EXPECT_FALSE(controlled_storage.get("3", 0));
  EXPECT_TRUE(controlled_storage.get_all().empty());
}


TEST(IndexUnpacker, ReadsChainedPackagesAtByteOffsetsAndResets) {
  auto package = [](const std::string &id) {
    std::string bytes;
    auto write = [&](const void *data, size_t len) {
      bytes.append(static_cast<const char *>(data), len);
      return len;
    };
    IndexPacker packer;
    EXPECT_TRUE(packer.setup(write));
    EXPECT_EQ(id.size(), packer.pack(write, id.data(), id.size()));
    std::vector<IndexPacker::SegmentMeta> segments{{id, id.size(), 0, 0}};
    EXPECT_TRUE(packer.finish(write, segments));
    return bytes;
  };
  auto first = package("first");
  const auto second = package("second");
  IndexFormat::MetaHeader header;
  IndexFormat::MetaFooter footer;
  const size_t footer_offset = first.size() - sizeof(footer);
  std::memcpy(&header, first.data(), sizeof(header));
  header.meta_footer_offset = static_cast<uint32_t>(footer_offset);
  IndexFormat::UpdateMetaHeader(&header);
  std::memcpy(first.data(), &header, sizeof(header));
  std::memcpy(&footer, first.data() + footer_offset, sizeof(footer));
  footer.next_meta_header_offset = first.size() + 1;
  IndexFormat::UpdateMetaFooter(&footer, 0);
  std::memcpy(first.data() + footer_offset, &footer, sizeof(footer));
  // Every pointer into the first package is unaligned, and the second package
  // has an end-relative footer inside a chain.
  std::string bytes = " " + first + " " + second;
  IndexUnpacker unpacker;
  auto unpack = [&](size_t start, size_t length) {
    return unpacker.unpack(
        [&](size_t offset, const void **data, size_t size) {
          if (offset > length || size > length - offset) return size_t{0};
          *data = bytes.data() + start + offset;
          return size;
        },
        length, true);
  };
  ASSERT_TRUE(unpack(1, bytes.size() - 1));
  const auto it = unpacker.segments().find("second");
  ASSERT_NE(unpacker.segments().end(), it);
  const auto &entry = it->second;
  EXPECT_EQ("second", bytes.substr(1 + entry.data_offset(), entry.data_size()));
  unpacker.reset();
  ASSERT_TRUE(unpack(2 + first.size(), second.size()));
  EXPECT_EQ(unpacker.segments().end(), unpacker.segments().find("first"));
}

TEST(FileReadStorage, MemoryBlockReadsRetainIndependentBuffers) {
  const std::string file_path = "file_read_storage_memory_block_test";
  const std::string first(64, 'A');
  const std::string second(8192, 'B');
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);
  ASSERT_EQ(0, dumper->create(file_path));
  ASSERT_EQ(first.size(), dumper->write(first.data(), first.size()));
  ASSERT_EQ(second.size(), dumper->write(second.data(), second.size()));
  ASSERT_EQ(0, dumper->append("payload", first.size() + second.size(), 0, 0));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(file_path, false));
  auto segment = storage->get("payload", 1);
  ASSERT_TRUE(segment);
  IndexStorage::MemoryBlock first_block;
  IndexStorage::MemoryBlock second_block;
  ASSERT_EQ(first.size(), segment->read(0, first_block, first.size()));
  ASSERT_EQ(second.size(),
            segment->read(first.size(), second_block, second.size()));
  EXPECT_EQ(first, std::string(static_cast<const char *>(first_block.data()),
                               first.size()));
  EXPECT_EQ(second, std::string(static_cast<const char *>(second_block.data()),
                                second.size()));
  ASSERT_EQ(0u, segment->read(segment->capacity() + 1, second_block, 1));
  EXPECT_EQ(nullptr, second_block.data());
  ASSERT_EQ(0, storage->close());
  segment.reset();
  EXPECT_EQ(first, std::string(static_cast<const char *>(first_block.data()),
                               first.size()));
  std::remove(file_path.c_str());
}

TEST(FileReadStorage, EmptyMappedSegmentsSurviveCloseAtAnyAlignment) {
  const std::string path = "file_read_empty_segment.idx";
  const auto cleanup =
      ailego::ScopeGuard::Make([&] { std::remove(path.c_str()); });
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, dumper->append("unaligned", 0, 0, 0));
  const size_t page = ailego::MemoryHelper::PageSize();
  const std::string padding(page - dumper->size() % page, 'x');
  ASSERT_EQ(padding.size(), dumper->write(padding.data(), padding.size()));
  ASSERT_EQ(0, dumper->append("padding", padding.size(), 0, 0));
  ASSERT_EQ(0, dumper->append("aligned", 0, 0, 0));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(storage);
  for (const char *name : {"unaligned", "aligned"}) {
    SCOPED_TRACE(name);
    ASSERT_EQ(0, storage->open(path, false));
    ASSERT_TRUE(storage->has(name));
    auto segment = storage->get(name, 0);
    ASSERT_TRUE(segment);
    EXPECT_EQ(0u, segment->capacity());
    auto clone = segment->clone();
    ASSERT_TRUE(clone);
    ASSERT_EQ(0, storage->close());
    segment.reset();
    IndexStorage::MemoryBlock block;
    EXPECT_EQ(0u, clone->read(0, block, 1));
    EXPECT_EQ(nullptr, block.data());
    IndexStorage::SegmentData range(0, 0);
    EXPECT_TRUE(clone->read(&range, 1));
  }
}

TEST(FileReadStorage, IndependentReadersKeepTheValidatedFile) {
  const std::string path = "file_read_generation.idx";
  const std::string moved = path + ".old";
  auto cleanup = ailego::ScopeGuard::Make([&] {
    ailego::File::Delete(path);
    ailego::File::Delete(moved);
  });
  auto write_package = [&](const std::string &bytes) {
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_TRUE(dumper);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(bytes.size(), dumper->write(bytes.data(), bytes.size()));
    ASSERT_EQ(0, dumper->append("data", bytes.size(), 0, 0));
    ASSERT_EQ(0, dumper->close());
  };
  auto expect_data = [](const IndexStorage::Segment::Pointer &segment,
                        const std::string &expected) {
    ASSERT_TRUE(segment);
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(expected.size(), segment->read(0, block, expected.size()));
    ASSERT_NE(nullptr, block.data());
    EXPECT_EQ(expected, std::string(static_cast<const char *>(block.data()),
                                    expected.size()));
  };

  ASSERT_NO_FATAL_FAILURE(write_package("first"));
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(storage);
  ailego::Params params;
  params.set("proxima.file.read_storage.alone_file_handle", true);
  ASSERT_EQ(0, storage->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  auto reader = storage->get("data", 1);
  ASSERT_TRUE(reader);
  ASSERT_TRUE(ailego::File::Rename(path, moved));
  EXPECT_EQ(IndexError_OpenFile, storage->open(path, false));
  expect_data(storage->get("data", 1), "first");
  ASSERT_NO_FATAL_FAILURE(write_package("later"));
  expect_data(storage->get("data", 0), "first");
  auto all = storage->get_all();
  ASSERT_EQ(1u, all.count("data"));
  expect_data(all.at("data"), "first");
  ASSERT_EQ(0, storage->open(path, false));
  expect_data(storage->get("data", 1), "later");
  ASSERT_EQ(0, storage->close());
  auto clone = reader->clone();
  ASSERT_TRUE(clone);
  expect_data(clone, "first");
  expect_data(clone->clone(), "first");
}

TEST(FileReadStorage, LargeOwnedCopiesFitAnExactNativeBudget) {
  constexpr size_t bytes = (2UL << 20) + 7;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto previous = pool.capacity();
  ASSERT_EQ(0, pool.init(64UL << 20));
  auto restore =
      ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(previous)); });
  ASSERT_GE(pool.available(), 2 * bytes);
  const size_t padding = pool.available() - 2 * bytes;
  ASSERT_TRUE(pool.try_charge_external(padding));
  auto release =
      ailego::ScopeGuard::Make([&] { pool.release_external(padding); });

  using Block = IndexStorage::MemoryBlock;
  auto first = Block::AllocateOwned(bytes);
  std::memset(first.data_, 0x5a, bytes);
  Block second = first;
  EXPECT_NE(first.data(), second.data());
  EXPECT_EQ(0u, pool.available());
  Block assigned;
  assigned = std::move(first);
  Block moved(std::move(second));
  EXPECT_EQ(0, std::memcmp(assigned.data(), moved.data(), bytes));
  moved.reset();
  EXPECT_EQ(bytes, pool.available());
  assigned.reset();
  EXPECT_EQ(2 * bytes, pool.available());
}

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST(FileReadStorage, ResidentReadIsCountedOnceAgainstRssLimit) {
  constexpr size_t bytes = 16UL << 20;
  const std::string path = "file_read_resident.idx";
  auto cleanup = ailego::ScopeGuard::Make([&] { ailego::File::Delete(path); });
  const std::string payload(bytes, 'q');
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(bytes, dumper->write(payload.data(), bytes));
  ASSERT_EQ(0, dumper->append("payload", bytes, 0, 0));
  ASSERT_EQ(0, dumper->close());

  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto previous = pool.capacity();
  const auto previous_limit = pool.process_rss_limit();
  ASSERT_EQ(0, pool.init(64UL << 20));
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0, pool.init(previous));
    EXPECT_TRUE(pool.configure_process_rss_limit(previous_limit));
  });
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->open(path, false));
  auto segment = storage->get("payload", 1);
  ASSERT_TRUE(segment);
  IndexStorage::MemoryBlock held;
  ASSERT_EQ(bytes, segment->read(0, held, bytes));
  const size_t baseline = ailego::MemoryHelper::SelfRSS();
  ASSERT_NE(0u, baseline);
  // There is room for one more result, but not for counting the first twice.
  ASSERT_TRUE(pool.configure_process_rss_limit(baseline + bytes + bytes / 2));
  IndexStorage::MemoryBlock second;
  ASSERT_EQ(bytes, segment->read(0, second, bytes));
  EXPECT_EQ(0, std::memcmp(held.data(), payload.data(), bytes));
  EXPECT_EQ(0, std::memcmp(second.data(), payload.data(), bytes));
}
#endif

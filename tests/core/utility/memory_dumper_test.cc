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
#include <limits>
#include <gtest/gtest.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_helper.h"
#include "zvec/core/framework/index_memory.h"

using namespace zvec;
using namespace zvec::core;

TEST(MemoryDumper, General) {
  std::string file_path = "memory_dumper_test_file";

  auto dumper = IndexFactory::CreateDumper("MemoryDumper");
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

  auto container = IndexFactory::CreateStorage("MemoryReadStorage");
  ASSERT_TRUE(container);

  ailego::Params params;
  params.set("memory.container.checksum_validation", true);
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
    // Copy reads must honor the segment's nonzero offset within the rope,
    // including a subrange and an end-of-segment request.
    std::string copied(hello.size(), '\0');
    ASSERT_EQ(copied.size(), seg->fetch(0, copied.data(), copied.size()));
    EXPECT_EQ(hello, copied);
    ASSERT_EQ(copied.size() - 2, seg->fetch(2, copied.data(), copied.size()));
    EXPECT_EQ(hello.substr(2), copied.substr(0, hello.size() - 2));
    EXPECT_EQ(0u, seg->fetch(hello.size(), copied.data(), copied.size()));
    EXPECT_EQ(0u, seg->fetch(std::numeric_limits<size_t>::max(), copied.data(),
                             copied.size()));
  }
}

TEST(MemoryDumper, ReadBlockKeepsItsPayloadAfterStorageClose) {
  const std::string path = "memory_dumper_retained_result";
  const std::string payload = "0123456789";
  auto dumper = IndexFactory::CreateDumper("MemoryDumper");
  ASSERT_TRUE(dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(payload.size(), dumper->write(payload.data(), payload.size()));
  ASSERT_EQ(0, dumper->append("data", payload.size(), 0, 0));
  ASSERT_EQ(0, dumper->close());
  auto storage = IndexFactory::CreateStorage("MemoryReadStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->open(path, false));
  auto seg = storage->get("data");
  ASSERT_TRUE(seg);
  char copy[4];
  ASSERT_EQ(sizeof(copy), seg->fetch(3, copy, sizeof(copy)));
  EXPECT_EQ("3456", std::string(copy, sizeof(copy)));

  IndexStorage::MemoryBlock block;
  ASSERT_EQ(sizeof(copy), seg->read(3, block, sizeof(copy)));
  std::weak_ptr<IndexMemory::Rope> owner = IndexMemory::Instance()->open(path);
  IndexMemory::Instance()->remove(path);
  ASSERT_EQ(0, storage->close());
  seg.reset();
  ASSERT_FALSE(owner.expired());
  EXPECT_EQ("3456",
            std::string(static_cast<const char *>(block.data()), sizeof(copy)));
  block.reset();
  EXPECT_TRUE(owner.expired());
}

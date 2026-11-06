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
#include "db/common/rocksdb_memtable_factory.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/db.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec {
namespace {

class StringComparator final : public rocksdb::MemTableRep::KeyComparator {
 public:
  rocksdb::Slice decode_key(const char *key) const override {
    return rocksdb::Slice(key);
  }
  int operator()(const char *left, const char *right) const override {
    return std::strcmp(left, right);
  }
  int operator()(const char *left, const rocksdb::Slice &right) const override {
    return rocksdb::Slice(left).compare(right);
  }
};

#if defined(__APPLE__)
bool IsMapped(const void *pointer) {
  const auto base = reinterpret_cast<vm_address_t>(pointer);
  auto address = base;
  vm_size_t size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  const auto result = ::vm_region_64(
      ::mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
      reinterpret_cast<vm_region_info_t>(&info), &count, &object);
  if (object != MACH_PORT_NULL)
    ::mach_port_deallocate(::mach_task_self(), object);
  EXPECT_TRUE(result == KERN_SUCCESS || result == KERN_INVALID_ADDRESS);
  return result == KERN_SUCCESS && address <= base && base - address < size;
}
#elif defined(__linux__)
bool IsMapped(const void *pointer) {
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  EXPECT_GT(page_size, 0);
  if (page_size <= 0) return false;
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  auto *page = reinterpret_cast<void *>(address / page_size * page_size);
  unsigned char resident = 0;
  const int result = ::mincore(page, static_cast<size_t>(page_size), &resident);
  EXPECT_TRUE(result == 0 || errno == ENOMEM);
  return result == 0;
}
#elif defined(_WIN32)
bool IsMapped(const void *pointer) {
  MEMORY_BASIC_INFORMATION info{};
  const auto result = ::VirtualQuery(pointer, &info, sizeof(info));
  EXPECT_EQ(sizeof(info), result);
  // Decommitting pages alone leaves a reservation behind. Destruction must
  // release the entire mapping, so only MEM_FREE counts as unmapped.
  return result == sizeof(info) && info.State != MEM_FREE;
}
#endif

TEST(RocksdbMemtableFactoryTest, HoldsBackingChargeUntilDestruction) {
  auto cache = rocksdb::NewLRUCache(4 * 1024 * 1024, 0, true);
  auto manager = std::make_shared<rocksdb::WriteBufferManager>(2 * 1024 * 1024,
                                                               cache, true);
  StringComparator compare;
  auto factory = std::make_unique<RocksdbMemtableFactory>(manager);
  std::unique_ptr<rocksdb::MemTableRep> table(
      factory->CreateMemTableRep(compare, nullptr, nullptr, nullptr));
  factory.reset();  // Each table owns the resources needed for its lifetime.
  char *data = nullptr;
  auto handle = table->Allocate(512 * 1024, &data);
  std::memset(data, 'a', 512 * 1024 - 1);
  data[512 * 1024 - 1] = '\0';
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(data) % alignof(void *));
  ASSERT_TRUE(table->InsertKey(handle));
  const size_t charge = manager->memory_usage();
  EXPECT_GE(charge, 512u * 1024);
  EXPECT_EQ(charge, table->ApproximateMemoryUsage());
  EXPECT_EQ(charge, manager->mutable_memtable_memory_usage());
  table->MarkReadOnly();
  table->MarkReadOnly();
  table->MarkFlushed();
  EXPECT_EQ(0u, manager->mutable_memtable_memory_usage());
  EXPECT_EQ(charge, manager->memory_usage());
  EXPECT_TRUE(table->Contains(data));
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  EXPECT_TRUE(IsMapped(data));
#endif
  table.reset();
  EXPECT_EQ(0u, manager->memory_usage());
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  EXPECT_FALSE(IsMapped(data));
#endif
}

TEST(RocksdbMemtableFactoryTest, MutableDestructionReturnsAllCharges) {
  auto manager = std::make_shared<rocksdb::WriteBufferManager>(1024 * 1024);
  StringComparator compare;
  RocksdbMemtableFactory factory(manager);
  for (size_t round = 0; round < 8; ++round) {
    {
      std::unique_ptr<rocksdb::MemTableRep> table(
          factory.CreateMemTableRep(compare, nullptr, nullptr, nullptr));
      char *data = nullptr;
      table->Allocate(256 * 1024, &data);
      std::memset(data, 7, 256 * 1024);
      EXPECT_GT(manager->memory_usage(), 256u * 1024);
    }
    EXPECT_EQ(0u, manager->memory_usage());
    EXPECT_EQ(0u, manager->mutable_memtable_memory_usage());
  }
}

TEST(RocksdbMemtableFactoryTest,
     ConcurrentInsertsRetainDefaultSkipListBehavior) {
  auto manager = std::make_shared<rocksdb::WriteBufferManager>(2 * 1024 * 1024);
  StringComparator compare;
  RocksdbMemtableFactory factory(manager);
  ASSERT_TRUE(factory.IsInsertConcurrentlySupported());
  ASSERT_TRUE(factory.CanHandleDuplicatedKey());
  std::unique_ptr<rocksdb::MemTableRep> table(
      factory.CreateMemTableRep(compare, nullptr, nullptr, nullptr));
  std::array<std::future<void>, 4> workers;
  for (size_t thread = 0; thread < workers.size(); ++thread) {
    workers[thread] = std::async(std::launch::async, [&table, thread]() {
      for (size_t i = 0; i < 1024; ++i) {
        char *data = nullptr;
        auto handle = table->Allocate(128, &data);
        std::snprintf(data, 128, "%04zu-%04zu", thread, i);
        EXPECT_TRUE(table->InsertKeyConcurrently(handle));
        EXPECT_TRUE(table->Contains(data));
      }
    });
  }
  for (auto &worker : workers) worker.get();
  char *duplicate = nullptr;
  const auto handle = table->Allocate(128, &duplicate);
  std::strcpy(duplicate, "0000-0000");
  EXPECT_FALSE(table->InsertKey(handle));
  std::unique_ptr<rocksdb::MemTableRep::Iterator> iterator(
      table->GetIterator());
  size_t count = 0;
  for (iterator->SeekToFirst(); iterator->Valid(); iterator->Next()) ++count;
  EXPECT_EQ(4096u, count);
  iterator.reset();
  table.reset();
  EXPECT_EQ(0u, manager->memory_usage());
}

class AppendMerge final : public rocksdb::AssociativeMergeOperator {
 public:
  bool Merge(const rocksdb::Slice &, const rocksdb::Slice *previous,
             const rocksdb::Slice &value, std::string *result,
             rocksdb::Logger *) const override {
    *result = previous ? previous->ToString() : "";
    result->append(value.data(), value.size());
    return true;
  }
  const char *Name() const override {
    return "MappedMemtableTestAppend";
  }
};

TEST(RocksdbMemtableFactoryTest, FlushIteratorMutationAndReopen) {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string path = (std::filesystem::temp_directory_path() /
                            ("zvec-mapped-memtable-" + std::to_string(suffix)))
                               .string();
  ASSERT_FALSE(std::filesystem::exists(path));
  auto manager = std::make_shared<rocksdb::WriteBufferManager>(4 * 1024 * 1024);
  rocksdb::Options options;
  options.create_if_missing = true;
  options.write_buffer_size = 1024 * 1024;
  options.write_buffer_manager = manager;
  options.memtable_factory = std::make_shared<RocksdbMemtableFactory>(manager);
  options.merge_operator = std::make_shared<AppendMerge>();
  rocksdb::DB *raw = nullptr;
  ASSERT_TRUE(rocksdb::DB::Open(options, path, &raw).ok());
  std::unique_ptr<rocksdb::DB> db(raw);
  for (size_t round = 0; round < 4; ++round) {
    rocksdb::WriteBatch batch;
    for (size_t i = 0; i < 1024; ++i)
      batch.Put(std::to_string(i), std::string(128, 'a' + round));
    batch.Put("merge", "first");
    batch.Merge("merge", "+second");
    batch.Delete("deleted");
    ASSERT_TRUE(db->Write(rocksdb::WriteOptions(), &batch).ok());
    std::unique_ptr<rocksdb::Iterator> pinned(
        db->NewIterator(rocksdb::ReadOptions()));
    const size_t before_flush = manager->memory_usage();
    EXPECT_GT(before_flush, 128u * 1024);
    rocksdb::FlushOptions flush;
    flush.wait = true;
    ASSERT_TRUE(db->Flush(flush).ok());
    EXPECT_GE(manager->memory_usage(), before_flush);
    pinned->Seek("merge");
    ASSERT_TRUE(pinned->Valid());
    EXPECT_EQ("first+second", pinned->value().ToString());
    EXPECT_TRUE(pinned->status().ok());
    pinned.reset();
    std::string value;
    ASSERT_TRUE(db->Get(rocksdb::ReadOptions(), "0", &value).ok());
    EXPECT_EQ(std::string(128, 'a' + round), value);
    EXPECT_TRUE(
        db->Get(rocksdb::ReadOptions(), "deleted", &value).IsNotFound());
  }
  db.reset();
  EXPECT_EQ(0u, manager->memory_usage());
  // The backing change must not introduce a new on-disk representation.
  options.memtable_factory = std::make_shared<rocksdb::SkipListFactory>();
  ASSERT_TRUE(rocksdb::DB::Open(options, path, &raw).ok());
  db.reset(raw);
  std::string value;
  ASSERT_TRUE(db->Get(rocksdb::ReadOptions(), "merge", &value).ok());
  EXPECT_EQ("first+second", value);
  db.reset();
  EXPECT_EQ(0u, manager->memory_usage());
  EXPECT_TRUE(rocksdb::DestroyDB(path, options).ok());
}

}  // namespace
}  // namespace zvec

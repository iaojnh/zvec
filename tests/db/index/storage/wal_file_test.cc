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

#include <zvec/db/status.h>

#ifdef _MSC_VER
#define _ALLOW_KEYWORD_MACROS
#endif
#define private public
#define protected public
#include "db/index/storage/wal/wal_file.h"
#undef private
#undef protected

#if defined(__linux__)
#include <sys/resource.h>
#include <csignal>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <gtest/gtest.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include "db/common/file_helper.h"
#include "tests/test_util.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec;
using SegmentID = uint32_t;

class WalFileTest : public testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestFiles("./data.wal.*");
  }

  void TearDown() override {}
};

TEST_F(WalFileTest, LargeRecordsRoundTripAcrossTheOldFourMiBBoundary) {
  const std::string path = "./data.wal.large";
  const size_t boundary = size_t{4} << 20;
  const std::vector<size_t> sizes{boundary - 1, boundary, boundary + 1,
                                  size_t{5} << 20};
  WalFilePtr wal;
  ASSERT_EQ(0, WalFile::CreateAndOpen(path, {0, true}, &wal));
  for (size_t size : sizes) {
    ASSERT_EQ(0, wal->append(std::string(size, 'x'))) << size;
  }
  ASSERT_EQ(0, wal->close());
  ASSERT_EQ(0, WalFile::CreateAndOpen(path, {0, false, true}, &wal));
  ASSERT_EQ(0, wal->prepare_for_read());
  for (size_t size : sizes) {
    EXPECT_EQ(std::string(size, 'x'), wal->next());
    ASSERT_TRUE(wal->read_status().ok());
  }
  EXPECT_TRUE(wal->next().empty());
  EXPECT_TRUE(wal->read_status().ok());
}

TEST_F(WalFileTest, WalRecoveryRemovesTornTailBeforeAppend) {
  for (const auto tail_bytes : {2, 6, 9}) {
    SCOPED_TRACE(tail_bytes);
    const auto wal_path = "./data.wal.torn." + std::to_string(tail_bytes);
    WalFilePtr wal;
    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, true}, &wal));
    ASSERT_EQ(0, wal->append("first"));
    ASSERT_EQ(0, wal->flush());
    const auto first_end = std::filesystem::file_size(wal_path);
    ASSERT_EQ(0, wal->append("incomplete"));
    ASSERT_EQ(0, wal->close());
    std::filesystem::resize_file(wal_path, first_end + tail_bytes);

    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, false, true}, &wal));
    ASSERT_EQ(0, wal->prepare_for_read());
    EXPECT_EQ("first", wal->next());
    EXPECT_TRUE(wal->next().empty());
    EXPECT_TRUE(wal->read_status().ok());
    ASSERT_EQ(0, wal->close());
    EXPECT_EQ(first_end + tail_bytes, std::filesystem::file_size(wal_path));

    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, false}, &wal));
    ASSERT_EQ(0, wal->prepare_for_read());
    EXPECT_EQ("first", wal->next());
    EXPECT_TRUE(wal->next().empty());
    EXPECT_TRUE(wal->read_status().ok());
    ASSERT_EQ(0, wal->close());

    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, false}, &wal));
    ASSERT_EQ(0, wal->append("second"));
    ASSERT_EQ(0, wal->prepare_for_read());
    EXPECT_EQ("first", wal->next());
    EXPECT_EQ("second", wal->next());
    EXPECT_TRUE(wal->next().empty());
    EXPECT_TRUE(wal->read_status().ok());
  }
}

#if defined(__linux__) && GTEST_HAS_DEATH_TEST
TEST_F(WalFileTest, PartialAppendCanRetryOnTheSameHandle) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        const std::string path = "./data.wal.short";
        WalFilePtr wal;
        if (WalFile::CreateAndOpen(path, {0, true}, &wal) != 0 ||
            wal->append("first") != 0)
          std::_Exit(1);
        const auto complete = std::filesystem::file_size(path);
        struct rlimit original;
        if (getrlimit(RLIMIT_FSIZE, &original) != 0) std::_Exit(2);
        auto limited = original;
        limited.rlim_cur = complete + 6;
        const auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
        if (old_signal == SIG_ERR || setrlimit(RLIMIT_FSIZE, &limited) != 0)
          std::_Exit(3);
        const int refused = wal->append("unfinished");
        if (setrlimit(RLIMIT_FSIZE, &original) != 0) std::_Exit(4);
        std::signal(SIGXFSZ, old_signal);
        if (refused == 0 || std::filesystem::file_size(path) != complete)
          std::_Exit(5);
        if (wal->append("second") != 0 || wal->prepare_for_read() != 0 ||
            wal->next() != "first" || wal->next() != "second" ||
            !wal->next().empty() || !wal->read_status().ok())
          std::_Exit(6);
        wal->close();
        wal->remove();
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}
#endif

TEST_F(WalFileTest, ReadOnlyRejectsWritesAndPreservesRecords) {
  const std::string path = "./data.wal.readonly";
  auto wal = WalFile::Create(path);
  WalOptions options;
  options.create_new = true;
  ASSERT_EQ(0, wal->open(options));
  ASSERT_EQ(0, wal->append("record"));
  ASSERT_EQ(0, wal->flush());
  ASSERT_EQ(0, wal->close());
  auto snapshot = [&]() {
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
  };
  const auto before = snapshot();

  options.create_new = false;
  options.read_only = true;
  ASSERT_EQ(0, wal->open(options));
  ASSERT_TRUE(wal->has_record());
  EXPECT_NE(0, wal->append("forbidden"));
  EXPECT_NE(0, wal->flush());
  EXPECT_NE(0, wal->remove());
  ASSERT_EQ(0, wal->prepare_for_read());
  EXPECT_EQ("record", wal->next());
  EXPECT_TRUE(wal->next().empty());
  ASSERT_EQ(0, wal->close());
  EXPECT_NE(0, wal->remove());
  EXPECT_EQ(before, snapshot());

  options.read_only = false;
  ASSERT_EQ(0, wal->open(options));
  ASSERT_EQ(0, wal->append("writable again"));
  ASSERT_EQ(0, wal->prepare_for_read());
  EXPECT_EQ("record", wal->next());
  EXPECT_EQ("writable again", wal->next());
  EXPECT_TRUE(wal->next().empty());
  EXPECT_EQ(0, wal->remove());
}

TEST_F(WalFileTest, ReadOnlyCannotCreateWal) {
  const std::string path = "./data.wal.readonly_create";
  auto wal = WalFile::Create(path);
  WalOptions options;
  options.create_new = true;
  options.read_only = true;
  EXPECT_NE(0, wal->open(options));
  EXPECT_FALSE(FileHelper::FileExists(path));
}

TEST_F(WalFileTest, TestGeneral) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);

  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 0;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 100 same record
  for (size_t i = 0; i < 100; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // add 100-200 record
  wal_option.create_new = false;
  wal_option.max_docs_wal_flush = 1;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 100; i < 200; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // reopen and add next 100 record
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 200; i < 300; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // reopen and add batch model 100 record
  wal_option.max_docs_wal_flush = 10;
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 300; i < 400; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    if (idx < 100) {
      ASSERT_EQ(record, "hello");
    } else {
      ASSERT_EQ(record, std::string("hello") + std::to_string(idx));
    }
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 400);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}

void do_append(WalFile *wal_file, size_t number) {
  std::string record = "hello" + std::to_string(number);
  int ret = wal_file->append(std::move(record));
  ASSERT_EQ(ret, 0);
}

TEST_F(WalFileTest, TestMultiThread) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  ailego::ThreadPool pool(10, false);
  for (size_t i = 0; i < 10000; i++) {
    pool.execute(do_append, wal_file.get(), i);
  }
  pool.wait_finish();
  wal_file->flush();
  wal_file->close();

  // reopen for batch model
  wal_option.create_new = false;
  wal_option.max_docs_wal_flush = 1000;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 10000; i < 20000; i++) {
    pool.execute(do_append, wal_file.get(), i);
  }
  pool.wait_finish();
  wal_file->flush();
  wal_file->close();

  // reopen for batch model
  wal_option.create_new = false;
  wal_option.max_docs_wal_flush = 0;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 20000; i < 30000; i++) {
    pool.execute(do_append, wal_file.get(), i);
  }
  pool.wait_finish();
  wal_file->flush();
  wal_file->close();

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 30000);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestBoundaryCondition) {
  // read empty file
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  uint32_t idx = 0;
  std::string record = wal_file->next();
  while (!record.empty()) {
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write and read binary struct
  std::vector<uint8_t> bin_v{0, 1, 2, 3};
  std::string str(bin_v.begin(), bin_v.end());
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  ret = wal_file->append(std::move(str));
  ASSERT_EQ(ret, 0);
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record.size(), 4);
    for (size_t i = 0; i < 4; i++) {
      ASSERT_EQ(record[i], i);
    }
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 1);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);


  // write very large record 4Mb
  size_t big_data_size = 4 * 1024 * 1024;
  std::vector<uint8_t> big_data(big_data_size);
  for (size_t i = 0; i < big_data_size; i++) {
    big_data[i] = i % 256;
  }
  str.clear();
  str.assign((const char *)big_data.data(), big_data_size);
  wal_option.create_new = true;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  ret = wal_file->append(std::move(str));
  ASSERT_EQ(ret, 0);
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record.size(), big_data_size);
    for (size_t i = 0; i < big_data_size; i++) {
      ASSERT_EQ((uint8_t)record[i], i % 256);
    }
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 1);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);

  // batch model 100, just add 99 record and close
  wal_option.max_docs_wal_flush = 100;
  wal_option.create_new = true;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 0; i < 99; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record, std::string("hello") + std::to_string(idx));
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 99);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}

TEST_F(WalFileTest, TestNotExistErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  // reopen for read
  WalOptions wal_option;
  wal_option.create_new = false;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, -1);
}


TEST_F(WalFileTest, TestFirstErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  // destory first record
  lseek(wal_fd, 64 + 8, SEEK_SET);
  // write err data
  char buf[6] = "nihao";
  write(wal_fd, buf, 5);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestMiddleErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  // destory middle record
  lseek(wal_fd, 64 + 13 * 5 + 8, SEEK_SET);
  // write err data
  char buf[6] = "nihao";
  write(wal_fd, buf, 5);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 5);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestLastErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // destory last record
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  off_t fsize = lseek(wal_fd, 0, SEEK_END);
  close(wal_fd);
  truncate(wal_path.c_str(), (fsize - 4));

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 9);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestLengthSmallErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write error length
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  uint32_t err_length = 2;
  lseek(wal_fd, 64, SEEK_SET);
  write(wal_fd, (const void *)&err_length, 4);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestLengthBigErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write error length
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  uint32_t err_length = 200;  // exceed file size 130

  lseek(wal_fd, 64, SEEK_SET);
  write(wal_fd, (const void *)&err_length, 4);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestCRCErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write error crc
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  // second record crc 64+(4+4+len(hello))+4
  lseek(wal_fd, 64 + 17, SEEK_SET);
  uint32_t err_crc = 123;
  write(wal_fd, (const void *)&err_crc, 4);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = wal_file->next();
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = wal_file->next();
    idx++;
  }
  ASSERT_EQ(idx, 1);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

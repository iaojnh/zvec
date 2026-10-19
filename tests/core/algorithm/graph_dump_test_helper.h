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

#pragma once

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>

namespace zvec::core::testing {
class GraphDumpCapture : public IndexDumper {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int create(const std::string &) override {
    return 0;
  }
  int close() override {
    return 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  size_t write(const void *data, size_t size) override {
    max_write = std::max(max_write, size);
    if (fail_write) return 0;
    if (size) pending.append(static_cast<const char *>(data), size);
    return size;
  }
  int append(const std::string &id, size_t size, size_t pad,
             uint32_t crc) override {
    EXPECT_EQ(size + pad, pending.size());
    EXPECT_EQ((32 - size % 32) % 32, pad);
    EXPECT_EQ(ailego::Crc32c::Hash(pending.data(), size), crc);
    EXPECT_EQ(std::string(pad, '\0'), pending.substr(size));
    EXPECT_TRUE(segments.emplace(id, pending.substr(0, size)).second);
    pending.clear();
    return 0;
  }
  template <class Record>
  std::vector<Record> records(const std::string &id) const {
    const auto &bytes = segments.at(id);
    EXPECT_EQ(0u, bytes.size() % sizeof(Record));
    std::vector<Record> result(bytes.size() / sizeof(Record));
    if (!bytes.empty()) std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
  }
  std::map<std::string, std::string> segments;
  std::string pending;
  size_t max_write{0};
  bool fail_write{false};
};
}  // namespace zvec::core::testing

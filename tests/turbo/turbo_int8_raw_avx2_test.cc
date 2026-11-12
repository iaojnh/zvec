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
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include "distance/avx2/record_quantized_int8/common.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__AVX2__)
namespace {

namespace kernels = zvec::turbo::avx2::internal;

void check_batch(const int8_t *query, std::vector<const void *> records,
                 size_t dim) {
  std::vector<float> results(records.size() + 1, -1234567.0f);
  kernels::raw_inner_product_batch(records.data(), query, records.size(), dim,
                                   results.data());
  for (size_t i = 0; i < records.size(); ++i) {
    int64_t expected = 0;
    for (size_t j = 0; j < dim; ++j) {
      expected += static_cast<int>(query[j]) *
                  static_cast<int>(static_cast<const int8_t *>(records[i])[j]);
    }
    ASSERT_EQ(kernels::raw_inner_product(
                  query, static_cast<const int8_t *>(records[i]), dim),
              static_cast<float>(expected));
    ASSERT_EQ(results[i], static_cast<float>(expected))
        << "dim=" << dim << " count=" << records.size() << " record=" << i;
  }
  ASSERT_EQ(results.back(), -1234567.0f);
}

TEST(TurboInt8RawAvx2, DimensionsCountsAndUnalignedPointers) {
  std::mt19937 rng(868);
  std::uniform_int_distribution<int> draw(-128, 127);
  for (size_t dim : {1U,  2U,  3U,  4U,   7U,   8U,   15U,  16U,  17U,  30U,
                     31U, 32U, 33U, 47U,  48U,  49U,  61U,  62U,  63U,  64U,
                     65U, 95U, 96U, 127U, 128U, 129U, 255U, 256U, 257U, 512U}) {
    for (size_t offset = 0; offset < 8; ++offset) {
      std::vector<int8_t> query(dim + offset);
      std::vector<std::vector<int8_t>> storage(
          65, std::vector<int8_t>(dim + offset));
      for (auto &v : query) v = static_cast<int8_t>(draw(rng));
      for (auto &record : storage) {
        for (auto &v : record) v = static_cast<int8_t>(draw(rng));
      }
      std::vector<const void *> records;
      for (size_t count = 0; count <= 65; ++count) {
        check_batch(query.data() + offset, records, dim);
        if (count < storage.size()) {
          records.push_back(storage[count].data() + offset);
        }
      }
    }
  }
}

TEST(TurboInt8RawAvx2, ExtremeSignedCodes) {
  for (size_t dim : {16U, 17U, 31U, 32U, 47U, 48U, 61U, 62U, 63U, 64U, 65U}) {
    for (int q : {-128, -127, 0, 127}) {
      std::vector<int8_t> query(dim, static_cast<int8_t>(q));
      for (int v : {-128, -127, 0, 127}) {
        std::vector<int8_t> record(dim, static_cast<int8_t>(v));
        for (size_t count :
             {1U, 2U, 3U, 4U, 7U, 8U, 11U, 12U, 13U, 23U, 31U, 32U}) {
          check_batch(query.data(),
                      std::vector<const void *>(count, record.data()), dim);
        }
      }
    }
  }
}

#if defined(__unix__) || defined(__APPLE__)
class GuardedBytes {
 public:
  explicit GuardedBytes(size_t size)
      : page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE))) {
    mapping_ = mmap(nullptr, page_size_ * 3, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    auto *page = static_cast<int8_t *>(mapping_) + page_size_;
    if (mprotect(page, page_size_, PROT_READ | PROT_WRITE) != 0) {
      munmap(mapping_, page_size_ * 3);
      throw std::runtime_error("mprotect failed");
    }
    data_ = page + page_size_ - size;
  }
  ~GuardedBytes() {
    munmap(mapping_, page_size_ * 3);
  }
  GuardedBytes(const GuardedBytes &) = delete;
  GuardedBytes &operator=(const GuardedBytes &) = delete;
  int8_t *data() {
    return data_;
  }

 private:
  size_t page_size_;
  void *mapping_;
  int8_t *data_;
};

TEST(TurboInt8RawAvx2, BareVectorsAtGuardPageBoundary) {
  for (size_t dim = 1; dim <= 129; ++dim) {
    GuardedBytes query(dim), record(dim);
    for (size_t i = 0; i < dim; ++i) {
      query.data()[i] = static_cast<int8_t>(static_cast<int>(i % 256) - 128);
      record.data()[i] = static_cast<int8_t>(127 - static_cast<int>(i % 256));
    }
    for (size_t count : {1U, 2U, 4U, 8U, 11U, 12U, 13U, 23U, 31U, 32U}) {
      check_batch(query.data(), std::vector<const void *>(count, record.data()),
                  dim);
    }
  }
}
#endif

}  // namespace

#else
TEST(TurboInt8RawAvx2, RequiresAvx2Build) {
  GTEST_SKIP() << "AVX2 is not enabled in this build";
}
#endif

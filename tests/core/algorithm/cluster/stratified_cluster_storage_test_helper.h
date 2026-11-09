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

#include <cmath>
#include <cstring>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/scope_guard.h"
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/cluster/holder_cluster.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
using ailego::Params;

// Single-pass input with one transient, unaligned row, duplicate keys and no
// corpus allocation. The first coordinate also encodes the source ordinal.
class GeneratedStratifiedHolder : public IndexHolder {
 public:
  GeneratedStratifiedHolder(IndexMeta meta, size_t count)
      : meta_(std::move(meta)), count_(count) {}
  class Cursor : public Iterator {
   public:
    explicit Cursor(GeneratedStratifiedHolder *owner)
        : owner_(owner), row_(owner->element_size() + 1) {}
    const void *data() const override {
      if (id_ == owner_->fail_at) {
        error_ = IndexError_ReadData;
        return nullptr;
      }
      owner_->fill(id_, row_.data() + 1);
      return row_.data() + 1;
    }
    uint64_t key() const override {
      return id_ % 3;
    }
    bool is_valid() const override {
      return id_ < owner_->count_;
    }
    int status() const override {
      return error_;
    }
    void next() override {
      std::fill(row_.begin(), row_.end(), '\xff');
      ++id_;
    }

   private:
    GeneratedStratifiedHolder *owner_;
    size_t id_{0};
    mutable std::vector<char> row_;
    mutable int error_{0};
  };
  size_t count() const override {
    return count_;
  }
  size_t dimension() const override {
    return meta_.dimension();
  }
  size_t element_size() const override {
    return meta_.element_size();
  }
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }
  bool multipass() const override {
    return false;
  }
  Iterator::Pointer create_iterator() override {
    ++iterations;
    if (iterations != 1) return nullptr;
    return std::make_unique<Cursor>(this);
  }
  void fill(size_t id, void *data) const {
    for (size_t d = 0; d < dimension(); ++d) {
      const float value =
          ordinal_values
              ? static_cast<float>(id)
              : static_cast<float>((id % 2) * 8 + (id / 2 + d) % 3) - 1.0f;
      auto *out = static_cast<char *>(data);
      if (data_type() == IndexMeta::DT_FP16) {
        const ailego::Float16 half(value);
        std::memcpy(out + d * sizeof(half), &half, sizeof(half));
      } else if (data_type() == IndexMeta::DT_INT8) {
        const auto number = static_cast<int8_t>(value);
        std::memcpy(out + d, &number, sizeof(number));
      } else {
        std::memcpy(out + d * sizeof(value), &value, sizeof(value));
      }
    }
  }
  bool ordinal_values{false};
  size_t iterations{0};
  size_t fail_at{std::numeric_limits<size_t>::max()};

 private:
  IndexMeta meta_;
  size_t count_;
};

struct PartitionCalls {
  size_t max_batch_bytes{0};
  size_t child_calls{0};
  size_t reject_child{0};
  bool invalid_label{false};
  bool retain_values{true};
  std::vector<std::vector<uint32_t>> groups;
  std::vector<uint32_t> suggestions;
} calls;

class PartitionRecordingCluster : public IndexCluster, public HolderCluster {
 public:
  int init(const IndexMeta &meta, const Params &params) override {
    meta_ = meta;
    first_ = params.get_as_bool("first");
    return 0;
  }
  int cleanup() override {
    return reset();
  }
  int reset() override {
    features_.reset();
    return 0;
  }
  int update(const Params &) override {
    return 0;
  }
  void suggest(uint32_t k) override {
    suggestion_ = k;
  }
  int mount(IndexFeatures::Pointer features) override {
    calls.max_batch_bytes = std::max(
        calls.max_batch_bytes, features->count() * features->element_size());
    features_ = std::move(features);
    return 0;
  }
  int cluster(IndexThreads::Pointer, CentroidList &) override {
    return IndexError_Unsupported;
  }
  int classify(IndexThreads::Pointer, CentroidList &) override {
    return IndexError_Unsupported;
  }
  int label(IndexThreads::Pointer, const CentroidList &,
            std::vector<uint32_t> *out) override {
    out->resize(features_->count());
    for (size_t i = 0; i < features_->count(); ++i) {
      float value;
      std::memcpy(&value, features_->element(i), sizeof(value));
      // Uneven split, with the third centroid deliberately empty.
      (*out)[i] = calls.invalid_label ? 3 : (uint32_t(value) % 4 == 0 ? 1 : 0);
    }
    return 0;
  }
  int cluster_holder(IndexThreads::Pointer, IndexHolder::Pointer holder,
                     CentroidList &cents) override {
    const auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_LE(pool.used(), pool.capacity());
    if (!first_) {
      ++calls.child_calls;
      if (calls.child_calls == calls.reject_child) return IndexError_NoMemory;
      calls.suggestions.push_back(suggestion_);
    }
    std::vector<uint32_t> values;
    size_t n = 0;
    auto iter = holder->create_iterator();
    if (!iter) return IndexError_Runtime;
    for (; iter->is_valid(); iter->next()) {
      const void *data = iter->data();
      if (iter->status() != 0) return iter->status();
      if (!data) return IndexError_ReadData;
      float value;
      std::memcpy(&value, data, sizeof(value));
      EXPECT_EQ(uint32_t(value) % 3, iter->key());
      if (!first_ && calls.retain_values) values.push_back(uint32_t(value));
      ++n;
    }
    if (iter->status() != 0) return iter->status();
    EXPECT_EQ(holder->count(), n);
    if (!first_) calls.groups.push_back(std::move(values));
    std::vector<float> row(meta_.dimension(), 0.0f);
    cents.clear();
    for (size_t i = 0; i < (first_ ? 3u : 1u); ++i) {
      cents.emplace_back(row.data(), meta_.element_size());
      cents.back().set_follows(n);
    }
    return 0;
  }

 private:
  IndexMeta meta_;
  IndexFeatures::Pointer features_;
  bool first_{false};
  uint32_t suggestion_{0};
};
INDEX_FACTORY_REGISTER_CLUSTER(PartitionRecordingCluster);

class StratifiedClusterStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    budget_ =
        3 * ailego::PoolPayloadBudget::storage_metadata_bytes(4096) +
        256 * 1024;
    ASSERT_EQ(0, pool.init(budget_));
    calls = {};
  }
  void TearDown() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(capacity_));
  }
  IndexMeta meta(IndexMeta::DataType type = IndexMeta::DT_FP32,
                 size_t dimension = 8) {
    IndexMeta result(type, dimension);
    result.set_metric("SquaredEuclidean", 0, Params());
    return result;
  }
  Params params(const std::string &child = "PartitionRecordingCluster",
                bool automatic = false) {
    Params first, second, result;
    first.set("first", true);
    first.set(OPTKMEANS_CLUSTER_MAX_ITERATIONS, 3);
    second.set(OPTKMEANS_CLUSTER_MAX_ITERATIONS, 3);
    result.set(STRATIFIED_CLUSTER_FIRST_CLASS, child);
    result.set(STRATIFIED_CLUSTER_SECOND_CLASS, child);
    result.set(STRATIFIED_CLUSTER_FIRST_COUNT, 2);
    result.set(STRATIFIED_CLUSTER_SECOND_COUNT, 2);
    result.set(STRATIFIED_CLUSTER_FIRST_PARAMS, first);
    result.set(STRATIFIED_CLUSTER_SECOND_PARAMS, second);
    result.set(STRATIFIED_CLUSTER_AUTO_TUNING, automatic);
    return result;
  }
  IndexThreads::Pointer threads =
      std::make_shared<SingleQueueIndexThreads>(1, false);
  size_t budget_{0};

 private:
  size_t capacity_{0};
};

}  // namespace
}  // namespace zvec::core

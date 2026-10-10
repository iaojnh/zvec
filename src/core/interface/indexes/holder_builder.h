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
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_converter.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {

inline constexpr uint64_t kInvalidKey = std::numeric_limits<uint64_t>::max();

// Borrow the sole owned direct-add input instead of copying its vector bytes
// into a second MultiPass holder. The index must freeze mutation while this
// view is used; shared ownership makes iterator lifetime independent of it.
class DocCacheHolder : public core::IndexHolder,
                       public core::OrdinalAccessHolder,
                       public std::enable_shared_from_this<DocCacheHolder> {
 public:
  using Cache = std::vector<std::pair<uint64_t, std::string>>;
  DocCacheHolder(core::IndexQueryMeta meta, std::shared_ptr<const Cache> cache)
      : meta_(std::move(meta)), cache_(std::move(cache)) {
    if (!cache_) return;
    dense_ = std::all_of(cache_->begin(), cache_->end(), [](const auto &row) {
      return row.first != kInvalidKey;
    });
    if (dense_) return;
    for (size_t i = 0; i < cache_->size(); ++i) {
      if ((*cache_)[i].first != kInvalidKey) ordinals_.push_back(i);
    }
  }
  size_t count() const override {
    return dense_ ? cache_->size() : ordinals_.size();
  }
  size_t dimension() const override {
    return meta_.dimension();
  }
  size_t element_size() const override {
    return meta_.dimension() * meta_.unit_size();
  }
  core::IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }
  bool multipass() const override {
    return true;
  }
  class Cursor : public core::IndexHolder::Iterator {
   public:
    explicit Cursor(std::shared_ptr<DocCacheHolder> owner)
        : owner_(std::move(owner)) {}
    const void *data() const override {
      return is_valid() ? row().second.data() : nullptr;
    }
    uint64_t key() const override {
      return is_valid() ? row().first : kInvalidKey;
    }
    bool is_valid() const override {
      return pos_ < owner_->count() && status() == 0;
    }
    int status() const override {
      return pos_ < owner_->count() &&
                     row().second.size() != owner_->element_size()
                 ? core::IndexError_InvalidLength
                 : 0;
    }
    void next() override {
      ++pos_;
    }

   private:
    const Cache::value_type &row() const {
      return owner_->row_at(pos_);
    }
    std::shared_ptr<DocCacheHolder> owner_;
    size_t pos_{0};
  };
  class OrdinalReader : public Reader {
   public:
    explicit OrdinalReader(std::shared_ptr<DocCacheHolder> owner)
        : owner_(std::move(owner)) {}
    int read(size_t pos, uint64_t *key, const void **data) override {
      if (!key || !data) return core::IndexError_InvalidArgument;
      if (pos >= owner_->count()) return core::IndexError_OutOfRange;
      const auto &row = owner_->row_at(pos);
      if (row.second.size() != owner_->element_size())
        return core::IndexError_InvalidLength;
      *key = row.first;
      *data = row.second.data();
      return 0;
    }
    void reset() override {}

   private:
    std::shared_ptr<DocCacheHolder> owner_;
  };
  Iterator::Pointer create_iterator() override {
    return std::make_unique<Cursor>(shared_from_this());
  }
  int create_ordinal_reader(Reader::Pointer *out) override {
    if (!out) return core::IndexError_InvalidArgument;
    *out = std::make_unique<OrdinalReader>(shared_from_this());
    return 0;
  }

 private:
  const Cache::value_type &row_at(size_t pos) const {
    return (*cache_)[dense_ ? pos : ordinals_[pos]];
  }
  core::IndexQueryMeta meta_;
  std::shared_ptr<const Cache> cache_;
  std::vector<size_t> ordinals_;
  bool dense_{false};
};

template <core::IndexMeta::DataType DT, typename T>
inline int BuildMultiPassHolderImpl(
    uint32_t dimension,
    const std::vector<std::pair<uint64_t, std::string>> &doc_cache,
    core::IndexHolder::Pointer *holder_out) {
  auto holder =
      std::make_shared<zvec::core::MultiPassIndexHolder<DT>>(dimension);
  for (const auto &doc : doc_cache) {
    if (doc.first == kInvalidKey) {
      continue;
    }
    ailego::NumericalVector<T> vec(doc.second);
    if (!holder->emplace(doc.first, vec)) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  }
  *holder_out = holder;
  return 0;
}

inline int BuildMultiPassHolder(
    DataType data_type, uint32_t dimension,
    const std::vector<std::pair<uint64_t, std::string>> &doc_cache,
    const core::IndexConverter::Pointer &converter,
    core::IndexHolder::Pointer *holder) {
  int ret = 0;
  switch (data_type) {
    case DataType::DT_FP16:
      ret = BuildMultiPassHolderImpl<DataType::DT_FP16, uint16_t>(
          dimension, doc_cache, holder);
      break;
    case DataType::DT_FP32:
      ret = BuildMultiPassHolderImpl<DataType::DT_FP32, float>(
          dimension, doc_cache, holder);
      break;
    case DataType::DT_INT8:
      ret = BuildMultiPassHolderImpl<DataType::DT_INT8, uint8_t>(
          dimension, doc_cache, holder);
      break;
    default:
      LOG_ERROR("data_type is not support");
      return core::IndexError_Runtime;
  }
  if (ret != 0) {
    return ret;
  }
  if (converter) {
    core::IndexConverter::TrainAndTransform(converter, *holder);
    *holder = converter->result();
  }
  return 0;
}

}  // namespace zvec::core_interface

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

#include "vector_build_source.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include "stable_vector_holder.h"
#include "temporary_buffer_storage.h"

namespace zvec::core {
namespace {
class BorrowedBuildIterator : public IndexHolder::Iterator {
 public:
  BorrowedBuildIterator(IndexHolder::Pointer holder,
                        IndexHolder::Iterator::Pointer iterator)
      : holder_(std::move(holder)), iterator_(std::move(iterator)) {}
  const void *data() const override {
    return iterator_->data();
  }
  uint64_t key() const override {
    return iterator_->key();
  }
  bool is_valid() const override {
    return iterator_->is_valid();
  }
  int status() const override {
    return iterator_->status();
  }
  void next() override {
    iterator_->next();
  }

 private:
  IndexHolder::Pointer holder_;
  IndexHolder::Iterator::Pointer iterator_;
};

class BorrowedBuildReader : public OrdinalAccessHolder::Reader {
 public:
  BorrowedBuildReader(IndexHolder::Pointer holder, Reader::Pointer reader)
      : holder_(std::move(holder)), reader_(std::move(reader)) {}
  int read(size_t ordinal, uint64_t *key, const void **data) override {
    try {
      return reader_->read(ordinal, key, data);
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_ReadData;
    }
  }
  void reset() override {
    reader_->reset();
  }

 private:
  IndexHolder::Pointer holder_;
  Reader::Pointer reader_;
};
}  // namespace

class VectorBuildSource::StableReader : public OrdinalAccessHolder::Reader {
 public:
  explicit StableReader(std::shared_ptr<VectorBuildSource> source)
      : source_(std::move(source)) {}

  int read(size_t ordinal, uint64_t *key, const void **data) override {
    if (!key || !data) return IndexError_InvalidArgument;
    if (ordinal >= source_->count_) return IndexError_OutOfRange;
    *key = source_->keys_[ordinal];
    *data = source_->vectors_[ordinal];
    return 0;
  }
  void reset() override {}

 private:
  std::shared_ptr<VectorBuildSource> source_;
};

// Ordinary IndexHolder consumers use the same reader contract as random
// access consumers. Cache one row so key()/data() do not duplicate I/O or
// invalidate a provider's scratch before the caller advances the iterator.
class VectorBuildSource::Cursor : public IndexHolder::Iterator {
 public:
  Cursor(std::shared_ptr<VectorBuildSource> source, Reader::Pointer reader,
         int status)
      : source_(std::move(source)), reader_(std::move(reader)), status_(status) {}

  const void *data() const override {
    load();
    return data_;
  }
  uint64_t key() const override {
    load();
    return key_;
  }
  bool is_valid() const override {
    return status_ == 0 && ordinal_ < source_->count_;
  }
  int status() const override {
    return status_;
  }
  void next() override {
    if (ordinal_ < source_->count_) ++ordinal_;
    loaded_ = false;
    data_ = nullptr;
    key_ = std::numeric_limits<uint64_t>::max();
  }

 private:
  void load() const {
    if (loaded_ || !is_valid()) return;
    loaded_ = true;
    status_ = reader_->read(ordinal_, &key_, &data_);
    if (status_ == 0 && !data_) status_ = IndexError_ReadData;
    if (status_ != 0) data_ = nullptr;
  }

  std::shared_ptr<VectorBuildSource> source_;
  Reader::Pointer reader_;
  size_t ordinal_{0};
  mutable int status_{0};
  mutable bool loaded_{false};
  mutable uint64_t key_{std::numeric_limits<uint64_t>::max()};
  mutable const void *data_{nullptr};
};

class VectorBuildSource::SpoolReader : public OrdinalAccessHolder::Reader {
 public:
  explicit SpoolReader(std::shared_ptr<VectorBuildSource> source)
      : source_(std::move(source)) {}

  int read(size_t ordinal, uint64_t *key, const void **data) override {
    if (!key || !data) return IndexError_InvalidArgument;
    if (ordinal >= source_->count_) return IndexError_OutOfRange;
    try {
      const size_t chunk = ordinal / source_->rows_per_chunk_;
      const size_t word_offset =
          (ordinal % source_->rows_per_chunk_) * source_->record_words_;
      const uint64_t *record = nullptr;
      if (source_->file_) {
        scratch_.resize(source_->record_words_);
        const size_t bytes = sizeof(uint64_t) + source_->element_bytes_;
        if (source_->segments_[chunk]->fetch(
                word_offset * sizeof(uint64_t), scratch_.data(), bytes) != bytes) {
          return IndexError_ReadData;
        }
        record = scratch_.data();
      } else {
        record = source_->memory_chunks_[chunk].data() + word_offset;
      }
      *key = record[0];
      *data = record + 1;
      return 0;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_ReadData;
    }
  }

  void reset() override {
    ailego::BudgetedVector<uint64_t>().swap(scratch_);
  }

 private:
  std::shared_ptr<VectorBuildSource> source_;
  ailego::BudgetedVector<uint64_t> scratch_;
};

int VectorBuildSource::Create(
    IndexHolder::Pointer holder, Pointer *out,
    const std::string &scratch_prefix) {
  if (!out || !holder) return IndexError_InvalidArgument;
  try {
    Pointer source(new VectorBuildSource());
    int ret = source->init(std::move(holder), scratch_prefix);
    if (ret == 0) *out = std::move(source);
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }
}

int VectorBuildSource::init(
    IndexHolder::Pointer holder, const std::string &scratch_prefix) {
  dimension_ = holder->dimension();
  data_type_ = holder->data_type();
  element_bytes_ = holder->element_size();
  const size_t expected_count = holder->count();
  const bool known_count = expected_count != std::numeric_limits<size_t>::max();
  if (known_count && expected_count > std::numeric_limits<uint32_t>::max()) {
    return IndexError_Overflow;
  }
  if (known_count) {
    auto *ordinal = dynamic_cast<OrdinalAccessHolder *>(holder.get());
    if (ordinal) {
      Reader::Pointer probe;
      int ret = ordinal->create_ordinal_reader(&probe);
      if (ret == 0) {
        if (!probe) return IndexError_Runtime;
        holder_ = std::move(holder);
        count_ = expected_count;
        return 0;
      }
      if (ret != IndexError_NotImplemented) return ret;
    }
    if (auto *stable = dynamic_cast<StableVectorHolder *>(holder.get())) {
      int ret = stable->retain_stable_data(&lifetime_);
      if (ret == 0) {
        if (!lifetime_) return IndexError_Runtime;
        keys_.reserve(expected_count);
        vectors_.reserve(expected_count);
      } else if (ret != IndexError_NotImplemented) {
        return ret;
      }
    }
  }

  if (element_bytes_ == 0 ||
      element_bytes_ > std::numeric_limits<size_t>::max() - 2 * sizeof(uint64_t)) {
    return IndexError_InvalidArgument;
  }
  record_words_ =
      1 + (element_bytes_ + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  rows_per_chunk_ =
      std::max<size_t>(1, (4U << 20) / (record_words_ * sizeof(uint64_t)));
  if (known_count && expected_count != 0) {
    rows_per_chunk_ = std::min(rows_per_chunk_, expected_count);
  }

  auto iter = holder->create_iterator();
  if (!iter) return IndexError_Runtime;
  const bool buffered = ailego::MemoryLimitPool::get_instance().capacity() != 0;
  for (; iter->is_valid(); iter->next()) {
    if (iter->status() != 0) return iter->status();
    if (known_count && count_ >= expected_count)
      return IndexError_InvalidLength;
    if (count_ == std::numeric_limits<uint32_t>::max())
      return IndexError_Overflow;
    const uint64_t key = iter->key();
    if (iter->status() != 0) return iter->status();
    const void *data = iter->data();
    if (iter->status() != 0) return iter->status();
    if (!data) return IndexError_ReadData;
    if (lifetime_) {
      keys_.push_back(key);
      vectors_.push_back(data);
      ++count_;
      continue;
    }
    if (buffered && !file_) {
      std::string prefix = scratch_prefix;
      if (prefix.empty()) {
        const int ret =
            TemporaryBufferStorage::DefaultPrefix("zvec-build-input", &prefix);
        if (ret != 0) return ret;
      }
      int ret = TemporaryBufferStorage::CreateEmpty(prefix, &file_);
      if (ret != 0) return ret;
    }
    int ret = append(key, data);
    if (ret != 0) return ret;
  }
  if (iter->status() != 0) return iter->status();
  if (known_count && count_ != expected_count)
    return IndexError_InvalidLength;
  return file_ ? file_->flush() : 0;
}

int VectorBuildSource::append(uint64_t key, const void *data) {
  const size_t chunk = count_ / rows_per_chunk_;
  const size_t word_offset =
      (count_ % rows_per_chunk_) * record_words_;
  const size_t chunk_words = rows_per_chunk_ * record_words_;
  if (file_) {
    if (chunk == segments_.size()) {
      std::string name = std::to_string(chunk);
      int ret =
          file_->storage()->append(name, chunk_words * sizeof(uint64_t));
      if (ret != 0) return ret;
      auto segment = file_->storage()->get(name);
      if (!segment) return IndexError_Runtime;
      segments_.push_back(std::move(segment));
    }
    auto &segment = segments_[chunk];
    const size_t offset = word_offset * sizeof(uint64_t);
    if (segment->write(offset, &key, sizeof(key)) != sizeof(key) ||
        segment->write(offset + sizeof(key), data, element_bytes_) !=
            element_bytes_) {
      return IndexError_WriteData;
    }
  } else {
    if (chunk == memory_chunks_.size())
      memory_chunks_.emplace_back(chunk_words);
    auto *record = memory_chunks_[chunk].data() + word_offset;
    record[0] = key;
    std::memcpy(record + 1, data, element_bytes_);
  }
  ++count_;
  return 0;
}

int VectorBuildSource::create_ordinal_reader(Reader::Pointer *out) {
  if (!out) return IndexError_InvalidArgument;
  try {
    if (holder_) {
      Reader::Pointer reader;
      auto *source = dynamic_cast<OrdinalAccessHolder *>(holder_.get());
      int ret = source->create_ordinal_reader(&reader);
      if (ret != 0) return ret;
      if (!reader) return IndexError_Runtime;
      *out = std::make_unique<BorrowedBuildReader>(holder_, std::move(reader));
    } else if (lifetime_) {
      *out = std::make_unique<StableReader>(shared_from_this());
    } else {
      *out = std::make_unique<SpoolReader>(shared_from_this());
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }
}

IndexHolder::Iterator::Pointer VectorBuildSource::create_iterator() {
  if (holder_ && holder_->multipass()) {
    auto iterator = holder_->create_iterator();
    if (!iterator) return nullptr;
    return std::make_unique<BorrowedBuildIterator>(holder_, std::move(iterator));
  }
  Reader::Pointer reader;
  const int status = create_ordinal_reader(&reader);
  return std::make_unique<Cursor>(shared_from_this(), std::move(reader), status);
}

}  // namespace zvec::core

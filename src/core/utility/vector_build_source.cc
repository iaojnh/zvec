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

struct IndexBuildSource::State {
  IndexHolder::Pointer holder_;
  std::shared_ptr<const void> lifetime_;
  ailego::BudgetedVector<uint64_t> keys_;
  ailego::BudgetedVector<const void *> vectors_;
  size_t count_{0};
  size_t dimension_{0};
  IndexMeta::DataType data_type_{IndexMeta::DT_UNDEFINED};
  size_t element_bytes_{0};
  size_t record_words_{0};
  size_t rows_per_chunk_{0};
  TemporaryBufferStorage::Pointer file_;
  // Drop segments before their temporary storage owner.
  ailego::BudgetedVector<IndexStorage::Segment::Pointer> segments_;
  std::vector<std::vector<uint64_t>> memory_chunks_;
};

IndexBuildSource::IndexBuildSource() : state_(std::make_unique<State>()) {}
IndexBuildSource::~IndexBuildSource() = default;

size_t IndexBuildSource::count() const {
  return state_->count_;
}
size_t IndexBuildSource::dimension() const {
  return state_->dimension_;
}
IndexMeta::DataType IndexBuildSource::data_type() const {
  return state_->data_type_;
}
size_t IndexBuildSource::element_size() const {
  return state_->element_bytes_;
}
bool IndexBuildSource::multipass() const {
  return true;
}

class IndexBuildSource::StableReader : public OrdinalAccessHolder::Reader {
 public:
  explicit StableReader(std::shared_ptr<IndexBuildSource> source)
      : source_(std::move(source)) {}

  int read(size_t ordinal, uint64_t *key, const void **data) override {
    if (!key || !data) return IndexError_InvalidArgument;
    if (ordinal >= source_->state_->count_) return IndexError_OutOfRange;
    *key = source_->state_->keys_[ordinal];
    *data = source_->state_->vectors_[ordinal];
    return 0;
  }
  void reset() override {}

 private:
  std::shared_ptr<IndexBuildSource> source_;
};

// Ordinary IndexHolder consumers use the same reader contract as random
// access consumers. Cache one row so key()/data() do not duplicate I/O or
// invalidate a provider's scratch before the caller advances the iterator.
class IndexBuildSource::Cursor : public IndexHolder::Iterator {
 public:
  Cursor(std::shared_ptr<IndexBuildSource> source, Reader::Pointer reader,
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
    return status_ == 0 && ordinal_ < source_->state_->count_;
  }
  int status() const override {
    return status_;
  }
  void next() override {
    if (ordinal_ < source_->state_->count_) ++ordinal_;
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

  std::shared_ptr<IndexBuildSource> source_;
  Reader::Pointer reader_;
  size_t ordinal_{0};
  mutable int status_{0};
  mutable bool loaded_{false};
  mutable uint64_t key_{std::numeric_limits<uint64_t>::max()};
  mutable const void *data_{nullptr};
};

class IndexBuildSource::SpoolReader : public OrdinalAccessHolder::Reader {
 public:
  explicit SpoolReader(std::shared_ptr<IndexBuildSource> source)
      : source_(std::move(source)) {}

  int read(size_t ordinal, uint64_t *key, const void **data) override {
    if (!key || !data) return IndexError_InvalidArgument;
    if (ordinal >= source_->state_->count_) return IndexError_OutOfRange;
    try {
      const size_t chunk = ordinal / source_->state_->rows_per_chunk_;
      const size_t word_offset =
          (ordinal % source_->state_->rows_per_chunk_) *
          source_->state_->record_words_;
      const uint64_t *record = nullptr;
      if (source_->state_->file_) {
        scratch_.resize(source_->state_->record_words_);
        const size_t bytes = sizeof(uint64_t) + source_->state_->element_bytes_;
        if (source_->state_->segments_[chunk]->fetch(
                word_offset * sizeof(uint64_t), scratch_.data(), bytes) != bytes) {
          return IndexError_ReadData;
        }
        record = scratch_.data();
      } else {
        record = source_->state_->memory_chunks_[chunk].data() + word_offset;
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
  std::shared_ptr<IndexBuildSource> source_;
  ailego::BudgetedVector<uint64_t> scratch_;
};

int IndexBuildSource::Create(IndexHolder::Pointer holder, Pointer *out,
                             const std::string &scratch_prefix) {
  if (!out || !holder) return IndexError_InvalidArgument;
  try {
    Pointer source(new IndexBuildSource());
    int ret = source->init(std::move(holder), scratch_prefix);
    if (ret == 0) *out = std::move(source);
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }
}

int IndexBuildSource::init(IndexHolder::Pointer holder,
                           const std::string &scratch_prefix) {
  state_->dimension_ = holder->dimension();
  state_->data_type_ = holder->data_type();
  state_->element_bytes_ = holder->element_size();
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
        state_->holder_ = std::move(holder);
        state_->count_ = expected_count;
        return 0;
      }
      if (ret != IndexError_NotImplemented) return ret;
    }
    if (auto *stable = dynamic_cast<StableVectorHolder *>(holder.get())) {
      int ret = stable->retain_stable_data(&state_->lifetime_);
      if (ret == 0) {
        if (!state_->lifetime_) return IndexError_Runtime;
        state_->keys_.reserve(expected_count);
        state_->vectors_.reserve(expected_count);
      } else if (ret != IndexError_NotImplemented) {
        return ret;
      }
    }
  }

  if (state_->element_bytes_ == 0 ||
      state_->element_bytes_ >
          std::numeric_limits<size_t>::max() - 2 * sizeof(uint64_t)) {
    return IndexError_InvalidArgument;
  }
  state_->record_words_ =
      1 + (state_->element_bytes_ + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  state_->rows_per_chunk_ =
      std::max<size_t>(1, (4U << 20) / (state_->record_words_ * sizeof(uint64_t)));
  if (known_count && expected_count != 0) {
    state_->rows_per_chunk_ = std::min(state_->rows_per_chunk_, expected_count);
  }

  auto iter = holder->create_iterator();
  if (!iter) return IndexError_Runtime;
  const bool buffered = ailego::MemoryLimitPool::get_instance().capacity() != 0;
  for (; iter->is_valid(); iter->next()) {
    if (iter->status() != 0) return iter->status();
    if (known_count && state_->count_ >= expected_count)
      return IndexError_InvalidLength;
    if (state_->count_ == std::numeric_limits<uint32_t>::max())
      return IndexError_Overflow;
    const uint64_t key = iter->key();
    if (iter->status() != 0) return iter->status();
    const void *data = iter->data();
    if (iter->status() != 0) return iter->status();
    if (!data) return IndexError_ReadData;
    if (state_->lifetime_) {
      state_->keys_.push_back(key);
      state_->vectors_.push_back(data);
      ++state_->count_;
      continue;
    }
    if (buffered && !state_->file_) {
      std::string prefix = scratch_prefix;
      if (prefix.empty()) {
        const int ret =
            TemporaryBufferStorage::DefaultPrefix("zvec-build-input", &prefix);
        if (ret != 0) return ret;
      }
      int ret = TemporaryBufferStorage::CreateEmpty(prefix, &state_->file_);
      if (ret != 0) return ret;
    }
    int ret = append(key, data);
    if (ret != 0) return ret;
  }
  if (iter->status() != 0) return iter->status();
  if (known_count && state_->count_ != expected_count)
    return IndexError_InvalidLength;
  return state_->file_ ? state_->file_->flush() : 0;
}

int IndexBuildSource::append(uint64_t key, const void *data) {
  const size_t chunk = state_->count_ / state_->rows_per_chunk_;
  const size_t word_offset =
      (state_->count_ % state_->rows_per_chunk_) * state_->record_words_;
  const size_t chunk_words = state_->rows_per_chunk_ * state_->record_words_;
  if (state_->file_) {
    if (chunk == state_->segments_.size()) {
      std::string name = std::to_string(chunk);
      int ret =
          state_->file_->storage()->append(name, chunk_words * sizeof(uint64_t));
      if (ret != 0) return ret;
      auto segment = state_->file_->storage()->get(name);
      if (!segment) return IndexError_Runtime;
      state_->segments_.push_back(std::move(segment));
    }
    auto &segment = state_->segments_[chunk];
    const size_t offset = word_offset * sizeof(uint64_t);
    if (segment->write(offset, &key, sizeof(key)) != sizeof(key) ||
        segment->write(offset + sizeof(key), data, state_->element_bytes_) !=
            state_->element_bytes_) {
      return IndexError_WriteData;
    }
  } else {
    if (chunk == state_->memory_chunks_.size())
      state_->memory_chunks_.emplace_back(chunk_words);
    auto *record = state_->memory_chunks_[chunk].data() + word_offset;
    record[0] = key;
    std::memcpy(record + 1, data, state_->element_bytes_);
  }
  ++state_->count_;
  return 0;
}

int IndexBuildSource::create_ordinal_reader(Reader::Pointer *out) {
  if (!out) return IndexError_InvalidArgument;
  try {
    if (state_->holder_) {
      Reader::Pointer reader;
      auto *source = dynamic_cast<OrdinalAccessHolder *>(state_->holder_.get());
      int ret = source->create_ordinal_reader(&reader);
      if (ret != 0) return ret;
      if (!reader) return IndexError_Runtime;
      *out = std::make_unique<BorrowedBuildReader>(state_->holder_,
                                                   std::move(reader));
    } else if (state_->lifetime_) {
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

IndexHolder::Iterator::Pointer IndexBuildSource::create_iterator() {
  if (state_->holder_) {
    auto iterator = state_->holder_->create_iterator();
    if (!iterator) return nullptr;
    return std::make_unique<BorrowedBuildIterator>(state_->holder_,
                                                   std::move(iterator));
  }
  Reader::Pointer reader;
  const int status = create_ordinal_reader(&reader);
  return std::make_unique<Cursor>(shared_from_this(), std::move(reader), status);
}

}  // namespace zvec::core

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

#include "merged_provider_index_holder.h"
#include <limits>
#include <new>
#include <utility>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace core {

namespace {

constexpr size_t kBitsPerWord = sizeof(uint64_t) * 8;

void AppendBit(std::vector<uint64_t> *bits, size_t ordinal, bool value) {
  if (ordinal % kBitsPerWord == 0) {
    bits->push_back(0);
  }
  if (value) {
    bits->back() |= uint64_t{1} << (ordinal % kBitsPerWord);
  }
}

}  // namespace

class MergedProviderIndexHolder::Iterator final : public IndexHolder::Iterator {
 public:
  explicit Iterator(MergedProviderIndexHolder *owner) : owner_(owner) {
    this->seek_to_kept();
  }

  const void *data(void) const override {
    if (!this->is_valid()) {
      return nullptr;
    }
    if (data_prepared_) {
      return data_;
    }
    if (owner_->canceled()) {
      return this->fail(IndexError_Canceled, "Read vector canceled");
    }

    const auto &source = owner_->sources_[source_index_];
    const void *source_data = source_iter_->data();
    if (source_data == nullptr) {
      return this->fail(IndexError_Runtime,
                        "Source provider returned a null vector");
    }

    if (!source.need_revert) {
      data_ = source_data;
      data_prepared_ = true;
      return data_;
    }

    revert_buffer_.clear();
    int ret = source.reformer->revert(source_data, source.provider_meta,
                                      &revert_buffer_);
    if (ret != 0) {
      return this->fail(ret, "Failed to revert source vector");
    }
    if (revert_buffer_.size() != owner_->element_size()) {
      return this->fail(IndexError_Mismatch,
                        "Reverted vector size does not match output meta");
    }

    data_ = revert_buffer_.data();
    data_prepared_ = true;
    return data_;
  }

  bool is_valid(void) const override {
    return owner_->status() == 0 && source_index_ < owner_->sources_.size() &&
           source_iter_ && source_iter_->is_valid();
  }

  uint64_t key(void) const override {
    return output_key_;
  }

  void next(void) override {
    if (!this->is_valid()) {
      return;
    }
    source_iter_->next();
    ++source_ordinal_;
    ++output_key_;
    data_ = nullptr;
    data_prepared_ = false;
    revert_buffer_.clear();
    this->seek_to_kept();
  }

 private:
  void seek_to_kept(void) {
    while (owner_->status() == 0 && source_index_ < owner_->sources_.size()) {
      if (owner_->canceled()) {
        owner_->set_status(IndexError_Canceled);
        return;
      }

      const auto &source = owner_->sources_[source_index_];
      if (!source_iter_) {
        source_iter_ = source.provider->create_iterator();
        source_ordinal_ = 0;
        if (!source_iter_) {
          LOG_ERROR("Failed to create source provider iterator, source=%zu",
                    source_index_);
          owner_->set_status(IndexError_Runtime);
          return;
        }
      }

      while (source_iter_->is_valid()) {
        if (source_ordinal_ >= source.iterated_count) {
          LOG_ERROR(
              "Source provider iteration grew after filter planning, "
              "source=%zu",
              source_index_);
          owner_->set_status(IndexError_Mismatch);
          return;
        }
        if (owner_->keep(source_index_, source_ordinal_)) {
          return;
        }
        source_iter_->next();
        ++source_ordinal_;
      }

      if (source_ordinal_ != source.iterated_count) {
        LOG_ERROR(
            "Source provider iteration changed after filter planning, "
            "source=%zu expected=%zu actual=%zu",
            source_index_, source.iterated_count, source_ordinal_);
        owner_->set_status(IndexError_Mismatch);
        return;
      }
      source_iter_.reset();
      ++source_index_;
    }
  }

  const void *fail(int status, const char *message) const {
    LOG_ERROR("%s, source=%zu ordinal=%zu ret=%d", message, source_index_,
              source_ordinal_, status);
    owner_->set_status(status);

    // Existing IndexHolder consumers expect a valid pointer after data(). Give
    // them one safe value for the current call; the shared status makes all
    // subsequent is_valid() checks false and the reducer returns the error.
    revert_buffer_.assign(owner_->element_size(), 0);
    data_ = revert_buffer_.data();
    data_prepared_ = true;
    return data_;
  }

  MergedProviderIndexHolder *owner_{nullptr};
  size_t source_index_{0};
  size_t source_ordinal_{0};
  uint64_t output_key_{0};
  IndexHolder::Iterator::Pointer source_iter_{};
  mutable std::string revert_buffer_{};
  mutable const void *data_{nullptr};
  mutable bool data_prepared_{false};
};

MergedProviderIndexHolder::MergedProviderIndexHolder(
    IndexQueryMeta output_meta, std::vector<Source> sources)
    : output_meta_(output_meta), sources_(std::move(sources)) {}

int MergedProviderIndexHolder::init(const IndexFilter &filter,
                                    std::atomic<bool> *stop_flag) {
  if (initialized_) {
    return IndexError_Logic;
  }
  if (output_meta_.meta_type() != IndexMeta::MetaType::MT_DENSE ||
      output_meta_.data_type() == IndexMeta::DataType::DT_UNDEFINED ||
      output_meta_.dimension() == 0 || output_meta_.element_size() == 0) {
    return IndexError_InvalidArgument;
  }

  stop_flag_ = stop_flag;
  has_filter_ = filter.is_valid();
  uint64_t logical_id_base = 0;

  for (size_t source_index = 0; source_index < sources_.size();
       ++source_index) {
    auto &source = sources_[source_index];
    if (!source.provider || (source.need_revert && !source.reformer)) {
      this->set_status(IndexError_InvalidArgument);
      return this->status();
    }
    if (!source.need_revert &&
        (source.provider->data_type() != output_meta_.data_type() ||
         source.provider->dimension() != output_meta_.dimension() ||
         source.provider->element_size() != output_meta_.element_size())) {
      LOG_ERROR(
          "Source provider meta does not match merged holder output, "
          "source=%zu",
          source_index);
      this->set_status(IndexError_Mismatch);
      return this->status();
    }
    if (source.provider->count() >
        std::numeric_limits<uint64_t>::max() - logical_id_base) {
      this->set_status(IndexError_Overflow);
      return this->status();
    }

    source.logical_id_base = logical_id_base;
    source.keep_bits.clear();
    if (has_filter_) {
      const size_t provider_count = source.provider->count();
      source.keep_bits.reserve(provider_count / kBitsPerWord +
                               (provider_count % kBitsPerWord != 0));
    }

    auto iter = source.provider->create_iterator();
    if (!iter) {
      LOG_ERROR("Failed to create source provider iterator, source=%zu",
                source_index);
      this->set_status(IndexError_Runtime);
      return this->status();
    }

    size_t ordinal = 0;
    bool data_validated = false;
    for (; iter->is_valid(); iter->next(), ++ordinal) {
      if (this->canceled()) {
        this->set_status(IndexError_Canceled);
        return this->status();
      }
      if (iter->key() >
          std::numeric_limits<uint64_t>::max() - source.logical_id_base) {
        this->set_status(IndexError_Overflow);
        return this->status();
      }

      bool keep_item =
          !has_filter_ || !filter(source.logical_id_base + iter->key());
      if (has_filter_) {
        AppendBit(&source.keep_bits, ordinal, keep_item);
      }
      if (keep_item) {
        if (!data_validated) {
          const void *source_data = iter->data();
          if (source_data == nullptr) {
            LOG_ERROR("Source provider returned a null vector, source=%zu",
                      source_index);
            this->set_status(IndexError_Runtime);
            return this->status();
          }
          if (source.need_revert) {
            std::string reverted;
            int ret = source.reformer->revert(source_data, source.provider_meta,
                                              &reverted);
            if (ret != 0) {
              LOG_ERROR("Failed to validate source reformer, source=%zu ret=%d",
                        source_index, ret);
              this->set_status(ret);
              return this->status();
            }
            if (reverted.size() != this->element_size()) {
              LOG_ERROR(
                  "Reverted vector size does not match output meta, "
                  "source=%zu expected=%zu actual=%zu",
                  source_index, this->element_size(), reverted.size());
              this->set_status(IndexError_Mismatch);
              return this->status();
            }
          }
          data_validated = true;
        }
        if (count_ == std::numeric_limits<size_t>::max()) {
          this->set_status(IndexError_Overflow);
          return this->status();
        }
        ++count_;
      } else {
        ++filtered_count_;
      }
    }
    source.iterated_count = ordinal;
    logical_id_base += source.provider->count();
  }

  initialized_ = true;
  return 0;
}

size_t MergedProviderIndexHolder::count(void) const {
  return count_;
}

size_t MergedProviderIndexHolder::dimension(void) const {
  return output_meta_.dimension();
}

IndexMeta::DataType MergedProviderIndexHolder::data_type(void) const {
  return output_meta_.data_type();
}

size_t MergedProviderIndexHolder::element_size(void) const {
  return output_meta_.element_size();
}

bool MergedProviderIndexHolder::multipass(void) const {
  return true;
}

IndexHolder::Iterator::Pointer MergedProviderIndexHolder::create_iterator(
    void) {
  if (!initialized_ || this->status() != 0) {
    return IndexHolder::Iterator::Pointer();
  }
  return IndexHolder::Iterator::Pointer(new (std::nothrow) Iterator(this));
}

size_t MergedProviderIndexHolder::filtered_count(void) const {
  return filtered_count_;
}

int MergedProviderIndexHolder::status(void) const {
  return status_.load(std::memory_order_relaxed);
}

void MergedProviderIndexHolder::set_stop_flag(std::atomic<bool> *stop_flag) {
  stop_flag_ = stop_flag;
}

bool MergedProviderIndexHolder::keep(size_t source_index,
                                     size_t ordinal) const {
  if (!has_filter_) {
    return true;
  }
  const auto &bits = sources_[source_index].keep_bits;
  size_t word = ordinal / kBitsPerWord;
  return word < bits.size() &&
         ((bits[word] >> (ordinal % kBitsPerWord)) & uint64_t{1}) != 0;
}

bool MergedProviderIndexHolder::canceled(void) const {
  return stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed);
}

void MergedProviderIndexHolder::set_status(int status) {
  if (status == 0) {
    return;
  }
  int expected = 0;
  status_.compare_exchange_strong(expected, status, std::memory_order_relaxed);
}

}  // namespace core
}  // namespace zvec

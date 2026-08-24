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
#include "mixed_streamer_reducer.h"
#include <ailego/pattern/defer.h>
#include <utility/sparse_utility.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include "mixed_reducer/mixed_reducer_params.h"

namespace zvec {
namespace core {

namespace {

bool matches_query_meta(const IndexMeta &meta,
                        const IndexQueryMeta &query_meta) {
  if (meta.meta_type() != query_meta.meta_type() ||
      meta.data_type() != query_meta.data_type() ||
      meta.unit_size() != query_meta.unit_size()) {
    return false;
  }
  // Sparse records carry their logical length per document. Their metadata
  // dimension and element size are not the size of an individual record.
  return meta.meta_type() == IndexMeta::MetaType::MT_SPARSE ||
         (meta.dimension() == query_meta.dimension() &&
          meta.element_size() == query_meta.element_size());
}

bool matches_index_meta(const IndexQueryMeta &query_meta,
                        const IndexMeta &meta) {
  return matches_query_meta(meta, query_meta);
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (rhs > (std::numeric_limits<uint64_t>::max)() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool checked_multiply(size_t lhs, size_t rhs, size_t *result) {
  if (lhs != 0 && rhs > (std::numeric_limits<size_t>::max)() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

}  // namespace

int MixedStreamerReducer::init(const ailego::Params &params) {
  enable_pk_rewrite_ =
      params.get_as_bool(PARAM_MIXED_STREAMER_REDUCER_ENABLE_PK_REWRITE);
  params.get(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS,
             &num_of_add_threads_);
  if (num_of_add_threads_ <= 0) {
    LOG_ERROR("Wrong parameter. %s must be set greater than 0.",
              PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS.c_str());
    return IndexError_InvalidArgument;
  }

  params_ = params;

  state_ = STATE_INITED;
  return 0;
}

int MixedStreamerReducer::cleanup(void) {
  int ret = 0;
  streamers_.clear();
  source_streamers_reformers_.clear();
  if (target_streamer_ != nullptr) {
    ret = target_streamer_->cleanup();
  }
  if (target_builder_ != nullptr) {
    const int builder_ret = target_builder_->cleanup();
    if (ret == 0) {
      ret = builder_ret;
    }
  }
  target_streamer_.reset();
  target_streamer_reformer_.reset();
  target_builder_.reset();
  target_builder_converter_.reset();
  doc_cache_.clear();
  mt_list_.reset();
  mt_list_.resume_consume();
  sparse_mt_list_.reset();
  sparse_mt_list_.resume_consume();

  stats_.clear_attributes();
  state_ = STATE_UNINITED;
  return ret;
}

int MixedStreamerReducer::set_target_streamer_wiht_info(
    const IndexBuilder::Pointer builder, const IndexStreamer::Pointer streamer,
    const IndexConverter::Pointer converter,
    const IndexReformer::Pointer reformer,
    const IndexQueryMeta &original_query_meta) {
  if (state_ != STATE_INITED) {
    LOG_ERROR("Set target streamer after init");
    return IndexError_Uninitialized;
  }
  if (!streamer ||
      original_query_meta.meta_type() != streamer->meta().meta_type()) {
    LOG_ERROR("Invalid target streamer or original query metadata");
    return IndexError_InvalidArgument;
  }

  target_builder_ = builder;
  target_streamer_ = streamer;
  target_builder_converter_ = converter;
  target_streamer_reformer_ = reformer;
  original_query_meta_ = original_query_meta;

  is_sparse_ =
      target_streamer_->meta().meta_type() == IndexMeta::MetaType::MT_SPARSE;

  state_ = STATE_STREAMER_SET;
  return 0;
}

int MixedStreamerReducer::feed_streamer_with_reformer(
    IndexStreamer::Pointer streamer, const IndexReformer::Pointer reformer) {
  if (!(state_ == STATE_STREAMER_SET || state_ == STATE_FEED)) {
    LOG_ERROR("Set target streamer or feed before feed");
    return IndexError_Uninitialized;
  }

  if (!streamer) {
    LOG_ERROR("Streamer nullptr");
    return IndexError_InvalidArgument;
  }

  const IndexMeta &target_meta = target_streamer_->meta();
  const IndexMeta &source_meta = streamer->meta();
  if (target_meta.meta_type() != source_meta.meta_type()) {
    LOG_ERROR("Streamer meta mismatch");
    return IndexError_InvalidArgument;
  }

  const bool same_reformer =
      target_meta.reformer_name() == source_meta.reformer_name();
  bool compatible = true;
  if (target_builder_ != nullptr) {
    compatible = reformer != nullptr ||
                 matches_query_meta(source_meta, original_query_meta_);
  } else if (same_reformer) {
    compatible = target_meta.data_type() == source_meta.data_type() &&
                 target_meta.dimension() == source_meta.dimension() &&
                 target_meta.unit_size() == source_meta.unit_size() &&
                 target_meta.element_size() == source_meta.element_size();
  } else {
    compatible = (reformer != nullptr ||
                  matches_query_meta(source_meta, original_query_meta_)) &&
                 (target_streamer_reformer_ != nullptr ||
                  matches_query_meta(target_meta, original_query_meta_));
  }
  if (!compatible) {
    LOG_ERROR("Streamer vector representation mismatch");
    return IndexError_InvalidArgument;
  }

  streamers_.push_back(streamer);
  source_streamers_reformers_.push_back(reformer);

  state_ = STATE_FEED;
  return 0;
}

int MixedStreamerReducer::reduce(const IndexFilter &filter) {
  if (state_ != STATE_FEED) {
    LOG_ERROR("Feed streamers first");
    return IndexError_Uninitialized;
  }
  if (thread_pool_ == nullptr) {
    LOG_ERROR("Thread pool is not set");
    return IndexError_Uninitialized;
  }

  ailego::ElapsedTime timer;


  std::vector<int> add_results(num_of_add_threads_, -1);
  auto add_group = thread_pool_->make_group();

  std::vector<int> read_results(streamers_.size(), -1);
  // TODO: use id instead of key
  // When merging into a non-empty target (e.g. reusing one input as base),
  // append new docs after the existing ones instead of overwriting from 0.
  uint64_t id_offset = 0;
  uint64_t next_id = 0;
  auto find_next_target_id = [&next_id](const auto &provider) -> int {
    const uint64_t target_count = provider->count();
    next_id = (std::max)(next_id, target_count);
    if (target_count == 0) {
      return 0;
    }
    auto iterator = provider->create_iterator();
    if (!iterator) {
      LOG_ERROR("Failed to create target provider iterator");
      return IndexError_Runtime;
    }
    while (iterator->is_valid()) {
      const uint64_t key = iterator->key();
      if (key == (std::numeric_limits<uint64_t>::max)()) {
        LOG_ERROR("Invalid target vector key");
        return IndexError_InvalidFormat;
      }
      next_id = (std::max)(next_id, key + 1);
      iterator->next();
    }
    return 0;
  };
  if (target_builder_ == nullptr) {
    if (is_sparse_) {
      auto provider = target_streamer_->create_sparse_provider();
      if (!provider) {
        LOG_ERROR("Failed to create target sparse provider");
        return IndexError_Runtime;
      }
      const int ret = find_next_target_id(provider);
      if (ret != 0) {
        return ret;
      }
    } else {
      auto provider = target_streamer_->create_provider();
      if (!provider) {
        LOG_ERROR("Failed to create target provider");
        return IndexError_Runtime;
      }
      const int ret = find_next_target_id(provider);
      if (ret != 0) {
        return ret;
      }
    }
  }

  if (is_sparse_) {
    for (size_t i = 0; i < num_of_add_threads_; i++) {
      add_group->submit(ailego::Closure::New(
          this, &MixedStreamerReducer::add_sparse_vec, &add_results[i]));
    }

    for (size_t i = 0; i < streamers_.size(); i++) {
      // due to filter, producing can't be parallel
      auto provider = streamers_[i]->create_sparse_provider();
      if (!provider) {
        LOG_ERROR("Failed to create source sparse provider, index=%zu", i);
        read_results[i] = IndexError_Runtime;
        break;
      }
      uint64_t source_span = 0;
      read_results[i] = read_sparse_vec(i, provider, filter, id_offset,
                                        &next_id, &source_span);
      if (read_results[i] != 0) {
        break;
      }
      if (!checked_add(id_offset, source_span, &id_offset)) {
        LOG_ERROR("Source sparse vector key range overflows");
        read_results[i] = IndexError_InvalidFormat;
        break;
      }
    }

    sparse_mt_list_.done();
  } else {
    for (size_t i = 0; i < num_of_add_threads_; i++) {
      add_group->submit(ailego::Closure::New(
          this, &MixedStreamerReducer::add_vec, &add_results[i]));
      // add_vec(&add_results[i]);
    }

    for (size_t i = 0; i < streamers_.size(); i++) {
      auto provider = streamers_[i]->create_provider();
      if (!provider) {
        LOG_ERROR("Failed to create source provider, index=%zu", i);
        read_results[i] = IndexError_Runtime;
        break;
      }
      uint64_t source_span = 0;
      read_results[i] =
          read_vec(i, provider, filter, id_offset, &next_id, &source_span);
      if (read_results[i] != 0) {
        break;
      }
      if (!checked_add(id_offset, source_span, &id_offset)) {
        LOG_ERROR("Source vector key range overflows");
        read_results[i] = IndexError_InvalidFormat;
        break;
      }
    }

    mt_list_.done();
  }
  add_group->wait_finish();

  auto check_results = [](const std::vector<int> &results) -> bool {
    return std::all_of(std::begin(results), std::end(results),
                       [](int item) { return item == 0; });
  };

  if (!check_results(read_results)) {
    LOG_ERROR("Get vector from entities failed");
    return IndexError_Runtime;
  }

  if (!check_results(add_results)) {
    LOG_ERROR("add vector failed");
    return IndexError_Runtime;
  }

  stats_.set_reduced_costtime(timer.seconds());
  state_ = STATE_REDUCE;
  if (target_builder_ != nullptr) {
    int ret = IndexBuild();
    if (ret != 0) {
      LOG_ERROR("Failed to build target index, ret=%d", ret);
      return ret;
    }
  }

  LOG_INFO("End brute force reduce. cost time: [%zu]s",
           (size_t)timer.seconds());
  return 0;
}

int MixedStreamerReducer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Begin brute force reducer dump");

  if (state_ != STATE_REDUCE) {
    LOG_WARN("Reduce first before dump");
    return IndexError_NoReady;
  }

  if (!dumper) {
    LOG_ERROR("Dumper is null");
    return IndexError_InvalidArgument;
  }

  ailego::ElapsedTime timer;
  int ret = 0;
  if (target_builder_ != nullptr) {
    ret = target_builder_->dump(dumper);
  } else {
    ret = target_streamer_->dump(dumper);
  }
  if (ret == IndexError_NotImplemented) {
    LOG_WARN("Dump index not implemented");
  } else if (ret < 0) {
    LOG_ERROR("Failed to dump in streamer");
  }

  return ret;
}

int MixedStreamerReducer::read_vec(size_t source_streamer_index,
                                   const IndexProvider::Pointer &provider,
                                   const IndexFilter &filter,
                                   uint64_t id_offset, uint64_t *next_id,
                                   uint64_t *source_span) {
  const auto &streamer = streamers_[source_streamer_index];
  const auto &reformer = source_streamers_reformers_[source_streamer_index];
  const IndexQueryMeta source_streamer_query_meta{streamer->meta().data_type(),
                                                  streamer->meta().dimension()};
  const bool same_reformer = target_streamer_->meta().reformer_name() ==
                             streamer->meta().reformer_name();
  const bool need_revert =
      reformer != nullptr && (target_builder_ != nullptr || !same_reformer);
  const bool need_convert = target_builder_ == nullptr && !same_reformer &&
                            target_streamer_reformer_ != nullptr;

  if (!provider) {
    LOG_ERROR("Source provider is null, index=%zu", source_streamer_index);
    return IndexError_Runtime;
  }
  *source_span = provider->count();
  IndexProvider::Iterator::Pointer iterator = provider->create_iterator();
  if (!iterator) {
    LOG_ERROR("Failed to create source provider iterator, index=%zu",
              source_streamer_index);
    return IndexError_Runtime;
  }

  while (iterator->is_valid()) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("read_vec cancelled.");
      return 0;
    }
    const uint64_t source_key = iterator->key();
    if (source_key == (std::numeric_limits<uint64_t>::max)()) {
      LOG_ERROR("Invalid source vector key");
      return IndexError_InvalidFormat;
    }
    *source_span = (std::max)(*source_span, source_key + 1);
    uint64_t global_id = 0;
    if (!checked_add(id_offset, source_key, &global_id)) {
      LOG_ERROR("Source vector key overflows global id range");
      return IndexError_InvalidFormat;
    }
    if (filter(global_id)) {
      (*stats_.mutable_filtered_count())++;
      iterator->next();
      continue;
    }

    const void *vector_data = iterator->data();
    if (!vector_data) {
      LOG_ERROR("Failed to read source vector, index=%zu key=%zu",
                source_streamer_index, static_cast<size_t>(iterator->key()));
      return IndexError_ReadData;
    }
    std::string reverted_vector;
    if (need_revert) {
      if (reformer->revert(vector_data, source_streamer_query_meta,
                           &reverted_vector) != 0) {
        LOG_ERROR("Failed to revert the vector");
        return IndexError_Runtime;
      }
      if (reverted_vector.size() != original_query_meta_.element_size()) {
        LOG_ERROR("Reverted vector has an invalid size: actual=%zu expected=%u",
                  reverted_vector.size(), original_query_meta_.element_size());
        return IndexError_Mismatch;
      }
      vector_data = reverted_vector.data();
    }

    std::string converted_vector;
    if (need_convert) {
      IndexQueryMeta converted_meta;
      if (target_streamer_reformer_->convert(vector_data, original_query_meta_,
                                             &converted_vector,
                                             &converted_meta) != 0) {
        LOG_ERROR("Failed to convert vector into target representation");
        return IndexError_Runtime;
      }
      if (!matches_index_meta(converted_meta, target_streamer_->meta())) {
        LOG_ERROR("Converted vector metadata does not match target streamer");
        return IndexError_Mismatch;
      }
      vector_data = converted_vector.data();
    }

    const size_t expected_size = target_builder_ != nullptr
                                     ? original_query_meta_.element_size()
                                     : target_streamer_->meta().element_size();
    const size_t source_size =
        need_convert
            ? converted_vector.size()
            : (need_revert ? reverted_vector.size() : provider->element_size());
    if (source_size != expected_size) {
      LOG_ERROR("Source vector has an invalid size: actual=%zu expected=%zu",
                source_size, expected_size);
      return IndexError_Mismatch;
    }

    std::vector<uint8_t> bytes(expected_size);
    if (!bytes.empty()) {
      memcpy(bytes.data(), vector_data, bytes.size());
    }

    // TODO: use id instead of key
    if (*next_id > (std::numeric_limits<uint32_t>::max)()) {
      LOG_ERROR("Target vector id overflows uint32 range");
      return IndexError_InvalidFormat;
    }
    if (!mt_list_.produce(VectorItem((*next_id)++, std::move(bytes)))) {
      LOG_ERROR("Produce vector to queue failed. key[%lu]",
                (size_t)iterator->key());
      return IndexError_Runtime;
    }
    iterator->next();
  }
  return 0;
}

void MixedStreamerReducer::add_vec(int *result) {
  if (target_builder_ != nullptr) {
    add_vec_with_builder(result);
    return;
  }
  ailego::ElapsedTime timer;
  auto target_streamer_context = target_streamer_->create_context();
  auto target_streamer_query_meta = IndexQueryMeta{
      IndexMeta::MetaType::MT_DENSE, target_streamer_->meta().data_type(),
      target_streamer_->meta().dimension()};

  AILEGO_DEFER([&]() {
    // make producer quit
    mt_list_.done();
  });

  VectorItem vector_item;
  while (mt_list_.consume(&vector_item)) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("add_vec cancelled.");
      return;
    }

    // TODO: use id instead of key
    int ret = target_streamer_->add_with_id_impl(
        (uint32_t)vector_item.pkey_, vector_item.vec_.data(),
        target_streamer_query_meta, target_streamer_context);
    if (ret != 0) {
      LOG_ERROR("Insert target streamer failed. ret[%d] reason[%s] pkey[%zu]",
                ret, IndexError::What(ret), (size_t)vector_item.pkey_);
      *result = ret;
      return;
    }
  }

  *result = 0;
  LOG_DEBUG("add_vec. cost time: [%zu]s", (size_t)timer.seconds());
  return;
}

void MixedStreamerReducer::add_vec_with_builder(int *result) {
  ailego::ElapsedTime timer;

  AILEGO_DEFER([&]() {
    // make producer quit
    mt_list_.done();
  });

  VectorItem vector_item;
  while (mt_list_.consume(&vector_item)) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("add_vec cancelled.");
      return;
    }

    const void *vector = vector_item.vec_.data();
    std::string out_vector_buffer = std::string(
        static_cast<const char *>(vector),
        original_query_meta_.dimension() * original_query_meta_.unit_size());
    PushToDocCache(original_query_meta_, (uint32_t)vector_item.pkey_,
                   out_vector_buffer);
  }

  *result = 0;
  LOG_DEBUG("add_vec. cost time: [%zu]s", (size_t)timer.seconds());
  return;
}

void MixedStreamerReducer::add_sparse_vec(int *result) {
  ailego::ElapsedTime timer;
  auto target_streamer_context = target_streamer_->create_context();
  auto target_streamer_query_meta = IndexQueryMeta{
      IndexMeta::MetaType::MT_SPARSE,
      target_streamer_->meta().data_type(),
  };

  AILEGO_DEFER([&]() {
    // make producer quit
    sparse_mt_list_.done();
  });

  SparseVectorItem sparse_vector_item;
  while (sparse_mt_list_.consume(&sparse_vector_item)) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("add_sparse_vec cancelled.");
      return;
    }
    auto sparse_count = sparse_vector_item.sparse_indices_.size();
    auto indices = sparse_vector_item.sparse_indices_.data();
    auto values = sparse_vector_item.sparse_values_.data();

    // TODO: use id instead of key
    int ret = target_streamer_->add_with_id_impl(
        (uint32_t)sparse_vector_item.pkey_, sparse_count, indices, values,
        target_streamer_query_meta, target_streamer_context);
    if (ret != 0) {
      LOG_ERROR("Insert target streamer failed. ret[%d] reason[%s] pkey[%zu]",
                ret, IndexError::What(ret), (size_t)sparse_vector_item.pkey_);
      *result = ret;
      return;
    }
  }

  *result = 0;
  LOG_DEBUG("add_sparse_vec. cost time: [%zu]s", (size_t)timer.seconds());
  return;
}


int MixedStreamerReducer::read_sparse_vec(
    size_t source_streamer_index,
    const IndexStreamer::SparseProvider::Pointer &provider,
    const IndexFilter &filter, uint64_t id_offset, uint64_t *next_id,
    uint64_t *source_span) {
  const auto &streamer = streamers_[source_streamer_index];
  const auto &reformer = source_streamers_reformers_[source_streamer_index];
  const bool same_reformer = target_streamer_->meta().reformer_name() ==
                             streamer->meta().reformer_name();
  const bool need_revert =
      reformer != nullptr && (target_builder_ != nullptr || !same_reformer);
  const bool need_convert = target_builder_ == nullptr && !same_reformer &&
                            target_streamer_reformer_ != nullptr;

  if (!provider) {
    LOG_ERROR("Source sparse provider is null, index=%zu",
              source_streamer_index);
    return IndexError_Runtime;
  }
  *source_span = provider->count();
  IndexStreamer::SparseProvider::Iterator::Pointer iterator =
      provider->create_iterator();
  if (!iterator) {
    LOG_ERROR("Failed to create source sparse provider iterator, index=%zu",
              source_streamer_index);
    return IndexError_Runtime;
  }

  while (iterator->is_valid()) {
    if (stop_flag_ != nullptr && stop_flag_->load(std::memory_order_relaxed)) {
      LOG_DEBUG("read_sparse_vec cancelled.");
      return 0;
    }
    const uint64_t source_key = iterator->key();
    if (source_key == (std::numeric_limits<uint64_t>::max)()) {
      LOG_ERROR("Invalid source sparse vector key");
      return IndexError_InvalidFormat;
    }
    *source_span = (std::max)(*source_span, source_key + 1);
    uint64_t global_id = 0;
    if (!checked_add(id_offset, source_key, &global_id)) {
      LOG_ERROR("Source sparse vector key overflows global id range");
      return IndexError_InvalidFormat;
    }
    if (filter(global_id)) {
      (*stats_.mutable_filtered_count())++;
      iterator->next();
      continue;
    }

    const auto sparse_count = iterator->sparse_count();
    const uint32_t *const source_indices = iterator->sparse_indices();
    const void *const source_values = iterator->sparse_data();
    if (sparse_count > 0 &&
        (source_indices == nullptr || source_values == nullptr)) {
      LOG_ERROR("Failed to read source sparse vector, index=%zu",
                source_streamer_index);
      return IndexError_ReadData;
    }
    std::vector<uint32_t> sparse_indices(sparse_count);
    if (!sparse_indices.empty()) {
      memcpy(sparse_indices.data(), source_indices,
             sparse_indices.size() * sizeof(uint32_t));
    }
    const void *values = source_values;
    std::string sparse_values;

    if (need_revert) {
      if (reformer->revert(sparse_count, source_indices, source_values,
                           {
                               IndexMeta::MetaType::MT_SPARSE,
                               streamer->meta().data_type(),
                           },
                           &sparse_values) != 0) {
        LOG_ERROR("Failed to revert the sparse vector");
        return IndexError_Runtime;
      }
      values = sparse_values.data();
    }

    std::string converted_sparse_values;
    if (need_convert) {
      IndexQueryMeta converted_meta;
      if (target_streamer_reformer_->convert(
              sparse_count, source_indices, values, original_query_meta_,
              &converted_sparse_values, &converted_meta) != 0) {
        LOG_ERROR("Failed to convert sparse vector into target representation");
        return IndexError_Runtime;
      }
      if (!matches_index_meta(converted_meta, target_streamer_->meta())) {
        LOG_ERROR(
            "Converted sparse vector metadata does not match target streamer");
        return IndexError_Mismatch;
      }
      values = converted_sparse_values.data();
    }

    const size_t expected_unit_size =
        target_builder_ != nullptr ? original_query_meta_.unit_size()
                                   : target_streamer_->meta().unit_size();
    size_t expected_size = 0;
    if (!checked_multiply(sparse_count, expected_unit_size, &expected_size)) {
      LOG_ERROR("Sparse vector size overflows");
      return IndexError_InvalidFormat;
    }
    size_t raw_source_size = 0;
    if (!checked_multiply(sparse_count, streamer->meta().unit_size(),
                          &raw_source_size)) {
      LOG_ERROR("Source sparse vector size overflows");
      return IndexError_InvalidFormat;
    }
    const size_t source_size =
        need_convert ? converted_sparse_values.size()
                     : (need_revert ? sparse_values.size() : raw_source_size);
    if (source_size != expected_size) {
      LOG_ERROR(
          "Source sparse vector has an invalid size: actual=%zu expected=%zu",
          source_size, expected_size);
      return IndexError_Mismatch;
    }
    if (!need_revert && !need_convert) {
      sparse_values.resize(expected_size);
      if (!sparse_values.empty()) {
        memcpy(sparse_values.data(), values, sparse_values.size());
      }
    } else if (need_convert) {
      sparse_values = std::move(converted_sparse_values);
    }

    // TODO: use id instead of key
    if (*next_id > (std::numeric_limits<uint32_t>::max)()) {
      LOG_ERROR("Target sparse vector id overflows uint32 range");
      return IndexError_InvalidFormat;
    }
    if (!sparse_mt_list_.produce(SparseVectorItem((*next_id)++,
                                                  std::move(sparse_indices),
                                                  std::move(sparse_values)))) {
      LOG_ERROR("Produce vector to queue failed. key[%lu]",
                (size_t)iterator->key());
      return IndexError_Runtime;
    }
    iterator->next();
  }
  return 0;
}

void MixedStreamerReducer::PushToDocCache(const IndexQueryMeta &meta,
                                          uint32_t doc_id, std::string &doc) {
  std::lock_guard<std::mutex> lock(mutex_);
  while (doc_cache_.size() <= doc_id) {
    std::string fake_data(meta.dimension() * meta.unit_size(), 0);
    doc_cache_.push_back(std::make_pair(kInvalidKey, fake_data));
  }
  doc_cache_[doc_id] = std::make_pair(doc_id, doc);
}

int MixedStreamerReducer::IndexBuild() {
  IndexHolder::Pointer target_holder;
  if (original_query_meta_.data_type() == core::IndexMeta::DataType::DT_FP16) {
    auto holder = std::make_shared<
        zvec::core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_FP16>>(
        original_query_meta_.dimension());
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<uint16_t> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    target_holder = holder;
  } else if (original_query_meta_.data_type() ==
             core::IndexMeta::DataType::DT_FP32) {
    auto holder = std::make_shared<
        zvec::core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_FP32>>(
        original_query_meta_.dimension());
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<float> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    target_holder = holder;
  } else if (original_query_meta_.data_type() ==
             core::IndexMeta::DataType::DT_INT8) {
    auto holder = std::make_shared<
        zvec::core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_INT8>>(
        original_query_meta_.dimension());
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<uint8_t> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    target_holder = holder;
  } else {
    LOG_ERROR("data_type is not support");
    return core::IndexError_Runtime;
  }
  if (target_builder_converter_) {
    int ret = core::IndexConverter::TrainAndTransform(target_builder_converter_,
                                                      target_holder);
    if (ret != 0) {
      LOG_ERROR("Failed to convert target holder, ret=%d", ret);
      return ret;
    }
    target_holder = target_builder_converter_->result();
    if (!target_holder) {
      LOG_ERROR("Target converter returned no result holder");
      return core::IndexError_Runtime;
    }
  }
  int ret = target_builder_->train(target_holder);
  if (ret != 0) {
    LOG_ERROR("Failed to train target builder, ret=%d", ret);
    return ret;
  }
  ret = target_builder_->build(target_holder);
  if (ret != 0) {
    LOG_ERROR("Failed to build target index, ret=%d", ret);
    return ret;
  }
  return 0;
}

INDEX_FACTORY_REGISTER_STREAMER_REDUCER_ALIAS(MixedStreamerReducer,
                                              MixedStreamerReducer);

}  // namespace core
}  // namespace zvec

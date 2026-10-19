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
#include "ivf_dumper.h"

namespace zvec {
namespace core {

int IVFDumper::reserve_keys(size_t count) try {
  if (keys_ready_ || header_.total_vector_count != 0)
    return IndexError_InvalidArgument;
  if (count > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<size_t>::max() / sizeof(KeyOrderRecord))
    return IndexError_Overflow;
  paged_keys_ = ailego::MemoryLimitPool::get_instance().capacity() != 0;
  if (paged_keys_) {
    size_t bytes = count * sizeof(KeyOrderRecord);
    if (bytes > 64 * 1024) {
      if (bytes > std::numeric_limits<size_t>::max() / 2)
        return IndexError_Overflow;
      bytes *= 2;  // Sort runs share this file's page table.
    }
    int ret = key_records_.init(bytes, 64 * 1024);
    if (ret != 0) return ret;
  } else {
    keys_.reserve(count);
  }
  key_capacity_ = count;
  keys_ready_ = true;
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

int IVFDumper::append_keys(const uint64_t *keys, size_t count) try {
  if (!keys_ready_) return IndexError_NoReady;
  if (count > key_capacity_ - key_count_) return IndexError_Overflow;
  if (!paged_keys_) {
    keys_.insert(keys_.end(), keys, keys + count);
    key_count_ += count;
    return 0;
  }
  constexpr size_t kBatch = 4096;
  key_scratch_.resize(std::min(count, kBatch));
  for (size_t begin = 0; begin < count;) {
    const size_t n = std::min(kBatch, count - begin);
    for (size_t i = 0; i < n; ++i) {
      key_scratch_[i].key = keys[begin + i];
      key_scratch_[i].id = static_cast<uint32_t>(key_count_ + begin + i);
    }
    int ret =
        key_records_.write((key_count_ + begin) * sizeof(KeyOrderRecord),
                           key_scratch_.data(), n * sizeof(KeyOrderRecord));
    if (ret != 0) return ret;
    begin += n;
  }
  key_count_ += count;
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

int IVFDumper::dump_key_column(bool mapping) const try {
  constexpr size_t kBatch = 4096;
  const size_t nmax = std::min(key_count_, kBatch);
  ailego::BudgetedVector<KeyOrderRecord> records(nmax);
  ailego::BudgetedVector<uint64_t> keys(mapping ? 0 : nmax);
  ailego::BudgetedVector<uint32_t> ids(mapping ? nmax : 0);
  const size_t stride = mapping ? sizeof(uint32_t) : sizeof(uint64_t);
  uint32_t crc = 0;
  for (size_t begin = 0; begin < key_count_;) {
    const size_t n = std::min(kBatch, key_count_ - begin);
    int ret = key_records_.read(begin * sizeof(KeyOrderRecord), records.data(),
                                n * sizeof(KeyOrderRecord));
    if (ret != 0) return ret;
    for (size_t i = 0; i < n; ++i) {
      if (mapping)
        ids[i] = records[i].id;
      else
        keys[i] = records[i].key;
    }
    const void *data = mapping ? static_cast<const void *>(ids.data())
                               : static_cast<const void *>(keys.data());
    if (dumper_->write(data, n * stride) != n * stride)
      return IndexError_WriteData;
    crc = ailego::Crc32c::Hash(data, n * stride, crc);
    begin += n;
  }
  const size_t bytes = key_count_ * stride;
  size_t padding = 0;
  int ret = dump_padding(bytes, &padding);
  if (ret != 0) return ret;
  ret = dumper_->append(mapping ? IVF_MAPPING_SEG_ID : IVF_KEYS_SEG_ID, bytes,
                        padding, crc);
  if (ret == 0) dumped_size_ += bytes + padding;
  return ret;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_WriteData;
}

int IVFDumper::dump_inverted_vector(uint32_t inverted_list_id, uint64_t key,
                                    const void *vec) {
  if (!keys_ready_) return IndexError_NoReady;
  if (header_.total_vector_count >= key_capacity_) return IndexError_Overflow;
  int ret = this->check_dump_inverted_list(inverted_list_id);
  ivf_check_error_code(ret);

  ++inverted_lists_meta_[cur_list_id_].vector_count;
  ++header_.total_vector_count;
  block_.emplace(key, vec, IndexMeta::MajorOrder::MO_ROW);
  if (block_.full()) {
    ret = this->dump_block();
    ivf_check_error_code(ret);
  }
  return 0;
}

int IVFDumper::dump_inverted_block(uint32_t inverted_list_id,
                                   const uint64_t *keys, const void *vecs,
                                   uint32_t vector_count, bool column_major) {
  if (!keys_ready_) return IndexError_NoReady;
  if (vector_count > key_capacity_ - header_.total_vector_count)
    return IndexError_Overflow;
  int ret = this->check_dump_inverted_list(inverted_list_id);
  ivf_check_error_code(ret);

  if (block_.match_order(column_major ? IndexMeta::MajorOrder::MO_COLUMN
                                      : IndexMeta::MajorOrder::MO_ROW) &&
      vector_count == block_.capacity()) {
    // Dump the block directly
    size_t size = vector_count * meta_.element_size();
    size_t pd_size = ailego_align(size, 32) - size;
    if (dumper_->write(vecs, size) != size) {
      LOG_ERROR("Failed to write data into dumper %s", dumper_->name().c_str());
      return IndexError_WriteData;
    }
    if (pd_size > 0) {
      std::string padding(pd_size, '\0');
      if (dumper_->write(padding.data(), pd_size) != pd_size) {
        return IndexError_WriteData;
      }
    }
    ret = append_keys(keys, vector_count);
    ivf_check_error_code(ret);
    ++inverted_lists_meta_[cur_list_id_].block_count;
    ++header_.block_count;
    header_.inverted_body_size += size + pd_size;
  } else {
    size_t step_size = meta_.element_size();
    if (column_major) {
      step_size = IndexMeta::AlignSizeof(meta_.data_type());
    }
    for (size_t i = 0; i < vector_count; ++i) {
      auto v = reinterpret_cast<const char *>(vecs) + i * step_size;
      block_.emplace(keys[i], v,
                     column_major ? IndexMeta::MajorOrder::MO_COLUMN
                                  : IndexMeta::MajorOrder::MO_ROW);
      if (block_.full()) {
        ret = this->dump_block();
        ivf_check_error_code(ret);
      }
    }
  }

  inverted_lists_meta_[cur_list_id_].vector_count += vector_count;
  header_.total_vector_count += vector_count;

  return 0;
}

int IVFDumper::dump_container_segment(const IndexStorage::Pointer &container,
                                      const std::string &segmemt_id) {
  auto seg = container->get(segmemt_id, 2);
  if (!seg) {
    LOG_ERROR("Failed to fetch segment %s from %s", segmemt_id.c_str(),
              container->name().c_str());
    return IndexError_InvalidFormat;
  }

  const size_t batch_size = 32 * 1024;
  const size_t total_size = seg->data_size() + seg->padding_size();
  size_t off = 0;
  while (off < total_size) {
    IndexStorage::MemoryBlock data;
    size_t rd_size = std::min(batch_size, total_size - off);
    if (seg->read(off, data, rd_size) != rd_size) {
      LOG_ERROR("Failed to read data, off=%zu size=%zu", off, rd_size);
      return IndexError_ReadData;
    }
    if (dumper_->write(data.data(), rd_size) != rd_size) {
      LOG_ERROR("Failed to write data, size=%zu", rd_size);
      return IndexError_WriteData;
    }
    off += rd_size;
  }

  int ret = dumper_->append(segmemt_id, seg->data_size(), seg->padding_size(),
                            seg->data_crc());
  ivf_check_with_msg(ret, "Failed to append %s", segmemt_id.c_str());

  dumped_size_ += total_size;

  return 0;
}

int IVFDumper::dump_inverted_vector_finished() {
  if (!keys_ready_) return IndexError_NoReady;
  //! Dump Inverted Index Segment
  if (!block_.empty()) {
    int ret = this->dump_block();
    ivf_check_error_code(ret);
  }
  header_.block_size = block_.block_size();
  size_t segment_size = header_.inverted_body_size;
  int ret = dumper_->append(IVF_INVERTED_BODY_SEG_ID, segment_size, 0, 0);
  if (ret != 0) {
    LOG_ERROR("Failed to append to segment %s, ret=%d",
              IVF_INVERTED_BODY_SEG_ID.c_str(), ret);
    return ret;
  }
  dumped_size_ += segment_size;

  //! Dump Inverted Index Header Segment
  std::string str;
  meta_.serialize(&str);
  header_.header_size = sizeof(header_) + str.size();
  header_.index_meta_size = str.size();
  header_.inverted_list_count = inverted_lists_meta_.size();
  if (dumper_->write(&header_, sizeof(header_)) != sizeof(header_)) {
    LOG_ERROR("Failed to write data, size %zu", sizeof(header_));
    return IndexError_WriteData;
  }
  if (dumper_->write(str.data(), str.size()) != str.size()) {
    LOG_ERROR("Failed to write data, size %zu", str.size());
    return IndexError_WriteData;
  }
  size_t padding_size = 0;
  ret = this->dump_padding(header_.header_size, &padding_size);
  ivf_check_error_code(ret);
  ret = dumper_->append(IVF_INVERTED_HEADER_SEG_ID, header_.header_size,
                        padding_size, 0);
  if (ret != 0) {
    LOG_ERROR("Failed to append to segment %s, ret:%d",
              IVF_INVERTED_HEADER_SEG_ID.c_str(), ret);
    return ret;
  }
  dumped_size_ += header_.header_size + padding_size;

  LOG_DEBUG(
      "Dump header info: blocks=%u block_size=%u block_vec_count=%u "
      "inverted_list_count=%u total_vecs=%u inverted_size=%zu",
      header_.block_count, header_.block_size, header_.block_vector_count,
      header_.inverted_list_count, header_.total_vector_count,
      static_cast<size_t>(header_.inverted_body_size));

  //! Dump Inverted Lists Meta Segment
  segment_size = inverted_lists_meta_.size() * sizeof(InvertedListMeta);
  ret = this->dump_segment(IVF_INVERTED_META_SEG_ID,
                           inverted_lists_meta_.data(), segment_size);
  ivf_check_error_code(ret);

  if (paged_keys_) {
    ret = dump_key_column(false);
    ivf_check_error_code(ret);
    if (!key_records_.paged()) {
      ret = key_records_.truncate(key_count_ * sizeof(KeyOrderRecord));
      ivf_check_error_code(ret);
    }
    ret = SortPagedRecords<KeyOrderRecord>(
        &key_records_,
        [](const auto &a, const auto &b) {
          return a.key != b.key ? a.key < b.key : a.id < b.id;
        },
        key_records_.paged() ? key_count_ : std::numeric_limits<size_t>::max());
    ivf_check_error_code(ret);
    ret = dump_key_column(true);
    ivf_check_error_code(ret);
    key_records_ = PagedBuildBuffer();
  } else {
    ret = this->dump_segment(IVF_KEYS_SEG_ID, keys_.data(),
                             keys_.size() * sizeof(keys_[0]));
    ivf_check_error_code(ret);
    // Keys have already been written. Only sorted ordinals are needed; do
    // not reorder the keys or allocate a second, inverse permutation.
    ailego::BudgetedVector<uint32_t> mapping(keys_.size());
    std::iota(mapping.begin(), mapping.end(), 0u);
    std::sort(mapping.begin(), mapping.end(), [&](uint32_t a, uint32_t b) {
      return keys_[a] != keys_[b] ? keys_[a] < keys_[b] : a < b;
    });
    ret = this->dump_segment(IVF_MAPPING_SEG_ID, mapping.data(),
                             mapping.size() * sizeof(uint32_t));
    ivf_check_error_code(ret);
    ailego::BudgetedVector<uint64_t>().swap(keys_);
  }

  //! Dump the Offsets Segment
  return this->dump_offsets_segment();
}

int IVFDumper::dump_centroid_index(const void *data, size_t size) {
  int ret = this->dump_segment(IVF_CENTROID_SEG_ID, data, size);
  ivf_check_error_code(ret);

  return 0;
}

int IVFDumper::dump_centroid_index(
    size_t size, const std::function<int(size_t, void *, size_t)> &read) {
  ailego::BudgetedVector<uint8_t> scratch(std::min<size_t>(size, 64 * 1024));
  uint32_t crc = 0;
  for (size_t offset = 0; offset < size;) {
    const size_t bytes = std::min(scratch.size(), size - offset);
    int ret = read(offset, scratch.data(), bytes);
    if (ret != 0) return ret;
    if (dumper_->write(scratch.data(), bytes) != bytes)
      return IndexError_WriteData;
    crc = ailego::Crc32c::Hash(scratch.data(), bytes, crc);
    offset += bytes;
  }
  size_t padding = 0;
  int ret = dump_padding(size, &padding);
  if (ret == 0) ret = dumper_->append(IVF_CENTROID_SEG_ID, size, padding, crc);
  if (ret == 0) dumped_size_ += size + padding;
  return ret;
}

int IVFDumper::dump_quantizer_params(
    const ailego::BudgetedVector<IndexConverter::Pointer> &quantizers) {
  if (meta_.reformer_name() != kInt8ReformerName &&
      meta_.reformer_name() != kInt4ReformerName) {
    // IntegerQuantizer params is support only
    return 0;
  }
  if (quantizers.size() == 1) {
    //! Donot dump, using reformer params in IndexMeta
    return 0;
  }

  if (quantizers.size() != header_.inverted_list_count) {
    LOG_ERROR("Mismatch size, quantizers=%zu, inverted_list_count=%u",
              quantizers.size(), header_.inverted_list_count);
    return IndexError_Logic;
  }
  bool int8_quantizer = meta_.reformer_name() == kInt8ReformerName;
  ailego::BudgetedVector<InvertedIntegerQuantizerParams> params;
  params.resize(header_.inverted_list_count);
  for (size_t i = 0; i < quantizers.size(); ++i) {
    auto &p = quantizers[i]->meta().reformer_params();
    auto &scale_key = int8_quantizer ? INT8_QUANTIZER_REFORMER_SCALE
                                     : INT4_QUANTIZER_REFORMER_SCALE;
    auto &bias_key = int8_quantizer ? INT8_QUANTIZER_REFORMER_BIAS
                                    : INT4_QUANTIZER_REFORMER_BIAS;
    if (inverted_lists_meta_[i].vector_count > 0 &&
        (!p.has(scale_key) || !p.has(bias_key))) {
      LOG_ERROR("Miss reformer params %s or %s", bias_key.c_str(),
                scale_key.c_str());
      return IndexError_Logic;
    }

    params[i].bias = p.get_as_float(bias_key);
    params[i].scale = p.get_as_float(scale_key);
  }

  return this->dump_segment(
      int8_quantizer ? IVF_INT8_QUANTIZED_PARAMS_SEG_ID
                     : IVF_INT4_QUANTIZED_PARAMS_SEG_ID,
      params.data(), params.size() * sizeof(InvertedIntegerQuantizerParams));
}

int IVFDumper::dump_turbo_quantizer(
    const turbo::Quantizer::Pointer &quantizer) {
  if (!quantizer) {
    return IndexError_InvalidArgument;
  }
  std::string state;
  int ret = quantizer->serialize(&state);
  ivf_check_with_msg(ret, "Failed to serialize Turbo IVF quantizer");
  if (quantizer->require_train() && state.empty()) {
    LOG_ERROR("Turbo IVF quantizer has no serialized training state");
    return IndexError_InvalidFormat;
  }
  // Stateless quantizers use the default empty serialization. The segment
  // still marks the index as having a persisted Turbo quantizer.
  return this->dump_segment(IVF_TURBO_QUANTIZER_SEG_ID, state.data(),
                            state.size());
}

int IVFDumper::dump_original_vector(const void *data, size_t size) {
  if (dumped_feature_count_ >= header_.total_vector_count) {
    LOG_ERROR("Dump too much orignal features, expect=%u",
              header_.total_vector_count);
    return IndexError_Logic;
  }

  if (dumper_->write(data, size) != size) {
    LOG_ERROR("Dumper write features failed");
    return IndexError_WriteData;
  }
  dumped_features_size_ += size;
  ++dumped_feature_count_;
  if (dumped_feature_count_ == header_.total_vector_count) {
    //! Dump features finished, dump the meta
    size_t padding_size = 0;
    int ret = this->dump_padding(size, &padding_size);
    ivf_check_error_code(ret);

    ret = dumper_->append(IVF_FEATURES_SEG_ID, dumped_features_size_,
                          padding_size, 0);
    if (ret != 0) {
      LOG_ERROR("Dumper append segment %s failed, ret:%d",
                IVF_FEATURES_SEG_ID.c_str(), ret);
      return ret;
    }
    dumped_size_ += dumped_features_size_;
  }

  return 0;
}

int IVFDumper::check_dump_inverted_list(uint32_t inverted_list_id) {
  if (inverted_list_id < cur_list_id_) {
    LOG_ERROR("Invalid backward vector dumping, want=%u cur=%u",
              inverted_list_id, cur_list_id_);
    return IndexError_Logic;
  }
  if (inverted_list_id >= inverted_lists_meta_.size()) {
    LOG_ERROR("Invalid inverted_list_id=%u, lists_size=%zu", inverted_list_id,
              inverted_lists_meta_.size());
    return IndexError_Logic;
  }
  if (inverted_list_id != cur_list_id_) {
    //! flush previous inverted_list block
    int ret = this->dump_block();
    ivf_check_error_code(ret);
    for (auto idx = cur_list_id_ + 1; idx <= inverted_list_id; ++idx) {
      inverted_lists_meta_[idx].offset = header_.inverted_body_size;
      inverted_lists_meta_[idx].id_offset = header_.total_vector_count;
    }
    cur_list_id_ = inverted_list_id;
  }

  return 0;
}

int IVFDumper::dump_offsets_segment() const {
  bool col_pri = meta_.major_order() == IndexMeta::MajorOrder::MO_COLUMN;
  size_t total_size = 0;
  constexpr size_t kBatch = 4096;
  ailego::BudgetedVector<InvertedVecLocation> offsets;
  offsets.reserve(kBatch);
  auto flush = [&]() -> int {
    const size_t bytes = offsets.size() * sizeof(InvertedVecLocation);
    if (bytes && dumper_->write(offsets.data(), bytes) != bytes)
      return IndexError_WriteData;
    total_size += bytes;
    offsets.clear();
    return 0;
  };
  for (size_t i = 0; i < inverted_lists_meta_.size(); ++i) {
    const auto &m = inverted_lists_meta_[i];
    size_t vec_cnt = m.vector_count;
    size_t idx = 0;
    uint64_t off = m.offset;
    size_t align_idx = vec_cnt - vec_cnt % block_vector_count_;
    for (size_t j = 0; j < vec_cnt; ++j) {
      if (col_pri && j < align_idx) {
        offsets.emplace_back(off + idx * block_.align_size(), true);
      } else {
        offsets.emplace_back(off + idx * block_.element_size(), false);
      }
      if (offsets.size() == kBatch) {
        int ret = flush();
        if (ret != 0) return ret;
      }
      ++idx;
      if (idx == block_vector_count_) {
        off += header_.block_size;
        idx = 0;
      }
    }
    if (idx != 0) {
      off += (vec_cnt - align_idx) * meta_.element_size();
    }
  }

  int status = flush();
  if (status != 0) return status;

  size_t padding_size = 0;
  int ret = this->dump_padding(total_size, &padding_size);
  ivf_check_error_code(ret);

  ret = dumper_->append(IVF_OFFSETS_SEG_ID, total_size, padding_size, 0);
  if (ret != 0) {
    LOG_ERROR("Dumper append segment %s failed, ret:%d",
              IVF_OFFSETS_SEG_ID.c_str(), ret);
    return ret;
  }

  dumped_size_ += total_size + padding_size;

  return 0;
}

int IVFDumper::dump_segment(const std::string &segment_id, const void *data,
                            size_t size) const {
  size_t len = dumper_->write(data, size);
  if (len != size) {
    LOG_ERROR("Dump segment %s data failed, expect=%zu, actual=%zu",
              segment_id.c_str(), size, len);
    return IndexError_WriteData;
  }

  size_t padding_size = 0;
  int ret = this->dump_padding(size, &padding_size);
  ivf_check_error_code(ret);

  uint32_t crc = ailego::Crc32c::Hash(data, size);
  ret = dumper_->append(segment_id, size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d", segment_id.c_str(), ret);
    return ret;
  }
  dumped_size_ += size + padding_size;

  return 0;
}

int IVFDumper::dump_padding(size_t data_size, size_t *padding_size) const {
  *padding_size = IVFUtility::AlignedSize(data_size) - data_size;
  if (*padding_size == 0) {
    return 0;
  }

  std::string padding(*padding_size, '\0');
  if (dumper_->write(padding.data(), *padding_size) != *padding_size) {
    LOG_ERROR("Append padding failed, size %lu", *padding_size);
    return IndexError_WriteData;
  }

  return 0;
}

int IVFDumper::dump_block() {
  if (block_.empty()) {
    return 0;
  }

  size_t size = ailego_align(block_.bytes(), 32);
  if (dumper_->write(block_.data(), size) != size) {
    LOG_ERROR("Failed to write data into dumper %s", dumper_->name().c_str());
    return IndexError_WriteData;
  }
  auto &keys = block_.keys();
  int ret = append_keys(keys.data(), keys.size());
  ivf_check_error_code(ret);
  ++inverted_lists_meta_[cur_list_id_].block_count;
  ++header_.block_count;
  header_.inverted_body_size += size;
  block_.clear();

  return 0;
}

}  // namespace core
}  // namespace zvec

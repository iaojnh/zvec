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

#include "diskann_pq_trainer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include "diskann_entity.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

DiskAnnPqTrainer::DiskAnnPqTrainer(uint32_t max_train_sample_count,
                                   double train_sample_ratio)
    : max_train_sample_count_{max_train_sample_count},
      train_sample_ratio_{train_sample_ratio} {}

DiskAnnPqTrainer::~DiskAnnPqTrainer() {}

int DiskAnnPqTrainer::gen_random_sample(IndexHolder::Pointer holder,
                                        const IndexMeta &meta,
                                        std::string &sample_data,
                                        size_t &sample_size) {
  sample_data.clear();
  sample_size = 0;
  if (!holder || holder->count() == 0 || meta.element_size() == 0 ||
      max_train_sample_count_ == 0 || !std::isfinite(train_sample_ratio_) ||
      train_sample_ratio_ <= 0.0 || train_sample_ratio_ > 1.0) {
    LOG_ERROR("Invalid DiskAnn PQ sampling configuration");
    return IndexError_InvalidArgument;
  }

  const size_t holder_count = holder->count();
  const size_t ratio_sample_count = static_cast<size_t>(
      std::ceil(static_cast<long double>(holder_count) * train_sample_ratio_));
  const size_t target_sample_count =
      (std::min)({holder_count, ratio_sample_count,
                  static_cast<size_t>(max_train_sample_count_)});
  const size_t vec_size = meta.element_size();
  if (target_sample_count == 0 ||
      target_sample_count > (std::numeric_limits<size_t>::max)() / vec_size) {
    LOG_ERROR("Invalid DiskAnn PQ sample buffer size");
    return IndexError_InvalidLength;
  }

  try {
    sample_data.resize(target_sample_count * vec_size);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::length_error &) {
    return IndexError_InvalidLength;
  }

  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }

  // Reservoir sampling avoids the previous prefix bias while keeping builds
  // deterministic for identical inputs.
  std::mt19937_64 generator(456321);
  size_t observed_count = 0;
  while (iter->is_valid()) {
    const void *vec = iter->data();
    if (vec == nullptr) {
      LOG_ERROR("Failed to read a vector for DiskAnn PQ training");
      return IndexError_ReadData;
    }

    size_t sample_index = observed_count;
    if (observed_count >= target_sample_count) {
      std::uniform_int_distribution<size_t> distribution(0, observed_count);
      sample_index = distribution(generator);
    }
    if (sample_index < target_sample_count) {
      std::memcpy(sample_data.data() + sample_index * vec_size, vec, vec_size);
    }
    ++observed_count;
    iter->next();
  }

  sample_size = (std::min)(observed_count, target_sample_count);
  if (sample_size == 0) {
    sample_data.clear();
    LOG_ERROR("DiskAnn PQ training holder contains no vectors");
    return IndexError_InvalidLength;
  }
  sample_data.resize(sample_size * vec_size);

  return 0;
}

template <typename T>
int DiskAnnPqTrainer::prepare_pq_train_data(
    const IndexMeta &meta, size_t num_train, std::string &train_data,
    bool use_zero_mean, std::vector<uint8_t> &centroid,
    std::shared_ptr<CompactIndexFeatures> &train_features) {
  uint32_t dim = meta.dimension();
  uint32_t vec_size = meta.element_size();
  if (num_train == 0 || dim == 0 || vec_size == 0 ||
      num_train > (std::numeric_limits<size_t>::max)() / vec_size ||
      train_data.size() < num_train * vec_size || !train_features) {
    LOG_ERROR("Invalid DiskAnn PQ training data");
    return IndexError_InvalidLength;
  }

  std::vector<T> train_data_processed;
  try {
    train_data_processed.resize(num_train * static_cast<size_t>(dim));
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::length_error &) {
    return IndexError_InvalidLength;
  }
  std::memcpy(train_data_processed.data(), train_data.data(),
              num_train * vec_size);

  // use fp32 to accumulate to avoid overflow
  std::vector<float> centroid_temp(dim);
  for (uint64_t d = 0; d < dim; d++) {
    centroid_temp[d] = 0;
  }

  T *train_data_processed_ptr = train_data_processed.data();

  if (use_zero_mean) {
    for (uint64_t d = 0; d < dim; d++) {
      for (uint64_t p = 0; p < num_train; p++) {
        centroid_temp[d] += train_data_processed_ptr[p * dim + d];
      }
      centroid_temp[d] /= num_train;
    }

    for (uint64_t d = 0; d < dim; d++) {
      for (uint64_t p = 0; p < num_train; p++) {
        train_data_processed_ptr[p * dim + d] -= centroid_temp[d];
      }
    }
  }

  for (size_t i = 0; i < num_train; ++i) {
    train_features->emplace(train_data_processed_ptr + i * dim);
  }

  // copy the centroid out
  centroid.resize(vec_size);
  for (uint64_t d = 0; d < dim; d++) {
    const T centroid_value = centroid_temp[d];
    std::memcpy(centroid.data() + d * sizeof(T), &centroid_value, sizeof(T));
  }

  return 0;
}

template <typename T>
int DiskAnnPqTrainer::convert_pivot_data(
    const IndexMeta &meta, uint32_t num_centers, uint32_t pq_chunk_num,
    const std::vector<uint32_t> &chunk_dims,
    const std::vector<uint32_t> &chunk_offsets,
    IndexCluster::CentroidList &centroids,
    std::vector<uint8_t> &full_pivot_data) {
  uint32_t dim = meta.dimension();
  uint32_t element_size = meta.element_size();

  full_pivot_data.resize(num_centers * element_size);

  for (size_t chunk = 0; chunk < pq_chunk_num; ++chunk) {
    for (size_t cluster = 0; cluster < num_centers; ++cluster) {
      size_t idx = chunk * num_centers + cluster;

      uint8_t *pivot_data_ptr =
          full_pivot_data.data() +
          (cluster * dim + chunk_offsets[chunk]) * sizeof(T);
      std::memcpy(pivot_data_ptr, centroids[idx].feature(),
                  chunk_dims[chunk] * sizeof(T));
    }
  }

  return 0;
}

template int DiskAnnPqTrainer::convert_pivot_data<float>(
    const IndexMeta &, uint32_t, uint32_t, const std::vector<uint32_t> &,
    const std::vector<uint32_t> &, IndexCluster::CentroidList &,
    std::vector<uint8_t> &);
template int DiskAnnPqTrainer::convert_pivot_data<ailego::Float16>(
    const IndexMeta &, uint32_t, uint32_t, const std::vector<uint32_t> &,
    const std::vector<uint32_t> &, IndexCluster::CentroidList &,
    std::vector<uint8_t> &);

int DiskAnnPqTrainer::train_pq(IndexThreads::Pointer threads,
                               const IndexMeta &meta, std::string &train_data,
                               size_t num_train, uint32_t num_centers,
                               uint32_t pq_chunk_num, uint32_t max_iterations,
                               bool use_zero_mean,
                               std::vector<uint8_t> &full_pivot_data,
                               std::vector<uint8_t> &centroid,
                               std::vector<uint32_t> &chunk_offsets) {
  uint32_t dim = meta.dimension();
  if (num_train == 0 || pq_chunk_num == 0 || pq_chunk_num > dim) {
    LOG_ERROR("Error: number of chunks more than dimension. chunk: %u, dim: %u",
              pq_chunk_num, dim);
    return IndexError_InvalidArgument;
  }

  std::shared_ptr<CompactIndexFeatures> train_features(
      new CompactIndexFeatures(meta));

  uint32_t type = meta.data_type();

  int ret;
  switch (type) {
    case IndexMeta::DataType::DT_FP32:
      ret = prepare_pq_train_data<float>(
          meta, num_train, train_data, use_zero_mean, centroid, train_features);
      if (ret != 0) {
        LOG_ERROR("Failed to prepare pq train data");
        return ret;
      }
      break;

    case IndexMeta::DataType::DT_FP16:
      ret = prepare_pq_train_data<ailego::Float16>(
          meta, num_train, train_data, use_zero_mean, centroid, train_features);
      if (ret != 0) {
        LOG_ERROR("Failed to prepare pq train data");
        return ret;
      }
      break;

    default:
      LOG_ERROR("Unsupported DiskAnn PQ training data type: %u", type);
      return IndexError_InvalidArgument;
  }

  // Do Train
  ailego::Params params;
  params.set(MULTI_CHUNK_CLUSTER_COUNT, num_centers);
  params.set(MULTI_CHUNK_CLUSTER_CHUNK_COUNT, pq_chunk_num);
  params.set(MULTI_CHUNK_CLUSTER_MAX_ITERATIONS, max_iterations);

  ret = chunk_cluster_.init(meta, params);
  if (ret != 0) {
    LOG_ERROR("Failed to get chunk cluster");
    return IndexError_InvalidArgument;
  }

  ret = chunk_cluster_.mount(train_features);
  if (ret != 0) {
    LOG_ERROR("Cannot mount train features");
    return ret;
  }


  std::vector<uint32_t> labels;

  ret = chunk_cluster_.cluster(threads, cluster_centroids_);
  if (ret != 0) {
    LOG_ERROR("Failed to cluster");
    return ret;
  }

  chunk_offsets = chunk_cluster_.chunk_dim_offsets();
  auto chunk_dims = chunk_cluster_.chunk_dims();

  switch (type) {
    case IndexMeta::DataType::DT_FP32:
      ret = convert_pivot_data<float>(meta, num_centers, pq_chunk_num,
                                      chunk_dims, chunk_offsets,
                                      cluster_centroids_, full_pivot_data);
      if (ret != 0) {
        LOG_ERROR("Failed to convert pivot data");
        return ret;
      }
      break;

    case IndexMeta::DataType::DT_FP16:
      ret = convert_pivot_data<ailego::Float16>(
          meta, num_centers, pq_chunk_num, chunk_dims, chunk_offsets,
          cluster_centroids_, full_pivot_data);
      if (ret != 0) {
        LOG_ERROR("Failed to convert pivot data");
        return ret;
      }
      break;

    default:
      return IndexError_InvalidArgument;
  }

  return 0;
}

int DiskAnnPqTrainer::train_quantized_data(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    const IndexMeta &meta, std::vector<uint8_t> &pq_full_pivot_data,
    std::vector<uint8_t> &pq_centroid, std::vector<uint32_t> &pq_chunk_offsets,
    size_t pq_chunk_num) {
  size_t train_size;
  std::string train_data;

  int ret = gen_random_sample(holder, meta, train_data, train_size);
  if (ret != 0) {
    LOG_ERROR("Get Random Sample Error, ret: %d", ret);
    return ret;
  }

  LOG_INFO("Training data with %zu samples loaded.", train_size);

  // bool use_zero_mean = (meta.metric_name() != "InnerProduct" ? true :
  // false);
  bool use_zero_mean = false;

  ret = train_pq(threads, meta, train_data, train_size, PQTable::kPQCentroidNum,
                 pq_chunk_num, PQTable::kMeanIterNum, use_zero_mean,
                 pq_full_pivot_data, pq_centroid, pq_chunk_offsets);
  if (ret != 0) {
    LOG_ERROR("Train PQ Error, ret: %d", ret);
    return ret;
  }

  return 0;
}

int DiskAnnPqTrainer::generate_pq(IndexThreads::Pointer threads,
                                  const IndexMeta &meta,
                                  IndexHolder::Pointer holder,
                                  uint32_t pq_chunk_num,
                                  std::vector<uint8_t> &centroid,
                                  std::vector<uint8_t> &block_compressed_data) {
  if (!holder) {
    LOG_ERROR("Cannot generate DiskAnn PQ data from a null holder");
    return IndexError_InvalidArgument;
  }

  uint32_t type = meta.data_type();
  uint32_t dim = meta.dimension();

  const size_t element_size = meta.element_size();
  if (pq_chunk_num == 0 || pq_chunk_num > dim || element_size == 0 ||
      centroid.size() < element_size) {
    LOG_ERROR(
        "Invalid DiskAnn PQ generation metadata: chunk=%u dim=%u "
        "element_size=%zu centroid_size=%zu",
        pq_chunk_num, dim, element_size, centroid.size());
    return IndexError_InvalidArgument;
  }

  // Do Label
  std::vector<uint32_t> labels;
  size_t num_vecs = holder->count();
  if (num_vecs == 0) {
    LOG_ERROR("Cannot generate DiskAnn PQ data for an empty holder");
    return IndexError_InvalidLength;
  }
  size_t batch_size =
      num_vecs <= compress_batch_size_ ? num_vecs : compress_batch_size_;
  if (num_vecs > (std::numeric_limits<size_t>::max)() / pq_chunk_num ||
      batch_size > (std::numeric_limits<size_t>::max)() / element_size) {
    LOG_ERROR("DiskAnn PQ output size overflows");
    return IndexError_InvalidLength;
  }

  std::vector<float> fp32_block_data;
  std::vector<ailego::Float16> fp16_block_data;
  switch (type) {
    case IndexMeta::DataType::DT_FP32:
      fp32_block_data.resize(batch_size * dim);
      break;
    case IndexMeta::DataType::DT_FP16:
      fp16_block_data.resize(batch_size * dim);
      break;
    default:
      return IndexError_InvalidArgument;
  }

  size_t block_num = DiskAnnUtil::div_round_up(num_vecs, batch_size);

  block_compressed_data.resize(num_vecs * pq_chunk_num);

  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }

  for (size_t block = 0; block < block_num; block++) {
    size_t start_id = block * batch_size;
    size_t end_id = std::min((block + 1) * batch_size, num_vecs);

    size_t cur_block_size = end_id - start_id;

    for (size_t i = 0; i < cur_block_size; i++) {
      if (!iter->is_valid()) {
        LOG_ERROR("DiskAnn holder ended before its declared vector count");
        return IndexError_InvalidLength;
      }
      const void *vec = iter->data();
      if (vec == nullptr) {
        LOG_ERROR("Failed to read a vector while generating DiskAnn PQ data");
        return IndexError_ReadData;
      }
      void *destination =
          type == IndexMeta::DataType::DT_FP32
              ? static_cast<void *>(fp32_block_data.data() + i * dim)
              : static_cast<void *>(fp16_block_data.data() + i * dim);
      std::memcpy(destination, vec, element_size);
      iter->next();
    }

    LOG_INFO("Processing Docs, Range: [%zu, %zu)..", start_id, end_id);

    std::shared_ptr<CompactIndexFeatures> block_features(
        new CompactIndexFeatures(meta));

    switch (type) {
      case IndexMeta::DataType::DT_FP32: {
        std::vector<float> typed_centroid(dim);
        std::memcpy(typed_centroid.data(), centroid.data(), element_size);
        DiskAnnUtil::convert_vector_to_residual<float>(
            fp32_block_data.data(), cur_block_size, dim, typed_centroid.data());
        for (size_t i = 0; i < cur_block_size; i++) {
          block_features->emplace(fp32_block_data.data() + i * dim);
        }
        break;
      }
      case IndexMeta::DataType::DT_FP16: {
        std::vector<ailego::Float16> typed_centroid(dim);
        std::memcpy(typed_centroid.data(), centroid.data(), element_size);
        DiskAnnUtil::convert_vector_to_residual<ailego::Float16>(
            fp16_block_data.data(), cur_block_size, dim, typed_centroid.data());
        for (size_t i = 0; i < cur_block_size; i++) {
          block_features->emplace(fp16_block_data.data() + i * dim);
        }
        break;
      }
      default:
        return IndexError_InvalidArgument;
    }

    int ret = chunk_cluster_.mount(block_features);
    if (ret != 0) {
      LOG_ERROR("Cannot mount block features");
      return ret;
    }

    ret = chunk_cluster_.label(threads, cluster_centroids_, &labels);
    if (ret != 0) {
      LOG_ERROR("Failed to label");
      return ret;
    }

    std::vector<uint8_t> compressed_data(cur_block_size * pq_chunk_num);

    DiskAnnUtil::convert_types_uint32_to_uint8(
        labels.data(), compressed_data.data(), cur_block_size, pq_chunk_num);

    memcpy(&(block_compressed_data[0]) + start_id * pq_chunk_num,
           compressed_data.data(), cur_block_size * pq_chunk_num);

    LOG_INFO("Generate PQ Data Done.");
  }
  if (iter->is_valid()) {
    LOG_ERROR("DiskAnn holder contains more vectors than its declared count");
    return IndexError_InvalidLength;
  }

  return 0;
}

int DiskAnnPqTrainer::generate_quantized_data(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    const IndexMeta &meta, std::vector<uint8_t> &pq_centroid,
    std::vector<uint8_t> &block_compressed_data, size_t pq_chunk_num) {
  // bool use_zero_mean = (meta.metric_name() != "InnerProduct" ? true :
  // false);

  int ret = generate_pq(threads, meta, holder, pq_chunk_num, pq_centroid,
                        block_compressed_data);
  if (ret != 0) {
    LOG_ERROR("Generate PQ Error, ret: %d", ret);
    return ret;
  }

  return 0;
}

}  // namespace core
}  // namespace zvec

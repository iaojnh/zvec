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

#include "rabitq_reformer.h"
#include <string>
#include <vector>
#include <rabitqlib/defines.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/core/framework/index_metric.h>
#include "core/algorithm/cluster/linear_seeker.h"
#include "utility/immutable_segment_view.h"
#include "utility/storage_segment_dumper.h"
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_features.h"
#include "zvec/core/framework/index_meta.h"
#include "zvec/core/framework/index_storage.h"
#include "hnsw_rabitq_query_entity.h"
#include "paged_rabitq_rotator.h"
#include "rabitq_converter.h"
#include "rabitq_utils.h"

namespace zvec {
namespace core {

// All rabitqlib types are confined to this translation unit via pimpl.
struct RabitqReformer::Impl {
  // RaBitQ parameters
  size_t num_clusters{0};
  size_t ex_bits{0};
  size_t dimension{0};
  size_t padded_dim{0};
  size_t size_bin_data{0};
  size_t size_ex_data{0};
  bool loaded{false};

  // Original centroids: num_clusters * dimension (for LinearSeeker query)
  ImmutableSegmentView centroids;
  // Rotated centroids: num_clusters * padded_dim (for quantization)
  ImmutableSegmentView rotated_centroids;

  rabitqlib::RotatorType rotator_type{rabitqlib::RotatorType::FhtKacRotator};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  rabitqlib::quant::RabitqConfig query_config;
  rabitqlib::quant::RabitqConfig config;
  rabitqlib::MetricType metric_type{rabitqlib::METRIC_L2};

  LinearSeeker::Pointer centroid_seeker;
  CoherentIndexFeatures::Pointer centroid_features;
  IndexMetric::Pointer centroid_metric;
  IndexMetric::MatrixDistance centroid_distance;

  // Translate local enum to rabitqlib enum (used only inside this .cc).
  static rabitqlib::MetricType to_rabitq(RabitqMetricType m) {
    return m == RabitqMetricType::kIP ? rabitqlib::METRIC_IP
                                      : rabitqlib::METRIC_L2;
  }

  // Translate rabitqlib enum to local enum.
  static RabitqMetricType from_rabitq(rabitqlib::MetricType m) {
    return m == rabitqlib::METRIC_IP ? RabitqMetricType::kIP
                                     : RabitqMetricType::kL2;
  }

  int quantize_vector(const float *raw_vector, uint32_t cluster_id,
                      std::string *quantized_data) const;
};

RabitqReformer::RabitqReformer() : impl_(std::make_unique<Impl>()) {}

RabitqReformer::~RabitqReformer() {
  this->cleanup();
}

size_t RabitqReformer::num_clusters() const {
  return impl_->num_clusters;
}

size_t RabitqReformer::ex_bits() const {
  return impl_->ex_bits;
}

RabitqMetricType RabitqReformer::rabitq_metric_type() const {
  return Impl::from_rabitq(impl_->metric_type);
}

int RabitqReformer::init(const ailego::Params &params) {
  std::string metric_name = params.get_as_string(PARAM_RABITQ_METRIC_NAME);
  if (metric_name == "SquaredEuclidean") {
    impl_->metric_type = rabitqlib::METRIC_L2;
  } else if (metric_name == "InnerProduct") {
    impl_->metric_type = rabitqlib::METRIC_IP;
  } else if (metric_name == "Cosine") {
    impl_->metric_type = rabitqlib::METRIC_IP;
  } else {
    LOG_ERROR("Unsupported metric name: %s", metric_name.c_str());
    return IndexError_InvalidArgument;
  }
  LOG_DEBUG("Rabitq reformer init done. metric_name=%s metric_type=%d",
            metric_name.c_str(), static_cast<int>(impl_->metric_type));
  return 0;
}

int RabitqReformer::cleanup() {
  impl_->centroids = ImmutableSegmentView();
  impl_->rotated_centroids = ImmutableSegmentView();
  impl_->centroid_seeker.reset();
  impl_->centroid_features.reset();
  impl_->centroid_metric.reset();
  impl_->centroid_distance = nullptr;
  impl_->loaded = false;
  impl_->rotator.reset();
  return 0;
}

int RabitqReformer::unload() {
  return this->cleanup();
}

int RabitqReformer::load(IndexStorage::Pointer storage) {
  if (!storage) {
    LOG_ERROR("Invalid storage for load");
    return IndexError_InvalidArgument;
  }

  auto segment = storage->get(RABITQ_CONVERTER_SEG_ID);
  if (!segment) {
    LOG_ERROR("Failed to get segment %s", RABITQ_CONVERTER_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  size_t offset = 0;
  RabitqConverterHeader header;
  IndexStorage::MemoryBlock block;
  size_t size = segment->read(offset, block, sizeof(header));
  if (size != sizeof(header)) {
    LOG_ERROR("Failed to read header");
    return IndexError_InvalidFormat;
  }
  memcpy(static_cast<void *>(&header), block.data(), sizeof(header));
  impl_->dimension = header.dim;
  impl_->padded_dim = header.padded_dim;
  impl_->ex_bits = header.ex_bits;
  impl_->num_clusters = header.num_clusters;
  impl_->rotator_type =
      static_cast<rabitqlib::RotatorType>(header.rotator_type);
  offset += sizeof(header);

  if (header.dim == 0 || header.padded_dim < header.dim ||
      header.padded_dim % 64 || header.num_clusters == 0 ||
      header.ex_bits > 8 || header.rotator_type > 1)
    return IndexError_InvalidFormat;
  const size_t max_count =
      std::numeric_limits<size_t>::max() / sizeof(float) / header.padded_dim;
  if (header.num_clusters > max_count || header.dim > max_count)
    return IndexError_InvalidFormat;
  const size_t rotated_bytes =
      size_t(header.num_clusters) * header.padded_dim * sizeof(float);
  const size_t centroid_bytes =
      size_t(header.num_clusters) * header.dim * sizeof(float);
  const size_t rotator_bytes =
      header.rotator_type == 0
          ? size_t(header.dim) * header.padded_dim * sizeof(float)
          : header.padded_dim / 2;
  size_t remaining = segment->data_size() - sizeof(header);
  for (size_t bytes : {rotated_bytes, centroid_bytes, rotator_bytes}) {
    if (bytes > remaining) return IndexError_InvalidFormat;
    remaining -= bytes;
  }
  if (header.rotator_size != rotator_bytes) return IndexError_InvalidFormat;
  int ret = impl_->rotated_centroids.load(storage, RABITQ_CONVERTER_SEG_ID,
                                          offset, rotated_bytes);
  if (ret != 0) return ret;
  offset += rotated_bytes;
  ret = impl_->centroids.load(storage, RABITQ_CONVERTER_SEG_ID, offset,
                              centroid_bytes);
  if (ret != 0) return ret;
  offset += centroid_bytes;
  ret = MakeRabitqRotator(impl_->dimension, impl_->rotator_type,
                          impl_->padded_dim, &impl_->rotator, false);
  if (ret != 0) return ret;
  ret = LoadRabitqRotator(
      impl_->rotator.get(), [&](size_t pos, void *out, size_t bytes) {
        return segment->fetch(offset + pos, out, bytes) == bytes
                   ? 0
                   : IndexError_ReadData;
      });
  if (ret != 0) return ret;

  impl_->query_config = rabitqlib::quant::faster_config(
      impl_->padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);
  impl_->config =
      rabitqlib::quant::faster_config(impl_->padded_dim, impl_->ex_bits + 1);

  impl_->size_bin_data =
      rabitqlib::BinDataMap<float>::data_bytes(impl_->padded_dim);
  impl_->size_ex_data = rabitqlib::ExDataMap<float>::data_bytes(
      impl_->padded_dim, impl_->ex_bits);

  // mmap keeps the original SIMD batch seeker; pageable tables use bounded
  // rows.
  IndexMeta centroid_meta;
  centroid_meta.set_data_type(IndexMeta::DataType::DT_FP32);
  centroid_meta.set_dimension(static_cast<uint32_t>(impl_->dimension));
  // Note:
  // 1. spherical kmeans is used for InnerProduct and Cosine, so centroids are
  // normalized.
  // 2. for Cosine metric, `transform_to_entity` input is normalized, need to
  // use InnerProduct metric as Cosine metric requires extra dimension which is
  // unsuitable for centroids.
  centroid_meta.set_metric(impl_->metric_type == rabitqlib::METRIC_L2
                               ? "SquaredEuclidean"
                               : "InnerProduct",
                           0, ailego::Params());

  impl_->centroid_seeker.reset();
  impl_->centroid_features.reset();
  if (impl_->centroids.data()) {
    impl_->centroid_features = std::make_shared<CoherentIndexFeatures>();
    impl_->centroid_features->mount(centroid_meta, impl_->centroids.data(),
                                    impl_->centroids.size());
    impl_->centroid_seeker = std::make_shared<LinearSeeker>();
    ret = impl_->centroid_seeker->init(centroid_meta);
    if (ret == 0) ret = impl_->centroid_seeker->mount(impl_->centroid_features);
    if (ret != 0) return ret;
  }
  impl_->centroid_metric =
      IndexFactory::CreateMetric(centroid_meta.metric_name());
  if (!impl_->centroid_metric) return IndexError_Unsupported;
  ret = impl_->centroid_metric->init(centroid_meta,
                                     centroid_meta.metric_params());
  if (ret != 0) return ret;
  impl_->centroid_distance = impl_->centroid_metric->distance_matrix(1, 1);
  if (!impl_->centroid_distance) return IndexError_Unsupported;

  LOG_INFO(
      "Rabitq reformer load done. dimension=%zu, padded_dim=%zu, "
      "ex_bits=%zu, num_clusters=%zu, size_bin_data=%zu, size_ex_data=%zu "
      "rotator_type=%d",
      impl_->dimension, impl_->padded_dim, impl_->ex_bits, impl_->num_clusters,
      impl_->size_bin_data, impl_->size_ex_data, (int)impl_->rotator_type);
  impl_->loaded = true;
  return 0;
}

int RabitqReformer::convert(const void *record, const IndexQueryMeta &rmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  if (!impl_->loaded) {
    LOG_ERROR("Centroids not loaded yet");
    return IndexError_NoReady;
  }

  if (!record || !out) {
    LOG_ERROR("Invalid arguments for convert");
    return IndexError_InvalidArgument;
  }

  // input may be transformed, require rmeta.dimension >= dimension
  if (rmeta.dimension() < impl_->dimension ||
      rmeta.data_type() != IndexMeta::DataType::DT_FP32) {
    LOG_ERROR("Invalid record meta: dimension=%zu, data_type=%d",
              static_cast<size_t>(rmeta.dimension()), (int)rmeta.data_type());
    return IndexError_InvalidArgument;
  }

  uint32_t cluster_id = 0;
  int ret = 0;
  if (impl_->centroid_seeker) {
    Seeker::Document doc;
    ret = impl_->centroid_seeker->seek(record, impl_->dimension * sizeof(float),
                                       &doc);
    if (ret != 0) return ret;
    cluster_id = doc.index;
  } else {
    float best = std::numeric_limits<float>::max();
    std::vector<float> centroid(impl_->dimension);
    for (size_t i = 0; i < impl_->num_clusters; ++i) {
      ret = impl_->centroids.read(i * impl_->dimension * sizeof(float),
                                  centroid.data(),
                                  impl_->dimension * sizeof(float));
      if (ret != 0) return ret;
      float distance;
      impl_->centroid_distance(centroid.data(), record, impl_->dimension,
                               &distance);
      if (distance < best) {
        best = distance;
        cluster_id = static_cast<uint32_t>(i);
      }
    }
  }

  const float *vector = static_cast<const float *>(record);
  ret = impl_->quantize_vector(vector, cluster_id, out);
  if (ret != 0) {
    LOG_ERROR("Failed to quantize vector");
    return ret;
  }

  ometa->set_meta(IndexMeta::DataType::DT_INT8, (uint32_t)out->size());
  return 0;
}

int RabitqReformer::transform(const void *, const IndexQueryMeta &,
                              std::string *, IndexQueryMeta *) const {
  return IndexError_NotImplemented;
}

int RabitqReformer::transform_to_entity(const void *query,
                                        HnswRabitqQueryEntity *entity) const {
  if (!impl_->loaded) {
    LOG_ERROR("Centroids not loaded yet");
    return IndexError_NoReady;
  }

  if (!query) {
    LOG_ERROR("Invalid arguments for transform");
    return IndexError_InvalidArgument;
  }

  const float *query_vector = static_cast<const float *>(query);

  // Apply rotator
  entity->rotated_query.resize(impl_->padded_dim);
  impl_->rotator->rotate(query_vector, entity->rotated_query.data());

  // Quantize query to 4-bit representation
  entity->query_wrapper = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
      entity->rotated_query.data(), impl_->padded_dim, impl_->ex_bits,
      impl_->query_config, impl_->metric_type);

  // Preprocess - get the distance from query to all centroids
  entity->q_to_centroids.resize(impl_->num_clusters);

  std::vector<float> centroid(impl_->padded_dim);
  if (impl_->metric_type == rabitqlib::METRIC_L2) {
    for (size_t i = 0; i < impl_->num_clusters; i++) {
      int ret = impl_->rotated_centroids.read(
          i * impl_->padded_dim * sizeof(float), centroid.data(),
          impl_->padded_dim * sizeof(float));
      if (ret != 0) return ret;
      entity->q_to_centroids[i] = std::sqrt(rabitqlib::euclidean_sqr(
          entity->rotated_query.data(), centroid.data(), impl_->padded_dim));
    }
  } else if (impl_->metric_type == rabitqlib::METRIC_IP) {
    entity->q_to_centroids.resize(impl_->num_clusters * 2);
    // first half as g_add, second half as g_error
    for (size_t i = 0; i < impl_->num_clusters; i++) {
      int ret = impl_->rotated_centroids.read(
          i * impl_->padded_dim * sizeof(float), centroid.data(),
          impl_->padded_dim * sizeof(float));
      if (ret != 0) return ret;
      entity->q_to_centroids[i] = rabitqlib::dot_product(
          entity->rotated_query.data(), centroid.data(), impl_->padded_dim);
      entity->q_to_centroids[i + impl_->num_clusters] = std::sqrt(
          rabitqlib::euclidean_sqr(entity->rotated_query.data(),
                                   centroid.data(), impl_->padded_dim));
    }
  }

  return 0;
}

int RabitqReformer::Impl::quantize_vector(const float *raw_vector,
                                          uint32_t cluster_id,
                                          std::string *quantized_data) const {
  std::vector<float> rotated_data(padded_dim);
  rotator->rotate(raw_vector, rotated_data.data());

  // quantized format: cluster_id + bin_data + ex_data
  quantized_data->resize(sizeof(cluster_id) + size_bin_data + size_ex_data);
  memcpy(&(*quantized_data)[0], &cluster_id, sizeof(cluster_id));
  int bin_data_offset = sizeof(cluster_id);
  int ex_data_offset = bin_data_offset + size_bin_data;
  std::vector<float> centroid(padded_dim);
  int ret =
      rotated_centroids.read(size_t(cluster_id) * padded_dim * sizeof(float),
                             centroid.data(), padded_dim * sizeof(float));
  if (ret != 0) return ret;
  rabitqlib::quant::quantize_split_single(
      rotated_data.data(), centroid.data(), padded_dim, ex_bits,
      &(*quantized_data)[bin_data_offset], &(*quantized_data)[ex_data_offset],
      metric_type, config);

  return 0;
}

int RabitqReformer::dump(const IndexDumper::Pointer &dumper) {
  if (!dumper) {
    LOG_ERROR("Null dumper");
    return IndexError_InvalidArgument;
  }

  if (!impl_->loaded || impl_->rotated_centroids.size() == 0 ||
      impl_->centroids.size() == 0) {
    LOG_ERROR("No centroids to dump");
    return IndexError_NoReady;
  }

  size_t dumped_size = 0;
  int ret = dump_rabitq_centroids(
      dumper, impl_->dimension, impl_->padded_dim, impl_->ex_bits,
      impl_->num_clusters, impl_->rotator_type,
      [&](size_t offset, void *out, size_t bytes) {
        return impl_->rotated_centroids.read(offset, out, bytes);
      },
      [&](size_t offset, void *out, size_t bytes) {
        return impl_->centroids.read(offset, out, bytes);
      },
      impl_->rotator, &dumped_size);
  if (ret != 0) {
    return ret;
  }

  LOG_INFO("RabitqReformer dump completed: %zu bytes", dumped_size);
  return 0;
}

int RabitqReformer::dump(const IndexStorage::Pointer &storage) {
  if (!storage) return IndexError_InvalidArgument;
  if (!impl_->loaded) return IndexError_NoReady;
  const size_t bytes = sizeof(RabitqConverterHeader) + impl_->centroids.size() +
                       impl_->rotated_centroids.size() +
                       impl_->rotator->dump_bytes();
  auto dumper = std::make_shared<StorageSegmentDumper>();
  int ret = dumper->prepare(storage, RABITQ_CONVERTER_SEG_ID,
                            (bytes + 31) & ~size_t(31));
  return ret == 0 ? dump(dumper) : ret;
}
}  // namespace core
}  // namespace zvec

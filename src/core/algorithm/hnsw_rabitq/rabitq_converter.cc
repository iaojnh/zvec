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

#include "rabitq_converter.h"
#include <cstring>
#include <limits>
#include <memory>
#include <rabitqlib/utils/rotator.hpp>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/string_helper.h>
#include "ailego/pattern/defer.h"
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/cluster/holder_cluster.h"
#include "algorithm/hnsw_rabitq/rabitq_reformer.h"
#include "utility/reservoir_sample_holder.h"
#include "utility/storage_segment_dumper.h"
#include "utility/temporary_buffer_storage.h"
#include "zvec/core/framework/index_cluster.h"
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_features.h"
#include "zvec/core/framework/index_holder.h"
#include "zvec/core/framework/index_memory.h"
#include "zvec/core/framework/index_meta.h"
#include "paged_rabitq_rotator.h"
#include "rabitq_params.h"
#include "rabitq_utils.h"

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#endif

namespace zvec {
namespace core {

namespace {
// Inner-product conversion may append a dimension that RaBitQ intentionally
// ignores. Expose the original prefix without materializing another corpus.
class RabitqTrainingHolder : public IndexHolder {
 public:
  RabitqTrainingHolder(IndexHolder::Pointer source, const IndexMeta &meta)
      : source_(std::move(source)), meta_(meta) {}
  size_t count() const override {
    return source_->count();
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
    return source_->multipass();
  }
  Iterator::Pointer create_iterator() override {
    return source_->create_iterator();
  }

 private:
  IndexHolder::Pointer source_;
  IndexMeta meta_;
};
}  // namespace

RabitqConverter::~RabitqConverter() {
  this->cleanup();
}

int RabitqConverter::init(const IndexMeta &meta, const ailego::Params &params) {
  // Copy meta and ensure it has metric information
  meta_ = meta;
  dimension_ = meta.dimension();

  if (meta_.metric_name().empty()) {
    LOG_ERROR("Meta metric is empty");
    return IndexError_InvalidArgument;
  }

  // Round up dimension to multiple of 64
  padded_dim_ = ((dimension_ + 63) / 64) * 64;

  // Get RaBitQ parameters with defaults
  uint32_t total_bits = 0;
  params.get(PARAM_RABITQ_TOTAL_BITS, &total_bits);
  if (total_bits == 0) {
    total_bits = kDefaultRabitqTotalBits;
  }
  if (total_bits < 1 || total_bits > 9) {
    LOG_ERROR("Invalid total_bits: %zu, must be in [1, 9]", (size_t)total_bits);
    return IndexError_InvalidArgument;
  }
  ex_bits_ = total_bits - 1;

  params.get(PARAM_RABITQ_NUM_CLUSTERS, &num_clusters_);
  if (num_clusters_ == 0) {
    num_clusters_ = kDefaultNumClusters;
  }

  if (ex_bits_ > 8) {
    LOG_ERROR("Invalid ex_bits: %zu, must be <= 8", ex_bits_);
    return IndexError_InvalidArgument;
  }

  if (meta.data_type() != IndexMeta::DataType::DT_FP32) {
    LOG_ERROR("RaBitQ only supports FP32 data type");
    return IndexError_Unsupported;
  }
  params.get(PARAM_RABITQ_SAMPLE_COUNT, &sample_count_);

  std::string rotator_type_str;
  params.get(PARAM_RABITQ_ROTATOR_TYPE, &rotator_type_str);
  if (rotator_type_str.empty()) {
    rotator_type_ = rabitqlib::RotatorType::FhtKacRotator;
  } else if (strncasecmp(rotator_type_str.c_str(), "fht", 3) == 0) {
    rotator_type_ = rabitqlib::RotatorType::FhtKacRotator;
  } else if (strncasecmp(rotator_type_str.c_str(), "matrix", 6) == 0) {
    rotator_type_ = rabitqlib::RotatorType::MatrixRotator;
  } else {
    LOG_ERROR("Invalid rotator_type: %s", rotator_type_str.c_str());
    return IndexError_InvalidArgument;
  }

  // Create rotator
  int ret =
      MakeRabitqRotator(dimension_, rotator_type_, padded_dim_, &rotator_);
  if (ret != 0) return ret;

  LOG_INFO(
      "RabitqConverter initialized: dim=%zu, padded_dim=%zu, "
      "num_clusters=%zu, ex_bits=%zu, rotator_type=%d[%s] sample_count[%zu]",
      dimension_, padded_dim_, num_clusters_, ex_bits_, (int)rotator_type_,
      rotator_type_str.c_str(), sample_count_);

  return 0;
}

int RabitqConverter::cleanup() {
  centroids_ = PagedBuildBuffer();
  rotated_centroids_ = PagedBuildBuffer();
  result_holder_.reset();
  rotator_.reset();
  return 0;
}

int RabitqConverter::train(IndexHolder::Pointer holder) {
  return this->train(std::move(holder), nullptr);
}

int RabitqConverter::train(IndexHolder::Pointer holder,
                           IndexThreads::Pointer threads) {
  if (!holder) {
    LOG_ERROR("Null holder for training");
    return IndexError_InvalidArgument;
  }

  ailego::ElapsedTime timer;

  if (!holder->is_matched(meta_)) {
    if (holder->data_type() != meta_.data_type() ||
        holder->dimension() < meta_.dimension() ||
        holder->element_size() < meta_.element_size()) {
      return IndexError_Mismatch;
    }
    holder = std::make_shared<RabitqTrainingHolder>(std::move(holder), meta_);
  }

  size_t vector_count = holder->count();
  if (vector_count == 0) {
    LOG_ERROR("No vectors for training");
    return IndexError_InvalidArgument;
  }

  // do sampling from all data
  size_t sample_count = vector_count;
  if (sample_count_ > 0) {
    sample_count = std::min(sample_count_, vector_count);
  }
  LOG_INFO("Training with %zu vectors from %zu of holder", sample_count,
           vector_count);
  // Create KmeansCluster for training centroids
  auto cluster = IndexFactory::CreateCluster("OptKmeansCluster");
  if (!cluster) {
    LOG_ERROR("Failed to create OptKmeansCluster");
    return IndexError_NoExist;
  }

  // Initialize cluster
  LOG_INFO(
      "Initializing KmeansCluster with meta: dim=%u, data_type=%d, metric=%s",
      meta_.dimension(), (int)meta_.data_type(), meta_.metric_name().c_str());
  ailego::Params cluster_params;
  if (ailego::MemoryLimitPool::get_instance().capacity() != 0) {
    std::string prefix;
    int status =
        TemporaryBufferStorage::DefaultPrefix("zvec-rabitq-train", &prefix);
    if (status != 0) return status;
    cluster_params.set(OPTKMEANS_CLUSTER_BUFFERED_STORAGE_PATH, prefix);
  }
  int ret = cluster->init(meta_, cluster_params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize KmeansCluster: %d", ret);
    return ret;
  }

  cluster->suggest(num_clusters_);

  // Perform clustering
  IndexCluster::CentroidList cents;
  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(0, false);
  }
  // Sampling the entire known corpus preserves its iteration order. Feed it
  // straight into the training matrix, without an intermediate FP32 copy.
  size_t trained_count = vector_count;
  ret = IndexError_NotImplemented;
  if (vector_count != std::numeric_limits<size_t>::max() &&
      sample_count == vector_count) {
    auto *streaming = dynamic_cast<HolderCluster *>(cluster.get());
    if (streaming) ret = streaming->cluster_holder(threads, holder, cents);
  }
  if (ret == IndexError_NotImplemented) {
    if (ailego::MemoryLimitPool::get_instance().capacity() != 0) {
      IndexHolder::Pointer sampled;
      ret = CreateReservoirSampleHolder(holder, sample_count, &sampled);
      if (ret != 0) return ret;
      trained_count = sampled->count();
      auto *streaming = dynamic_cast<HolderCluster *>(cluster.get());
      if (!streaming) return IndexError_NotImplemented;
      ret = streaming->cluster_holder(threads, sampled, cents);
    } else {
      auto sampler =
          std::make_shared<SampleIndexFeatures<CompactIndexFeatures>>(
              meta_, sample_count);
      auto iter = holder->create_iterator();
      if (!iter) return IndexError_Runtime;
      for (; iter->is_valid(); iter->next()) {
        if (iter->status() != 0) return iter->status();
        const void *data = iter->data();
        if (iter->status() != 0) return iter->status();
        if (!data) return IndexError_ReadData;
        sampler->emplace(data);
      }
      if (iter->status() != 0) return iter->status();
      trained_count = sampler->count();
      if (trained_count == 0) return IndexError_InvalidLength;
      ret = cluster->mount(sampler);
      if (ret != 0) return ret;
      ret = cluster->cluster(threads, cents);
    }
  }
  if (ret != 0) {
    LOG_ERROR("Failed to perform clustering: %d", ret);
    return ret;
  }
  // Centroids own their buffers. Release any mounted sample before building
  // the original/rotated centroid tables.
  cluster.reset();
  holder.reset();

  if (cents.size() != num_clusters_) {
    LOG_WARN("Expected %zu clusters, got %zu", num_clusters_, cents.size());
    num_clusters_ = cents.size();
  }
  ret = centroids_.init(num_clusters_ * dimension_ * sizeof(float));
  if (ret != 0) return ret;
  ret = rotated_centroids_.init(num_clusters_ * padded_dim_ * sizeof(float));
  if (ret != 0) return ret;
  std::vector<float> row(padded_dim_);
  std::vector<float> center(dimension_);
  for (size_t i = 0; i < num_clusters_; ++i) {
    ret = cents[i].read_feature(center.data(), center.size() * sizeof(float));
    if (ret != 0) return ret;
    const float *cent_data = center.data();
    ret = centroids_.write(i * dimension_ * sizeof(float), cent_data,
                           dimension_ * sizeof(float));
    if (ret != 0) return ret;
    rotator_->rotate(cent_data, row.data());
    ret = rotated_centroids_.write(i * padded_dim_ * sizeof(float), row.data(),
                                   padded_dim_ * sizeof(float));
    if (ret != 0) return ret;
    // The cluster API returns owning rows. Release each once persisted rather
    // than retaining another full centroid table through the handoff.
    cents[i] = IndexCluster::Centroid();
  }

  stats_.set_trained_count(trained_count);
  stats_.set_trained_costtime(timer.milli_seconds());

  LOG_INFO("Training completed: %zu centroids, cost %zu ms", num_clusters_,
           static_cast<size_t>(timer.milli_seconds()));

  return 0;
}


int RabitqConverter::transform(IndexHolder::Pointer holder) {
  if (!holder) {
    LOG_ERROR("Null holder for transformation");
    return IndexError_InvalidArgument;
  }

  if (rotated_centroids_.size() == 0) {
    LOG_ERROR("Centroids not trained yet");
    return IndexError_NoReady;
  }

  LOG_ERROR("Not implemented");
  return IndexError_NotImplemented;
}

int RabitqConverter::dump(const IndexDumper::Pointer &dumper) {
  if (!dumper) {
    LOG_ERROR("Null dumper");
    return IndexError_InvalidArgument;
  }

  if (rotated_centroids_.size() == 0 || centroids_.size() == 0) {
    LOG_ERROR("No centroids to dump");
    return IndexError_NoReady;
  }

  ailego::ElapsedTime timer;
  size_t dumped_size = 0;

  int ret = dump_rabitq_centroids(
      dumper, dimension_, padded_dim_, ex_bits_, num_clusters_, rotator_type_,
      [&](size_t offset, void *out, size_t bytes) {
        return rotated_centroids_.read(offset, out, bytes);
      },
      [&](size_t offset, void *out, size_t bytes) {
        return centroids_.read(offset, out, bytes);
      },
      rotator_, &dumped_size);
  if (ret != 0) {
    return ret;
  }

  stats_.set_dumped_size(dumped_size);
  stats_.set_dumped_costtime(timer.milli_seconds());

  LOG_INFO("Dump completed: %zu bytes, cost %zu ms", stats_.dumped_size(),
           static_cast<size_t>(timer.milli_seconds()));
  return 0;
}

int RabitqConverter::training_storage(IndexStorage::Pointer *storage) {
  if (!storage) return IndexError_InvalidArgument;
  if (!rotator_ || centroids_.size() == 0 || rotated_centroids_.size() == 0)
    return IndexError_NoReady;
  if (ailego::MemoryLimitPool::get_instance().capacity() != 0) {
    int ret = CreateTemporaryIndexStorage(storage);
    if (ret != 0) return ret;
    auto dumper = std::make_shared<StorageSegmentDumper>();
    size_t bytes = sizeof(RabitqConverterHeader) + centroids_.size() +
                   rotated_centroids_.size() + rotator_->dump_bytes();
    ret = dumper->prepare(*storage, RABITQ_CONVERTER_SEG_ID,
                          (bytes + 31) & ~size_t(31));
    return ret == 0 ? dump(dumper) : ret;
  }
  auto dumper = IndexFactory::CreateDumper("MemoryDumper");
  dumper->init(ailego::Params());
  std::string id = ailego::StringHelper::Concat(
      "rabitq_converter_", ailego::Monotime::MilliSeconds(), rand());
  AILEGO_DEFER([&id]() { IndexMemory::Instance()->remove(id); });
  int ret = dumper->create(id);
  if (ret == 0) ret = dump(dumper);
  if (ret == 0) ret = dumper->close();
  if (ret != 0) return ret;
  *storage = IndexFactory::CreateStorage("MemoryReadStorage");
  return (*storage)->open(id, false);
}

int RabitqConverter::to_reformer(IndexReformer::Pointer *reformer) {
  IndexStorage::Pointer storage;
  int ret = training_storage(&storage);
  if (ret != 0) return ret;
  auto res = std::make_shared<RabitqReformer>();
  ailego::Params params;
  params.set(PARAM_RABITQ_METRIC_NAME, meta_.metric_name());
  ret = res->init(params);
  if (ret == 0) ret = res->load(storage);
  if (ret == 0) *reformer = std::move(res);
  return ret;
}
}  // namespace core
}  // namespace zvec

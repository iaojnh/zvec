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
#include <limits>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/core/framework/index_build_partitions.h>
#include <zvec/core/framework/index_builder.h>
#include <zvec/core/framework/index_meta.h>
#include "utility/vector_build_source.h"
#include "ivf_centroid_index.h"

namespace zvec {
namespace core {

/*! IVF Builder
 */
class IVFBuilder : public IndexBuilder {
 public:
  //! Constructor
  IVFBuilder();

  //! Destructor
  ~IVFBuilder() override;

  //! Disable them
  IVFBuilder(const IVFBuilder &) = delete;
  IVFBuilder &operator=(const IVFBuilder &) = delete;

 public:
  //! Initialize the builder
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Initialize with an initialized Turbo quantizer for row-major postings.
  //! Meta describes raw input vectors, retained for centroid training and
  //! assignment. Meta (or quantizer->meta()) must carry the matching factory
  //! name and initialization params in its quantizer descriptor for reload.
  int init(const IndexMeta &meta, const ailego::Params &params,
           const std::shared_ptr<zvec::turbo::Quantizer> &quantizer) override;

  //! Cleanup the builder
  int cleanup() override;

  //! Train the data
  int train(IndexThreads::Pointer threads,
            IndexHolder::Pointer holder) override;

  //! Train the data
  int train(const IndexTrainer::Pointer &trainer) override;

  //! Build the index
  int build(IndexThreads::Pointer threads,
            IndexHolder::Pointer holder) override;

  //! Dump index into file system
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const Stats &stats() const override {
    return stats_;
  }

  IVFCentroidIndex::Pointer centroid_index() const {
    return centroid_index_;
  }

 private:
  //! Check MajorOrder in meta, and update the major order if needed
  int check_and_update_major_order(IndexMeta &meta);

  //! Parse params
  int parse_centroids_num(const ailego::Params &params);
  int parse_clustering_params(const ailego::Params &params);
  int parse_general_params(const ailego::Params &params);

  //! Prepare params for trainer
  int prepare_trainer_params(ailego::Params &params);

  //! Build the index
  int build_label_index(IndexThreads *threads,
                        const IndexHolder::Pointer &holder);

  //! Dump the index to dumper
  int dump_index(const IndexDumper::Pointer &dumper);

  //! Read one original vector; the returned data is consumed before next read.
  int read_vector(size_t id, uint64_t *key, const void **data);
  size_t label_count(size_t centroid) const;
  int read_label(size_t centroid, size_t index, uint32_t *id) const;
  IndexHolder::Pointer label_holder(size_t centroid) const;

  //! Prepare the quantizer for inverted index
  int prepare_quantizer(IndexThreads *threads);

  //! Quantize the centrods list
  int quantize_centroids();

  //! Create converter and init with params
  static IndexConverter::Pointer CreateAndInitConverter(
      const IndexMeta &meta, const std::string &name,
      const ailego::Params &params) {
    auto converter = IndexFactory::CreateConverter(name);
    if (!converter) {
      LOG_ERROR("Failed to create converter %s", name.c_str());
      return IndexConverter::Pointer();
    }
    int ret = converter->init(meta, params);
    if (ret != 0) {
      LOG_ERROR("Failed to initialize converter %s for %s", name.c_str(),
                IndexError::What(ret));
      return IndexConverter::Pointer();
    }
    return converter;
  }

  //! Select the nearest centroid id for the vector
  void label(const char *vectors, size_t element_size, uint32_t first_id,
             size_t count) try {
    for (size_t i = 0; i < count; ++i) {
      const uint32_t id = first_id + static_cast<uint32_t>(i);
      uint32_t centroid_idx = IVFCentroidIndex::kInvalidID;
      const int label_ret = centroid_index_->search_nearest_centroid(
          vectors + i * element_size, element_size, &centroid_idx);
      if (label_ret != 0) {
        LOG_ERROR("Failed to search nearest centroid in CentroidIndex");
        if (!error_.exchange(true)) {
          err_code_ = label_ret;
        }
        return;
      }
      const int ret = labels_.assign(id, centroid_idx);
      if (ret != 0) {
        if (!error_.exchange(true)) err_code_ = ret;
        return;
      }
    }
  } catch (const std::bad_alloc &) {
    if (!error_.exchange(true)) err_code_ = IndexError_NoMemory;
  }


 private:
  //! Constants
  static constexpr size_t kLabelMemoryBudget = 4u * 1024u * 1024u;
  static constexpr size_t kBatchSize = 10u;
  static constexpr size_t kDefaultBlockCount = 32u;

  enum BuilderState { INIT = 0, INITED = 1, TRAINED = 2, BUILT = 3 };

  //! Members
  BuilderState state_{INIT};
  Stats stats_{};
  ailego::Params params_{};
  IndexMeta meta_{};

  std::vector<uint32_t> centroid_num_vec_{};
  std::string cluster_class_{};
  std::string converter_class_{};
  std::vector<ailego::Params> cluster_params_{};

  IndexBuildPartitions labels_;
  IVFCentroidIndex::Pointer centroid_index_{};
  IVFCentroidIndex::Pointer searcher_centroid_index_{};
  VectorBuildSource::Pointer holder_{};
  // Keep the immutable source alive through dump, including repeated dumps.
  // The reader owns only a key map and at most one provider, never all vectors.
  OrdinalAccessHolder::Reader::Pointer source_reader_{};
  IndexMeta converted_meta_{};
  IndexConverter::Pointer converter_{};
  IndexMeta quantized_meta_{};
  ailego::BudgetedVector<IndexConverter::Pointer> quantizers_{};
  std::shared_ptr<zvec::turbo::Quantizer> turbo_quantizer_{};

  std::atomic_bool error_{false};
  int err_code_{0};

  uint32_t thread_count_{0};
  uint32_t sample_count_{0};
  float sample_ratio_{0.0};
  uint32_t block_vector_count_{kDefaultBlockCount};
  bool cluster_auto_tuning_{false};
  bool store_original_features_{false};
  bool quantize_by_centroid_{false};
};

}  // namespace core
}  // namespace zvec

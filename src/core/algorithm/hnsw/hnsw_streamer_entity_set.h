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

#include <zvec/core/framework/index_streamer.h>
#include "hnsw_streamer_entity.h"
#include "hnsw_streamer_bench_entity.h"

namespace zvec {
namespace core {

class HnswStreamerEntitySet {
 public:
  enum Options {
    kUnknown = 0,
    kMMap = 1,
    kMMapBench = 2,
    kBufferPool = 3,
  };

 public:
  HnswStreamerEntitySet(Options opt, IndexStreamer::Stats &stats) : options_(opt), stats_(stats) {
    if(opt == kMMap) {
      normal_entity_ = new HnswStreamerEntity(stats);
    } else if(opt == kMMapBench) {
      bench_entity_ = new HnswStreamerBenchEntity(stats);
    }
  }
  
  ~HnswStreamerEntitySet() {
    delete normal_entity_;
    delete bench_entity_;
  }

  

 private:
  IndexStreamer::Stats &stats_;
  Options options_;

 private:
  HnswStreamerEntity *normal_entity_{nullptr};
  HnswStreamerBenchEntity *bench_entity_{nullptr};
};

}  // namespace core
}  // namespace zvec
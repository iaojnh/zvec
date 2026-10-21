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

#include <functional>
#include <zvec/db/query.h>
#include <zvec/db/status.h>
#include "db/common/profiler.h"
#include "db/index/segment/segment.h"

namespace zvec::sqlengine {

class SQLEngine {
 public:
  using Ptr = std::shared_ptr<SQLEngine>;
  virtual ~SQLEngine();

  virtual Result<DocPtrList> execute(
      CollectionSchema::Ptr collection, SearchQuery query,
      const std::vector<Segment::Ptr> &segments) = 0;

  // Borrowed ID batches are valid only during the callback. The caller must
  // keep the scanned collection stable and must not mutate it in the callback.
  using DocIdConsumer = std::function<Status(const uint64_t *, size_t)>;
  virtual Status scan_doc_ids(CollectionSchema::Ptr collection,
                              const std::string &filter,
                              const std::vector<Segment::Ptr> &segments,
                              const DocIdConsumer &consume) = 0;

  virtual Result<GroupResults> execute_group_by(
      CollectionSchema::Ptr collection,
      const GroupByVectorQuery &group_by_query,
      const std::vector<Segment::Ptr> &segments) = 0;

 public:
  static SQLEngine::Ptr create(zvec::Profiler::Ptr profiler);
};

}  // namespace zvec::sqlengine

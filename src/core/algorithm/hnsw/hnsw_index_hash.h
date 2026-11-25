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

#include "utility/paged_graph_index_hash.h"
#include "hnsw_chunk.h"

namespace zvec::core {

template <class Key, class Val, Val EmptyVal = 0U,
          typename =
              typename std::enable_if<std::is_integral<Key>::value>::type>
using HnswIndexHashMap = PagedGraphIndexHash<Key, Val, ChunkBroker, EmptyVal>;

}  // namespace zvec::core

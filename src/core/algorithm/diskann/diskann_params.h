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

#include <string>

namespace zvec {
namespace core {

static const std::string PARAM_DISKANN_BUILDER_MAX_DEGREE(
    "zvec.diskann.builder.max_degree");
static const std::string PARAM_DISKANN_BUILDER_LIST_SIZE(
    "zvec.diskann.builder.list_size");
static const std::string PARAM_DISKANN_BUILDER_MEMORY_LIMIT(
    "zvec.diskann.builder.memory_limit");
static const std::string PARAM_DISKANN_BUILDER_MEMORY_BUDGET(
    "zvec.diskann.builder.memory_budget");
static const std::string PARAM_DISKANN_BUILDER_DISK_PQ_DIM(
    "zvec.diskann.builder.disk_pq_dim");
static const std::string PARAM_DISKANN_BUILDER_THREAD_COUNT(
    "zvec.diskann.builder.thread_count");
static const std::string PARAM_DISKANN_BUILDER_MAX_TRAIN_SAMPLE_COUNT(
    "zvec.diskann.builder.max_train_sample_count");
static const std::string PARAM_DISKANN_BUILDER_TRAIN_SAMPLE_RATIO(
    "zvec.diskann.builder.train_sample_ratio");
static const std::string PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM(
    "zvec.diskann.builder.max_pq_chunk_num");

static const std::string PARAM_DISKANN_SEARCHER_LIST_SIZE(
    "zvec.diskann.searcher.list_size");
static const std::string PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM(
    "zvec.diskann.searcher.cache_node_num");
static const std::string PARAM_DISKANN_SEARCHER_CACHE_NODE_BUDGET_BYTES(
    "zvec.diskann.searcher.cache_node_budget_bytes");
static const std::string PARAM_DISKANN_SEARCHER_CACHE_NODE_PAGE_POLICY(
    "zvec.diskann.searcher.cache_node_page_policy");
static const std::string DISKANN_CACHE_NODE_PAGE_POLICY_KEEP("keep");
static const std::string DISKANN_CACHE_NODE_PAGE_POLICY_EVICT("evict");
static const std::string PARAM_DISKANN_SEARCHER_CACHE_NODE_LIST_PATH(
    "zvec.diskann.searcher.cache_node_list_path");
static const std::string PARAM_DISKANN_SEARCHER_WARMUP_MODE(
    "zvec.diskann.searcher.warmup_mode");
static const std::string PARAM_DISKANN_SEARCHER_WARMUP_NODE_NUM(
    "zvec.diskann.searcher.warmup_node_num");
static const std::string DISKANN_WARMUP_MODE_NONE("none");
static const std::string DISKANN_WARMUP_MODE_BFS("bfs");
static const std::string DISKANN_WARMUP_MODE_QUERY_SAMPLE("query_sample");
static const std::string DISKANN_WARMUP_MODE_CACHE_LIST("cache_list");

static const std::string PARAM_DISKANN_REDUCER_INDEX_NAME(
    "zvec.diskann.reducer.index_name");
static const std::string PARAM_DISKANN_REDUCER_WORKING_PATH(
    "zvec.diskann.reducer.working_path");
static const std::string PARAM_DISKANN_REDUCER_NUM_OF_ADD_THREADS(
    "zvec.diskann.reducer.num_of_add_threads");

}  // namespace core
}  // namespace zvec

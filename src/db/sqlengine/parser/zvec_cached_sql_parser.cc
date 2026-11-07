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

#include "zvec_cached_sql_parser.h"
#include <exception>
#include <typeinfo>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/string_helper.h>
#include "atn/ParserATNSimulator.h"
#include "db/sqlengine/antlr/gen/SQLLexer.h"
#include "db/sqlengine/antlr/gen/SQLParser.h"
#include "db/sqlengine/common/util.h"
#include "case_changing_charstream.h"
#include "error_verbose_listener.h"
#include "node.h"
#include "select_info.h"
#include "selected_elem_info.h"

using namespace antlr4;
using namespace tree;
using namespace atn;

namespace zvec::sqlengine {

namespace {
// Parser caches are optional and configuration-sized, not corpus storage.
// Cap each input and reserve an AST/key allowance. This is deliberately a
// planning estimate, not an allocator-level bound for the parser itself.
constexpr size_t kMaxCachedSqlBytes = 32 * 1024;
template <typename T>
std::shared_ptr<T> admit_cached_tree(std::shared_ptr<T> tree, size_t bytes) {
  if (bytes > kMaxCachedSqlBytes) return nullptr;
  struct Owner {
    ailego::MemoryReservation memory;
    std::shared_ptr<T> tree;
  };
  try {
    auto owner = std::make_shared<Owner>();
    owner->memory.add(4096 + bytes * 128);
    owner->tree = std::move(tree);
    auto *value = owner->tree.get();
    return std::shared_ptr<T>(std::move(owner), value);
  } catch (const std::bad_alloc &) {
    // A cache miss must not change query results or fail a valid query.
    return nullptr;
  }
}
}  // namespace

std::unordered_map<std::string, SQLInfo::Ptr>
    ZVecCachedSQLParser::sql_info_map_{};
std::unordered_map<std::string, Node::Ptr> ZVecCachedSQLParser::filter_map_;
std::atomic<uint32_t> ZVecCachedSQLParser::hit{0};
std::atomic<uint32_t> ZVecCachedSQLParser::miss{0};

// Optional parser caches participate in the same pressure eviction as pages.
// Trees already returned to callers keep their allowance through aliasing
// shared ownership, even after the cache drops its reference.
class ZVecCachedSQLParser::CacheReclaimer : public ailego::EvictableBlockOwner {
 public:
  CacheReclaimer() {
    ailego::BlockEvictionQueue::get_instance().set_valid(this);
  }
  ~CacheReclaimer() override {
    ailego::BlockEvictionQueue::get_instance().set_invalid(this);
  }
  bool is_dead_block(size_t, size_t version) override {
    std::shared_lock guard(shared_mutex_);
    return version != generation_ ||
           (sql_info_map_.empty() && filter_map_.empty());
  }
  bool evict_block(size_t) override {
    std::unique_lock guard(shared_mutex_);
    queued_ = false;
    const auto before = ailego::MemoryLimitPool::get_instance().external_used();
    sql_info_map_.clear();
    filter_map_.clear();
    return ailego::MemoryLimitPool::get_instance().external_used() < before;
  }
  size_t recover_eviction_queue() override {
    std::unique_lock guard(shared_mutex_);
    return enqueue_locked() ? 1 : 0;
  }
  void eviction_requeue_failed(size_t, size_t version) override {
    std::unique_lock guard(shared_mutex_);
    if (generation_ == version) queued_ = false;
  }
  bool enqueue_locked() {
    if (queued_ || (sql_info_map_.empty() && filter_map_.empty())) return false;
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    generation_ = queue.next_version();
    queued_ = queue.add_single_block({0, generation_, this}, 0);
    return queued_;
  }

 private:
  size_t generation_{0};
  bool queued_{false};
};

ZVecCachedSQLParser::CacheReclaimer &ZVecCachedSQLParser::cache_reclaimer() {
  // Constructed after the static maps; unregisters before their destruction.
  static CacheReclaimer reclaimer;
  return reclaimer;
}

ZVecCachedSQLParser::ZVecCachedSQLParser(uint32_t cache_count)
    : cache_count_(cache_count) {
  cache_reclaimer();
}

ZVecCachedSQLParser::~ZVecCachedSQLParser() = default;

SQLInfo::Ptr ZVecCachedSQLParser::parse(const std::string &query,
                                        bool need_formatted_tree) {
  std::string query_cache_key{""};
  SQLInfo::Ptr cached_sql_info = get_from_cache(query, &query_cache_key);
  if (cached_sql_info != nullptr) {
    return cached_sql_info;
  }

  SQLInfo::Ptr new_sql_info = real_parser_.parse(query, need_formatted_tree);
  if (new_sql_info == nullptr) {
    // no need to cache parse failed sql. just return.
    err_msg_ = real_parser_.err_msg();
    return nullptr;
  }

  if (cache_count_ && !query_cache_key.empty()) {
    auto cached = admit_cached_tree(new_sql_info, query.size());
    if (cached) {
      put_into_cache(query_cache_key, cached);
      return cached;
    }
  }

  return new_sql_info;
}

void ZVecCachedSQLParser::put_into_cache(const std::string &query_cache_key,
                                         const SQLInfo::Ptr &new_sql_info) {
  {
    std::unique_lock guard(shared_mutex_);
    if (sql_info_map_.size() >= cache_count_) {
      // if full, clear to refresh new sql
      sql_info_map_.clear();
      hit = miss = 0;
    }
    sql_info_map_.emplace(query_cache_key, new_sql_info);
    cache_reclaimer().enqueue_locked();
  }

  LOG_DEBUG("cache emplaced. [%s] [%s] ", query_cache_key.c_str(),
            new_sql_info->to_string().c_str());

  return;
}

SQLInfo::Ptr ZVecCachedSQLParser::get_from_cache(const std::string &query,
                                                 std::string *query_cache_key) {
  if (!cache_count_ || query.size() > kMaxCachedSqlBytes) return nullptr;
  // find [ and ], must only one occurrence.
  std::string::size_type left_pos, right_pos;
  left_pos = query.find("[");
  if (left_pos == query.npos) {
    return nullptr;
  }
  // find from left_pos+1
  right_pos = query.rfind("]");
  if (right_pos == query.npos) {
    return nullptr;
  }

  // ok, let's find it.
  *query_cache_key = query.substr(0, left_pos);
  query_cache_key->append(query.begin() + right_pos + 1, query.end());
  std::string vector_text = query.substr(left_pos, right_pos - left_pos + 1);

  SQLInfo::Ptr cached_sql_info = nullptr;
  SQLInfo::Ptr copied_sql_info = nullptr;
  {  // lock only in this block. after sql_info is copied, just unlock.
    std::shared_lock guard(shared_mutex_);
    auto iter = sql_info_map_.find(*query_cache_key);
    if (iter == sql_info_map_.end()) {
      ++miss;
      LOG_DEBUG("cache miss. key: [%s]", query_cache_key->c_str());
      return nullptr;
    }

    cached_sql_info = iter->second;
    // copy cached_sql_info
    copied_sql_info = std::make_shared<SQLInfo>(*cached_sql_info);
  }

  // parse vector part
  Node::Ptr vector_node = parse_vector_text(&vector_text);
  if (vector_node == nullptr) {
    LOG_DEBUG("wrong vector format: [%s]", vector_text.c_str());
    return nullptr;
  }
  // replace vector in copied_sql_info
  if (replace_vector_node(copied_sql_info, vector_node) != 0) {
    LOG_WARN("replace_vector_node failed. [%s][%s]", query.c_str(),
             vector_text.c_str());
    return nullptr;
  }

  ++hit;
  LOG_DEBUG("cache hit. key: [%s] sql_info: [%s]", query_cache_key->c_str(),
            copied_sql_info->to_string().c_str());
  return copied_sql_info;
}

int ZVecCachedSQLParser::replace_vector_node(SQLInfo::Ptr cached_sql_info,
                                             Node::Ptr vector_node) {
  SelectInfo::Ptr cached_select_info =
      std::dynamic_pointer_cast<SelectInfo>(cached_sql_info->base_info());
  if (cached_select_info == nullptr) {
    LOG_WARN("wrong select_info in cache. [%s]",
             cached_sql_info->to_string().c_str());
    return -1;
  }

  Node::Ptr search_cond = cached_select_info->mutable_search_cond();
  if (search_cond == nullptr) {
    LOG_WARN("wrong search_cond in cache. [%s]",
             cached_sql_info->to_string().c_str());
    return -1;
  }

  replace_flag_ = false;
  if (traverse_to_replace(search_cond, vector_node) != 0 ||
      replace_flag_ == false) {
    LOG_WARN("replace search_cond in cache failed. [%s]",
             cached_sql_info->to_string().c_str());
    return -1;
  }

  return 0;
}

int ZVecCachedSQLParser::traverse_to_replace(Node::Ptr ptr,
                                             Node::Ptr vector_node) {
  if (ptr->op() == NodeOp::T_VECTOR_MATRIX_VALUE) {
    Node *parent = dynamic_cast<Node *>(ptr->parent());
    if (parent == nullptr) {
      LOG_WARN("wrong parent node in cache. [%s]", ptr->to_string().c_str());
      return -1;
    }
    if (parent->left() == ptr) {
      parent->set_left(vector_node);
      replace_flag_ = true;
    } else if (parent->right() == ptr) {
      parent->set_right(vector_node);
      replace_flag_ = true;
    } else {
      LOG_WARN("wrong node in cache. [%s]", ptr->to_string().c_str());
      return -1;
    }
    return 0;
  }

  if (ptr->left() != nullptr) {
    if (traverse_to_replace(ptr->left(), vector_node) < 0) {
      return -1;
    }
    if (replace_flag_) {
      return 0;
    }
  }
  if (ptr->right() != nullptr) {
    if (traverse_to_replace(ptr->right(), vector_node) != 0) {
      return -1;
    }
    if (replace_flag_) {
      return 0;
    }
  }

  return 0;
}

Node::Ptr ZVecCachedSQLParser::parse_filter(const std::string &filter,
                                            bool need_formatted_tree) {
  if (cache_count_ && filter.size() <= kMaxCachedSqlBytes) {
    std::shared_lock guard(shared_mutex_);
    auto iter = filter_map_.find(filter);
    if (iter != filter_map_.end()) {
      ++hit;
      return iter->second;
    }
    ++miss;
  }
  auto node = real_parser_.parse_filter(filter, need_formatted_tree);
  err_msg_ = real_parser_.err_msg();
  formatted_tree_ = real_parser_.formatted_tree();
  auto cached =
      cache_count_ && node ? admit_cached_tree(node, filter.size()) : nullptr;
  if (cached != nullptr) {
    std::unique_lock guard(shared_mutex_);
    if (filter_map_.size() >= cache_count_) {
      // clear cache if full
      filter_map_.clear();
      hit = miss = 0;
    }
    filter_map_.emplace(filter, cached);
    cache_reclaimer().enqueue_locked();
    return cached;
  }
  return node;
}

}  // namespace zvec::sqlengine

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

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <zvec/ailego/io/file.h>
#include <zvec/db/status.h>
#include "db/common/rocksdb_context.h"


namespace zvec {


class IDMap {
 public:
  using Ptr = std::shared_ptr<IDMap>;

  explicit IDMap(std::string collection_name)
      : collection_name_(std::move(collection_name)) {};

  ~IDMap() {
    if (opened_) {
      close();
    }
  }

  static Ptr CreateAndOpen(const std::string &collection_name,
                           const std::string &working_dir,
                           bool create_if_missing, bool read_only);


  IDMap(const IDMap &) = delete;
  IDMap &operator=(const IDMap &) = delete;
  IDMap &operator=(IDMap &&) = delete;

 private:
 public:
  Status open(const std::string &working_dir, bool create_if_missing,
              bool read_only);

  Status close();

  Status create_snapshot(const std::string &snapshot_dir);

  Status flush();

  // Recovery runs before publishing the collection, under its exclusive lock.
  // Reads observe staged changes; persistent mappings change in one DB write
  // only after the complete replay succeeds. No concurrent use is permitted.
  Status begin_recovery();
  Status commit_recovery();
  void discard_recovery();

  Status upsert(const std::string &key, uint64_t doc_id);

  void remove(const std::string &key);

  // Return false only for a confirmed missing key. Cache admission and I/O
  // failures must reach readers and writers before they act on the lookup.
  Result<bool> lookup(const std::string &key, uint64_t *doc_id = nullptr) const;

  Status multi_get(const std::vector<std::string> &keys,
                   std::vector<uint64_t> *doc_ids) const;

  size_t storage_size_in_bytes();

  size_t count();


  const std::string &collection_name() const {
    return collection_name_;
  }

  const std::string &working_dir() const {
    return working_dir_;
  }


 private:
  using FILE = ailego::File;


  const std::string collection_name_{};
  std::string working_dir_{};

  RocksdbContext rocksdb_context_{};
  std::optional<std::unordered_map<std::string, std::optional<uint64_t>>>
      recovery_changes_;
  bool opened_{false};
};


}  // namespace zvec

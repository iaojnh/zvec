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
#include <memory>
#include <string>
#include <string_view>
#include <zvec/db/status.h>


namespace zvec {

class WalFile;

using WalFilePtr = std::shared_ptr<WalFile>;

struct WalOptions {
  uint32_t max_docs_wal_flush{0};
  bool create_new{false};
};

class WalFile {
 public:
  static constexpr size_t kMaxRecordSize = 4 * 1024 * 1024;

  //! Constructor
  WalFile() = default;

  //! Destructor
  virtual ~WalFile() = default;  // LCOV_EXCL_LINE

  //! Create an instance
  static WalFilePtr Create(const std::string &wal_path);

  //! Crate an instance and open
  static int CreateAndOpen(const std::string &wal_path,
                           const WalOptions &wal_options, WalFilePtr *wal_file);

 public:
  // Synchronously consumes the view; callers keep its backing bytes alive
  // until append returns. The WAL does not retain a second document copy.
  virtual int append(std::string_view data) = 0;
  virtual int prepare_for_read() = 0;
  // Called after validating the record length, before allocating its payload.
  // The caller keeps the admitted bytes alive until the returned string dies.
  using ReadAdmission = std::function<void(size_t)>;
  virtual std::string next(const ReadAdmission &admit = {}) = 0;

  // An empty next() result is EOF only when this status is OK.
  virtual Status read_status() const {
    return Status::OK();
  }

 public:
  //! Open and initialize WalFile
  virtual int open(const WalOptions &wal_options) = 0;

  //! Close WalFile
  virtual int close() = 0;

  //! Remove wal disk file
  virtual int remove() = 0;

  //! Flush WalFile's memory to disk file
  virtual int flush() = 0;

  virtual bool has_record() = 0;
};

};  // namespace zvec

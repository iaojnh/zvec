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
#include <rocksdb/memtablerep.h>
#include <rocksdb/write_buffer_manager.h>

namespace zvec {

// Keep the default SkipList representation, but give its nodes and records
// one mapping-backed allocation lifetime. WBM charges are held until those
// mappings are actually unmapped, including while old iterators pin a table.
class RocksdbMemtableFactory final : public rocksdb::SkipListFactory {
 public:
  explicit RocksdbMemtableFactory(
      std::shared_ptr<rocksdb::WriteBufferManager> manager);

  const char *Name() const override {
    return "ZvecMappedSkipList";
  }

  using rocksdb::SkipListFactory::CreateMemTableRep;
  rocksdb::MemTableRep *CreateMemTableRep(
      const rocksdb::MemTableRep::KeyComparator &compare,
      rocksdb::Allocator *allocator, const rocksdb::SliceTransform *transform,
      rocksdb::Logger *logger) override;

 private:
  std::shared_ptr<rocksdb::WriteBufferManager> manager_;
};

}  // namespace zvec

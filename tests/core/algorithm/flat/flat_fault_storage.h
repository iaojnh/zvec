// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <zvec/core/framework/index_storage.h>

namespace zvec::core::test {
// One-shot failures at the storage boundary, shared by cloned segments.
struct StorageFault {
  std::string segment;
  bool read{false};
  bool write{false};
  size_t block_reads{0};
  bool hit(const std::string &id, bool writing) {
    bool &armed = writing ? write : read;
    if (!armed || id != segment) return false;
    armed = false;
    return true;
  }
};
class FaultSegment : public IndexStorage::Segment {
 public:
  FaultSegment(Pointer source, std::string id,
               std::shared_ptr<StorageFault> fault)
      : source_(std::move(source)),
        id_(std::move(id)),
        fault_(std::move(fault)) {}
  size_t data_size() const override {
    return source_->data_size();
  }
  uint32_t data_crc() const override {
    return source_->data_crc();
  }
  size_t padding_size() const override {
    return source_->padding_size();
  }
  size_t capacity() const override {
    return source_->capacity();
  }
  size_t fetch(size_t off, void *out, size_t len) const override {
    return fault_->hit(id_, false) ? 0 : source_->fetch(off, out, len);
  }
  size_t read(size_t off, IndexStorage::MemoryBlock &out, size_t len) override {
    ++fault_->block_reads;
    return fault_->hit(id_, false) ? 0 : source_->read(off, out, len);
  }
  size_t write(size_t off, const void *data, size_t len) override {
    return fault_->hit(id_, true) ? 0 : source_->write(off, data, len);
  }
  bool write_batch(const IndexStorage::SegmentData *writes,
                   size_t count) override {
    return !fault_->hit(id_, true) && source_->write_batch(writes, count);
  }
  size_t resize(size_t size) override {
    return source_->resize(size);
  }
  void update_data_crc(uint32_t crc) override {
    source_->update_data_crc(crc);
  }
  Pointer clone() override {
    return std::make_shared<FaultSegment>(source_->clone(), id_, fault_);
  }

 private:
  Pointer source_;
  std::string id_;
  std::shared_ptr<StorageFault> fault_;
};
class FaultStorage : public IndexStorage {
 public:
  explicit FaultStorage(Pointer source) : source_(std::move(source)) {}
  int init(const ailego::Params &p) override {
    return source_->init(p);
  }
  int cleanup() override {
    return source_->cleanup();
  }
  int open(const std::string &p, bool create) override {
    return source_->open(p, create);
  }
  int flush() override {
    return source_->flush();
  }
  int close() override {
    return source_->close();
  }
  int append(const std::string &id, size_t size) override {
    return source_->append(id, size);
  }
  void refresh(uint64_t point) override {
    source_->refresh(point);
  }
  uint64_t check_point() const override {
    return source_->check_point();
  }
  Segment::Pointer get(const std::string &id, int level = -1) override {
    auto segment = source_->get(id, level);
    return segment ? std::make_shared<FaultSegment>(segment, id, fault)
                   : nullptr;
  }
  bool has(const std::string &id) const override {
    return source_->has(id);
  }
  uint32_t magic() const override {
    return source_->magic();
  }
  MemoryBlock::MemoryBlockType memory_block_type() const override {
    return source_->memory_block_type();
  }
  std::shared_ptr<ailego::VecBufferPool> vec_buffer_pool() const override {
    return source_->vec_buffer_pool();
  }
  std::shared_ptr<StorageFault> fault = std::make_shared<StorageFault>();

 private:
  Pointer source_;
};
}  // namespace zvec::core::test

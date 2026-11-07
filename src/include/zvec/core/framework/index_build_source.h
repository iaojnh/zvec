// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <zvec/core/framework/index_holder.h>

namespace zvec::core {

// Immutable, multipass build input for algorithms and converters. Readers use
// iteration ordinals, preserving duplicate keys and input order. The owner
// chooses borrowing, managed temporary storage or the unconfigured heap path.
// Keep the source immutable while using this object or any of its readers.
class IndexBuildSource final
    : public IndexHolder,
      public OrdinalAccessHolder,
      public std::enable_shared_from_this<IndexBuildSource> {
 public:
  using Pointer = std::shared_ptr<IndexBuildSource>;
  IndexBuildSource(const IndexBuildSource &) = delete;
  IndexBuildSource &operator=(const IndexBuildSource &) = delete;
  ~IndexBuildSource() override;

  // On failure, output is unchanged. Only an unsupported source capability
  // permits materializing input; budget and I/O errors never select a heap
  // fallback. scratch_prefix is used only when temporary storage is needed.
  static int Create(IndexHolder::Pointer holder, Pointer *output,
                    const std::string &scratch_prefix = "");

  size_t count() const override;
  size_t dimension() const override;
  IndexMeta::DataType data_type() const override;
  size_t element_size() const override;
  bool multipass() const override;
  IndexHolder::Iterator::Pointer create_iterator() override;
  // Each reader retains its backing and is single-threaded. Its returned
  // pointer remains valid until that reader's next read/reset or destruction.
  int create_ordinal_reader(Reader::Pointer *output) override;

 private:
  IndexBuildSource();
  class Cursor;
  class StableReader;
  class SpoolReader;
  struct State;
  int init(IndexHolder::Pointer holder, const std::string &scratch_prefix);
  int append(uint64_t key, const void *data);
  std::unique_ptr<State> state_;
};

}  // namespace zvec::core

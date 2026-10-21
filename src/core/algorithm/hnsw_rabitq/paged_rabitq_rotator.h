// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <rabitqlib/core/rotator.hpp>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include "utility/paged_rotation_matrix.h"

namespace zvec::core {
// The third-party FHT owns a standard vector. Its budget base is constructed
// first and destroyed last, covering allocation and the full mask lifetime.
struct RabitqFhtBudget {
  explicit RabitqFhtBudget(size_t padded) {
    memory.add_array(padded / 8, 4);
  }
  ailego::MemoryReservation memory;
};
class BudgetedRabitqFhtRotator : private RabitqFhtBudget,
                                 public rabitqlib::rotator_impl::FhtKacRotator {
 public:
  BudgetedRabitqFhtRotator(size_t dim, size_t padded)
      : RabitqFhtBudget(padded),
        rabitqlib::rotator_impl::FhtKacRotator(dim, padded) {}
};

class PagedRabitqRotator : public rabitqlib::Rotator<float> {
 public:
  PagedRabitqRotator(size_t dim, size_t padded)
      : rabitqlib::Rotator<float>(dim, padded) {}
  int generate() {
    return matrix_.generate(dim_, padded_dim_);
  }
  void rotate(const float *in, float *out) const override {
    matrix_.apply(in, out);
  }
  void load(const char *data) override {
    PagedRotationMatrix::check(
        load_reader([data](size_t offset, void *out, size_t bytes) {
          std::memcpy(out, data + offset, bytes);
          return 0;
        }));
  }
  void save(char *data) const override {
    PagedRotationMatrix::check(matrix_.read(0, data, dump_bytes()));
  }
  void load(std::ifstream &input) override {
    PagedRotationMatrix::check(
        load_reader([&input](size_t, void *out, size_t bytes) {
          input.read(static_cast<char *>(out), bytes);
          return input ? 0 : IndexError_ReadData;
        }));
  }
  void save(std::ofstream &output) const override {
    std::array<char, 65536> buffer;
    for (size_t offset = 0; offset < dump_bytes();) {
      const size_t bytes = std::min(buffer.size(), dump_bytes() - offset);
      PagedRotationMatrix::check(matrix_.read(offset, buffer.data(), bytes));
      output.write(buffer.data(), bytes);
      if (!output) throw std::runtime_error("Failed to save rotation matrix");
      offset += bytes;
    }
  }
  size_t dump_bytes() const override {
    return dim_ * padded_dim_ * sizeof(float);
  }
  int read(size_t offset, void *data, size_t bytes) const {
    return matrix_.read(offset, data, bytes);
  }
  int load_reader(const PagedRotationMatrix::Reader &read) {
    return matrix_.load(dim_, padded_dim_, read);
  }

 private:
  PagedRotationMatrix matrix_;
};

inline int MakeRabitqRotator(size_t dim, rabitqlib::RotatorType type,
                             size_t padded,
                             std::unique_ptr<rabitqlib::Rotator<float>> *out,
                             bool generate = true) try {
  if (!out || !dim || padded < dim || padded % 64)
    return IndexError_InvalidArgument;
  if (type == rabitqlib::RotatorType::MatrixRotator) {
    auto matrix = std::make_unique<PagedRabitqRotator>(dim, padded);
    int ret = generate ? matrix->generate() : 0;
    if (ret != 0) return ret;
    *out = std::move(matrix);
  } else if (type == rabitqlib::RotatorType::FhtKacRotator) {
    *out = std::make_unique<BudgetedRabitqFhtRotator>(dim, padded);
  } else {
    return IndexError_InvalidArgument;
  }
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}
inline int LoadRabitqRotator(rabitqlib::Rotator<float> *rotator,
                             const PagedRotationMatrix::Reader &read) try {
  auto *matrix = dynamic_cast<PagedRabitqRotator *>(rotator);
  if (matrix) return matrix->load_reader(read);
  ailego::BudgetedVector<char> buffer(rotator->dump_bytes());
  int ret = read(0, buffer.data(), buffer.size());
  if (ret == 0) rotator->load(buffer.data());
  return ret;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}
inline int ReadRabitqRotator(const rabitqlib::Rotator<float> *rotator,
                             size_t offset, void *out, size_t bytes) try {
  auto *matrix = dynamic_cast<const PagedRabitqRotator *>(rotator);
  if (matrix) return matrix->read(offset, out, bytes);
  ailego::BudgetedVector<char> buffer(rotator->dump_bytes());
  if (offset > buffer.size() || bytes > buffer.size() - offset)
    return IndexError_InvalidArgument;
  rotator->save(buffer.data());
  std::memcpy(out, buffer.data() + offset, bytes);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}
}  // namespace zvec::core

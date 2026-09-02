#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr size_t kCount = 50'000;
constexpr size_t kDim = 384;
constexpr size_t kOpenVikingStride = 400;  // 384 int8 + scale + id + offset.
constexpr size_t kZvecStride = 448;        // 404-byte code rounded to 64 bytes.
constexpr size_t kRepeats = 100;

volatile float sink;

__attribute__((noinline)) float dot_float_accumulator(const int8_t* lhs,
                                                       const int8_t* rhs) {
  float sum = 0.0F;
  for (size_t i = 0; i < kDim; ++i) {
    sum += static_cast<float>(lhs[i] * rhs[i]);
  }
  return sum;
}

__attribute__((noinline)) int32_t dot_int32_accumulator(const int8_t* lhs,
                                                        const int8_t* rhs) {
  int32_t sum = 0;
  for (size_t i = 0; i < kDim; ++i) {
    sum += static_cast<int32_t>(lhs[i]) * static_cast<int32_t>(rhs[i]);
  }
  return sum;
}

template <typename Dot>
double run(const std::vector<int8_t>& records, size_t stride,
           const std::array<int8_t, kDim>& query, Dot dot) {
  const auto begin = std::chrono::steady_clock::now();
  float checksum = 0.0F;
  for (size_t repeat = 0; repeat < kRepeats; ++repeat) {
    float repeat_sum = 0.0F;
    for (size_t i = 0; i < kCount; ++i) {
      repeat_sum += static_cast<float>(dot(records.data() + i * stride,
                                          query.data()));
    }
    checksum += repeat_sum;
  }
  sink = checksum;
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count() /
         kRepeats;
}

}  // namespace

int main() {
  std::mt19937 rng(20260901);
  std::uniform_int_distribution<int> distribution(-127, 127);
  std::array<int8_t, kDim> query{};
  for (auto& value : query) {
    value = static_cast<int8_t>(distribution(rng));
  }

  std::vector<int8_t> openviking_records(kCount * kOpenVikingStride);
  std::vector<int8_t> zvec_records(kCount * kZvecStride);
  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = 0; j < kDim; ++j) {
      const auto value = static_cast<int8_t>(distribution(rng));
      openviking_records[i * kOpenVikingStride + j] = value;
      zvec_records[i * kZvecStride + j] = value;
    }
  }

  // Warm both code and data paths.
  sink = static_cast<float>(dot_int32_accumulator(openviking_records.data(),
                                                  query.data()));
  sink = dot_float_accumulator(zvec_records.data(), query.data());

  const double int_tight = run(openviking_records, kOpenVikingStride, query,
                               dot_int32_accumulator);
  const double int_padded =
      run(zvec_records, kZvecStride, query, dot_int32_accumulator);
  const double float_padded =
      run(zvec_records, kZvecStride, query, dot_float_accumulator);

  std::printf("int32 accumulator, 400-byte stride: %.3f ms/query\n", int_tight);
  std::printf("int32 accumulator, 448-byte stride: %.3f ms/query\n", int_padded);
  std::printf("float accumulator, 448-byte stride: %.3f ms/query\n",
              float_padded);
  std::printf("float/int32 ratio at 448 stride: %.2fx\n",
              float_padded / int_padded);
  std::printf("448/400 stride penalty with int32: %.2fx\n",
              int_padded / int_tight);
}

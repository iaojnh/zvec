// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

// Reuse the exact publication benchmark, in a separate executable, so Linux
// diagnostics do not cancel or change the original eight-platform CI run.
#include "diskann_platform_benchmark_test.cc"

#if defined(__linux__) && !defined(__ANDROID__) && DISKANN_SUPPORTED
#include <fcntl.h>
#include <sys/resource.h>
#include <map>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/ailego/logger/logger.h>
#include "algorithm/diskann/diskann_file_reader.h"

namespace zvec::core_interface {
namespace {

std::map<std::string, uint64_t> ProcessIO() {
  std::map<std::string, uint64_t> result;
  std::ifstream input("/proc/self/io");
  std::string name;
  uint64_t value;
  while (input >> name >> value) result[name] = value;
  return result;
}

double CpuSeconds(const timeval &value) {
  return value.tv_sec + value.tv_usec / 1e6;
}

class DiagnosticListener : public ::testing::EmptyTestEventListener {
 public:
  void OnTestStart(const ::testing::TestInfo &) override {
    start_ = Clock::now();
    getrusage(RUSAGE_SELF, &before_);
    io_before_ = ProcessIO();
  }

  void OnTestEnd(const ::testing::TestInfo &) override {
    if (Env("ZVEC_BENCH_MODE").empty()) return;
    rusage after{};
    getrusage(RUSAGE_SELF, &after);
    const auto io_after = ProcessIO();
    std::cout << "ZVEC_IO_PROCESS_RESULT {\"wall_seconds\":"
              << Milliseconds(start_) / 1000
              << ",\"user_cpu_seconds\":"
              << CpuSeconds(after.ru_utime) - CpuSeconds(before_.ru_utime)
              << ",\"system_cpu_seconds\":"
              << CpuSeconds(after.ru_stime) - CpuSeconds(before_.ru_stime)
              << ",\"voluntary_context_switches\":"
              << after.ru_nvcsw - before_.ru_nvcsw
              << ",\"involuntary_context_switches\":"
              << after.ru_nivcsw - before_.ru_nivcsw;
    for (const auto &entry : io_after) {
      const auto previous = io_before_.find(entry.first);
      const auto before = previous == io_before_.end() ? 0 : previous->second;
      std::string key = entry.first;
      if (!key.empty() && key.back() == ':') key.pop_back();
      std::cout << ",\"" << key << "\":" << entry.second - before;
    }
    std::cout << "}" << std::endl;
  }

 private:
  Clock::time_point start_;
  rusage before_{};
  std::map<std::string, uint64_t> io_before_;
};

const bool kRegisterDiagnostics = [] {
  ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);
  ::testing::UnitTest::GetInstance()->listeners().Append(new DiagnosticListener);
  return true;
}();

const char *BackendName(ailego::IOBackendType type) {
  switch (type) {
    case ailego::IOBackendType::kIoUring:
      return "io_uring";
    case ailego::IOBackendType::kLibAio:
      return "libaio";
    case ailego::IOBackendType::kPread:
      return "pread";
    default:
      return "other";
  }
}

TEST(DiskAnnLinuxIO, RandomReads) {
  if (Env("ZVEC_BENCH_DATA").empty()) GTEST_SKIP() << "Opt-in diagnostics";
  const std::string path = Env("ZVEC_BENCH_DATA") + "/diskann.index";
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(input.good());
  const uint64_t bytes = static_cast<uint64_t>(input.tellg());
  ASSERT_GT(bytes, 4096U);
  input.close();
  const uint64_t blocks = bytes / 4096;
  void *memory = nullptr;
  ASSERT_EQ(0, posix_memalign(&memory, 4096, 8 * 4096));
  std::unique_ptr<void, decltype(&std::free)> allocation(memory, &std::free);
  core::IOContext ctx = nullptr;
  ASSERT_EQ(0, core::setup_io_ctx(ctx));
  ASSERT_NE(nullptr, ctx);
  core::LinuxAlignedFileReader reader;
  reader.open(path);
  uint64_t seed = 20261105;
  for (uint32_t batch_size : {1U, 8U}) {
    std::vector<double> latencies;
    uint64_t checksum = 0;
    for (uint32_t round = 0; round < 256; ++round) {
      std::vector<core::AlignedRead> requests;
      for (uint32_t i = 0; i < batch_size; ++i) {
        seed = seed * 6364136223846793005ULL + 1;
        requests.emplace_back((seed % blocks) * 4096, 4096,
                              static_cast<char *>(memory) + i * 4096);
      }
      const auto start = Clock::now();
      const int ret = reader.read(requests, ctx);
      const double ms = Milliseconds(start);
      ASSERT_EQ(0, ret);
      latencies.push_back(ms);
      checksum += *static_cast<unsigned char *>(memory);
    }
    std::cout << "ZVEC_IO_PROBE_RESULT {\"backend\":\""
              << BackendName(ctx->type) << "\",\"batch\":" << batch_size
              << ",\"iterations\":256,\"block_bytes\":4096"
              << ",\"p50_ms\":" << Percentile(latencies, 0.5)
              << ",\"p95_ms\":" << Percentile(latencies, 0.95)
              << ",\"p99_ms\":" << Percentile(latencies, 0.99)
              << ",\"checksum\":" << checksum << "}" << std::endl;
  }
  reader.close();
  ASSERT_EQ(0, core::destroy_io_ctx(ctx));
}

}  // namespace
}  // namespace zvec::core_interface
#endif

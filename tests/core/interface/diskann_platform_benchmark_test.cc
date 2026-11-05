// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/ailego/io/io_backend.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

#include "diskann_benchmark_counters.h"
#include "diskann_controlled_io.h"
#include "diskann_paced_io.h"
#if defined(__APPLE__)
#include "../../apple_io_phase_gate.h"
#endif

namespace zvec::core_interface {
namespace {

using Clock = std::chrono::steady_clock;

std::string Env(const char *key, const std::string &fallback = "") {
  const char *value = std::getenv(key);
  return value ? value : fallback;
}

uint64_t RssBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS info{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
    throw std::runtime_error("GetProcessMemoryInfo failed");
  }
  return info.WorkingSetSize;
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    throw std::runtime_error("task_info failed");
  }
  return info.resident_size;
#else
  uint64_t pages = 0, resident = 0;
  std::ifstream input("/proc/self/statm");
  if (!(input >> pages >> resident)) {
    throw std::runtime_error("Cannot read /proc/self/statm");
  }
  return resident * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
#endif
}

template <typename T>
struct Matrix {
  uint32_t rows{0}, columns{0};
  std::vector<T> values;
};

// Shared little-endian format: uint32 rows, uint32 columns, row-major payload.
// Every runner in this workflow is little-endian.
template <typename T>
Matrix<T> ReadMatrix(const std::string &path, bool header_only = false) {
  Matrix<T> matrix;
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(&matrix.rows), sizeof(uint32_t));
  input.read(reinterpret_cast<char *>(&matrix.columns), sizeof(uint32_t));
  if (!input || !matrix.rows || !matrix.columns) {
    throw std::runtime_error("Invalid matrix: " + path);
  }
  if (!header_only) {
    matrix.values.resize(static_cast<size_t>(matrix.rows) * matrix.columns);
    input.read(reinterpret_cast<char *>(matrix.values.data()),
               matrix.values.size() * sizeof(T));
    if (!input) throw std::runtime_error("Truncated matrix: " + path);
  }
  return matrix;
}

double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double Percentile(std::vector<double> values, double percentile) {
  std::sort(values.begin(), values.end());
  return values.at(static_cast<size_t>(std::ceil(values.size() * percentile)) -
                   1);
}

TEST(DiskAnnPlatformBenchmark, Measure) {
  if (Env("ZVEC_BENCH_MODE").empty()) {
    GTEST_SKIP() << "Opt-in benchmark; run via scripts/benchmark_diskann_platform.py";
  }
#if !DISKANN_SUPPORTED
  FAIL() << "DiskANN is not supported on this target";
#else
  const std::string root = Env("ZVEC_BENCH_DATA");
  const std::string algorithm = Env("ZVEC_BENCH_INDEX");
  const std::string mode = Env("ZVEC_BENCH_MODE");
  ASSERT_TRUE(algorithm == "diskann" || algorithm == "hnsw");
  ASSERT_TRUE(mode == "build" || mode == "query");
  const bool controlled_io = Env("ZVEC_BENCH_CONTROLLED_IO") == "1";
  const bool paced_io = Env("ZVEC_BENCH_PACED_IO") == "1";
  ASSERT_FALSE(controlled_io && paced_io);
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
  std::unique_ptr<zvec::core::benchmark::IOModel> io_model;
  std::unique_ptr<zvec::core::benchmark::ControlledScope> controlled_scope;
  std::unique_ptr<zvec::core::benchmark::NativeRateLimiter> limiter;
  std::unique_ptr<zvec::core::benchmark::PacedScope> paced_scope;
  if (paced_io && algorithm == "diskann" && mode == "query") {
    zvec::core::benchmark::PaceConfig config;
    config.iops = std::stoull(Env("ZVEC_BENCH_IOPS", "2000"));
    config.bytes_per_second = std::stoull(Env("ZVEC_BENCH_PACE_BPS", "16777216"));
    config.depth = static_cast<uint32_t>(std::stoul(Env("ZVEC_BENCH_PACE_DEPTH", "8")));
    limiter = std::make_unique<zvec::core::benchmark::NativeRateLimiter>(config);
    paced_scope = std::make_unique<zvec::core::benchmark::PacedScope>(*limiter);
  }
  if (controlled_io && algorithm == "diskann" && mode == "query") {
    io_model = std::make_unique<zvec::core::benchmark::IOModel>(
        std::stoull(Env("ZVEC_BENCH_IO_LATENCY_NS", "200000")),
        std::stoul(Env("ZVEC_BENCH_IO_QUEUE_DEPTH", "8")),
        std::stoull(Env("ZVEC_BENCH_IO_BYTES_PER_SECOND", "1048576000")));
    controlled_scope = std::make_unique<zvec::core::benchmark::ControlledScope>(*io_model);
  }
#else
  ASSERT_FALSE(controlled_io) << "Rebuild with DISKANN_CONTROLLED_IO_BENCHMARK=ON";
  ASSERT_FALSE(paced_io) << "Rebuild with DISKANN_CONTROLLED_IO_BENCHMARK=ON";
#endif
  const auto base_header = ReadMatrix<float>(root + "/base.fbin", true);
  const auto dimension = base_header.columns;
  const auto index_path = root + "/" + algorithm + ".index";

  BaseIndexParam::Pointer param;
  if (algorithm == "diskann") {
    param = DiskAnnIndexParamBuilder()
                .with_dimension(dimension)
                .with_data_type(DataType::DT_FP32)
                .with_metric_type(MetricType::kL2sq)
                .with_max_degree(64)
                .with_list_size(100)
                .with_pq_chunk_num(dimension / 4)
                .build();
  } else {
    param = HNSWIndexParamBuilder()
                .with_dimension(dimension)
                .with_data_type(DataType::DT_FP32)
                .with_metric_type(MetricType::kL2sq)
                .with_m(32)
                .with_ef_construction(200)
                .build();
  }
  if (mode == "build") {
    auto base = ReadMatrix<float>(root + "/base.fbin");
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(nullptr, index);
    ASSERT_EQ(0, index->open(index_path,
                             {StorageOptions::StorageType::kMMAP, true}));
    const auto start = Clock::now();
    for (uint32_t i = 0; i < base.rows; ++i) {
      ASSERT_EQ(0, index->add(VectorData{DenseVector{
                                 base.values.data() + size_t(i) * dimension}},
                             i));
    }
    ASSERT_EQ(0, index->train());
    const double build_ms = Milliseconds(start);
    ASSERT_EQ(base.rows, index->get_doc_count());
    ASSERT_EQ(0, index->close());
    std::ostringstream record;
    record << "{\"phase\":\"build\",\"index\":\""
              << algorithm << "\",\"build_seconds\":" << build_ms / 1000.0
              << "}";
    BenchmarkEmit(record, Env("ZVEC_BENCH_RESULT_PATH"));
    return;
  }

  // The build ran in another process. Never load base vectors in this process.
  const auto queries = ReadMatrix<float>(root + "/queries.fbin");
  const auto truth = ReadMatrix<uint32_t>(root + "/neighbors.ibin");
  ASSERT_EQ(dimension, queries.columns);
  ASSERT_EQ(queries.rows, truth.rows);
  ASSERT_EQ(10U, truth.columns);
  const uint64_t baseline_rss = RssBytes();
  const auto before_open = BenchmarkSnapshot();
  const auto open_start = Clock::now();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(index_path,
                           {StorageOptions::StorageType::kMMAP, false, true}));
  ASSERT_EQ(base_header.rows, index->get_doc_count());
  const auto index_open_ms = Milliseconds(open_start);
  const auto after_open = BenchmarkSnapshot();
  const uint32_t repetitions = std::stoul(Env("ZVEC_BENCH_REPETITIONS", "3"));
  ASSERT_GT(repetitions, 0U);
  const uint32_t calibration_count = std::min(200U, queries.rows);
  const double target = std::stod(Env("ZVEC_BENCH_TARGET_RECALL", "0.95"));

  auto search = [&](uint32_t q, uint32_t budget, SearchResult *result) {
    BaseIndexQueryParam::Pointer query_param;
    if (algorithm == "diskann") {
      auto p = std::make_shared<DiskAnnQueryParam>();
      p->list_size = budget;
      query_param = p;
    } else {
      auto p = std::make_shared<HNSWQueryParam>();
      p->ef_search = budget;
      query_param = p;
    }
    query_param->topk = 10;
    return index->search(VectorData{DenseVector{
                             queries.values.data() + size_t(q) * dimension}},
                         query_param, result);
  };
  auto hits = [&](uint32_t q, const SearchResult &result) {
    uint32_t count = 0;
    for (const auto &doc : result.doc_list_) {
      auto begin = truth.values.begin() + size_t(q) * 10;
      if (std::find(begin, begin + 10, doc.key()) != begin + 10) ++count;
    }
    return count;
  };

  // Use a disjoint calibration prefix to select the smallest budget reaching
  // target recall, then report all remaining queries (no tuning on test queries).
  uint32_t budget = 0;
  double calibration_recall = 0;
  const auto fixed_budget = Env("ZVEC_BENCH_FIXED_BUDGET");
  if (!fixed_budget.empty()) {
    budget = std::stoul(fixed_budget);
    ASSERT_GT(budget, 0U);
  } else {
    for (uint32_t candidate : {32U, 48U, 64U, 96U, 128U, 192U, 256U, 384U,
                               512U, 768U, 1024U, 1536U, 2048U}) {
      uint64_t correct = 0;
      for (uint32_t q = 0; q < calibration_count; ++q) {
        SearchResult result;
        ASSERT_EQ(0, search(q, candidate, &result));
        correct += hits(q, result);
      }
      calibration_recall = double(correct) / (calibration_count * 10);
      budget = candidate;
      if (calibration_recall >= target) break;
    }
    ASSERT_GE(calibration_recall, target) << "Calibration recall target not reached";
  }
  ASSERT_GT(queries.rows, calibration_count);
  const bool warmup = Env("ZVEC_BENCH_WARMUP", "1") == "1";
  const auto before_warmup = BenchmarkSnapshot();
  // Query-workload warmup does not imply OS cache hits for direct readers.
  if (warmup) {
    for (uint32_t q = calibration_count; q < queries.rows; ++q) {
      SearchResult result;
      ASSERT_EQ(0, search(q, budget, &result));
    }
  }
#if defined(__APPLE__)
  testing_util::AppleIOGate("ZVEC_BENCH_IO_CONTROL", "query", "before");
#endif
  const auto before_timing = BenchmarkSnapshot();
  uint64_t peak_query_rss = RssBytes();
  uint64_t correct = 0;
  double elapsed_ms = 0;
  std::vector<double> latencies;
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
  std::vector<double> model_latencies;
  uint64_t model_reads = 0, model_bytes = 0, model_batches = 0;
  const auto pace_before = limiter ? limiter->stats() : zvec::core::benchmark::PaceStats{};
#endif
  const auto block_start = Clock::now();
  for (uint32_t round = 0; round < repetitions; ++round) {
    for (uint32_t q = calibration_count; q < queries.rows; ++q) {
      SearchResult result;
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
      if (io_model) io_model->reset();
#endif
      const auto start = Clock::now();
      const int ret = search(q, budget, &result);
      const double ms = Milliseconds(start);
      ASSERT_EQ(0, ret);
      ASSERT_EQ(10U, result.doc_list_.size());
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
      if (io_model) {
        model_latencies.push_back(io_model->now_ns() / 1000000.0);
        model_reads += io_model->requests();
        model_bytes += io_model->bytes();
        model_batches += io_model->batches();
      }
#endif
      latencies.push_back(ms);
      elapsed_ms += ms;
      correct += hits(q, result);
      if (q % 50 == 0) peak_query_rss = std::max(peak_query_rss, RssBytes());
    }
  }
  const double timed_block_ms = Milliseconds(block_start);
  const auto after_timing = BenchmarkSnapshot();
#if defined(__APPLE__)
  testing_util::AppleIOGate("ZVEC_BENCH_IO_CONTROL", "query", "after");
#endif
  peak_query_rss = std::max(peak_query_rss, RssBytes());
  const auto steady_rss = RssBytes();
  std::ostringstream record;
  record << std::setprecision(12)
            << "{\"phase\":\"query\",\"index\":\""
            << algorithm << "\",\"vectors\":" << base_header.rows
            << ",\"dimension\":" << dimension
            << ",\"queries\":" << queries.rows - calibration_count
            << ",\"repetitions\":" << repetitions
            << ",\"budget\":" << budget
            << ",\"fixed_budget\":" << (fixed_budget.empty() ? "false" : "true")
            << ",\"workload_warmup\":" << (warmup ? "true" : "false")
            << ",\"index_open_ms\":" << index_open_ms
            << ",\"io_backend_type\":" << static_cast<int>(ailego::current_io_backend_type())
            << ",\"calibration_recall_at_10\":"
            << (fixed_budget.empty() ? std::to_string(calibration_recall) : "null")
            << ",\"recall_at_10\":" << double(correct) / (latencies.size() * 10)
            << ",\"p50_ms\":" << Percentile(latencies, 0.50)
            << ",\"p95_ms\":" << Percentile(latencies, 0.95)
            << ",\"p99_ms\":" << Percentile(latencies, 0.99)
            << ",\"qps\":" << latencies.size() * 1000.0 / (paced_io ? timed_block_ms : elapsed_ms)
            << ",\"search_only_qps\":" << latencies.size() * 1000.0 / elapsed_ms
            << ",\"timed_block_wall_ms\":" << timed_block_ms
            << ",\"baseline_rss_bytes\":" << baseline_rss
            << ",\"steady_rss_bytes\":" << steady_rss
            << ",\"sampled_peak_query_rss_bytes\":" << peak_query_rss
            << ",\"index_rss_delta_bytes\":"
            << (steady_rss > baseline_rss ? steady_rss - baseline_rss : 0)
            << ",\"io\":{\"source\":\"" << after_timing.source
            << "\",\"open_read_bytes\":";
  BenchmarkBytes(record, before_open, after_open);
  record << ",\"warmup_read_bytes\":";
  BenchmarkBytes(record, before_warmup, before_timing);
  record << ",\"timed_read_bytes\":";
  BenchmarkBytes(record, before_timing, after_timing);
  record << ",\"timed_cpu_seconds\":"
         << after_timing.cpu_seconds - before_timing.cpu_seconds
         << ",\"timed_major_faults\":"
         << after_timing.major_faults - before_timing.major_faults << "}";
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
  if (limiter) {
    const auto reader = zvec::core::benchmark::ActivePacedReader().lock();
    ASSERT_NE(nullptr, reader);
    const auto after = limiter->stats();
    const auto reads = after.reads - pace_before.reads;
    record << ",\"native_pacing\":{\"kind\":\"real-time-native-file-io\""
           << ",\"iops_limit\":" << limiter->config().iops
           << ",\"bytes_per_second_limit\":" << limiter->config().bytes_per_second
           << ",\"max_in_flight\":" << limiter->config().depth
           << ",\"reads\":" << reads
           << ",\"bytes\":" << after.bytes - pace_before.bytes
           << ",\"batches\":" << after.batches - pace_before.batches
           << ",\"actual_wait_ns\":" << after.wait_ns - pace_before.wait_ns
           << ",\"achieved_iops\":" << reads * 1000.0 / timed_block_ms
           << ",\"resident_backing_bytes\":0,\"cache_policy\":\"" << reader->cache_policy()
           << "\",\"timer\":\"" << limiter->timer_kind() << "\"}";
  }
  if (io_model) {
    const auto reader = zvec::core::benchmark::ActiveReader().lock();
    ASSERT_NE(nullptr, reader) << "Controlled reader was not installed";
    ASSERT_GT(model_reads, 0U);
    double model_ms = 0;
    for (double value : model_latencies) model_ms += value;
    record << ",\"controlled_io\":{\"kind\":\"discrete-event-no-cache\""
           << ",\"latency_ns\":" << io_model->latency_ns()
           << ",\"queue_depth\":" << io_model->queue_depth()
           << ",\"bytes_per_second\":" << io_model->bytes_per_second()
           << ",\"resident_backing_bytes\":" << reader->image_bytes()
           << ",\"reads\":" << model_reads << ",\"bytes\":" << model_bytes
           << ",\"batches\":" << model_batches
           << ",\"io_only_p50_ms\":" << Percentile(model_latencies, 0.50)
           << ",\"io_only_p95_ms\":" << Percentile(model_latencies, 0.95)
           << ",\"io_only_p99_ms\":" << Percentile(model_latencies, 0.99)
           << ",\"io_only_qps\":" << model_latencies.size() * 1000.0 / model_ms
           << ",\"memory_is_product_rss\":false"
           << ",\"wall_qps_is_native_disk_qps\":false}";
  }
#endif
  record << "}";
  ASSERT_EQ(0, index->close());
  BenchmarkEmit(record, Env("ZVEC_BENCH_RESULT_PATH"));
#endif
}

#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK) && DISKANN_SUPPORTED
TEST(DiskAnnPacedIOProbe, Measure) {
  if (Env("ZVEC_BENCH_MODE") != "probe") GTEST_SKIP() << "Opt-in native pacing probe";
  using namespace zvec::core;
  using namespace zvec::core::benchmark;
  const auto path = Env("ZVEC_BENCH_DATA") + "/diskann.index";
  std::ifstream size_input(path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(size_input.good());
  const uint64_t sectors = static_cast<uint64_t>(size_input.tellg()) / 4096;
  ASSERT_GT(sectors, 0U);
  size_input.close();
  PaceConfig target;
  target.iops = std::stoull(Env("ZVEC_BENCH_IOPS", "2000"));
  target.bytes_per_second = std::stoull(Env("ZVEC_BENCH_PACE_BPS", "16777216"));
  target.depth = static_cast<uint32_t>(std::stoul(Env("ZVEC_BENCH_PACE_DEPTH", "8")));
  const uint32_t operations = 2048;
  std::ostringstream record;
  record << std::setprecision(12) << "{\"phase\":\"probe\",\"index\":\"diskann\",\"cases\":[";
  for (int case_index = 0; case_index < 4; ++case_index) {
    const bool buffered = case_index == 0;
    const std::string name = buffered ? "buffered-warm" : case_index == 1 ? "native-unlimited" :
                             case_index == 2 ? "paced-target" : "paced-half";
    PaceConfig config = target;
    if (case_index < 2) { config.iops = 0; config.bytes_per_second = 0; }
    if (case_index == 3) { config.iops = std::max<uint64_t>(1, target.iops / 2);
                          config.bytes_per_second = std::max<uint64_t>(1, target.bytes_per_second / 2); }
    NativeRateLimiter rate(config);
    std::shared_ptr<RateLimitedReader> reader;
    std::ifstream input;
    IOContext ctx = nullptr;
    if (buffered) {
      input.open(path, std::ios::binary);
      ASSERT_TRUE(input.good());
    } else {
      auto native = std::make_shared<PlatformAlignedFileReader>();
      native->open(path);
      reader = std::make_shared<RateLimitedReader>(std::move(native), rate);
      ASSERT_EQ(0, setup_io_ctx(ctx));
    }
    std::vector<char> allocation(size_t(target.depth) * 4096 + 4096);
    auto buffer = reinterpret_cast<char *>((reinterpret_cast<uintptr_t>(allocation.data()) + 4095) & ~uintptr_t(4095));
#if defined(__APPLE__)
    testing_util::AppleIOGate("ZVEC_BENCH_IO_CONTROL", name, "before");
#endif
    const auto before = BenchmarkSnapshot();
    const auto start = Clock::now();
    uint64_t checksum = 1469598103934665603ULL;
    for (uint32_t first = 0; first < operations; first += target.depth) {
      const uint32_t count = std::min(target.depth, operations - first);
      std::vector<AlignedRead> requests;
      for (uint32_t n = 0; n < count; ++n) {
        const uint64_t offset = ((uint64_t(first + n) * 104729 + 17) % sectors) * 4096;
        requests.emplace_back(offset, 4096, buffer + size_t(n) * 4096);
        if (buffered) {
          input.seekg(static_cast<std::streamoff>(offset));
          input.read(buffer + size_t(n) * 4096, 4096);
          ASSERT_TRUE(input.good());
        }
      }
      if (reader) ASSERT_EQ(0, reader->read(requests, ctx));
      for (size_t n = 0; n < size_t(count) * 4096; ++n) {
        checksum ^= static_cast<unsigned char>(buffer[n]);
        checksum *= 1099511628211ULL;
      }
    }
    const double seconds = Milliseconds(start) / 1000;
    const auto after = BenchmarkSnapshot();
#if defined(__APPLE__)
    testing_util::AppleIOGate("ZVEC_BENCH_IO_CONTROL", name, "after");
#endif
    if (case_index) record << ',';
    record << "{\"state\":\"" << name << "\",\"operations\":" << operations
           << ",\"requested_bytes\":" << uint64_t(operations) * 4096
           << ",\"seconds\":" << seconds << ",\"achieved_iops\":" << operations / seconds
           << ",\"iops_limit\":" << config.iops << ",\"actual_wait_ns\":" << rate.stats().wait_ns
           << ",\"checksum\":\"" << checksum << "\",\"cache_policy\":\""
           << (reader ? reader->cache_policy() : "buffered-file-control") << "\",\"read_bytes\":";
    BenchmarkBytes(record, before, after);
    record << ",\"counter_source\":\"" << after.source << "\"}";
    if (reader) { reader->release_io_ctx(ctx); reader->close(); ASSERT_EQ(0, destroy_io_ctx(ctx)); }
  }
  record << "]}";
  BenchmarkEmit(record, Env("ZVEC_BENCH_RESULT_PATH"));
}
#endif

}  // namespace
}  // namespace zvec::core_interface

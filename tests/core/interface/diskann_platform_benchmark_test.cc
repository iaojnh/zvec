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
  const auto before_timing = BenchmarkSnapshot();
  uint64_t peak_query_rss = RssBytes();
  uint64_t correct = 0;
  double elapsed_ms = 0;
  std::vector<double> latencies;
  for (uint32_t round = 0; round < repetitions; ++round) {
    for (uint32_t q = calibration_count; q < queries.rows; ++q) {
      SearchResult result;
      const auto start = Clock::now();
      const int ret = search(q, budget, &result);
      const double ms = Milliseconds(start);
      ASSERT_EQ(0, ret);
      ASSERT_EQ(10U, result.doc_list_.size());
      latencies.push_back(ms);
      elapsed_ms += ms;
      correct += hits(q, result);
      if (q % 50 == 0) peak_query_rss = std::max(peak_query_rss, RssBytes());
    }
  }
  const auto after_timing = BenchmarkSnapshot();
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
            << ",\"qps\":" << latencies.size() * 1000.0 / elapsed_ms
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
         << after_timing.major_faults - before_timing.major_faults << "}}";
  ASSERT_EQ(0, index->close());
  BenchmarkEmit(record, Env("ZVEC_BENCH_RESULT_PATH"));
#endif
}

}  // namespace
}  // namespace zvec::core_interface

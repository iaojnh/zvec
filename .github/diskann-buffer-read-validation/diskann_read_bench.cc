// Validation-only executable; not part of the product change.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <condition_variable>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include <random>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "diskann/diskann_params.h"

#if defined(_WIN32)
#include <Windows.h>
#include <Psapi.h>
#else
#include <sys/resource.h>
#endif
using namespace zvec;
using namespace zvec::core_interface;
namespace {
constexpr size_t kRows = 20000, kDim = 256, kQueries = 256, kTopk = 10;
int setting(const char *key, int fallback) {
  const char *s = std::getenv(key); return s ? std::atoi(s) : fallback;
}
struct BenchIndex : DiskAnnIndex {
  using DiskAnnIndex::open;
  using DiskAnnIndex::add;
  using DiskAnnIndex::train;
  int initialize(const BaseIndexParam &p) {
    proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_THREAD_COUNT, 2U);
    return init(p);
  }
  std::shared_ptr<ailego::VecBufferPool> pool() { return storage_->vec_buffer_pool(); }
};
double process_cpu_seconds() {
#if defined(_WIN32)
  FILETIME created{}, exited{}, kernel{}, user{};
  if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user))
    return -1;
  const uint64_t k = (static_cast<uint64_t>(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime;
  const uint64_t u = (static_cast<uint64_t>(user.dwHighDateTime) << 32) | user.dwLowDateTime;
  return static_cast<double>(k + u) * 1e-7;
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage)) return -1;
  return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
         1e-6 * (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
#endif
}
size_t peak_rss() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS m{};
  return GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof(m)) ? m.PeakWorkingSetSize : 0;
#else
  rusage r{}; if (getrusage(RUSAGE_SELF, &r)) return 0;
#if defined(__APPLE__)
  return static_cast<size_t>(r.ru_maxrss);
#else
  return static_cast<size_t>(r.ru_maxrss) * 1024;
#endif
#endif
}
}
TEST(DiskAnnReadBench, PairedQuery) {
  ASSERT_EQ(4096u, ailego::kVectorPageSize);
#if defined(_WIN32)
  SYSTEM_INFO native_system_info{};
  GetSystemInfo(&native_system_info);
  ASSERT_EQ(4096u, native_system_info.dwPageSize);
#endif
  const char *dir = std::getenv("ZVEC_READ_BENCH_DIR");
  ASSERT_NE(nullptr, dir);
  const int seed = setting("ZVEC_READ_BENCH_SEED", 721);
  const char *cache_arg = std::getenv("ZVEC_READ_BENCH_QUERY_CACHE");
  const std::filesystem::path cache_dir = cache_arg ? cache_arg : "";
  const auto query_path = cache_dir / ("queries-" + std::to_string(seed) + ".bin");
  const auto truth_path = cache_dir / ("truth-" + std::to_string(seed) + ".bin");
  const bool cached = cache_arg && std::filesystem::exists(query_path) &&
                      std::filesystem::exists(truth_path) &&
                      !setting("ZVEC_READ_BENCH_PREPARE", 0);
  std::vector<float> queries(kQueries * kDim), values;
  if (!cached) {
    values.resize(kRows * kDim);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> centers(64 * kDim);
  for (float &v : centers) v = normal(rng);
  for (size_t i = 0; i < kRows; ++i)
    for (size_t d = 0; d < kDim; ++d)
      values[i * kDim + d] = centers[(i % 64) * kDim + d] + 0.3f * normal(rng);
  for (size_t i = 0; i < kQueries; ++i) {
    const size_t row = rng() % kRows;
    for (size_t d = 0; d < kDim; ++d)
      queries[i * kDim + d] = values[row * kDim + d] + 0.05f * normal(rng);
  }
  } else {
    std::ifstream in(query_path, std::ios::binary);
    ASSERT_TRUE(in.good());
    in.read(reinterpret_cast<char *>(queries.data()), queries.size() * sizeof(float));
    ASSERT_EQ(static_cast<std::streamsize>(queries.size() * sizeof(float)), in.gcount());
  }
  const auto param = DiskAnnIndexParamBuilder().with_dimension(static_cast<int>(kDim))
      .with_metric_type(MetricType::kL2sq).with_data_type(DataType::DT_FP32)
      .with_max_degree(32).with_list_size(128).with_pq_chunk_num(16).build();
  auto &budget = ailego::MemoryLimitPool::get_instance();
  std::filesystem::create_directories(dir);
  const auto path = (std::filesystem::path(dir) / ("index-" + std::to_string(seed))).string();
  if (setting("ZVEC_READ_BENCH_PREPARE", 0)) {
    ASSERT_EQ(0, budget.init(512UL << 20));
    auto index = std::make_shared<BenchIndex>(); ASSERT_EQ(0, index->initialize(*param));
    ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
    for (uint32_t i = 0; i < kRows; ++i)
      ASSERT_EQ(0, index->add(VectorData{DenseVector{values.data() + i * kDim}}, i));
    ASSERT_EQ(0, index->train()); ASSERT_EQ(0, index->close());
    return;
  }
  std::vector<std::vector<uint32_t>> truth(kQueries);
  if (!cached) {
  for (size_t q = 0; q < kQueries; ++q) {
    std::vector<std::pair<float, uint32_t>> distances(kRows);
    for (size_t i = 0; i < kRows; ++i) {
      float sum = 0;
      for (size_t d = 0; d < kDim; ++d) {
        const float delta = queries[q * kDim + d] - values[i * kDim + d]; sum += delta * delta;
      }
      distances[i] = {sum, static_cast<uint32_t>(i)};
    }
    std::partial_sort(distances.begin(), distances.begin() + kTopk, distances.end());
    for (size_t i = 0; i < kTopk; ++i) truth[q].push_back(distances[i].second);
  }
    if (cache_arg) {
      std::filesystem::create_directories(cache_dir);
      std::ofstream qout(query_path, std::ios::binary);
      qout.write(reinterpret_cast<const char *>(queries.data()), queries.size() * sizeof(float));
      ASSERT_TRUE(qout.good());
      std::ofstream tout(truth_path, std::ios::binary);
      for (const auto &row : truth)
        tout.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(uint32_t));
      ASSERT_TRUE(tout.good());
    }
  } else {
    std::ifstream in(truth_path, std::ios::binary);
    ASSERT_TRUE(in.good());
    for (auto &row : truth) {
      row.resize(kTopk);
      in.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(uint32_t));
      ASSERT_EQ(static_cast<std::streamsize>(row.size() * sizeof(uint32_t)), in.gcount());
    }
  }
  const size_t capacity = static_cast<size_t>(setting("ZVEC_READ_BENCH_BUDGET", 64)) << 20;
  ASSERT_EQ(0, budget.init(capacity));
  auto index = std::make_shared<BenchIndex>(); ASSERT_EQ(0, index->initialize(*param));
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kBufferPool, false, true}));
  auto pool = index->pool(); ASSERT_NE(nullptr, pool);
  ASSERT_EQ(capacity, budget.capacity());
  const int workers = setting("ZVEC_READ_BENCH_WORKERS", 1);
  const int passes = setting("ZVEC_READ_BENCH_PASSES", 4);
  const bool cold = setting("ZVEC_READ_BENCH_COLD", 0) != 0;
  const size_t nq = kQueries * passes;
  auto qp = std::make_shared<DiskAnnQueryParam>(); qp->topk = kTopk; qp->list_size = static_cast<uint32_t>(setting("ZVEC_READ_BENCH_LIST", 512));
  if (!cold) {
    // Each measured process warms the same query sequence, under its own quota.
    for (size_t q = 0; q < kQueries; ++q) {
      SearchResult result; ASSERT_EQ(0, index->search(VectorData{DenseVector{queries.data() + q * kDim}}, qp, &result));
    }
  }
  auto before = pool->stats();
  std::vector<double> latency(nq);
  std::vector<uint64_t> hashes(nq);
  std::atomic<size_t> next{0}, hits{0}, errors{0};
  auto one = [&](size_t n) {
      const size_t q = n % kQueries;
      SearchResult result;
      const auto start = std::chrono::steady_clock::now();
      const int ret = index->search(VectorData{DenseVector{queries.data() + q * kDim}}, qp, &result);
      latency[n] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
      if (ret || result.doc_list_.size() != kTopk) { ++errors; return; }
      uint64_t hash = 1469598103934665603ULL;
      for (const auto &doc : result.doc_list_) {
        const auto key = doc.key(); uint32_t bits; const float score = doc.score(); std::memcpy(&bits, &score, sizeof(bits));
        hash = (hash ^ key) * 1099511628211ULL; hash = (hash ^ bits) * 1099511628211ULL;
        if (std::find(truth[q].begin(), truth[q].end(), key) != truth[q].end()) ++hits;
      }
      hashes[n] = hash;
  };
  const bool steady = setting("ZVEC_READ_BENCH_STEADY", 0) != 0 && !cold;
  std::mutex gate_mutex;
  std::condition_variable gate;
  int warm_turn = 0;
  bool go = false;
  std::atomic<unsigned> warm_errors{0};
  auto run = [&](int worker) {
    if (steady) {
#if defined(__linux__)
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      CPU_SET(worker, &cpus);
      if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) != 0)
        ++warm_errors;
#endif
      std::unique_lock<std::mutex> lock(gate_mutex);
      gate.wait(lock, [&] { return warm_turn == worker; });
      lock.unlock();
      SearchResult warm_result;
      if (index->search(VectorData{DenseVector{queries.data() + worker * kDim}}, qp,
                        &warm_result) != 0 || warm_result.doc_list_.size() != kTopk)
        ++warm_errors;
      lock.lock();
      ++warm_turn;
      gate.notify_all();
      gate.wait(lock, [&] { return go; });
      lock.unlock();
      if (warm_errors.load() == 0)
        for (size_t n = static_cast<size_t>(worker); n < nq; n += workers) one(n);
    } else {
      for (;;) {
        const size_t n = next.fetch_add(1);
        if (n >= nq) break;
        one(n);
      }
    }
  };
  double cpu_start = process_cpu_seconds();
  auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int w = 0; w < workers; ++w) threads.emplace_back(run, w);
  if (steady) {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate.wait(lock, [&] { return warm_turn == workers; });
    before = pool->stats();
    cpu_start = process_cpu_seconds();
    start = std::chrono::steady_clock::now();
    go = true;
    gate.notify_all();
  }
  for (auto &t : threads) t.join();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const double cpu_seconds = process_cpu_seconds() - cpu_start;
  EXPECT_GT(cpu_seconds, 0);
  EXPECT_EQ(0u, warm_errors.load());
  const auto after = pool->stats();
  std::sort(latency.begin(), latency.end());
  uint64_t digest = 0; for (size_t i = 0; i < hashes.size(); ++i) digest ^= hashes[i] + i * 1099511628211ULL;
  const double recall = static_cast<double>(hits) / (nq * kTopk);
  std::cout << "READ_BENCH {\"seed\":" << seed << ",\"page_bytes\":" << ailego::kVectorPageSize
      << ",\"workers\":" << workers << ",\"budget_bytes\":" << capacity << ",\"queries\":" << nq
      << ",\"steady_workers\":" << steady << ",\"cold_managed_cache\":" << cold << ",\"qps\":" << std::setprecision(12) << nq / seconds
      << ",\"cpu_seconds\":" << cpu_seconds << ",\"p50_us\":" << latency[nq/2] << ",\"p99_us\":" << latency[static_cast<size_t>(nq*0.99)]
      << ",\"recall\":" << recall << ",\"errors\":" << errors.load() << ",\"digest\":" << digest
      << ",\"peak_rss\":" << peak_rss() << ",\"pool_used\":" << budget.used()
      << ",\"cache_hits\":" << after.hit-before.hit << ",\"cache_misses\":" << after.miss-before.miss
      << ",\"bypass_bytes\":" << after.bypass_bytes-before.bypass_bytes << "}" << std::endl;
  EXPECT_EQ(0u, errors.load()); EXPECT_GE(recall, 0.9); EXPECT_LE(budget.used(), budget.capacity());
  ASSERT_EQ(0, index->close()); pool.reset(); index.reset(); EXPECT_EQ(budget.queue_metadata_used() + budget.stats().slab_header_bytes, budget.used());
}

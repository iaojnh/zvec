#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task_info.h>
#else
#include <unistd.h>
#endif

#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/schema.h>

namespace {

using Clock = std::chrono::steady_clock;
using zvec::Collection;
using zvec::CollectionOptions;
using zvec::MetricType;
constexpr char kField[] = "embedding";
constexpr uint32_t kTopK = 10;
constexpr uint32_t kCalibration = 200;
constexpr uint32_t kPasses = 3;
constexpr char kPrefix[] = "ZVEC_COLLECTION_BENCH_RESULT ";

void Check(const zvec::Status &status) {
  if (!status.ok()) {
    throw std::runtime_error(status.message());
  }
}

template <class T>
T Value(zvec::Result<T> result) {
  if (!result.has_value()) {
    throw std::runtime_error(result.error().message());
  }
  return std::move(result).value();
}

template <class T>
struct Matrix {
  uint32_t rows = 0;
  uint32_t dimension = 0;
  std::vector<T> values;

  const T *row(uint32_t i) const {
    return values.data() + static_cast<size_t>(i) * dimension;
  }
};

template <class T>
Matrix<T> ReadMatrix(const std::filesystem::path &path, uint32_t row_limit) {
  std::ifstream input(path, std::ios::binary);
  Matrix<T> matrix;
  if (!input.read(reinterpret_cast<char *>(&matrix.rows), sizeof(matrix.rows)) ||
      !input.read(reinterpret_cast<char *>(&matrix.dimension), sizeof(matrix.dimension)) ||
      matrix.rows == 0 || matrix.rows > row_limit || matrix.dimension == 0 ||
      matrix.dimension > 4096) {
    throw std::runtime_error("Invalid matrix header: " + path.string());
  }
  matrix.values.resize(static_cast<size_t>(matrix.rows) * matrix.dimension);
  if (!input.read(reinterpret_cast<char *>(matrix.values.data()),
                  matrix.values.size() * sizeof(T)) ||
      input.peek() != EOF) {
    throw std::runtime_error("Invalid matrix size: " + path.string());
  }
  return matrix;
}

uint64_t ProcessRss() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    throw std::runtime_error("GetProcessMemoryInfo failed");
  }
  return counters.WorkingSetSize;
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    throw std::runtime_error("task_info failed");
  }
  return info.resident_size;
#else
  uint64_t virtual_pages = 0;
  uint64_t resident_pages = 0;
  std::ifstream status("/proc/self/statm");
  const long page_size = sysconf(_SC_PAGESIZE);
  if (!(status >> virtual_pages >> resident_pages) || page_size <= 0) {
    throw std::runtime_error("Cannot read process RSS");
  }
  return resident_pages * static_cast<uint64_t>(page_size);
#endif
}

double ElapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

uint32_t ReadDocCount(const std::filesystem::path &path) {
  uint32_t rows;
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char *>(&rows), sizeof(rows)) || rows == 0) {
    throw std::runtime_error("Cannot read base vector count");
  }
  return rows;
}

void CheckCollection(const Collection &collection, bool diskann, uint32_t rows) {
  const auto schema = Value(collection.schema());
  const auto *field = schema.get_field(kField);
  const auto expected = diskann ? zvec::IndexType::DISKANN : zvec::IndexType::HNSW;
  const auto stats = Value(collection.stats());
  const auto completeness = stats.index_completeness.find(kField);
  if (!field || !field->index_params() || field->index_params()->type() != expected ||
      stats.doc_count != rows || completeness == stats.index_completeness.end() ||
      completeness->second != 1.0f) {
    throw std::runtime_error("Unexpected index type, document count, or index completeness");
  }
}

void Build(const std::filesystem::path &data, const std::string &path, bool diskann,
           std::ostream &result) {
  const auto base = ReadMatrix<float>(data / "base.fbin", 1'000'000);
  zvec::IndexParams::Ptr params;
  if (diskann) {
    params = std::make_shared<zvec::DiskAnnIndexParams>(MetricType::L2, 64, 100, 32);
  } else {
    params = std::make_shared<zvec::HnswIndexParams>(MetricType::L2, 32, 200);
  }
  zvec::CollectionSchema schema("collection_benchmark");
  Check(schema.add_field(std::make_shared<zvec::FieldSchema>(
      kField, zvec::DataType::VECTOR_FP32, base.dimension, false, params)));

  const auto start = Clock::now();
  auto collection = Value(Collection::CreateAndOpen(path, schema, CollectionOptions{false, true}));
  constexpr uint32_t kBatch = 256;
  for (uint32_t first = 0; first < base.rows; first += kBatch) {
    const uint32_t end = std::min(first + kBatch, base.rows);
    std::vector<zvec::Doc> docs;
    docs.reserve(end - first);
    for (uint32_t i = first; i < end; ++i) {
      zvec::Doc doc;
      doc.set_pk(std::to_string(i));
      doc.set(kField, std::vector<float>(base.row(i), base.row(i) + base.dimension));
      docs.emplace_back(std::move(doc));
    }
    const auto statuses = Value(collection->insert(docs));
    if (statuses.size() != docs.size()) {
      throw std::runtime_error("Insert status count differs from batch size");
    }
    for (const auto &status : statuses) {
      Check(status);
    }
  }
  Check(collection->flush());
  const double write_ms = ElapsedMs(start);
  const auto optimize_start = Clock::now();
  Check(collection->optimize(zvec::OptimizeOptions{2}));
  const double optimize_ms = ElapsedMs(optimize_start);
  CheckCollection(*collection, diskann, base.rows);
  Check(collection->close());
  result << kPrefix << "{\"phase\":\"build\",\"index\":\""
            << (diskann ? "diskann" : "hnsw") << "\",\"vectors\":" << base.rows
            << ",\"index_complete\":true,\"write_ms\":" << write_ms
            << ",\"optimize_ms\":" << optimize_ms << "}\n";
}

double Quantile(const std::vector<double> &sorted, double percentile) {
  return sorted[static_cast<size_t>(std::ceil(percentile * sorted.size())) - 1];
}

void Query(const std::filesystem::path &data, const std::string &path, bool diskann,
           std::ostream &result) {
  const auto base_rows = ReadDocCount(data / "base.fbin");
  const auto queries = ReadMatrix<float>(data / "queries.fbin", 20'000);
  const auto truth = ReadMatrix<uint32_t>(data / "neighbors.ibin", 20'000);
  if (queries.rows <= kCalibration || truth.rows != queries.rows || truth.dimension != kTopK) {
    throw std::runtime_error("Query and truth matrices disagree");
  }

  const auto baseline = ProcessRss();
  const auto opening = Clock::now();
  auto collection = Value(Collection::Open(path, CollectionOptions{true, true}));
  const auto open_ms = ElapsedMs(opening);
  const auto after_open = ProcessRss();
  CheckCollection(*collection, diskann, base_rows);

  zvec::SearchQuery query;
  query.topk_ = kTopK;
  query.target_.field_name_ = kField;
  query.output_fields_ = std::vector<std::string>{};
  auto eval = [&](uint32_t i, double *latency) {
    query.target_.set_vector(std::string(reinterpret_cast<const char *>(queries.row(i)),
                                         queries.dimension * sizeof(float)));
    const auto start = Clock::now();
    auto hits = Value(collection->query(query));
    if (latency) {
      *latency = ElapsedMs(start);
    }
    if (hits.size() != kTopK) {
      throw std::runtime_error("Invalid top-k result size");
    }
    size_t matched = 0;
    std::array<uint32_t, kTopK> ids{};
    size_t seen = 0;
    for (const auto &hit : hits) {
      if (!hit || !std::isfinite(hit->score())) {
        throw std::runtime_error("Invalid result");
      }
      const auto &pk = hit->pk_ref();
      uint32_t id = 0;
      const auto parsed = std::from_chars(pk.data(), pk.data() + pk.size(), id);
      if (parsed.ec != std::errc{} || parsed.ptr != pk.data() + pk.size() || id >= base_rows ||
          std::find(ids.begin(), ids.begin() + seen, id) != ids.begin() + seen) {
        throw std::runtime_error("Invalid or duplicate result primary key");
      }
      ids[seen++] = id;
      const auto exact = truth.row(i);
      matched += std::find(exact, exact + kTopK, id) != exact + kTopK;
    }
    return matched;
  };

  int budget = 0;
  double calibration_recall = 0;
  for (const int candidate : {32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048}) {
    if (diskann) {
      query.target_.query_params_ = std::make_shared<zvec::DiskAnnQueryParams>(candidate);
    } else {
      query.target_.query_params_ = std::make_shared<zvec::HnswQueryParams>(candidate);
    }
    size_t correct = 0;
    for (uint32_t i = 0; i < kCalibration; ++i) correct += eval(i, nullptr);
    calibration_recall = static_cast<double>(correct) / (kTopK * kCalibration);
    if (calibration_recall >= 0.95) {
      budget = candidate;
      break;
    }
  }
  if (!budget) throw std::runtime_error("Could not reach 95% calibration recall");

  for (uint32_t i = kCalibration; i < queries.rows; ++i) eval(i, nullptr);
  const auto after_warmup = ProcessRss();
  auto sampled_peak = after_warmup;
  std::vector<double> latencies;
  latencies.reserve((queries.rows - kCalibration) * kPasses);
  size_t correct = 0;
  double elapsed = 0;
  for (uint32_t pass = 0; pass < kPasses; ++pass) {
    for (uint32_t i = kCalibration; i < queries.rows; ++i) {
      double latency;
      correct += eval(i, &latency);
      latencies.push_back(latency);
      elapsed += latency;
      if (i % 50 == 0) sampled_peak = std::max(sampled_peak, ProcessRss());
    }
  }
  const auto after_query = ProcessRss();
  sampled_peak = std::max(sampled_peak, after_query);
  const auto recall = static_cast<double>(correct) / (kTopK * latencies.size());
  Check(collection->close());
  std::sort(latencies.begin(), latencies.end());
  result << kPrefix << "{\"phase\":\"query\",\"index\":\""
            << (diskann ? "diskann" : "hnsw")
            << "\",\"vectors\":" << base_rows << ",\"dimension\":" << queries.dimension
            << ",\"queries_per_pass\":" << queries.rows - kCalibration
            << ",\"passes\":" << kPasses << ",\"budget\":" << budget
            << ",\"calibration_recall_at_10\":" << calibration_recall
            << ",\"recall_at_10\":" << recall << ",\"qps\":" << latencies.size() * 1000 / elapsed
            << ",\"p95_ms\":" << Quantile(latencies, 0.95)
            << ",\"p99_ms\":" << Quantile(latencies, 0.99) << ",\"open_ms\":" << open_ms
            << ",\"baseline_rss_bytes\":" << baseline << ",\"after_open_rss_bytes\":" << after_open
            << ",\"after_warmup_rss_bytes\":" << after_warmup
            << ",\"after_query_rss_bytes\":" << after_query
            << ",\"sampled_peak_query_rss_bytes\":" << sampled_peak
            << ",\"query_rss_delta_bytes\":"
            << static_cast<int64_t>(after_query) - static_cast<int64_t>(baseline) << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5 && argc != 6) {
      throw std::runtime_error(
          "Usage: benchmark build|query hnsw|diskann DATA_DIR COLLECTION_DIR [RESULT_FILE]");
    }
    const std::string_view phase(argv[1]), index(argv[2]);
    if ((phase != "build" && phase != "query") || (index != "hnsw" && index != "diskann")) {
      throw std::runtime_error("Invalid phase or index algorithm");
    }
    zvec::GlobalConfig::ConfigData config;
    config.query_thread_count = 1;
    config.optimize_thread_count = 2;
    Check(zvec::GlobalConfig::Instance().initialize(config));
    std::ostringstream result;
    result << std::setprecision(12);
    if (phase == "build") {
      Build(argv[3], argv[4], index == "diskann", result);
    } else {
      Query(argv[3], argv[4], index == "diskann", result);
    }
    const auto line = result.str();
    std::cout << line << std::flush;
    if (argc == 6) {
      // simctl can lose console output when a short-lived app exits during launch.
      const std::string temporary = std::string(argv[5]) + ".tmp";
      std::ofstream output(temporary);
      output << line;
      output.close();
      if (!output) {
        throw std::runtime_error("Cannot write benchmark result");
      }
      std::filesystem::rename(temporary, argv[5]);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

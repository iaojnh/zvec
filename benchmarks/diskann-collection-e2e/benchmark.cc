#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task_info.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif
#if defined(__ANDROID__)
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif
#if defined(__ANDROID__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
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

#if defined(__ANDROID__) || defined(__APPLE__)
class IndexCacheMapping {
 public:
  explicit IndexCacheMapping(const std::filesystem::path &file)
      : path_(file.string()),
        bytes_(std::filesystem::file_size(file)),
        residency_((bytes_ + getpagesize() - 1) / getpagesize()) {
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ == -1) Fail(errno, "open");
#if defined(__ANDROID__)
    // DONTNEED only removes clean pages.
    if (fsync(fd_) != 0) {
      const int error = errno;
      close(fd_);
      Fail(error, "fsync");
    }
#endif
    address_ = mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (address_ == MAP_FAILED) {
      const int error = errno;
      close(fd_);
      Fail(error, "mmap");
    }
  }

  ~IndexCacheMapping() {
    munmap(address_, bytes_);
    close(fd_);
  }

  IndexCacheMapping(const IndexCacheMapping &) = delete;
  IndexCacheMapping &operator=(const IndexCacheMapping &) = delete;

  void Evict() {
    for (int attempt = 0; attempt < 8; ++attempt) {
#if defined(__ANDROID__)
      const int error = posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
      if (error) Fail(error, "posix_fadvise");
      auto *pages = residency_.data();
#else
      if (msync(address_, bytes_, MS_INVALIDATE | MS_SYNC) != 0) Fail(errno, "msync");
      auto *pages = reinterpret_cast<char *>(residency_.data());
#endif
      if (mincore(address_, bytes_, pages) != 0) Fail(errno, "mincore");
      if (std::none_of(residency_.begin(), residency_.end(), [](auto page) { return page & 1; })) {
        return;
      }
    }
    throw std::runtime_error("Index file pages remained resident before query: " + path_);
  }

 private:
  [[noreturn]] void Fail(int error, const char *operation) const {
    throw std::system_error(error, std::generic_category(), operation + (" " + path_));
  }

  std::string path_;
  size_t bytes_;
  std::vector<unsigned char> residency_;
  int fd_;
  void *address_;
};
#endif

#if defined(__APPLE__)
void ObserveSimulatorBoundary(const std::filesystem::path &control, const char *point) {
  const auto temporary = control / (std::string(point) + ".tmp");
  {
    std::ofstream marker(temporary);
    marker << getpid() << '\n';
    marker.close();
    if (!marker) throw std::runtime_error("Could not write simulator I/O marker");
  }
  std::filesystem::rename(temporary, control / point);
  const auto resume = control / (std::string(point) + ".go");
  const auto deadline = Clock::now() + std::chrono::seconds(30);
  while (!std::filesystem::exists(resume)) {
    if (Clock::now() >= deadline) {
      throw std::runtime_error("Simulator I/O observer did not acknowledge query boundary");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}
#endif

#if defined(__ANDROID__)
struct AndroidReadCounters {
  std::optional<uint64_t> read_ios;
  std::optional<uint64_t> device_bytes;
  std::optional<uint64_t> process_bytes;
};

std::optional<uint64_t> AndroidProcessReadBytes() {
  std::ifstream process("/proc/self/io");
  std::string key;
  uint64_t count;
  while (process >> key >> count) {
    if (key == "read_bytes:") return count;
  }
  return std::nullopt;
}

std::string AndroidDevice(const std::string &collection) {
  struct stat info {};
  if (stat(collection.c_str(), &info) != 0) {
    throw std::runtime_error("Could not identify the Android collection device");
  }
  return std::to_string(major(info.st_dev)) + ":" + std::to_string(minor(info.st_dev));
}

AndroidReadCounters ReadAndroidCounters(const std::string &device) {
  AndroidReadCounters result;
  uint64_t reads, merges, sectors;
  std::ifstream disk("/sys/dev/block/" + device + "/stat");
  if (disk >> reads >> merges >> sectors) {
    result.read_ios = reads;
    result.device_bytes = sectors * 512;
  } else {
    std::ifstream devices("/proc/diskstats");
    for (std::string line; std::getline(devices, line);) {
      std::istringstream row(line);
      unsigned major_id, minor_id;
      std::string name;
      if (row >> major_id >> minor_id >> name >> reads >> merges >> sectors) {
        if (std::to_string(major_id) + ":" + std::to_string(minor_id) == device) {
          result.read_ios = reads;
          result.device_bytes = sectors * 512;
          break;
        }
      }
    }
  }
  result.process_bytes = AndroidProcessReadBytes();
  return result;
}

void WriteDifference(std::ostream &out, std::optional<uint64_t> before,
                     std::optional<uint64_t> after) {
  if (before && after && *after >= *before) {
    out << *after - *before;
  } else {
    out << "null";
  }
}
#endif

void Query(const std::filesystem::path &data, const std::string &path, bool diskann,
           std::ostream &result) {
#if defined(__ANDROID__) || defined(__APPLE__)
  const bool cold = diskann && std::getenv("ZVEC_BENCH_COLD_STORAGE");
  std::vector<std::unique_ptr<IndexCacheMapping>> cache_mappings;
#else
  const bool cold = false;
#endif
#if defined(__ANDROID__)
  uint64_t cold_storage_bytes = 0;
  uint64_t cold_min_bytes = UINT64_MAX;
#endif
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

  if (!cold) {
    for (uint32_t i = kCalibration; i < queries.rows; ++i) eval(i, nullptr);
  }
#if defined(__ANDROID__) || defined(__APPLE__)
  if (cold) {
    for (const auto &file : std::filesystem::recursive_directory_iterator(path)) {
      const auto name = file.path().filename().string();
      if (file.is_regular_file() && name.rfind("embedding.index.", 0) == 0 &&
          file.path().extension() == ".proxima") {
        cache_mappings.push_back(std::make_unique<IndexCacheMapping>(file.path()));
      }
    }
    if (cache_mappings.empty()) throw std::runtime_error("No persisted DiskANN index files found");
  }
#endif
  const auto after_warmup = ProcessRss();
  auto sampled_peak = after_warmup;
  std::vector<double> latencies;
  latencies.reserve((queries.rows - kCalibration) * kPasses);
  size_t correct = 0;
  double elapsed = 0;
#if defined(__ANDROID__)
  const bool observe = diskann && std::getenv("ZVEC_BENCH_QUERY_IO");
  const auto device = observe ? AndroidDevice(path) : std::string{};
  const auto before_io = observe ? ReadAndroidCounters(device) : AndroidReadCounters{};
#elif defined(__APPLE__)
  const char *control = diskann ? std::getenv("ZVEC_BENCH_IO_CONTROL") : nullptr;
  const bool observe = control != nullptr;
  if (observe) ObserveSimulatorBoundary(control, "before");
#else
  const bool observe = false;
#endif
  const auto query_begin = Clock::now();
  for (uint32_t pass = 0; pass < kPasses; ++pass) {
    for (uint32_t i = kCalibration; i < queries.rows; ++i) {
#if defined(__ANDROID__) || defined(__APPLE__)
      for (const auto &mapping : cache_mappings) mapping->Evict();
#endif
#if defined(__ANDROID__)
      const auto cold_before = cold ? AndroidProcessReadBytes() : std::nullopt;
#endif
      double latency;
      correct += eval(i, &latency);
#if defined(__ANDROID__)
      if (cold) {
        const auto cold_after = AndroidProcessReadBytes();
        if (!cold_before || !cold_after || *cold_after <= *cold_before) {
          throw std::runtime_error("Cold query has no accounted Android storage reads");
        }
        const auto bytes = *cold_after - *cold_before;
        cold_storage_bytes += bytes;
        cold_min_bytes = std::min(cold_min_bytes, bytes);
      }
#endif
      latencies.push_back(latency);
      elapsed += latency;
      if (i % 50 == 0) sampled_peak = std::max(sampled_peak, ProcessRss());
    }
  }
  const auto query_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - query_begin).count();
#if defined(__ANDROID__)
  const auto after_io = observe ? ReadAndroidCounters(device) : AndroidReadCounters{};
#elif defined(__APPLE__)
  if (observe) ObserveSimulatorBoundary(control, "after");
#endif
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
            << static_cast<int64_t>(after_query) - static_cast<int64_t>(baseline);
  if (observe) {
    result << ",\"query_window_ns\":" << query_ns;
#if defined(__ANDROID__)
    result << ",\"storage_observation\":{\"scope\":\"Android guest device " << device
           << "\",\"cache_policy\":\"buffered DiskANN pread\",\"device_read_ios\":";
    WriteDifference(result, before_io.read_ios, after_io.read_ios);
    result << ",\"device_read_bytes\":";
    WriteDifference(result, before_io.device_bytes, after_io.device_bytes);
    result << ",\"process_storage_read_bytes\":";
    WriteDifference(result, before_io.process_bytes, after_io.process_bytes);
    result << "}";
#endif
  }
  if (cold) {
    result << ",\"cold_storage\":{\"verified_queries\":" << latencies.size()
           << ",\"query_ns\":" << static_cast<uint64_t>(elapsed * 1e6);
#if defined(__ANDROID__)
    result << ",\"storage_read_bytes\":" << cold_storage_bytes
           << ",\"min_storage_read_bytes\":" << cold_min_bytes;
#endif
    result << "}";
  }
  result << "}\n";
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

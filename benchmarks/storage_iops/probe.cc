// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
// Standalone storage probe: deliberately has no dependency on Zvec or GTest.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE
#include <libproc.h>
#endif
#include <sys/mount.h>
#else
#include <sys/vfs.h>
#endif
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kBlock = 4096;
constexpr uint64_t kSeed = 20261105;

struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

uint64_t Word(uint64_t index) {
  uint64_t value = index + kSeed + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::string Quote(const std::string &value) {
  std::string result = "\"";
  for (char c : value) {
    if (c == '"' || c == '\\') result += '\\';
    if (c == '\n') result += "\\n";
    else if (c == '\r') result += "\\r";
    else result += c;
  }
  return result + '"';
}

struct Options {
  std::string file;
  std::string result_file;
  std::string mode{"buffered"};
  unsigned threads{1};
  unsigned seconds{10};
  uint64_t mib{256};
};

Options Parse(int argc, char **argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (i + 1 == argc) throw std::runtime_error("Missing argument: " + key);
    std::string value = argv[++i];
    if (key == "--file") result.file = value;
    else if (key == "--result") result.result_file = value;
    else if (key == "--mode") result.mode = value;
    else if (key == "--threads") result.threads = std::stoul(value);
    else if (key == "--seconds") result.seconds = std::stoul(value);
    else if (key == "--size-mib") result.mib = std::stoull(value);
    else throw std::runtime_error("Unknown option: " + key);
  }
  if (result.file.empty() || !result.threads || result.threads > 32 ||
      !result.seconds || !result.mib || result.mib > 4096 ||
      (result.mode != "prepare" && result.mode != "buffered" &&
       result.mode != "uncached" && result.mode != "warmhint" && result.mode != "evict")) {
    throw std::runtime_error("Invalid probe options");
  }
  return result;
}

void Prepare(const Options &options, std::ostream &output_record) {
  const uint64_t bytes = options.mib * 1024 * 1024;
  if (std::filesystem::exists(options.file)) {
    std::ifstream input(options.file, std::ios::binary);
    uint64_t first = 0;
    input.read(reinterpret_cast<char *>(&first), sizeof(first));
    if (!input || first != Word(0) ||
        std::filesystem::file_size(options.file) != bytes) {
      throw std::runtime_error("Refusing to replace an unrelated existing file");
    }
  } else {
    auto parent = std::filesystem::path(options.file).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream output(options.file, std::ios::binary);
    std::vector<uint64_t> block(1024 * 1024 / sizeof(uint64_t));
    for (uint64_t start = 0; start < bytes; start += block.size() * sizeof(uint64_t)) {
      for (size_t i = 0; i < block.size(); ++i) block[i] = Word(start / 8 + i);
      output.write(reinterpret_cast<const char *>(block.data()), block.size() * 8);
    }
    output.close();
    if (!output) throw std::runtime_error("Failed to create fixture");
  }
#if defined(_WIN32)
  HANDLE file = CreateFileW(std::filesystem::path(options.file).wstring().c_str(),
                            GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot flush fixture");
  BOOL ok = FlushFileBuffers(file);
  CloseHandle(file);
  if (!ok) throw std::runtime_error("FlushFileBuffers failed");
#else
  int fd = ::open(options.file.c_str(), O_RDWR);
  if (fd < 0) throw std::runtime_error("Cannot flush fixture");
  int ret = ::fsync(fd);
  ::close(fd);
  if (ret != 0) throw std::runtime_error("fsync failed");
#endif
  output_record << "IOPS_RESULT {\"prepared\":true,\"file_bytes\":" << bytes
            << ",\"seed\":" << kSeed << "}" << std::endl;
}

struct Reader {
#if defined(_WIN32)
  HANDLE handle{INVALID_HANDLE_VALUE};
#else
  int fd{-1};
#endif
  void *buffer{nullptr};
  std::string policy;
  bool evict{false};

  explicit Reader(const Options &options) {
    const bool no_cache = options.mode == "uncached" || options.mode == "warmhint";
    evict = options.mode == "evict";
#if defined(_WIN32)
    if (evict) throw Unsupported("Per-read cache eviction is a Linux/Android diagnostic");
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if (no_cache) flags |= FILE_FLAG_NO_BUFFERING;
    handle = CreateFileW(std::filesystem::path(options.file).wstring().c_str(),
                         GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                         nullptr, OPEN_EXISTING, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("CreateFileW failed");
    buffer = VirtualAlloc(nullptr, kBlock, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    policy = no_cache ? "ReadFile FILE_FLAG_NO_BUFFERING" : "ReadFile buffered warm";
#else
    int flags = O_RDONLY;
#if !defined(__APPLE__)
    if (no_cache) flags |= O_DIRECT;
#else
    if (evict) throw Unsupported("Per-read cache eviction is a Linux/Android diagnostic");
#endif
    fd = ::open(options.file.c_str(), flags);
    if (fd < 0) {
      if (no_cache && (errno == EINVAL || errno == ENOTSUP)) {
        throw Unsupported("Uncached file open unsupported, errno=" + std::to_string(errno));
      }
      throw std::runtime_error("open failed, errno=" + std::to_string(errno));
    }
    if (posix_memalign(&buffer, kBlock, kBlock) != 0) buffer = nullptr;
#if defined(__APPLE__)
    if (no_cache) {
      if (::fcntl(fd, F_NOCACHE, 1) != 0 || ::fcntl(fd, F_RDAHEAD, 0) != 0) {
        ::close(fd);
        std::free(buffer);
        throw Unsupported("F_NOCACHE/F_RDAHEAD unavailable");
      }
      policy = "pread F_NOCACHE + F_RDAHEAD(0), advisory";
    } else {
      policy = "pread buffered warm";
    }
#else
    policy = no_cache ? "pread O_DIRECT" : "pread buffered warm";
    if (evict) {
      if (posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM) != 0 ||
          posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0) {
        ::close(fd);
        std::free(buffer);
        throw Unsupported("Per-file POSIX_FADV_DONTNEED unavailable");
      }
      policy = "pread + per-read POSIX_FADV_DONTNEED";
    }
#endif
#endif
    if (!buffer) {
#if defined(_WIN32)
      CloseHandle(handle);
#else
      ::close(fd);
#endif
      throw std::runtime_error("Aligned buffer allocation failed");
    }
  }
  ~Reader() {
#if defined(_WIN32)
    if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    if (buffer) VirtualFree(buffer, 0, MEM_RELEASE);
#else
    if (fd >= 0) ::close(fd);
    std::free(buffer);
#endif
  }
  Reader(const Reader &) = delete;
  Reader &operator=(const Reader &) = delete;
  bool Read(uint64_t offset) {
#if defined(_WIN32)
    LARGE_INTEGER location{};
    location.QuadPart = static_cast<LONGLONG>(offset);
    DWORD bytes = 0;
    if (!SetFilePointerEx(handle, location, nullptr, FILE_BEGIN) ||
        !ReadFile(handle, buffer, static_cast<DWORD>(kBlock), &bytes, nullptr) ||
        bytes != kBlock) return false;
#else
    if (::pread(fd, buffer, kBlock, static_cast<off_t>(offset)) !=
        static_cast<ssize_t>(kBlock)) return false;
#endif
    const bool valid = *static_cast<const uint64_t *>(buffer) == Word(offset / 8);
#if defined(__linux__)
    if (evict && posix_fadvise(fd, static_cast<off_t>(offset), kBlock,
                               POSIX_FADV_DONTNEED) != 0) return false;
#endif
    return valid;
  }
};

struct Counters {
  bool available{false};
  uint64_t read_bytes{0};
  std::string source;
};

Counters ReadCounters([[maybe_unused]] const char *path = "/proc/self/io") {
  Counters result;
#if defined(_WIN32)
  IO_COUNTERS counters{};
  result.available = GetProcessIoCounters(GetCurrentProcess(), &counters) != 0;
  result.read_bytes = counters.ReadTransferCount;
  result.source = "GetProcessIoCounters ReadTransferCount (logical I/O)";
#elif defined(__APPLE__)
#if !TARGET_OS_IPHONE
  rusage_info_v4 counters{};
  result.available = proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                                    reinterpret_cast<rusage_info_t *>(&counters)) == 0;
  result.read_bytes = counters.ri_diskio_bytesread;
  result.source = "proc_pid_rusage ri_diskio_bytesread (OS disk-I/O accounting)";
#else
  result.source = "No disk-byte counter in the public iOS SDK";
#endif
#else
  std::ifstream input(path);
  std::string key;
  uint64_t value;
  while (input >> key >> value) {
    if (key == "read_bytes:") {
      result.available = true;
      result.read_bytes = value;
    }
  }
  result.source = "/proc/self/io read_bytes (storage I/O accounting)";
#endif
  return result;
}

void Measure(const Options &options, std::ostream &output_record) {
  if (std::filesystem::file_size(options.file) != options.mib * 1024 * 1024) {
    throw std::runtime_error("Wrong fixture size");
  }
  if (options.mode == "buffered" || options.mode == "warmhint") {
    std::ifstream input(options.file, std::ios::binary);
    std::vector<char> chunk(1024 * 1024);
    while (input.read(chunk.data(), chunk.size()) || input.gcount()) {}
    if (!input.eof()) throw std::runtime_error("Cannot warm fixture");
  }
  std::vector<std::unique_ptr<Reader>> readers;
  for (unsigned i = 0; i < options.threads; ++i) {
    readers.push_back(std::make_unique<Reader>(options));
  }
  struct Worker {
    uint64_t operations{0};
    bool failed{false};
    bool io_available{false};
    uint64_t io_bytes{0};
    std::vector<double> latencies;
  };
  std::vector<Worker> workers(options.threads);
  std::vector<std::thread> threads;
  std::mutex mutex;
  std::condition_variable cv;
  bool go = false;
  std::atomic<bool> failed{false};
  Clock::time_point deadline;
  const uint64_t blocks = options.mib * 1024 * 1024 / kBlock;
  for (unsigned i = 0; i < options.threads; ++i) {
    threads.emplace_back([&, i] {
      auto &worker = workers[i];
      worker.latencies.reserve(8192);
      uint64_t state = kSeed + i;
#if defined(__linux__)
      const auto io_start = ReadCounters("/proc/thread-self/io");
#endif
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return go; });
      }
      while (!failed.load(std::memory_order_relaxed)) {
        const auto start = Clock::now();
        if (start >= deadline) break;
        state = state * 6364136223846793005ULL + 1;
        // Use high bits; low LCG bits would repeatedly visit a small subset.
        const uint64_t offset = ((state >> 16) % blocks) * kBlock;
        if (!readers[i]->Read(offset)) {
          worker.failed = true;
          failed.store(true);
          break;
        }
        if ((worker.operations & 63) == 0) {
          worker.latencies.push_back(
              std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        }
        ++worker.operations;
      }
#if defined(__linux__)
      const auto io_end = ReadCounters("/proc/thread-self/io");
      worker.io_available = io_start.available && io_end.available &&
                            io_end.read_bytes >= io_start.read_bytes;
      if (worker.io_available) worker.io_bytes = io_end.read_bytes - io_start.read_bytes;
#endif
    });
  }
  const auto before = ReadCounters();
  const auto start = Clock::now();
  {
    std::lock_guard<std::mutex> lock(mutex);
    deadline = start + std::chrono::seconds(options.seconds);
    go = true;
  }
  cv.notify_all();
  for (auto &thread : threads) thread.join();
  const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  const auto after = ReadCounters();
  uint64_t operations = 0;
  uint64_t worker_io_bytes = 0;
  bool worker_io_available = true;
  std::vector<double> latencies;
  for (const auto &worker : workers) {
    if (worker.failed) throw std::runtime_error("Read failed or fixture data mismatch");
    operations += worker.operations;
    worker_io_available = worker_io_available && worker.io_available;
    worker_io_bytes += worker.io_bytes;
    latencies.insert(latencies.end(), worker.latencies.begin(), worker.latencies.end());
  }
  if (latencies.empty()) throw std::runtime_error("No completed I/O");
  std::sort(latencies.begin(), latencies.end());
  auto percentile = [&](double value) {
    return latencies.at(static_cast<size_t>(std::ceil(latencies.size() * value)) - 1);
  };
  output_record << std::setprecision(12)
            << "IOPS_RESULT {\"supported\":true,\"mode\":" << Quote(options.mode)
            << ",\"policy\":" << Quote(readers[0]->policy)
            << ",\"block_bytes\":4096,\"file_bytes\":" << options.mib * 1024 * 1024
            << ",\"threads\":" << options.threads
            << ",\"elapsed_seconds\":" << elapsed
            << ",\"operations\":" << operations
            << ",\"iops\":" << operations / elapsed
            << ",\"mib_per_second\":" << operations * double(kBlock) / elapsed / 1048576
            << ",\"p50_ms\":" << percentile(0.5)
            << ",\"p95_ms\":" << percentile(0.95)
            << ",\"p99_ms\":" << percentile(0.99)
            << ",\"latency_samples\":" << latencies.size()
            << ",\"payload_validated\":true,\"os_read_bytes_source\":" << Quote(after.source)
            << ",\"os_accounted_read_bytes\":";
  if (before.available && after.available && after.read_bytes >= before.read_bytes) {
    output_record << after.read_bytes - before.read_bytes;
  } else {
    output_record << "null";
  }
  output_record << ",\"worker_accounted_read_bytes\":";
  if (worker_io_available) output_record << worker_io_bytes;
  else output_record << "null";
  output_record << "}" << std::endl;
}

void Emit(const Options &options, const std::string &record) {
  if (!options.result_file.empty()) {
    std::ofstream output(options.result_file, std::ios::binary | std::ios::trunc);
    output << record;
    output.close();
    if (!output) throw std::runtime_error("Cannot write native result file");
  }
  std::cout << record << std::flush;
}
}  // namespace

int main(int argc, char **argv) {
  Options options;
  try {
    options = Parse(argc, argv);
    std::ostringstream record;
    if (options.mode == "prepare") Prepare(options, record);
    else Measure(options, record);
    Emit(options, record.str());
    return 0;
  } catch (const Unsupported &error) {
    std::ostringstream record;
    record << "IOPS_RESULT {\"supported\":false,\"reason\":"
           << Quote(error.what()) << "}" << std::endl;
    Emit(options, record.str());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "IOPS_ERROR " << error.what() << std::endl;
    return 1;
  }
}

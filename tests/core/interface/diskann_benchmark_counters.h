// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/resource.h>
#include <unistd.h>
#if !TARGET_OS_IPHONE
#include <libproc.h>
#endif
#else
#include <sys/resource.h>
#endif

namespace zvec::core_interface {
namespace {

struct BenchmarkCounters {
  bool has_read_bytes{false};
  uint64_t read_bytes{0};
  double cpu_seconds{0};
  int64_t major_faults{0};
  const char *source{"unavailable"};
};

BenchmarkCounters BenchmarkSnapshot() {
  BenchmarkCounters result;
#if defined(_WIN32)
  IO_COUNTERS io{};
  if (GetProcessIoCounters(GetCurrentProcess(), &io)) {
    result.has_read_bytes = true;
    result.read_bytes = io.ReadTransferCount;
  }
  result.source = "GetProcessIoCounters logical transfer bytes; not physical I/O";
  FILETIME creation{}, exit{}, kernel{}, user{};
  if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
    ULARGE_INTEGER k{}, u{};
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;
    u.HighPart = user.dwHighDateTime;
    result.cpu_seconds = (k.QuadPart + u.QuadPart) / 1e7;
  }
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error("getrusage failed");
  }
  result.cpu_seconds = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
                       usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
  result.major_faults = usage.ru_majflt;
#if defined(__APPLE__)
#if !TARGET_OS_IPHONE
  rusage_info_v4 io{};
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                     reinterpret_cast<rusage_info_t *>(&io)) == 0) {
    result.has_read_bytes = true;
    result.read_bytes = io.ri_diskio_bytesread;
  }
  result.source = "proc_pid_rusage OS disk-I/O bytes";
#else
  result.source = "unavailable in the public iOS SDK";
#endif
#else
  std::ifstream input("/proc/self/io");
  std::string name;
  uint64_t value = 0;
  while (input >> name >> value) {
    if (name == "read_bytes:") {
      result.has_read_bytes = true;
      result.read_bytes = value;
    }
  }
  result.source = "/proc/self/io OS storage-read bytes (guest on Android)";
#endif
#endif
  return result;
}

void BenchmarkBytes(std::ostream &output, const BenchmarkCounters &before,
                    const BenchmarkCounters &after) {
  if (before.has_read_bytes && after.has_read_bytes &&
      after.read_bytes >= before.read_bytes) {
    output << after.read_bytes - before.read_bytes;
  } else {
    output << "null";
  }
}

void BenchmarkEmit(const std::ostringstream &record, const std::string &path) {
  const auto line = "ZVEC_BENCH_RESULT " + record.str() + "\n";
  if (!path.empty()) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << line;
    file.close();
    if (!file) throw std::runtime_error("Cannot persist benchmark result");
  }
  std::cout << line << std::flush;
}

}  // namespace
}  // namespace zvec::core_interface

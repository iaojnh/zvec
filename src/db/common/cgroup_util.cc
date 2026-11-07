// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "db/common/cgroup_util.h"
#include <algorithm>
#include <charconv>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <system_error>
#include "db/common/cgroup_memory.h"

#if defined(PLATFORM_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec {

// Static member definitions
int CgroupUtil::cpu_cores_ = 0;
uint64_t CgroupUtil::memory_limit_ = 0;
unsigned long long CgroupUtil::last_idle_time_ = 0;
unsigned long long CgroupUtil::last_total_time_ = 0;
std::chrono::steady_clock::time_point CgroupUtil::last_cpu_check_;

namespace {

bool HasController(const std::string &list, const std::string &name) {
  std::istringstream stream(list);
  std::string token;
  while (std::getline(stream, token, ','))
    if (token == name) return true;
  return false;
}

bool ParseMemoryBytes(const std::string &text, uint64_t *bytes) {
  std::istringstream stream(text);
  std::string token, trailing;
  if (!(stream >> token) || stream >> trailing) return false;
  const auto parsed =
      std::from_chars(token.data(), token.data() + token.size(), *bytes);
  return parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size();
}

// Kernel paths are absolute. Never let malformed metadata or a cgroup outside
// the visible namespace escape the mount through a dot component.
bool ValidCgroupPath(const std::string &path) {
  if (path.empty() || path.front() != '/' ||
      path.find('\0') != std::string::npos)
    return false;
  std::istringstream stream(path);
  std::string part;
  while (std::getline(stream, part, '/'))
    if (part == "." || part == "..") return false;
  return path.size() == 1 || path.back() != '/';
}

std::string DecodeMountPath(const std::string &path) {
  std::string decoded;
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '\\') {
      // mountinfo escapes whitespace and backslashes as octal triplets.
      if (i + 3 >= path.size()) return {};
      unsigned value = 0;
      for (size_t j = 1; j <= 3; ++j) {
        if (path[i + j] < '0' || path[i + j] > '7') return {};
        value = value * 8 + path[i + j] - '0';
      }
      if (value == 0 || value > 255) return {};
      decoded += static_cast<char>(value);
      i += 3;
    } else {
      decoded += path[i];
    }
  }
  return decoded;
}

bool ReadMemoryBytes(const detail::CgroupFileReader &read_file,
                     const std::string &path, uint64_t *bytes) {
  std::string text;
  return read_file(path, &text) && ParseMemoryBytes(text, bytes);
}

std::string ChildPath(const std::string &parent, const std::string &child) {
  return parent == "/" ? "/" + child : parent + "/" + child;
}

void ReadHierarchyLimits(const std::string &leaf, const std::string &mount,
                         bool v2, const detail::CgroupFileReader &read_file,
                         uint64_t *limit) {
  std::string directory = leaf;
  while (true) {
    uint64_t value = 0;
    // Old v1 kernels can disable hierarchical accounting. Their parent limit
    // does not constrain descendants unless memory.use_hierarchy is enabled.
    const bool applies =
        v2 || directory == leaf ||
        (ReadMemoryBytes(
             read_file, ChildPath(directory, "memory.use_hierarchy"), &value) &&
         value == 1);
    if (applies) {
      if (ReadMemoryBytes(
              read_file,
              ChildPath(directory, v2 ? "memory.max" : "memory.limit_in_bytes"),
              &value))
        *limit = (std::min)(*limit, value);
      if (!v2) {
        // The kernel's v1 effective limit also covers ancestors hidden above
        // a subtree mount. It respects memory.use_hierarchy on older kernels.
        std::string text;
        if (read_file(ChildPath(directory, "memory.stat"), &text)) {
          std::istringstream lines(text);
          std::string line;
          while (std::getline(lines, line)) {
            std::istringstream fields(line);
            std::string key, rest;
            if (fields >> key && key == "hierarchical_memory_limit" &&
                std::getline(fields, rest) && ParseMemoryBytes(rest, &value))
              *limit = (std::min)(*limit, value);
          }
        }
      }
    }
    if (directory == mount) break;
    const auto separator = directory.rfind('/');
    directory.resize(separator == 0 ? 1 : separator);
  }
}

}  // namespace

uint64_t detail::CgroupMemoryLimit(uint64_t physical_bytes,
                                   const std::string &membership,
                                   const std::string &mountinfo,
                                   const CgroupFileReader &read_file) {
  // Limits may only tighten detected physical RAM. This also handles v1's
  // page-rounded, very large unlimited sentinel without assuming a page size.
  uint64_t limit = physical_bytes;
  std::string v1_path, v2_path, line;
  std::istringstream groups(membership);
  while (std::getline(groups, line)) {
    const auto first = line.find(':');
    if (first == std::string::npos) continue;
    const auto second = line.find(':', first + 1);
    if (second == std::string::npos) continue;
    const auto path = line.substr(second + 1);
    if (!ValidCgroupPath(path)) continue;
    const auto controllers = line.substr(first + 1, second - first - 1);
    if (line.substr(0, first) == "0" && controllers.empty()) v2_path = path;
    if (HasController(controllers, "memory")) v1_path = path;
  }

  std::istringstream mounts(mountinfo);
  while (std::getline(mounts, line)) {
    const auto separator = line.find(" - ");
    if (separator == std::string::npos) continue;
    std::istringstream fields(line.substr(0, separator));
    std::string id, parent, device, root, mount, options;
    if (!(fields >> id >> parent >> device >> root >> mount >> options))
      continue;
    std::istringstream filesystem(line.substr(separator + 3));
    std::string type, source, super_options;
    if (!(filesystem >> type >> source >> super_options)) continue;
    const bool v2 = type == "cgroup2";
    if (!v2 && (type != "cgroup" || !HasController(super_options, "memory")))
      continue;
    const auto &path = v2 ? v2_path : v1_path;
    if (path.empty()) continue;
    root = DecodeMountPath(root);
    mount = DecodeMountPath(mount);
    if (!ValidCgroupPath(root) || !ValidCgroupPath(mount)) continue;
    // A bind mount can expose only part of the hierarchy. Match whole path
    // components, not prefixes such as /tenant versus /tenant-other.
    std::string suffix;
    if (root == "/") {
      suffix = path.substr(1);
    } else if (path == root) {
      suffix.clear();
    } else if (path.size() > root.size() &&
               path.compare(0, root.size(), root) == 0 &&
               path[root.size()] == '/') {
      suffix = path.substr(root.size() + 1);
    } else {
      continue;
    }
    ReadHierarchyLimits(suffix.empty() ? mount : ChildPath(mount, suffix),
                        mount, v2, read_file, &limit);
    // Inspect every matching mount: a broader view may reveal an ancestor
    // hidden above a bind mount of the current cgroup.
  }
  if (membership.empty() || mountinfo.empty()) {
    // Retain the old root-only fallback when proc metadata is unavailable.
    // Do not guess a root if readable metadata identifies a different group.
    uint64_t value = 0;
    for (const auto *path : {"/sys/fs/cgroup/memory.max",
                             "/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
      if (ReadMemoryBytes(read_file, path, &value))
        limit = (std::min)(limit, value);
    }
  }
  return limit;
}

// Static initialization method
void CgroupUtil::initialize() {
  // Concurrent ConfigData construction must wait for the entire hierarchy
  // scan, never observe the physical-memory ceiling before it is tightened.
  static std::once_flag once;
  std::call_once(once, [] {
    update_cpu_cores();
    update_memory_limit();
    initialize_cpu_stats();
  });
}

int CgroupUtil::get_cpu_limit() {
  initialize();
  return cpu_cores_;
}

uint64_t CgroupUtil::get_memory_limit() {
  initialize();
  return memory_limit_;
}

// Other static methods implementation
double CgroupUtil::get_cpu_usage() {
  initialize();
  return calculate_cpu_usage();
}

uint64_t CgroupUtil::get_memory_usage() {
  initialize();
  return get_current_memory_usage();
}

uint64_t CgroupUtil::get_uptime() {
#if defined(PLATFORM_LINUX)
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return info.uptime;
  }
#elif defined(PLATFORM_MACOS)
  struct timeval boottime;
  size_t len = sizeof(boottime);
  int mib[2] = {CTL_KERN, KERN_BOOTTIME};
  if (sysctl(mib, 2, &boottime, &len, nullptr, 0) == 0) {
    time_t bsec = boottime.tv_sec;
    time_t csec = time(nullptr);
    return csec - bsec;
  }
#elif defined(PLATFORM_WINDOWS)
  return GetTickCount64() / 1000;
#endif
  return 0;
}

void CgroupUtil::update_cpu_cores() {
  if (read_cpu_cgroup()) {
    return;
  }

#if defined(PLATFORM_MACOS)
  int cores;
  size_t len = sizeof(cores);
  if (sysctlbyname("hw.ncpu", &cores, &len, nullptr, 0) == 0) {
    cpu_cores_ = cores;
  } else {
    cpu_cores_ = 1;
  }
#elif defined(PLATFORM_LINUX)
  cpu_cores_ = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu_cores_ <= 0) {
    cpu_cores_ = 1;
  }
#elif defined(PLATFORM_WINDOWS)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  cpu_cores_ = static_cast<int>(sysinfo.dwNumberOfProcessors);
  if (cpu_cores_ <= 0) {
    cpu_cores_ = 1;
  }
#endif
}

bool CgroupUtil::read_cpu_cgroup() {
#if defined(PLATFORM_LINUX)
  // cgroup v2
  std::ifstream file("/sys/fs/cgroup/cpu.max");
  if (file.is_open()) {
    std::string cpu_max;
    std::getline(file, cpu_max);

    int cpu_cores = 0;
    if (parse_cpu_max(cpu_max, &cpu_cores)) {
      cpu_cores_ = cpu_cores;
      return true;
    }
    return false;
  }

  // cgroup v1
  std::ifstream quota_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
  std::ifstream period_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us");

  if (quota_file.is_open() && period_file.is_open()) {
    long long quota, period;
    quota_file >> quota;
    period_file >> period;
    quota_file.close();
    period_file.close();

    if (quota > 0 && period > 0) {
      cpu_cores_ =
          static_cast<int>(std::ceil(static_cast<double>(quota) / period));
      return true;
    }
  }
#endif
  return false;
}

bool CgroupUtil::parse_cpu_max(const std::string &cpu_max, int *cpu_cores) {
  if (cpu_cores == nullptr) {
    return false;
  }

  std::istringstream stream(cpu_max);
  std::string quota_token;
  uint64_t period = 0;
  if (!(stream >> quota_token >> period) || period == 0) {
    return false;
  }

  std::string trailing_token;
  if (stream >> trailing_token) {
    return false;
  }
  if (quota_token == "max") {
    return false;
  }

  uint64_t quota = 0;
  const auto result = std::from_chars(
      quota_token.data(), quota_token.data() + quota_token.size(), quota);
  if (result.ec != std::errc{} ||
      result.ptr != quota_token.data() + quota_token.size() || quota == 0) {
    return false;
  }

  const uint64_t cores = quota / period + (quota % period != 0 ? 1 : 0);
  if (cores > static_cast<uint64_t>((std::numeric_limits<int>::max)())) {
    return false;
  }

  *cpu_cores = static_cast<int>(cores);
  return true;
}

void CgroupUtil::update_memory_limit() {
#if defined(PLATFORM_MACOS)
  uint64_t mem;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    memory_limit_ = mem;
  } else {
    memory_limit_ = 0;
  }
#elif defined(PLATFORM_LINUX)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    memory_limit_ = static_cast<uint64_t>(pages) * page_size;
  } else {
    memory_limit_ = 0;
  }
  read_memory_cgroup();
#elif defined(PLATFORM_WINDOWS)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    memory_limit_ = statex.ullTotalPhys;
  } else {
    memory_limit_ = 0;
  }
#endif
}

void CgroupUtil::read_memory_cgroup() {
#if defined(PLATFORM_LINUX)
  const auto read_file = [](const std::string &path, std::string *text) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    text->assign(std::istreambuf_iterator<char>(file),
                 std::istreambuf_iterator<char>());
    return !file.bad();
  };
  std::string membership, mountinfo;
  read_file("/proc/self/cgroup", &membership);
  read_file("/proc/self/mountinfo", &mountinfo);
  memory_limit_ = detail::CgroupMemoryLimit(memory_limit_, membership,
                                            mountinfo, read_file);
#endif
}

void CgroupUtil::initialize_cpu_stats() {
  last_cpu_check_ = std::chrono::steady_clock::now();
#if defined(PLATFORM_LINUX)
  read_proc_stat();
#endif
}

#if defined(PLATFORM_LINUX)
bool CgroupUtil::read_proc_stat() {
  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  if (!std::getline(file, line)) {
    return false;
  }

  std::istringstream iss(line);
  std::string cpu_label;
  iss >> cpu_label;

  if (cpu_label != "cpu") {
    return false;
  }

  unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
  iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

  unsigned long long idle_time = idle + iowait;
  unsigned long long total_time =
      user + nice + system + irq + softirq + steal + idle_time;

  last_idle_time_ = idle_time;
  last_total_time_ = total_time;

  return true;
}
#endif

uint64_t CgroupUtil::get_current_memory_usage() {
#if defined(PLATFORM_LINUX)
  // cgroup
  uint64_t usage = read_memory_usage_cgroup();
  if (usage > 0) {
    return usage;
  }

  // back to /proc/meminfo
  return read_memory_usage_proc();
#elif defined(PLATFORM_MACOS)
  return get_mac_os_memory_usage();
#elif defined(PLATFORM_WINDOWS)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    return statex.ullTotalPhys - statex.ullAvailPhys;
  }
  return 0;
#else
  return 0;
#endif
}

#if defined(PLATFORM_LINUX)
uint64_t CgroupUtil::read_memory_usage_cgroup() {
  // cgroup v2
  std::ifstream file("/sys/fs/cgroup/memory.current");
  if (file.is_open()) {
    uint64_t usage;
    file >> usage;
    file.close();
    return usage;
  }

  // cgroup v1
  std::ifstream v1_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
  if (v1_file.is_open()) {
    uint64_t usage;
    v1_file >> usage;
    v1_file.close();
    return usage;
  }

  return 0;
}

uint64_t CgroupUtil::read_memory_usage_proc() {
  std::ifstream file("/proc/meminfo");
  if (!file.is_open()) {
    return 0;
  }

  std::string line;
  uint64_t total_mem = 0;
  uint64_t free_mem = 0;
  uint64_t available_mem = 0;
  uint64_t buffers = 0;
  uint64_t cached = 0;

  while (std::getline(file, line)) {
    if (line.find("MemTotal:") == 0) {
      total_mem = extract_memory_value(line);
    } else if (line.find("MemFree:") == 0) {
      free_mem = extract_memory_value(line);
    } else if (line.find("MemAvailable:") == 0) {
      available_mem = extract_memory_value(line);
    } else if (line.find("Buffers:") == 0) {
      buffers = extract_memory_value(line);
    } else if (line.find("Cached:") == 0) {
      cached = extract_memory_value(line);
    }
  }

  if (available_mem > 0 && total_mem > available_mem) {
    return total_mem - available_mem;
  }

  if (total_mem > 0 && free_mem > 0) {
    return total_mem - free_mem - buffers - cached;
  }

  return 0;
}
#endif

#if defined(PLATFORM_MACOS)
uint64_t CgroupUtil::get_mac_os_memory_usage() {
  mach_port_t host_port = mach_host_self();
  mach_msg_type_number_t host_size =
      sizeof(vm_statistics64_data_t) / sizeof(integer_t);
  vm_size_t page_size;
  vm_statistics64_data_t vm_stat;

  if (host_page_size(host_port, &page_size) != KERN_SUCCESS) {
    return 0;
  }

  if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat,
                        &host_size) != KERN_SUCCESS) {
    return 0;
  }

  uint64_t used_memory =
      ((vm_stat.active_count + vm_stat.inactive_count + vm_stat.wire_count) *
       page_size);

  return used_memory;
}
#endif

uint64_t CgroupUtil::extract_memory_value(const std::string &line) {
  size_t colon_pos = line.find(':');
  if (colon_pos == std::string::npos) {
    return 0;
  }

  std::string value_str = line.substr(colon_pos + 1);
  std::istringstream iss(value_str);
  uint64_t value;
  std::string unit;

  iss >> value;
  if (iss >> unit) {
    if (unit == "kB") {
      value *= 1024;
    }
  }

  return value;
}

double CgroupUtil::calculate_cpu_usage() {
#if defined(PLATFORM_LINUX)
  return calculate_linux_cpu_usage();
#elif defined(PLATFORM_MACOS)
  return calculate_mac_os_cpu_usage();
#else
  return 0.0;
#endif
}

#if defined(PLATFORM_LINUX)
double CgroupUtil::calculate_linux_cpu_usage() {
  if (!read_proc_stat()) {
    return 0.0;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    return 0.0;
  }

  std::string line;
  if (!std::getline(file, line)) {
    return 0.0;
  }

  std::istringstream iss(line);
  std::string cpu_label;
  iss >> cpu_label;

  if (cpu_label != "cpu") {
    return 0.0;
  }

  unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
  iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

  unsigned long long current_idle = idle + iowait;
  unsigned long long current_total =
      user + nice + system + irq + softirq + steal + current_idle;

  unsigned long long idle_delta = current_idle - last_idle_time_;
  unsigned long long total_delta = current_total - last_total_time_;

  last_idle_time_ = current_idle;
  last_total_time_ = current_total;

  if (total_delta == 0) {
    return 0.0;
  }

  double cpu_usage =
      100.0 * (1.0 - static_cast<double>(idle_delta) / total_delta);
  return std::max(0.0, std::min(100.0, cpu_usage));
}
#endif

#if defined(PLATFORM_MACOS)
double CgroupUtil::calculate_mac_os_cpu_usage() {
  host_cpu_load_info_data_t cpuinfo;
  mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

  if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                      (host_info_t)&cpuinfo, &count) != KERN_SUCCESS) {
    return 0.0;
  }

  unsigned long long total_tick =
      cpuinfo.cpu_ticks[CPU_STATE_USER] + cpuinfo.cpu_ticks[CPU_STATE_SYSTEM] +
      cpuinfo.cpu_ticks[CPU_STATE_NICE] + cpuinfo.cpu_ticks[CPU_STATE_IDLE];

  unsigned long long idle_tick = cpuinfo.cpu_ticks[CPU_STATE_IDLE];

  static unsigned long long prev_total = 0;
  static unsigned long long prev_idle = 0;

  if (prev_total == 0) {
    prev_total = total_tick;
    prev_idle = idle_tick;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return calculate_mac_os_cpu_usage();
  }

  unsigned long long total_delta = total_tick - prev_total;
  unsigned long long idle_delta = idle_tick - prev_idle;

  prev_total = total_tick;
  prev_idle = idle_tick;

  if (total_delta == 0) {
    return 0.0;
  }

  double cpu_usage =
      100.0 * (1.0 - static_cast<double>(idle_delta) / total_delta);
  return std::max(0.0, std::min(100.0, cpu_usage));
}
#endif

}  // namespace zvec

// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace zvec::detail {

using CgroupFileReader =
    std::function<bool(const std::string &, std::string *)>;

// Resolve /proc/self/cgroup against /proc/self/mountinfo, then tighten the
// physical-memory ceiling with readable memory-controller limits. Injection
// of file reads keeps hierarchy tests independent of the host's cgroups.
uint64_t CgroupMemoryLimit(uint64_t physical_bytes,
                           const std::string &membership,
                           const std::string &mountinfo,
                           const CgroupFileReader &read_file);

}  // namespace zvec::detail

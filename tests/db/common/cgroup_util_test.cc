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
#include <future>
#include <map>
#include <gtest/gtest.h>
#include "db/common/cgroup_memory.h"
#include "db/common/mobile_memory.h"

using namespace zvec;

TEST(CgroupUtil, ParseCpuMaxQuota) {
  int cpu_cores = 0;

  ASSERT_TRUE(CgroupUtil::parse_cpu_max("100000 100000", &cpu_cores));
  EXPECT_EQ(1, cpu_cores);

  ASSERT_TRUE(CgroupUtil::parse_cpu_max("150000 100000\n", &cpu_cores));
  EXPECT_EQ(2, cpu_cores);

  ASSERT_TRUE(CgroupUtil::parse_cpu_max("400000 100000", &cpu_cores));
  EXPECT_EQ(4, cpu_cores);
}

TEST(CgroupUtil, ParseCpuMaxUnlimitedOrInvalid) {
  int cpu_cores = 7;

  EXPECT_FALSE(CgroupUtil::parse_cpu_max("max 100000", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parse_cpu_max("100000/100000", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parse_cpu_max("100000 0", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parse_cpu_max("0 100000", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parse_cpu_max("100000 100000 trailing", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parse_cpu_max("invalid", &cpu_cores));
  EXPECT_FALSE(CgroupUtil::parse_cpu_max("100000 100000", nullptr));

  EXPECT_EQ(7, cpu_cores);
}

namespace {

constexpr uint64_t kMiB = uint64_t{1} << 20;
constexpr uint64_t kPhysicalMemory = 4096 * kMiB;

class CgroupMemoryTest : public testing::Test {
 protected:
  uint64_t detect(uint64_t physical = kPhysicalMemory) {
    return detail::CgroupMemoryLimit(
        physical, membership, mounts,
        [this](const std::string &path, std::string *text) {
          reads.push_back(path);
          const auto found = files.find(path);
          if (found == files.end()) return false;
          *text = found->second;
          return true;
        });
  }
  std::string membership = "0::/tenant/app\n";
  std::string mounts = "12 1 0:28 / /cg rw,nosuid shared:3 - cgroup2 none rw\n";
  std::map<std::string, std::string> files;
  std::vector<std::string> reads;
};

TEST_F(CgroupMemoryTest, UnlimitedLeafStillUsesTighterAncestor) {
  files["/cg/memory.max"] = std::to_string(2048 * kMiB);
  files["/cg/tenant/memory.max"] = std::to_string(128 * kMiB);
  files["/cg/tenant/app/memory.max"] = "max\n";
  const auto detected = detect();
  EXPECT_EQ(128 * kMiB, detected);
  // Android's conservative default must use the effective container budget,
  // not the larger mount-root/host value (which would select 128 MiB).
  EXPECT_EQ(16 * kMiB, detail::MobileDefaultMemoryLimit(detected));
}

TEST_F(CgroupMemoryTest, BroaderMountRevealsParentAboveBindMount) {
  mounts =
      "12 1 0:28 /tenant/app /bound rw - cgroup2 none rw\n"
      "13 1 0:28 / /full rw - cgroup2 none rw\n";
  files["/bound/memory.max"] = std::to_string(512 * kMiB);
  files["/full/tenant/memory.max"] = std::to_string(128 * kMiB);
  EXPECT_EQ(128 * kMiB, detect());
}

TEST_F(CgroupMemoryTest, V1CombinedControllersHonorHierarchicalParents) {
  membership = "4:cpu,memory:/tenant/app\n";
  mounts = "12 1 0:28 / /dev/memcg rw - cgroup none rw,cpu,memory\n";
  files["/dev/memcg/tenant/app/memory.limit_in_bytes"] = "9223372036854771712";
  files["/dev/memcg/tenant/memory.use_hierarchy"] = "1\n";
  files["/dev/memcg/tenant/memory.limit_in_bytes"] = std::to_string(256 * kMiB);
  EXPECT_EQ(256 * kMiB, detect());
}

TEST_F(CgroupMemoryTest, HybridLayoutFindsV1MemoryInsteadOfOtherV2Controllers) {
  membership = "0::/unified/app\n4:memory:/tenant/app\n5:notmemory:/wrong\n";
  mounts +=
      "13 1 0:29 / /mem rw - cgroup cgroup rw,memory\n"
      "14 1 0:30 / /other rw - cgroup cgroup rw,notmemory\n";
  files["/mem/tenant/app/memory.limit_in_bytes"] = std::to_string(256 * kMiB);
  files["/other/wrong/memory.limit_in_bytes"] = "1";
  EXPECT_EQ(256 * kMiB, detect());
}

TEST_F(CgroupMemoryTest, MalformedMetadataCannotEscapeVisibleMount) {
  for (const auto *path :
       {"relative", "/../tenant", "/tenant/../app", "/tenant/./app"}) {
    membership = std::string("0::") + path + "\n";
    EXPECT_EQ(kPhysicalMemory, detect());
    EXPECT_TRUE(reads.empty());
  }
  membership = "0::/tenant/app\n";
  for (const auto *bad : {"malformed", "12 1 0:28 / /cg rw - cgroup2",
                          "12 1 0:28 / /cg\\000 rw - cgroup2 none rw",
                          "12 1 0:28 / /cg\\04x rw - cgroup2 none rw"}) {
    mounts = bad;
    EXPECT_EQ(kPhysicalMemory, detect());
    EXPECT_TRUE(reads.empty());
  }
}

}  // namespace

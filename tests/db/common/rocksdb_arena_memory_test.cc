// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory/arena.h>
#include <port/port.h>
#include <rocksdb/write_buffer_manager.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace zvec {
namespace {

TEST(RocksdbArenaMemoryTest, TracksMappedSlabAndReleasesBackingStorage) {
  constexpr size_t kBlockBytes = 64 * 1024;
  const size_t page_size = rocksdb::port::kPageSize;
  const size_t requested = kBlockBytes + 17;
  const size_t rounded = (requested + page_size - 1) / page_size * page_size;
  rocksdb::WriteBufferManager manager(8 * 1024 * 1024);
  for (size_t round = 0; round < 32; ++round) {
    SCOPED_TRACE(round);
    rocksdb::AllocTracker tracker(&manager);
    void *address = nullptr;
    {
      rocksdb::Arena arena(kBlockBytes, &tracker);
      address = arena.Allocate(requested);
      ASSERT_NE(nullptr, address);
      EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(address) % page_size);
      std::memset(address, 0x5a, requested);
      EXPECT_EQ(rounded + rocksdb::Arena::kInlineSize,
                arena.MemoryAllocatedBytes());
      EXPECT_EQ(arena.MemoryAllocatedBytes(), manager.memory_usage());
      // Mark immutable, as flush does; this must not refund physical storage.
      tracker.DoneAllocating();
      EXPECT_EQ(arena.MemoryAllocatedBytes(), manager.memory_usage());
    }
    // No allocations between destruction and the OS check: confirm the slab
    // is unmapped, not merely freed to malloc while its WBM charge disappears.
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(0u, ::VirtualQuery(address, &info, sizeof(info)));
    EXPECT_EQ(MEM_FREE, info.State);
#elif defined(__APPLE__)
    // Darwin mincore can succeed for a hole in the address map, so its return
    // code alone does not prove that a mapping still exists. Ask for the next
    // VM region and ensure it no longer covers the released address.
    mach_vm_address_t next = reinterpret_cast<uintptr_t>(address);
    mach_vm_size_t length = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(
        mach_task_self(), &next, &length, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) {
      ::mach_port_deallocate(mach_task_self(), object);
    }
    EXPECT_TRUE(result == KERN_INVALID_ADDRESS ||
                (result == KERN_SUCCESS &&
                 next > reinterpret_cast<uintptr_t>(address)));
#else
    unsigned char state = 0;
    errno = 0;
    EXPECT_EQ(-1, ::mincore(address, page_size, &state));
    EXPECT_EQ(ENOMEM, errno);
#endif
    EXPECT_EQ(0u, manager.memory_usage());
    EXPECT_EQ(0u, manager.mutable_memtable_memory_usage());
  }
}

}  // namespace
}  // namespace zvec

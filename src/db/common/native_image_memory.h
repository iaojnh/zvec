// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <zvec/ailego/buffer/memory_reservation.h>
#if defined(__linux__)
#include <link.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <unistd.h>
#endif
#endif

namespace zvec::detail {

#if defined(__linux__)
// Load segments are ordered by virtual address. Count only pages covered by the
// image, excluding gaps. Overlapping segments on the same page count once;
// p_memsz includes zero-initialized BSS.
inline bool ElfLoadImageBytes(const ElfW(Phdr) * headers, size_t count,
                              size_t page_size, size_t *bytes) noexcept {
  if (!headers || !bytes || !page_size) return false;
  constexpr uintptr_t kMax = std::numeric_limits<uintptr_t>::max();
  uintptr_t covered_end = 0;
  uintptr_t previous_start = 0;
  size_t total = 0;
  bool found = false;
  for (size_t i = 0; i < count; ++i) {
    const auto &header = headers[i];
    if (header.p_type != PT_LOAD || !header.p_memsz) continue;
    const uintptr_t start = header.p_vaddr;
    if (header.p_memsz > kMax - start || (found && start < previous_start))
      return false;
    const uintptr_t end = start + header.p_memsz;
    const uintptr_t padding = (page_size - end % page_size) % page_size;
    if (padding > kMax - end) return false;
    const uintptr_t page_start = start - start % page_size;
    const uintptr_t page_end = end + padding;
    const uintptr_t begin =
        found && covered_end > page_start ? covered_end : page_start;
    const uintptr_t extra = page_end > begin ? page_end - begin : 0;
    if (extra > std::numeric_limits<size_t>::max() - total) return false;
    total += extra;
    if (page_end > covered_end) covered_end = page_end;
    previous_start = start;
    found = true;
  }
  if (!found) return false;
  *bytes = total;
  return true;
}
#endif

#if defined(__APPLE__) && TARGET_OS_OSX
// Parse only the bounded header/load-command area. Use memcpy so malformed or
// unaligned test inputs cannot cause an unaligned structure access. This counts
// potentially resident image capacity, not an RSS sample or the file's size.
inline bool MachOLoadImageBytes(const void *image, size_t header_bytes,
                               size_t page_size, size_t *bytes) noexcept {
  if (!image || !bytes || !page_size ||
      header_bytes < sizeof(mach_header_64))
    return false;
  mach_header_64 header;
  std::memcpy(&header, image, sizeof(header));
  if (header.magic != MH_MAGIC_64 ||
      header.sizeofcmds > header_bytes - sizeof(header) ||
      header.ncmds > header.sizeofcmds / sizeof(load_command))
    return false;
  const auto *data = static_cast<const unsigned char *>(image);
  const size_t commands_end = sizeof(header) + header.sizeofcmds;
  constexpr uintptr_t kMax = std::numeric_limits<uintptr_t>::max();
  const auto visit = [&](auto &&visitor) noexcept {
    size_t offset = sizeof(header);
    for (uint32_t i = 0; i < header.ncmds; ++i) {
      if (commands_end - offset < sizeof(load_command)) return false;
      load_command command;
      std::memcpy(&command, data + offset, sizeof(command));
      if (command.cmdsize < sizeof(command) || command.cmdsize % 8 != 0 ||
          command.cmdsize > commands_end - offset)
        return false;
      if (command.cmd == LC_SEGMENT_64) {
        if (command.cmdsize < sizeof(segment_command_64)) return false;
        segment_command_64 segment;
        std::memcpy(&segment, data + offset, sizeof(segment));
        if (segment.nsects >
                (command.cmdsize - sizeof(segment)) / sizeof(section_64) ||
            segment.filesize > segment.vmsize)
          return false;
        // __PAGEZERO and inaccessible guard mappings have no resident backing.
        // Zero-fill data in an accessible segment still consumes capacity.
        if (segment.vmsize &&
            std::strncmp(segment.segname, SEG_PAGEZERO,
                         sizeof(segment.segname)) != 0 &&
            (segment.initprot &
             (VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE))) {
          if (segment.vmaddr > kMax || segment.vmsize > kMax - segment.vmaddr)
            return false;
          const uintptr_t start = static_cast<uintptr_t>(segment.vmaddr);
          const uintptr_t end = start + segment.vmsize;
          const uintptr_t padding = (page_size - end % page_size) % page_size;
          if (padding > kMax - end) return false;
          visitor(start - start % page_size, end + padding);
        }
      }
      offset += command.cmdsize;
    }
    return offset == commands_end;
  };

  // Select distinct starts in order without allocating temporary descriptors.
  // Load-command order need not match virtual-address order. Each pass consumes
  // at least one segment start, including overlapping and duplicate ranges.
  bool have_previous = false;
  uintptr_t previous_start = 0;
  uintptr_t covered_end = 0;
  size_t total = 0;
  while (true) {
    bool found = false;
    uintptr_t next_start = 0;
    uintptr_t next_end = 0;
    if (!visit([&](uintptr_t start, uintptr_t end) noexcept {
          if (have_previous && start <= previous_start) return;
          if (!found || start < next_start) {
            next_start = start;
            next_end = end;
            found = true;
          } else if (start == next_start) {
            next_end = std::max(next_end, end);
          }
        }))
      return false;
    if (!found) break;
    const uintptr_t begin = std::max(next_start, covered_end);
    const uintptr_t extra = next_end > begin ? next_end - begin : 0;
    if (extra > std::numeric_limits<size_t>::max() - total) return false;
    total += extra;
    covered_end = std::max(covered_end, next_end);
    previous_start = next_start;
    have_previous = true;
  }
  if (!have_previous) return false;
  *bytes = total;
  return true;
}
#endif

// anchor must belong to the translation unit containing the product
// implementation, not an inline function supplied by the caller. A statically
// linked main executable mixes product and caller code, so exclude the entire
// executable. Keep the existing policy on other platforms. This does not read
// RSS or touch pages to warm them up.
inline size_t NativeImageBytesForAddress(const void *anchor) {
#if defined(__linux__)
  const long native_page_size = sysconf(_SC_PAGESIZE);
  if (!anchor || native_page_size <= 0) throw std::bad_alloc();
  struct Search {
    uintptr_t address;
    size_t page_size;
    size_t bytes{0};
    bool found{false};
    bool valid{false};
  } search{reinterpret_cast<uintptr_t>(anchor),
           static_cast<size_t>(native_page_size)};
  dl_iterate_phdr(
      [](dl_phdr_info *info, size_t, void *opaque) noexcept {
        auto &result = *static_cast<Search *>(opaque);
        if (result.address < info->dlpi_addr) return 0;
        const uintptr_t relative = result.address - info->dlpi_addr;
        for (size_t i = 0; i < info->dlpi_phnum; ++i) {
          const auto &header = info->dlpi_phdr[i];
          if (header.p_type != PT_LOAD || relative < header.p_vaddr ||
              relative - header.p_vaddr >= header.p_memsz)
            continue;
          result.found = true;
          if (!info->dlpi_name || !*info->dlpi_name) {
            result.valid = true;
          } else {
            result.valid = ElfLoadImageBytes(info->dlpi_phdr, info->dlpi_phnum,
                                             result.page_size, &result.bytes);
          }
          return 1;
        }
        return 0;
      },
      &search);
  if (!search.found || !search.valid) throw std::bad_alloc();
  return search.bytes;
#elif defined(__APPLE__) && TARGET_OS_OSX
  const long native_page_size = sysconf(_SC_PAGESIZE);
  Dl_info owner{};
  if (!anchor || native_page_size <= 0 || !dladdr(anchor, &owner) ||
      !owner.dli_fbase)
    throw std::bad_alloc();
  mach_header_64 header;
  std::memcpy(&header, owner.dli_fbase, sizeof(header));
  if (header.magic != MH_MAGIC_64) throw std::bad_alloc();
  if (header.filetype == MH_EXECUTE) return 0;
  const size_t command_bytes = header.sizeofcmds;
  if ((header.filetype != MH_DYLIB && header.filetype != MH_BUNDLE) ||
      command_bytes > std::numeric_limits<size_t>::max() - sizeof(header))
    throw std::bad_alloc();
  size_t bytes = 0;
  if (!MachOLoadImageBytes(owner.dli_fbase, sizeof(header) + command_bytes,
                          static_cast<size_t>(native_page_size), &bytes))
    throw std::bad_alloc();
  return bytes;
#else
  (void)anchor;
  return 0;
#endif
}

inline ailego::MemoryReservation &NativeImageReservation() {
  // The image remains live through global resource destruction. As with
  // process-wide thread reservations, do not refund it during collection.close
  // or static destruction: the pool might already be destroyed, and the
  // capacity must not be reused.
  static auto *reservation = new ailego::MemoryReservation;
  return *reservation;
}

inline void ReserveNativeImage(const void *anchor) {
  const size_t bytes = NativeImageBytesForAddress(anchor);
  if (!bytes) return;
  auto &reservation = NativeImageReservation();
  if (bytes > reservation.bytes()) reservation.add(bytes - reservation.bytes());
}

}  // namespace zvec::detail

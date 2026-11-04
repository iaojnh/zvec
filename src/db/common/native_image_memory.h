// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <zvec/ailego/buffer/memory_reservation.h>
#if defined(__linux__)
#include <link.h>
#include <unistd.h>
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

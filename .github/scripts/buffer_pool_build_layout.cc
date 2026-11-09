// Compile against the exact measured revision; never link this helper to zvec.
#include <cstddef>
#include <iostream>
#include <zvec/ailego/buffer/block_eviction_queue.h>
namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static void print() {
    static_assert(sizeof(size_t) == 8, "64-bit diagnostic ABI only");
    std::condition_variable condition;
    const size_t native_offset = reinterpret_cast<uintptr_t>(condition.native_handle()) -
                                 reinterpret_cast<uintptr_t>(&condition);
    std::cout << "{\"pool_sizeof\":" << sizeof(MemoryLimitPool)
              << ",\"capacity_cv_offset\":" << offsetof(MemoryLimitPool, capacity_cv_) + native_offset
              << ",\"fields\":{";
#define FIELD(label, member) \
    std::cout << "\"" label "\":" << offsetof(MemoryLimitPool, member)
    FIELD("pool_size", pool_size_); std::cout << ',';
    FIELD("used", used_size_); std::cout << ',';
    FIELD("committed", committed_size_); std::cout << ',';
    FIELD("external_used", external_used_size_); std::cout << ',';
    FIELD("metadata_used", metadata_used_size_); std::cout << ',';
    FIELD("workspace_waiting", workspace_waiting_size_); std::cout << ',';
    FIELD("capacity_waits", capacity_waits_); std::cout << ',';
    FIELD("capacity_wait_timeouts", capacity_wait_timeouts_); std::cout << ',';
    FIELD("high_watermark_hits", high_watermark_hits_); std::cout << ',';
    FIELD("bg_evict_rounds", bg_evict_rounds_); std::cout << ',';
    FIELD("bg_evicted_buffers", bg_evicted_buffers_); std::cout << ',';
    FIELD("bg_no_progress_sleeps", bg_no_progress_sleeps_); std::cout << ',';
    FIELD("slab_reclaimed_pages", slab_reclaimed_pages_); std::cout << ',';
    FIELD("retained_block_reuses", retained_block_reuses_);
#undef FIELD
    std::cout << "}}\n";
  }
};
}
int main() { zvec::ailego::MemoryLimitPoolTestPeer::print(); }

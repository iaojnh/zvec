// Diagnostic-only observer. Product sources and mapped product code stay intact.
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <new>
#include <pthread.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace {
constexpr size_t kFields = 16;
constexpr uint64_t kMagic = 0x5a56454342505231ULL;
struct Shared {
  uint64_t magic{kMagic};
  std::atomic<uint64_t> sequence{0}, sample_ns{0};
  std::atomic<uint64_t> calls{0}, completed{0}, wait_ns{0}, timeouts{0};
  std::atomic<uint64_t> active{0}, max_wait_ns{0};
  std::atomic<uint64_t> fields[kFields]{};
};
static_assert(sizeof(Shared) <= 4096, "bounded diagnostic allocation");
std::atomic<Shared *> shared{nullptr};
std::atomic<uintptr_t> capacity_cv{0};
std::atomic<bool> running{false};
pthread_t sampler;
pthread_mutex_t sample_mutex = PTHREAD_MUTEX_INITIALIZER;
uintptr_t pool_address = 0;
size_t offsets[kFields]{};
size_t field_count = 0;
thread_local bool inside_wait = false;
using TimedWait = int (*)(pthread_cond_t *, pthread_mutex_t *, const timespec *);
using ClockWait = int (*)(pthread_cond_t *, pthread_mutex_t *, clockid_t, const timespec *);
TimedWait original_timed = nullptr;
ClockWait original_clock = nullptr;
pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
void resolve_originals() {
  original_timed = reinterpret_cast<TimedWait>(dlsym(RTLD_NEXT, "pthread_cond_timedwait"));
  original_clock = reinterpret_cast<ClockWait>(dlsym(RTLD_NEXT, "pthread_cond_clockwait"));
}

uint64_t now_ns() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
void take_sample() {
  auto *p = shared.load(std::memory_order_acquire);
  if (!p) return;
  pthread_mutex_lock(&sample_mutex);
  p->sequence.fetch_add(1, std::memory_order_acq_rel);
  for (size_t i = 0; i < field_count; ++i) {
    const auto *value = reinterpret_cast<const std::atomic<uint64_t> *>(
        pool_address + offsets[i]);
    p->fields[i].store(value->load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
  }
  p->sample_ns.store(now_ns(), std::memory_order_relaxed);
  p->sequence.fetch_add(1, std::memory_order_release);
  pthread_mutex_unlock(&sample_mutex);
}
void *sample_loop(void *) {
  while (running.load(std::memory_order_acquire)) {
    take_sample();
    timespec delay{0, 100000000};
    nanosleep(&delay, nullptr);
  }
  take_sample();
  return nullptr;
}
template <class F>
int timed_wait(pthread_cond_t *condition, F call) {
  auto *p = shared.load(std::memory_order_acquire);
  if (!p || inside_wait ||
      reinterpret_cast<uintptr_t>(condition) !=
          capacity_cv.load(std::memory_order_relaxed)) return call();
  inside_wait = true;
  p->calls.fetch_add(1, std::memory_order_relaxed);
  p->active.fetch_add(1, std::memory_order_relaxed);
  const auto start = now_ns();
  const int result = call();
  const int saved_errno = errno;
  const auto duration = now_ns() - start;
  p->wait_ns.fetch_add(duration, std::memory_order_relaxed);
  p->completed.fetch_add(1, std::memory_order_relaxed);
  if (result == ETIMEDOUT) p->timeouts.fetch_add(1, std::memory_order_relaxed);
  auto maximum = p->max_wait_ns.load(std::memory_order_relaxed);
  while (maximum < duration && !p->max_wait_ns.compare_exchange_weak(
             maximum, duration, std::memory_order_relaxed)) {}
  p->active.fetch_sub(1, std::memory_order_relaxed);
  inside_wait = false;
  errno = saved_errno;
  return result;
}
}

extern "C" int pthread_cond_timedwait(pthread_cond_t *condition,
                                      pthread_mutex_t *mutex,
                                      const timespec *deadline) {
  pthread_once(&resolve_once, resolve_originals);
  if (!original_timed) return ENOSYS;
  return timed_wait(condition, [&] { return original_timed(condition, mutex, deadline); });
}
extern "C" int pthread_cond_clockwait(pthread_cond_t *condition,
                                      pthread_mutex_t *mutex, clockid_t clock,
                                      const timespec *deadline) {
  pthread_once(&resolve_once, resolve_originals);
  if (!original_clock) return ENOSYS;
  return timed_wait(condition, [&] { return original_clock(condition, mutex, clock, deadline); });
}

extern "C" int zvec_build_probe_start(uintptr_t pool, size_t cv_offset,
                                      const size_t *fields, size_t count,
                                      const char *path) {
  if (!pool || !fields || count > kFields || shared.load()) return EINVAL;
  const int fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0) return errno;
  if (ftruncate(fd, 4096)) { const int e = errno; close(fd); return e; }
  void *mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED) return errno;
  pool_address = pool;
  field_count = count;
  std::memcpy(offsets, fields, count * sizeof(size_t));
  auto *p = new (mapping) Shared();
  shared.store(p, std::memory_order_release);
  capacity_cv.store(pool + cv_offset, std::memory_order_release);
  running.store(true, std::memory_order_release);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 128 * 1024);
  const int result = pthread_create(&sampler, &attr, sample_loop, nullptr);
  pthread_attr_destroy(&attr);
  if (result) { running.store(false); capacity_cv.store(0); return result; }
  take_sample();
  return 0;
}
extern "C" void zvec_build_probe_stop() {
  capacity_cv.store(0, std::memory_order_release);
  if (running.exchange(false, std::memory_order_acq_rel)) pthread_join(sampler, nullptr);
  take_sample();
  // Keep this single 4 KiB mapping until exit: an already-entered waiter can
  // finish after stop. It must never write through an unmapped observer pointer.
}

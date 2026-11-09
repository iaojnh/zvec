// Validate matching CV selection, timeout semantics and concurrent accounting.
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
struct Fixture {
  pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
  std::atomic<uint64_t> value{17};
};
int main(int argc, char **argv) {
  assert(argc == 2);
  Fixture fixture;
  auto start = reinterpret_cast<int (*)(uintptr_t, size_t, const size_t *, size_t, const char *)>(dlsym(RTLD_DEFAULT, "zvec_build_probe_start"));
  auto stop = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "zvec_build_probe_stop"));
  assert(start && stop);
  size_t offset = offsetof(Fixture, value);
  assert(!start(reinterpret_cast<uintptr_t>(&fixture), offsetof(Fixture, condition), &offset, 1, argv[1]));
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t unrelated = PTHREAD_COND_INITIALIZER;
  auto wait = [&](pthread_cond_t *condition) {
    timespec deadline{};
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 20000000;
    if (deadline.tv_nsec >= 1000000000) { ++deadline.tv_sec; deadline.tv_nsec -= 1000000000; }
    pthread_mutex_lock(&mutex);
    assert(pthread_cond_timedwait(condition, &mutex, &deadline) == ETIMEDOUT);
    pthread_mutex_unlock(&mutex);
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) threads.emplace_back([&] { wait(&fixture.condition); wait(&unrelated); });
  for (auto &thread : threads) thread.join();
  unsigned expected = 4;
#ifdef __GLIBC__
  timespec deadline{};
  clock_gettime(CLOCK_MONOTONIC, &deadline);
  deadline.tv_nsec += 10000000;
  if (deadline.tv_nsec >= 1000000000) { ++deadline.tv_sec; deadline.tv_nsec -= 1000000000; }
  pthread_mutex_lock(&mutex);
  assert(pthread_cond_clockwait(&fixture.condition, &mutex, CLOCK_MONOTONIC, &deadline) == ETIMEDOUT);
  pthread_mutex_unlock(&mutex);
  ++expected;
#endif
  stop();
  const int fd = open(argv[1], O_RDONLY);
  assert(fd >= 0);
  auto *values = static_cast<const uint64_t *>(mmap(nullptr, 4096, PROT_READ, MAP_SHARED, fd, 0));
  assert(values != MAP_FAILED && values[0] == 0x5a56454342505231ULL);
  assert(values[3] == expected && values[4] == expected && values[6] == expected);
  assert(values[5] >= 50000000 && values[7] == 0 && values[9] == 17);
  munmap(const_cast<uint64_t *>(values), 4096);
  close(fd);
  pthread_cond_destroy(&fixture.condition);
  pthread_cond_destroy(&unrelated);
  pthread_mutex_destroy(&mutex);
  return 0;
}

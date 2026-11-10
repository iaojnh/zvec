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

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>

#if defined(_WIN32)
#include <Windows.h>
#endif

using namespace zvec;
using namespace zvec::ailego;

namespace {
struct BlockingLogState {
  std::promise<void> entered;
  std::promise<void> finish;
  std::shared_future<void> released = finish.get_future().share();
  std::atomic<bool> destroyed{false};
};

class BlockingLogger : public Logger {
 public:
  explicit BlockingLogger(std::shared_ptr<BlockingLogState> state)
      : state_(std::move(state)) {}
  ~BlockingLogger() override {
    state_->destroyed.store(true);
  }
  int init(const Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  void log(int, const char *, int, const char *, va_list) override {
    auto state = state_;
    state->entered.set_value();
    state->released.wait();
  }

 private:
  std::shared_ptr<BlockingLogState> state_;
};
}  // namespace

TEST(Logger, UnregisterRetainsInFlightSinkUntilCallFinishes) {
  auto state = std::make_shared<BlockingLogState>();
  auto entered = state->entered.get_future();
  auto previous =
      LoggerBroker::Register(std::make_shared<BlockingLogger>(state));
  auto restore =
      ScopeGuard::Make([&] { LoggerBroker::Register(std::move(previous)); });
  std::thread worker([] {
    LoggerBroker::Log(Logger::LEVEL_FATAL, __FILE__, __LINE__, "in flight");
  });
  auto join = ScopeGuard::Make([&] {
    if (worker.joinable()) {
      state->finish.set_value();
      worker.join();
    }
  });
  ASSERT_EQ(std::future_status::ready,
            entered.wait_for(std::chrono::seconds(2)));
  LoggerBroker::Unregister();
  EXPECT_FALSE(state->destroyed.load());
  EXPECT_FALSE(LoggerBroker::IsLevelEnabled(Logger::LEVEL_FATAL));
  state->finish.set_value();
  worker.join();
  EXPECT_TRUE(state->destroyed.load());
}

static void DoLogging() {
  static int log_count = 0;
  LOG_INFO("DoLogging: %d", ++log_count);
}

static void DoErrLogging() {
  static int err_log_count = 0;
  LOG_ERROR("DoErrLogging: %d", ++err_log_count);
}

TEST(Logger, General) {
  ASSERT_TRUE(ailego::Factory<Logger>::Has("ConsoleLogger"));

  for (int i = 0; i < 10; ++i) {
    LoggerBroker::SetLevel(i);
    LOG_DEBUG("level: %d, %s", i, "LOG_DEBUG");
    LOG_INFO("level: %d, %s", i, "LOG_INFO");
    LOG_WARN("level: %d, %s", i, "LOG_WARN");
    LOG_ERROR("level: %d, %s", i, "LOG_ERROR");
    LOG_FATAL("level: %d, %s", i, "LOG_FATAL");
  }

  LoggerBroker::SetLevel(0);
  LOG_DEBUG("%s", std::string("LOG_DEBUG").c_str());
  LOG_INFO("%s", std::string("LOG_INFO").c_str());
  LOG_WARN("%s", std::string("LOG_WARN").c_str());
  LOG_ERROR("%s", std::string("LOG_ERROR").c_str());
  LOG_FATAL("%s", std::string("LOG_FATAL").c_str());

  ThreadPool pool;
  for (uint32_t i = 0; i < 20; ++i) {
    pool.enqueue(Closure::New(DoLogging));
  }
  for (uint32_t i = 0; i < 20; ++i) {
    pool.enqueue(Closure::New(DoErrLogging));
  }
  pool.wake_all();
  pool.wait_finish();

  LoggerBroker::Unregister();
  LOG_DEBUG("%s", "LOG_DEBUG");
  LOG_INFO("%s", "LOG_INFO");
  LOG_WARN("%s", "LOG_WARN");
  LOG_ERROR("%s", "LOG_ERROR");
  LOG_FATAL("%s", "LOG_FATAL");
}

#if defined(_WIN32)
namespace {

void LogToConsole(Logger &logger, int level, const char *format, ...) {
  va_list args;
  va_start(args, format);
  logger.log(level, __FILE__, __LINE__, format, args);
  va_end(args);
}

}  // namespace

TEST(Logger, FollowsRedirectedStandardHandles) {
  auto logger = ailego::Factory<Logger>::MakeShared("ConsoleLogger");
  ASSERT_NE(logger, nullptr);

  for (int level : {Logger::LEVEL_INFO, Logger::LEVEL_WARN}) {
    const DWORD device = level <= Logger::LEVEL_INFO ? STD_OUTPUT_HANDLE
                                                   : STD_ERROR_HANDLE;
    const HANDLE original = GetStdHandle(device);
    // Replace the process handle without touching this CRT's stdout/stderr,
    // as happens when an embedding host redirects through a different CRT.
    for (int redirect = 0; redirect < 2; ++redirect) {
      HANDLE reader = nullptr;
      HANDLE writer = nullptr;
      ASSERT_TRUE(CreatePipe(&reader, &writer, nullptr, 0));
      if (!SetStdHandle(device, writer)) {
        CloseHandle(writer);
        CloseHandle(reader);
        FAIL() << "Failed to redirect standard handle";
      }

      LogToConsole(*logger, level, "redirected-console-message-%d", redirect);

      const BOOL restored = SetStdHandle(device, original);
      CloseHandle(writer);
      char buffer[4096];
      DWORD size = 0;
      const BOOL received =
          ReadFile(reader, buffer, sizeof(buffer), &size, nullptr);
      CloseHandle(reader);
      ASSERT_TRUE(restored);
      ASSERT_TRUE(received);
      const std::string message(buffer, size);
      EXPECT_NE(message.find("redirected-console-message-" +
                             std::to_string(redirect)),
                std::string::npos);
    }
  }
}
#endif

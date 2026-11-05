// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once
#if defined(__APPLE__)
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace zvec::testing_util {
// CI-only handshake lets the simulator host sample the live app's public
// macOS disk-I/O counter; no private iOS APIs or app entitlements are required.
inline void AppleIOGate(const char *variable, const std::string &name,
                        const std::string &point) {
  const char *root = std::getenv(variable);
  if (root == nullptr || *root == '\0') return;
  const std::string directory(root);
  const auto temporary = directory + "/status.tmp";
  const auto status = directory + "/status.json";
  {
    std::ofstream out(temporary);
    out << "{\"case\":\"" << name << "\",\"point\":\"" << point
        << "\",\"pid\":" << ::getpid() << '}';
    out.close();
    if (!out) throw std::runtime_error("Cannot write Apple I/O gate");
  }
  if (::rename(temporary.c_str(), status.c_str()) != 0) {
    throw std::runtime_error("Cannot publish Apple I/O gate");
  }
  const auto ack = directory + "/" + name + "." + point + ".go";
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
  while (::access(ack.c_str(), F_OK) != 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("Apple I/O host gate timed out");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
}  // namespace zvec::testing_util
#endif

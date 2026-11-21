// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

namespace {
char image_zero_fill[4UL << 20];
}

extern "C" __declspec(dllexport) const void *ZvecNativeImageFixtureAnchor() {
  return image_zero_fill;
}

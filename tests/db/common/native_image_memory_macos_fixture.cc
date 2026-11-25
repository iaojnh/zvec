// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

// Keep untouched zero-fill capacity in the shared-library fixture. Looking up
// its address must resolve the image, not the executable.
namespace {
char image_zero_fill[32768];
}

extern "C" const void *ZvecNativeImageFixtureAnchor() {
  return image_zero_fill;
}

// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

// This hook is absent from ordinary builds. It must never select a synthetic
// reader in a production library or through a production environment variable.
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
#include <memory>
#include <string>
#include "diskann_file_reader.h"

namespace zvec::core::benchmark {
using ReaderFactory = std::shared_ptr<AlignedFileReader> (*)(const std::string &);
ReaderFactory &ControlledReaderFactory();
}  // namespace zvec::core::benchmark
#endif

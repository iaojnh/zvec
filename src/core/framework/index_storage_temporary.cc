// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_storage_temporary.h>

namespace zvec::core {
IndexStorage::Pointer IndexStorage::Temporary::CreateBackend() {
  return IndexFactory::CreateStorage("BufferStorage");
}
}  // namespace zvec::core

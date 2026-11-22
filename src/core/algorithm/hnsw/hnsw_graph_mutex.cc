// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "hnsw_graph_mutex.h"

namespace zvec::core {

thread_local const HnswGraphLock *HnswGraphLock::active_ = nullptr;

HnswGraphLock::HnswGraphLock(HnswGraphMutex *mutex, Mode mode) : mode_(mode) {
  // A callback already holding a graph lock must never wait for a writer.
  if (mode_ == Mode::kWrite && active()) {
    blocked_ = true;
    return;
  }
  if (!mutex) return;
  if (mode_ == Mode::kWrite) {
    mutex->lock();
  } else {
    // A filter or provider can read the index this thread already holds.
    for (auto *guard = active_; guard; guard = guard->previous_) {
      if (guard->mutex_ == mutex) return;
    }
    // Another index's callback may be waiting on a lock this thread holds.
    // Join readers without waiting, and report an active write to the caller.
    if (active_) {
      if (!mutex->try_lock_shared_for_nested_read()) {
        blocked_ = true;
        return;
      }
    } else {
      mutex->lock_shared();
    }
  }
  mutex_ = mutex;
  previous_ = active_;
  active_ = this;
}

HnswGraphLock::~HnswGraphLock() {
  if (!mutex_) return;
  active_ = previous_;
  if (mode_ == Mode::kWrite) {
    mutex_->unlock();
  } else {
    mutex_->unlock_shared();
  }
}

bool HnswGraphLock::active() {
  return active_ != nullptr;
}

bool HnswGraphLock::has_writer(const HnswGraphMutex *mutex) {
  for (auto *guard = active_; guard; guard = guard->previous_) {
    if (guard->mode_ == Mode::kWrite && (!mutex || guard->mutex_ == mutex)) {
      return true;
    }
  }
  return false;
}

}  // namespace zvec::core

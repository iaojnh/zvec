// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "db/common/managed_arrow.h"

#include <arrow/acero/options.h>
#include <arrow/io/api.h>
#include <arrow/util/thread_pool.h>
#include <mutex>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/parallel/worker_memory.h>
#include <zvec/db/config.h>

namespace zvec::detail {
namespace {

struct ArrowExecutors {
  ArrowExecutors(int cpu_workers, int io_workers)
      : stacks(static_cast<size_t>(cpu_workers) + io_workers) {
    auto cpu_pool = arrow::internal::ThreadPool::MakeEternal(cpu_workers);
    auto io_pool = arrow::internal::ThreadPool::MakeEternal(io_workers);
    if (!cpu_pool.ok() || !io_pool.ok()) throw std::bad_alloc();
    cpu = std::move(*cpu_pool);
    io = std::move(*io_pool);
  }
  ailego::WorkerMemoryLease stacks;
  std::shared_ptr<arrow::internal::ThreadPool> cpu;
  std::shared_ptr<arrow::internal::ThreadPool> io;
};

}  // namespace

arrow::Status ConfigureManagedDbRss() {
  static std::mutex mutex;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  std::lock_guard<std::mutex> lock(mutex);
  if (pool.process_rss_limit() == 0 && pool.capacity() != 0 &&
      !pool.configure_process_rss_limit(
          GlobalConfig::Instance().memory_limit_bytes())) {
    return arrow::Status::OutOfMemory("Cannot initialize DB RSS admission");
  }
  return arrow::Status::OK();
}

arrow::Result<arrow::acero::QueryOptions> ManagedArrowOptions(
    arrow::MemoryPool *pool, bool use_threads) try {
  ARROW_RETURN_NOT_OK(ConfigureManagedDbRss());
  // Arrow starts workers lazily. Admit their backing before publishing either
  // executor and retain the lease for the lifetime of the process-wide pools.
  static auto *executors = new ArrowExecutors(
      arrow::GetCpuThreadPoolCapacity(), arrow::io::GetIOThreadPoolCapacity());
  arrow::acero::QueryOptions options;
  options.use_threads = use_threads;
  if (use_threads) options.custom_cpu_executor = executors->cpu.get();
  options.custom_io_executor = executors->io.get();
  options.memory_pool = pool;
  return options;
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("Managed Arrow executors exceed budget");
}

void BindManagedArrowIo(arrow::acero::Declaration &declaration,
                        arrow::internal::Executor *executor) {
  if (declaration.factory_name == "record_batch_reader_source") {
    const auto &source =
        static_cast<const arrow::acero::RecordBatchReaderSourceNodeOptions &>(
            *declaration.options);
    declaration.options =
        std::make_shared<arrow::acero::RecordBatchReaderSourceNodeOptions>(
            source.reader, executor);
  }
  for (auto &input : declaration.inputs) {
    if (auto *child = std::get_if<arrow::acero::Declaration>(&input)) {
      BindManagedArrowIo(*child, executor);
    }
  }
}

Status ArrowQueryError(const arrow::Status &status) {
  return status.IsOutOfMemory()
             ? Status(StatusCode::RESOURCE_EXHAUSTED, status.ToString())
             : Status::InternalError(status.ToString());
}

}  // namespace zvec::detail

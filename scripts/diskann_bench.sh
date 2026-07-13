#!/bin/bash
# =============================================================================
# DiskAnn 基准测试脚本 (build -> recall -> bench)
#
# 目的：一键跑通 DiskAnn 的「构建 / 召回 / 压测」，产出一份结构化、
#       可直接复制粘贴的报告（见脚本末尾 REPORT 块），用于生成类似
#       docs/diskann.md 的测试文档。
#
# 覆盖维度（与 docs/diskann.md 对齐）：
#   - 数据类型：FP32 / FP16（通过 ConverterName 切换）
#   - list_size 多档位
#   - bench 线程数多档位（1 / 2 / 4 ...）
#   - 指标：Recall@K、QPS、Avg/百分位延迟、峰值内存(RSS)、索引大小、构建耗时
#
# 用法：
#   1) 按需修改下方 "==== 配置区 ====" 的变量（或用环境变量覆盖）
#   2) chmod +x scripts/diskann_bench.sh
#   3) ./scripts/diskann_bench.sh 2>&1 | tee diskann_bench.log
#   4) 把最后的 "ZVEC DISKANN BENCH REPORT ... END REPORT" 整段发给我
#
# 说明：
#   - IO backend 由代码运行时自动探测（libaio 优先，缺失则回退 pread）；
#     当前实现只有 libaio / pread 两种，io_uring 不在当前代码里。
#   - 仅在 Linux x86_64 / arm64 / riscv64 上受支持（见 CMakeLists.txt）。
# =============================================================================
set -u

# ==== 配置区（可用环境变量覆盖，例：BUILD_DIR=build_release ./scripts/diskann_bench.sh）====
ROOT_DIR="${ROOT_DIR:-/home/admin/yinzefeng.yzf/zvec}"
BUILD_DIR="${BUILD_DIR:-build_release}"            # 二进制所在的 cmake 构建目录
BIN_DIR="${ROOT_DIR}/${BUILD_DIR}/bin"

# 使用的工具（DiskAnn 用旧格式 BuilderClass/SearcherClass -> *_original 系列）
BUILDER_BIN="${BUILDER_BIN:-${BIN_DIR}/local_builder_original}"
RECALL_BIN="${RECALL_BIN:-${BIN_DIR}/recall_original}"
BENCH_BIN="${BENCH_BIN:-${BIN_DIR}/bench_original}"

# ---- 数据集 ----
DATASET_NAME="${DATASET_NAME:-Cohere-1M}"
BUILD_FILE="${BUILD_FILE:-/home/admin/yinzefeng.yzf/test_luofeng/cohere_1m/cohere_train_vector_1m.new.txt.vecs}"
TRAIN_FILE="${TRAIN_FILE:-${BUILD_FILE}}"
QUERY_FILE="${QUERY_FILE:-/home/admin/yinzefeng.yzf/test_luofeng/cohere_1m/cohere_test_vector_1000.new.txt}"
GROUNDTRUTH_FILE="${GROUNDTRUTH_FILE:-/home/admin/yinzefeng.yzf/test_luofeng/cohere_1m/neighbors.txt}"
METRIC_NAME="${METRIC_NAME:-Cosine}"
QUERY_TYPE="${QUERY_TYPE:-float}"
QUERY_FIRST_SEP="${QUERY_FIRST_SEP:-;}"
QUERY_SECOND_SEP="${QUERY_SECOND_SEP:- }"

# ---- 待测的数据类型（converter）。留一个即可只测一种 ----
#   FP32 -> CosineFp32Converter ；FP16 -> CosineFp16Converter
#   若 metric 非 Cosine，请改成对应的 Converter 名字
CONVERTERS="${CONVERTERS:-CosineFp32Converter CosineFp16Converter}"

# ---- 构建参数 ----
BUILD_THREADS="${BUILD_THREADS:-8}"
MAX_DEGREE="${MAX_DEGREE:-32}"
BUILD_LIST_SIZE="${BUILD_LIST_SIZE:-50}"
MEMORY_LIMIT="${MEMORY_LIMIT:-100}"
MAX_PQ_CHUNK_NUM="${MAX_PQ_CHUNK_NUM:-384}"

# ---- 检索参数 ----
TOPK="${TOPK:-1,10,50}"                    # 逗号分隔
CACHE_NODE_NUM="${CACHE_NODE_NUM:-10000}"
SEARCH_LIST_SIZES="${SEARCH_LIST_SIZES:-100 300 500}"  # 空格分隔，多档位
BENCH_THREADS="${BENCH_THREADS:-1 2 4}"                 # 空格分隔，多档位
BENCH_SECS="${BENCH_SECS:-30}"
BENCH_ITER_COUNT="${BENCH_ITER_COUNT:-10000000}"
RECALL_THREAD_COUNT="${RECALL_THREAD_COUNT:-16}"
RECALL_SCORE_PRECISION="${RECALL_SCORE_PRECISION:-1e-4}"
CONTAINER_TYPE="${CONTAINER_TYPE:-FileReadStorage}"    # 或 MMapFileReadStorage(全内存)

# ---- 行为开关 ----
DO_BUILD="${DO_BUILD:-1}"       # 0 = 跳过构建，直接用已存在的索引
DO_RECALL="${DO_RECALL:-1}"
DO_BENCH="${DO_BENCH:-1}"

# ---- 工作目录 ----
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/diskann_bench_work}"
mkdir -p "${WORK_DIR}"
REPORT_FILE="${WORK_DIR}/report_$(date +%Y%m%d_%H%M%S).txt"

# =============================================================================
# 内部函数
# =============================================================================

# 找 /usr/bin/time，用于抓峰值内存；没有则退化为无内存统计
TIME_BIN=""
if [ -x /usr/bin/time ]; then TIME_BIN="/usr/bin/time -v"; fi

# converter -> 数据类型短名（FP32/FP16）
dtype_of() {
  case "$1" in
    *Fp16*|*FP16*|*fp16*) echo "FP16" ;;
    *Fp32*|*FP32*|*fp32*) echo "FP32" ;;
    *) echo "$1" ;;
  esac
}

index_path_of() {
  # 每种 converter 一个独立索引文件
  echo "${WORK_DIR}/diskann_$(dtype_of "$1").index"
}

human_size() {
  # 字节 -> 人类可读
  local b="$1"
  awk -v b="$b" 'BEGIN{
    split("B KB MB GB TB", u, " ");
    i=1; while (b>=1024 && i<5){b/=1024; i++}
    printf("%.2f%s", b, u[i]);
  }'
}

gen_build_yaml() {
  local converter="$1" index_path="$2" out="$3"
  cat > "$out" <<EOF
BuilderCommon:
  BuilderClass: DiskAnnBuilder
  BuildFile: ${BUILD_FILE}
  NeedTrain: true
  TrainFile: ${TRAIN_FILE}
  DumpPath: ${index_path}
  IndexPath: ${index_path}
  MetricName: ${METRIC_NAME}
  ConverterName: ${converter}
  ThreadCount: ${BUILD_THREADS}
  LogLevel: Info

BuilderParams:
  zvec.general.builder.thread_count: !!int ${BUILD_THREADS}
  zvec.diskann.builder.thread_count: !!int ${BUILD_THREADS}
  zvec.diskann.builder.max_degree: !!int ${MAX_DEGREE}
  zvec.diskann.builder.list_size: !!int ${BUILD_LIST_SIZE}
  zvec.diskann.builder.memory_limit: !!float ${MEMORY_LIMIT}
  zvec.diskann.builder.max_pq_chunk_num: !!int ${MAX_PQ_CHUNK_NUM}
EOF
}

gen_search_yaml() {
  local index_path="$1" list_size="$2" bench_threads="$3" out="$4"
  cat > "$out" <<EOF
SearcherCommon:
  SearcherClass: DiskAnnSearcher
  IndexPath: ${index_path}
  TopK: ${TOPK}
  QueryFile: ${QUERY_FILE}
  QueryType: ${QUERY_TYPE}
  QueryFirstSep: "${QUERY_FIRST_SEP}"
  QuerySecondSep: "${QUERY_SECOND_SEP}"
  GroundTruthFile: ${GROUNDTRUTH_FILE}
  RecallThreadCount: ${RECALL_THREAD_COUNT}
  RecallScorePrecision: ${RECALL_SCORE_PRECISION}
  BenchThreadCount: ${bench_threads}
  BenchSecs: ${BENCH_SECS}
  BenchIterCount: ${BENCH_ITER_COUNT}
  CompareById: true
  ContainerType: ${CONTAINER_TYPE}
  LogLevel: Info

SearcherParams:
  zvec.diskann.searcher.cache_node_num: ${CACHE_NODE_NUM}
  zvec.diskann.searcher.list_size: ${list_size}
EOF
}

# 从 build 日志里提取 io backend（libaio / pread）
detect_io_backend() {
  local log="$1"
  if grep -qiE "backend '?libaio'?|libaio.*loaded|async I/O enabled" "$log" 2>/dev/null; then
    echo "libaio"
  elif grep -qiE "synchronous pread|fall back to synchronous pread|no async I/O" "$log" 2>/dev/null; then
    echo "pread"
  else
    echo "unknown"
  fi
}

# =============================================================================
# 采集环境信息
# =============================================================================
ENV_HOST="$(hostname 2>/dev/null)"
ENV_OS="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-$(uname -s)}")"
ENV_KERNEL="$(uname -r)"
ENV_ARCH="$(uname -m)"
ENV_CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/.*: //')"
ENV_CPU_CORES="$(nproc 2>/dev/null)"
ENV_MEM_TOTAL="$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{printf "%.1fGiB", $2/1024/1024}')"
ENV_GIT="$(cd "$ROOT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"

# =============================================================================
# 主流程
# =============================================================================
{
echo "=============================================================="
echo "ZVEC DISKANN BENCH REPORT"
echo "generated_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
echo "[ENV]"
echo "host=${ENV_HOST}"
echo "os=${ENV_OS}"
echo "kernel=${ENV_KERNEL}"
echo "arch=${ENV_ARCH}"
echo "cpu_model=${ENV_CPU_MODEL}"
echo "cpu_cores=${ENV_CPU_CORES}"
echo "mem_total=${ENV_MEM_TOTAL}"
echo "git_commit=${ENV_GIT}"
echo "builder_bin=${BUILDER_BIN}"
echo "recall_bin=${RECALL_BIN}"
echo "bench_bin=${BENCH_BIN}"
echo "container_type=${CONTAINER_TYPE}"
echo ""
echo "[DATASET]"
echo "name=${DATASET_NAME}"
echo "build_file=${BUILD_FILE}"
echo "query_file=${QUERY_FILE}"
echo "groundtruth_file=${GROUNDTRUTH_FILE}"
echo "metric=${METRIC_NAME}"
echo "topk=${TOPK}"
echo ""
echo "[BUILD_PARAMS]"
echo "build_threads=${BUILD_THREADS}"
echo "max_degree=${MAX_DEGREE}"
echo "build_list_size=${BUILD_LIST_SIZE}"
echo "memory_limit=${MEMORY_LIMIT}"
echo "max_pq_chunk_num=${MAX_PQ_CHUNK_NUM}"
echo "cache_node_num=${CACHE_NODE_NUM}"
echo ""
} | tee "$REPORT_FILE"

# 基本检查
for b in "$BUILDER_BIN" "$RECALL_BIN" "$BENCH_BIN"; do
  if [ ! -x "$b" ]; then
    echo "WARN: binary not found or not executable: $b" | tee -a "$REPORT_FILE"
  fi
done

for converter in $CONVERTERS; do
  DT="$(dtype_of "$converter")"
  INDEX_PATH="$(index_path_of "$converter")"

  echo "==============================================================" | tee -a "$REPORT_FILE"
  echo "### DATA_TYPE=${DT} converter=${converter}" | tee -a "$REPORT_FILE"
  echo "==============================================================" | tee -a "$REPORT_FILE"

  # -------- 构建 --------
  IO_BACKEND="unknown"
  if [ "$DO_BUILD" = "1" ]; then
    BUILD_YAML="${WORK_DIR}/build_${DT}.yml"
    BUILD_LOG="${WORK_DIR}/build_${DT}.log"
    gen_build_yaml "$converter" "$INDEX_PATH" "$BUILD_YAML"
    echo "[BUILD] dtype=${DT} yaml=${BUILD_YAML}" | tee -a "$REPORT_FILE"
    BUILD_START=$(date +%s.%N)
    "$BUILDER_BIN" "$BUILD_YAML" > "$BUILD_LOG" 2>&1
    BUILD_RC=$?
    BUILD_END=$(date +%s.%N)
    BUILD_WALL=$(awk -v a="$BUILD_START" -v b="$BUILD_END" 'BEGIN{printf "%.3f", b-a}')
    IO_BACKEND="$(detect_io_backend "$BUILD_LOG")"
    echo "build_rc=${BUILD_RC}" | tee -a "$REPORT_FILE"
    echo "build_wall_sec=${BUILD_WALL}" | tee -a "$REPORT_FILE"
    echo "io_backend=${IO_BACKEND}" | tee -a "$REPORT_FILE"
    # 尝试从日志抓取 train/build/dump 分段耗时（如实现有打印）
    grep -iE "train|build|dump|index|elapsed|cost|time" "$BUILD_LOG" \
      | grep -iE "[0-9].*(s|ms|sec|second)" | sed 's/^/build_log> /' \
      | head -30 | tee -a "$REPORT_FILE"
  else
    echo "[BUILD] skipped (DO_BUILD=0), using existing index ${INDEX_PATH}" | tee -a "$REPORT_FILE"
  fi

  # 索引大小
  if [ -e "$INDEX_PATH" ]; then
    SZ_BYTES=$(du -sb "$INDEX_PATH" 2>/dev/null | awk '{print $1}')
    echo "index_size_bytes=${SZ_BYTES}" | tee -a "$REPORT_FILE"
    echo "index_size_human=$(human_size "${SZ_BYTES:-0}")" | tee -a "$REPORT_FILE"
  else
    echo "index_size_bytes=NA (index not found: ${INDEX_PATH})" | tee -a "$REPORT_FILE"
  fi
  echo "" | tee -a "$REPORT_FILE"

  # -------- 召回 (每个 list_size 一次) --------
  if [ "$DO_RECALL" = "1" ]; then
    for ls in $SEARCH_LIST_SIZES; do
      SEARCH_YAML="${WORK_DIR}/search_${DT}_ls${ls}.yml"
      RECALL_LOG="${WORK_DIR}/recall_${DT}_ls${ls}.log"
      gen_search_yaml "$INDEX_PATH" "$ls" "1" "$SEARCH_YAML"
      "$RECALL_BIN" "$SEARCH_YAML" > "$RECALL_LOG" 2>&1
      echo "[RECALL] dtype=${DT} list_size=${ls}" | tee -a "$REPORT_FILE"
      grep -E "^Recall@" "$RECALL_LOG" | sed 's/: /=/' | tee -a "$REPORT_FILE"
      echo "" | tee -a "$REPORT_FILE"
    done
  fi

  # -------- 压测 (list_size × threads) --------
  if [ "$DO_BENCH" = "1" ]; then
    for ls in $SEARCH_LIST_SIZES; do
      for th in $BENCH_THREADS; do
        SEARCH_YAML="${WORK_DIR}/search_${DT}_ls${ls}_t${th}.yml"
        BENCH_LOG="${WORK_DIR}/bench_${DT}_ls${ls}_t${th}.log"
        gen_search_yaml "$INDEX_PATH" "$ls" "$th" "$SEARCH_YAML"
        echo "[BENCH] dtype=${DT} list_size=${ls} threads=${th}" | tee -a "$REPORT_FILE"
        if [ -n "$TIME_BIN" ]; then
          $TIME_BIN "$BENCH_BIN" "$SEARCH_YAML" > "$BENCH_LOG" 2>&1
        else
          "$BENCH_BIN" "$SEARCH_YAML" > "$BENCH_LOG" 2>&1
        fi
        # QPS + 平均延迟
        grep -E "qps:" "$BENCH_LOG" | tail -1 \
          | sed -E 's/.*Avg latency: ([0-9.]+)ms qps: ([0-9.]+)/avg_latency_ms=\1\nqps=\2/' \
          | tee -a "$REPORT_FILE"
        # 百分位延迟
        grep -E "Percentile:" "$BENCH_LOG" \
          | sed -E 's/^([0-9]+) Percentile:\s*([0-9.]+) ms/p\1_ms=\2/' \
          | tee -a "$REPORT_FILE"
        # 峰值内存
        grep -E "Maximum resident set size" "$BENCH_LOG" \
          | sed -E 's/.*: ([0-9]+)/max_rss_kb=\1/' | tee -a "$REPORT_FILE"
        echo "" | tee -a "$REPORT_FILE"
      done
    done
  fi
done

{
echo "=============================================================="
echo "END REPORT"
echo "report_file=${REPORT_FILE}"
echo "work_dir=${WORK_DIR}"
echo "=============================================================="
} | tee -a "$REPORT_FILE"

echo ""
echo ">>> 完整报告已保存：${REPORT_FILE}"
echo ">>> 请把上面 'ZVEC DISKANN BENCH REPORT ... END REPORT' 整段发给我，用于生成测试文档。"

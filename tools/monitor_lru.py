#!/usr/bin/env python3
# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LRU Cache Effectiveness Monitor for zvec Benchmark
====================================================
通过监控 /proc/[pid]/ 下的内存和 I/O 统计，实时验证 VectorPageTable
的 LRU Cache 是否有效工作。

关键指标说明
-----------
  VmRSS        : 实际驻留内存，LRU 有效时应稳定在 pool_size 以内
  rchar        : 进程逻辑读字节数（含缓存命中）
  read_bytes   : 实际磁盘读字节数（仅缓存未命中时产生）
  cache_hit%   : 1 - Δread_bytes / Δrchar，越高说明 LRU 越有效
  rss_util%    : VmRSS / pool_size_mb × 100，内存池利用率

LRU 有效性判断
--------------
  - cache_hit% 持续 > 70%               → LRU 命中效果良好
  - rss_util% 稳定（不持续增长）         → LRU 淘汰在守住内存上限
  - rss_util% 稳定在 80–100% 之间       → pool 被充分利用，热块在 cache 中
  - 若 rss_util% 不断超过 100%          → LRU 可能未能有效淘汰，存在问题

Usage
-----
  # 直接指定 PID
  python3 monitor_lru.py --pid <PID> [options]

  # 自动查找 bench 进程
  python3 monitor_lru.py --name bench [options]

  # 完整示例（pool=3GB，每秒采样，持续60秒，保存CSV）
  python3 monitor_lru.py --name bench --pool-size 3072 --interval 1 \\
      --duration 60 --output lru_report.csv
"""

import argparse
import csv
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime

# ─── ANSI 颜色 ────────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ─── /proc 读取工具 ────────────────────────────────────────────────────────────

def read_proc_status(pid: int) -> dict:
    """解析 /proc/[pid]/status，返回 key→value 字典（数值已去单位）。"""
    result = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]  # 去掉 kB 等单位
                    result[key] = val
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return result


def read_proc_io(pid: int) -> dict:
    """解析 /proc/[pid]/io，返回各 I/O 计数器。"""
    result = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    result[parts[0].strip()] = int(parts[1].strip())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return result


def read_proc_stat(pid: int) -> list:
    """返回 /proc/[pid]/stat 的字段列表（用于计算 CPU）。"""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def read_system_jiffies() -> int:
    """返回系统总 jiffies (user+nice+system+idle+iowait+...)"""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
            fields = line.split()[1:]
            return sum(int(x) for x in fields)
    except Exception:
        return 0


def find_pid_by_name(name: str) -> list:
    """在 /proc 下查找匹配进程名的所有 PID。"""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/comm") as f:
                comm = f.read().strip()
            if name in comm:
                pids.append(int(entry))
        except Exception:
            pass
    return pids


def pid_alive(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}/status")


def get_descendant_pids(root_pid: int) -> list[int]:
    """BFS 遍历进程树，返回 root_pid 及其所有后代的 PID 列表。"""
    # 构建 ppid → [children] 映射（只扫一次 /proc，效率更高）
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status") as f:
                ppid = None
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
            if ppid is not None:
                children.setdefault(ppid, []).append(int(entry))
        except Exception:
            pass

    result = []
    queue  = [root_pid]
    while queue:
        pid = queue.pop()
        result.append(pid)
        queue.extend(children.get(pid, []))
    return result


# ─── 采样与指标计算 ────────────────────────────────────────────────────────────

class Sample:
    __slots__ = ("ts", "rss_kb", "vmsize_kb", "rchar", "read_bytes", "wchar",
                 "syscr", "utime", "stime", "sys_jiffies")

    def __init__(self, ts, rss_kb, vmsize_kb, rchar, read_bytes, wchar, syscr,
                 utime, stime, sys_jiffies):
        self.ts          = ts
        self.rss_kb      = rss_kb
        self.vmsize_kb   = vmsize_kb
        self.rchar       = rchar
        self.read_bytes  = read_bytes
        self.wchar       = wchar
        self.syscr       = syscr
        self.utime       = utime
        self.stime       = stime
        self.sys_jiffies = sys_jiffies


def take_sample(pid: int) -> Sample | None:
    """采样单个进程。"""
    ts = time.time()
    status = read_proc_status(pid)
    io     = read_proc_io(pid)
    stat   = read_proc_stat(pid)
    sysjif = read_system_jiffies()

    if not status or not io or len(stat) < 15:
        return None

    rss_kb    = int(status.get("VmRSS",  0))
    vmsize_kb = int(status.get("VmSize", 0))
    utime  = int(stat[13])
    stime  = int(stat[14])

    return Sample(
        ts          = ts,
        rss_kb      = rss_kb,
        vmsize_kb   = vmsize_kb,
        rchar       = io.get("rchar", 0),
        read_bytes  = io.get("read_bytes", 0),
        wchar       = io.get("wchar", 0),
        syscr       = io.get("syscr", 0),
        utime       = utime,
        stime       = stime,
        sys_jiffies = sysjif,
    )


def take_tree_sample(root_pid: int) -> tuple[Sample | None, int]:
    """聚合 root_pid 整棵进程树的采样，返回 (Sample, 活跃进程数)。

    RSS / VmSize 直接相加；IO 计数器相加；CPU jiffies 相加。
    sys_jiffies 只取一次（全局系统时钟，不应累加）。
    """
    pids   = get_descendant_pids(root_pid)
    ts     = time.time()
    sysjif = read_system_jiffies()

    agg = dict(rss_kb=0, vmsize_kb=0, rchar=0, read_bytes=0,
               wchar=0, syscr=0, utime=0, stime=0)
    alive = 0

    for pid in pids:
        status = read_proc_status(pid)
        io     = read_proc_io(pid)
        stat   = read_proc_stat(pid)
        if not status or not io or len(stat) < 15:
            continue
        alive += 1
        agg["rss_kb"]    += int(status.get("VmRSS",  0))
        agg["vmsize_kb"] += int(status.get("VmSize", 0))
        agg["rchar"]      += io.get("rchar",      0)
        agg["read_bytes"] += io.get("read_bytes", 0)
        agg["wchar"]      += io.get("wchar",      0)
        agg["syscr"]      += io.get("syscr",      0)
        agg["utime"]      += int(stat[13])
        agg["stime"]      += int(stat[14])

    if alive == 0:
        return None, 0

    sample = Sample(
        ts          = ts,
        rss_kb      = agg["rss_kb"],
        vmsize_kb   = agg["vmsize_kb"],
        rchar       = agg["rchar"],
        read_bytes  = agg["read_bytes"],
        wchar       = agg["wchar"],
        syscr       = agg["syscr"],
        utime       = agg["utime"],
        stime       = agg["stime"],
        sys_jiffies = sysjif,
    )
    return sample, alive


def compute_metrics(prev: Sample, curr: Sample,
                    pool_size_kb: int,
                    limit_size_kb: int | None = None) -> dict:
    dt = max(curr.ts - prev.ts, 1e-6)

    d_rchar      = max(curr.rchar       - prev.rchar,      0)
    d_read_bytes = max(curr.read_bytes  - prev.read_bytes, 0)
    d_syscr      = max(curr.syscr       - prev.syscr,      0)

    # 缓存命中率：逻辑读中不走磁盘的比例
    cache_hit_pct = (1.0 - d_read_bytes / d_rchar * 1.0) * 100.0 \
        if d_rchar > 0 else None

    # 磁盘读速率 (MB/s)
    disk_read_mbps = d_read_bytes / 1024 / 1024 / dt

    # 逻辑读速率 (MB/s)
    logical_read_mbps = d_rchar / 1024 / 1024 / dt

    # Pool% ：RSS / pool_only（反映 LRU 充填程度）
    rss_util_pct = curr.rss_kb / pool_size_kb * 100.0 if pool_size_kb > 0 else None
    pool_size_mb_eff = pool_size_kb / 1024.0

    # WARN 检查：RSS / limit（pool + overhead）
    _limit_kb = limit_size_kb if limit_size_kb else pool_size_kb
    rss_over_limit_pct = curr.rss_kb / _limit_kb * 100.0 if _limit_kb > 0 else None
    limit_size_mb_eff  = _limit_kb / 1024.0

    # CPU
    d_utime  = max(curr.utime  - prev.utime,       0)
    d_stime  = max(curr.stime  - prev.stime,       0)
    d_sysjif = max(curr.sys_jiffies - prev.sys_jiffies, 1)
    cpu_pct  = (d_utime + d_stime) / d_sysjif * 100.0

    return {
        "ts":                  curr.ts,
        "rss_mb":              curr.rss_kb    / 1024,
        "vmsize_mb":           curr.vmsize_kb / 1024,
        "rss_util_pct":        rss_util_pct,        # RSS / pool
        "rss_over_limit_pct": rss_over_limit_pct,  # RSS / (pool+overhead)
        "limit_size_mb_eff":  limit_size_mb_eff,
        "cache_hit_pct":       cache_hit_pct,
        "disk_read_mbps":      disk_read_mbps,
        "logical_read_mbps":   logical_read_mbps,
        "syscr_per_sec":       d_syscr / dt,
        "cpu_pct":             cpu_pct,
        "d_read_bytes":        d_read_bytes,
        "d_rchar":             d_rchar,
        "proc_count":          0,
        "pool_size_mb_eff":    pool_size_mb_eff,
        "expected_procs":      0,
    }


# ─── 实时输出 ──────────────────────────────────────────────────────────────────

HEADER = (
    f"{'Time':>8}  {'Procs':>5}  {'RSS(MB)':>8}  {'VmSize(MB)':>10}  {'RSS/Virt%':>9}  {'Pool%':>6}  "
    f"{'Hit%':>7}  {'DiskRd MB/s':>12}  {'LogRd MB/s':>11}  {'SysRd/s':>8}  "
    f"{'CPU%':>6}  {'Status'}"
)
DIVIDER = "─" * len(HEADER)


def lru_status(metrics: dict) -> tuple[str, str]:
    """根据关键指标判断 LRU 健康状态，返回 (颜色, 文字)。"""
    hit    = metrics["cache_hit_pct"]
    # WARN 检查用 RSS/(pool+overhead)，不用纯 pool 利用率
    util   = metrics.get("rss_over_limit_pct") or metrics["rss_util_pct"]
    procs  = metrics.get("proc_count", 1)
    exp    = metrics.get("expected_procs", 0)

    # 进程还未全部报到（启动期或风冷期），一律不报 WARN
    if exp > 0 and procs < exp:
        if hit is None:
            return CYAN,   f"RAMP   starting up ({procs}/{exp})"
        if hit >= 80:
            return GREEN,  f"RAMP   starting up ({procs}/{exp}), hit={hit:.0f}%"
        return YELLOW,     f"RAMP   starting up ({procs}/{exp}), hit={hit:.0f}%"

    if hit is None:
        return CYAN, "IDLE   (no read activity)"

    if util is not None and util > 110:
        return RED,    f"WARN   RSS exceeds limit ({util:.0f}%)"
    if hit >= 80:
        return GREEN,  "GOOD   LRU effective"
    if hit >= 50:
        return YELLOW, "OK     moderate cache hit"
    return RED,        "MISS   low cache hit – check LRU"


def format_row(metrics: dict, t0: float) -> str:
    elapsed = metrics["ts"] - t0
    hit_str = f"{metrics['cache_hit_pct']:6.1f}%" \
        if metrics["cache_hit_pct"] is not None else "   N/A%"
    util_str = f"{metrics['rss_util_pct']:5.1f}%" \
        if metrics["rss_util_pct"] is not None else "  N/A%"
    vmsize_mb = metrics["vmsize_mb"]
    rss_mb    = metrics["rss_mb"]
    rss_virt_pct = rss_mb / vmsize_mb * 100.0 if vmsize_mb > 0 else 0.0
    procs = metrics.get("proc_count", 1)
    color, status = lru_status(metrics)
    return (
        f"{elapsed:>7.1f}s  "
        f"{procs:>5}  "
        f"{rss_mb:>8.1f}  "
        f"{vmsize_mb:>10.1f}  "
        f"{rss_virt_pct:>8.1f}%  "
        f"{util_str:>6}  "
        f"{hit_str:>7}  "
        f"{metrics['disk_read_mbps']:>12.2f}  "
        f"{metrics['logical_read_mbps']:>11.2f}  "
        f"{metrics['syscr_per_sec']:>8.0f}  "
        f"{metrics['cpu_pct']:>6.1f}  "
        f"{color}{status}{RESET}"
    )


# ─── 最终报告 ──────────────────────────────────────────────────────────────────

def print_report(all_metrics: list,
                 pool_size_mb: float | None,
                 pool_per_proc_mb: float | None = None):
    if not all_metrics:
        print("无采样数据，无法生成报告。")
        return

    import math

    valid_hit  = [m["cache_hit_pct"]  for m in all_metrics if m["cache_hit_pct"]  is not None]
    valid_util = [m["rss_util_pct"]   for m in all_metrics if m["rss_util_pct"]   is not None]
    disk_reads = [m["disk_read_mbps"] for m in all_metrics]
    log_reads  = [m["logical_read_mbps"] for m in all_metrics]

    avg_hit   = sum(valid_hit)  / len(valid_hit)  if valid_hit  else 0
    avg_util  = sum(valid_util) / len(valid_util) if valid_util else 0
    peak_rss  = max(m["rss_mb"] for m in all_metrics)
    avg_disk  = sum(disk_reads) / len(disk_reads) if disk_reads else 0
    avg_log   = sum(log_reads)  / len(log_reads)  if log_reads  else 0

    rss_vals = [m["rss_mb"] for m in all_metrics]
    rss_mean = sum(rss_vals) / len(rss_vals)
    rss_std  = math.sqrt(sum((x - rss_mean)**2 for x in rss_vals) / len(rss_vals))

    peak_vmsize   = max(m["vmsize_mb"]  for m in all_metrics)
    peak_procs    = max(m.get("proc_count", 1) for m in all_metrics)
    peak_rss_virt = peak_rss / peak_vmsize * 100.0 if peak_vmsize > 0 else 0.0

    # 报告使用的 pool 上限：per_proc 模式取峰值时刻的有效值
    peak_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["rss_mb"])
    if pool_per_proc_mb is not None:
        peak_pool_mb  = float(all_metrics[peak_idx]["pool_size_mb_eff"])   # 纯 LRU pool
        peak_limit_mb = float(all_metrics[peak_idx]["limit_size_mb_eff"])  # pool + overhead
        pool_desc = (f"{pool_per_proc_mb:.0f} MB × {peak_procs} procs"
                     f" = {peak_pool_mb:.0f} MB pool"
                     + (f" + {peak_limit_mb - peak_pool_mb:.0f} MB overhead"
                        f" = {peak_limit_mb:.0f} MB limit"
                        if peak_limit_mb > peak_pool_mb else "")
                     + "（峰值时刻）")
    else:
        peak_pool_mb  = pool_size_mb
        peak_limit_mb = float(all_metrics[peak_idx]["limit_size_mb_eff"])
        pool_desc = (f"{pool_size_mb:.0f} MB pool"
                     + (f" + {peak_limit_mb - peak_pool_mb:.0f} MB overhead"
                        f" = {peak_limit_mb:.0f} MB limit"
                        if peak_limit_mb > peak_pool_mb else "")
                     + "（固定总量）")

    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  LRU Cache 有效性报告{RESET}")
    print(f"{'='*60}")
    print(f"  采样点数          : {len(all_metrics)}")
    print(f"  内存池配置        : {pool_desc}")
    print(f"  峰值进程数        : {peak_procs}")
    print(f"  峰值 VmSize       : {peak_vmsize:.1f} MB  (虚拟内存，含未 fault 页)")
    print(f"  峰值 RSS          : {peak_rss:.1f} MB  "
          f"({peak_rss/peak_pool_mb*100:.1f}% pool"
          + (f" / {peak_rss/peak_limit_mb*100:.1f}% limit"
             if peak_limit_mb > peak_pool_mb else "")
          + f" / {peak_rss_virt:.1f}% virt)")
    print(f"  RSS 均值 ± 标准差 : {rss_mean:.1f} ± {rss_std:.1f} MB")
    print(f"  平均池利用率      : {avg_util:.1f}%")
    print(f"  平均缓存命中率    : {avg_hit:.1f}%")
    print(f"  平均磁盘读速率    : {avg_disk:.2f} MB/s")
    print(f"  平均逻辑读速率    : {avg_log:.2f} MB/s")
    print()

    issues = []
    verdicts = []

    if avg_hit >= 80:
        verdicts.append(f"{GREEN}[PASS] 缓存命中率 {avg_hit:.1f}% ≥ 80%，LRU 命中效果良好{RESET}")
    elif avg_hit >= 50:
        verdicts.append(f"{YELLOW}[WARN] 缓存命中率 {avg_hit:.1f}%，中等，可适当增大 pool_size{RESET}")
    else:
        issues.append("缓存命中率偏低")
        verdicts.append(f"{RED}[FAIL] 缓存命中率 {avg_hit:.1f}% 偏低，LRU 效果不理想{RESET}")

    if peak_rss <= peak_limit_mb * 1.05:
        verdicts.append(f"{GREEN}[PASS] RSS 峰值未超出 limit（{peak_rss:.0f}/{peak_limit_mb:.0f} MB），内存守限正常{RESET}")
    else:
        issues.append("RSS 超出 limit")
        verdicts.append(f"{RED}[FAIL] RSS 峰值 {peak_rss:.0f} MB 超出 limit {peak_limit_mb:.0f} MB，LRU 淘汰可能滞后{RESET}")

    if rss_std / max(rss_mean, 1) < 0.15:
        verdicts.append(f"{GREEN}[PASS] RSS 标准差 {rss_std:.1f} MB（{rss_std/rss_mean*100:.1f}%），内存使用稳定{RESET}")
    else:
        verdicts.append(f"{YELLOW}[WARN] RSS 波动较大（std={rss_std:.1f} MB），可能存在间歇性内存压力{RESET}")

    for v in verdicts:
        print(f"  {v}")

    print()
    if not issues:
        print(f"  {BOLD}{GREEN}总结：LRU Cache 工作正常，有效控制了内存用量并保持高命中率。{RESET}")
    else:
        print(f"  {BOLD}{RED}总结：存在问题 [{', '.join(issues)}]，建议检查 LRU 参数配置。{RESET}")
    print(f"{'='*60}")


# ─── CSV 输出 ──────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "elapsed_s", "proc_count", "rss_mb", "vmsize_mb", "rss_util_pct",
    "cache_hit_pct", "disk_read_mbps", "logical_read_mbps",
    "syscr_per_sec", "cpu_pct", "d_read_bytes_mb", "d_rchar_mb",
]


def write_csv_header(writer):
    writer.writerow(CSV_FIELDS)


def write_csv_row(writer, metrics: dict, t0: float):
    writer.writerow([
        datetime.fromtimestamp(metrics["ts"]).strftime("%H:%M:%S.%f")[:-3],
        f"{metrics['ts'] - t0:.2f}",
        f"{metrics.get('proc_count', 1)}",
        f"{metrics['rss_mb']:.2f}",
        f"{metrics['vmsize_mb']:.2f}",
        f"{metrics['rss_util_pct']:.2f}" if metrics["rss_util_pct"] is not None else "",
        f"{metrics['cache_hit_pct']:.2f}" if metrics["cache_hit_pct"] is not None else "",
        f"{metrics['disk_read_mbps']:.4f}",
        f"{metrics['logical_read_mbps']:.4f}",
        f"{metrics['syscr_per_sec']:.1f}",
        f"{metrics['cpu_pct']:.2f}",
        f"{metrics['d_read_bytes'] / 1024 / 1024:.4f}",
        f"{metrics['d_rchar']    / 1024 / 1024:.4f}",
    ])


# ─── 主流程 ────────────────────────────────────────────────────────────────────

_stop = False


def _sig_handler(sig, frame):
    global _stop
    _stop = True


def run_monitor(pid: int, pool_size_mb: float, interval: float,
                duration: float | None, output: str | None,
                tree: bool = True,
                pool_per_proc_mb: float | None = None,
                expected_procs: int = 0,
                overhead_per_proc_mb: float = 0.0):
    global _stop
    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # pool_per_proc_mb 模式：pool 上限 = pool_per_proc_mb × 实时进程数
    # pool_size_mb 模式：固定总量
    pool_size_kb = int(pool_size_mb * 1024)  # 初始值，per_proc 模式下每轮覆盖
    peak_procs   = 0  # 追踪历史最大进程数，保证 pool 只升不降
    t0 = time.time()
    t_end = (t0 + duration) if duration else None

    all_metrics: list = []
    csv_file = None
    csv_writer = None

    if output:
        csv_file   = open(output, "w", newline="")
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

    mode_str = "进程树聚合" if tree else "单进程"
    print(f"\n{BOLD}zvec LRU Cache 内存监控{RESET}")
    print(f"  Root PID   : {pid}")
    print(f"  模式       : {mode_str}（{'含所有子进程' if tree else '仅根进程'}）")
    if pool_per_proc_mb is not None:
        print(f"  Pool size  : {pool_per_proc_mb:.0f} MB × Procs（每进程独立 pool，动态计算）")
        if overhead_per_proc_mb > 0:
            print(f"  Overhead   : {overhead_per_proc_mb:.0f} MB × Procs（非内存池开销预算）")
            print(f"  WARN阈値   : (pool + overhead) × Procs × 110%")
    else:
        print(f"  Pool size  : {pool_size_mb:.0f} MB（固定总量）")
        if overhead_per_proc_mb > 0:
            print(f"  Overhead   : {overhead_per_proc_mb:.0f} MB（非内存池开销预算）")
            print(f"  WARN阈値   : (pool + overhead) × 110% = {(pool_size_mb + overhead_per_proc_mb) * 1.1:.0f} MB")
    print(f"  Interval   : {interval}s")
    print(f"  Duration   : {'∞' if duration is None else f'{duration}s'}")
    print(f"  Output CSV : {output or '(不保存)'}")
    print(f"\n  {CYAN}提示: 热度阈值 level1=60% pool / level2=80% pool{RESET}")
    print()
    print(DIVIDER)
    print(HEADER)
    print(DIVIDER)

    if tree:
        prev, _ = take_tree_sample(pid)
    else:
        prev = take_sample(pid)
    if prev is None:
        print(f"{RED}无法读取 PID {pid} 的 /proc 数据，请确认进程存在且有权限读取。{RESET}")
        return

    row_count = 0

    while not _stop:
        time.sleep(interval)

        if not pid_alive(pid):
            print(f"\n{YELLOW}进程 {pid} 已退出，监控结束。{RESET}")
            break
        if t_end and time.time() >= t_end:
            print(f"\n达到指定监控时长 {duration}s，结束。")
            break

        if tree:
            curr, proc_count = take_tree_sample(pid)
        else:
            curr = take_sample(pid)
            proc_count = 1
        if curr is None:
            continue

        # exec() 独立进程模式：pool 上限随存活进程数动态变化
        # 用 max(实时procs, peak_procs, expected_procs) 作为底，保证 pool 只升不降
        # - 启动期: peak_procs 随进程数增长， pool 逐步扰大
        # - 风冷期: peak_procs 锁定在峰值， pool 不缩小，避免误报
        if pool_per_proc_mb is not None and proc_count > 0:
            peak_procs = max(peak_procs, proc_count)
            effective_procs = max(peak_procs, expected_procs)
            pool_size_kb  = int(pool_per_proc_mb  * 1024 * effective_procs)
            limit_size_kb = int((pool_per_proc_mb + overhead_per_proc_mb) * 1024 * effective_procs)
        else:
            limit_size_kb = int((pool_size_mb + overhead_per_proc_mb) * 1024)

        metrics = compute_metrics(prev, curr, pool_size_kb, limit_size_kb)
        metrics["proc_count"]    = proc_count
        metrics["expected_procs"] = expected_procs
        all_metrics.append(metrics)

        # 每 20 行重打表头
        if row_count % 20 == 0 and row_count > 0:
            print(DIVIDER)
            print(HEADER)
            print(DIVIDER)

        print(format_row(metrics, t0))
        row_count += 1

        if csv_writer:
            write_csv_row(csv_writer, metrics, t0)
            csv_file.flush()

        prev = curr

    if csv_file:
        csv_file.close()
        print(f"\nCSV 数据已保存至: {output}")

    # 报告中使用各采样点实际生效的 pool_size
    print_report(all_metrics,
                 pool_size_mb if pool_per_proc_mb is None else None,
                 pool_per_proc_mb)


# ─── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="监控 zvec bench 进程的 LRU Cache 有效性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pid",  type=int,   help="直接指定进程 PID")
    group.add_argument("--name", type=str,   help="按进程名自动查找（如 bench）")

    pool_group = parser.add_mutually_exclusive_group()
    pool_group.add_argument(
        "--pool-size", type=float, default=None,
        metavar="MB",
        help="固定总 pool 大小（MB）。fork 共享池时使用，默认 3072 MB",
    )
    pool_group.add_argument(
        "--pool-per-proc", type=float, default=None,
        metavar="MB",
        help="每个子进程独立的 pool 大小（MB）。exec() 独立启动时使用，"
             "Pool%% = RSS / (pool_per_proc × 实时进程数)，默认 3072 MB",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        metavar="SEC",
        help="采样间隔（秒），默认 1.0",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        metavar="SEC",
        help="最长监控时长（秒），默认不限（直到进程结束或 Ctrl+C）",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        metavar="FILE",
        help="将采样数据保存为 CSV 文件（可用于后续绘图）",
    )
    parser.add_argument(
        "--overhead-per-proc", type=float, default=0.0,
        metavar="MB",
        help="每进程预期的非 pool 内存开销（MB），包括代码段、堆、共享库、索引元数据等。"
             "WARN 阈値 = (pool + overhead) × procs × 110%，默认 0"
    )
    parser.add_argument(
        "--expected-procs", type=int, default=0,
        metavar="N",
        help="预期工作进程数（exec 模式下有效）。"
             "pool 上限用 max(实时procs, N) 计算，"
             "避免启动期进程未全部拉起时误报 WARN"
    )
    parser.add_argument(
        "--no-tree", action="store_true", default=False,
        help="禁用进程树聚合，仅监控根进程自身（默认开启树聚合）",
    )

    args = parser.parse_args()

    # 解析 PID
    if args.pid:
        pid = args.pid
        if not pid_alive(pid):
            print(f"{RED}错误：PID {pid} 不存在或无权限访问。{RESET}")
            sys.exit(1)
    else:
        pids = find_pid_by_name(args.name)
        if not pids:
            print(f"{RED}错误：找不到名称含 '{args.name}' 的进程。{RESET}")
            sys.exit(1)
        if len(pids) > 1:
            print(f"{YELLOW}找到多个匹配进程: {pids}，使用第一个 PID={pids[0]}{RESET}")
        pid = pids[0]
        print(f"自动选择进程: PID={pid}  名称={args.name}")

    # 处理 pool 参数默认值
    pool_per_proc = args.pool_per_proc   # None 或用户指定值
    if args.pool_size is not None:
        pool_total = args.pool_size
    elif pool_per_proc is not None:
        pool_total = pool_per_proc       # 占位，run_monitor 会动态覆盖
    else:
        pool_total = 3072.0              # 默认 3 GB

    run_monitor(
        pid                  = pid,
        pool_size_mb         = pool_total,
        interval             = args.interval,
        duration             = args.duration,
        output               = args.output,
        tree                 = not args.no_tree,
        pool_per_proc_mb     = pool_per_proc,
        expected_procs       = args.expected_procs,
        overhead_per_proc_mb = args.overhead_per_proc,
    )


if __name__ == "__main__":
    main()

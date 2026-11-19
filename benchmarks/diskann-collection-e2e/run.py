#!/usr/bin/env python3
"""Run the same Collection workload for HNSW and DiskANN in fresh processes."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
from statistics import median
import subprocess
import sys
import tempfile
import time
import uuid


PREFIX = "ZVEC_COLLECTION_BENCH_RESULT "
INDEXES = ("hnsw", "diskann")
FIXTURE_FILES = ("base.fbin", "queries.fbin", "neighbors.ibin")
IOS_BUNDLE = "com.zvec.zvec_collection_benchmark"


@contextmanager
def prepare_transport(args, binary, dataset_dir, output):
    if args.transport == "native":
        if args.python_sdk:
            command = [sys.executable, str(Path(__file__).with_name("python_benchmark.py").resolve())]
        else:
            command = [str(binary)]
        yield command, str(dataset_dir), str(output)
        return
    if args.transport == "android":
        remote = f"/data/local/tmp/zvec-collection-{uuid.uuid4().hex}"
        executable = f"{remote}/benchmark"
        subprocess.run(["adb", "shell", "mkdir", remote], check=True)
        try:
            subprocess.run(["adb", "push", str(binary), executable], check=True)
            subprocess.run(["adb", "shell", "chmod", "755", executable], check=True)
            for name in FIXTURE_FILES:
                subprocess.run(["adb", "push", str(dataset_dir / name), f"{remote}/{name}"], check=True)
            yield ["adb", "shell", executable], remote, remote
        finally:
            subprocess.run(["adb", "shell", "rm", "-rf", remote], check=False)
        return

    simctl = ["xcrun", "simctl"]
    subprocess.run([*simctl, "install", args.device, str(binary)], check=True)
    container = subprocess.check_output(
        [*simctl, "get_app_container", args.device, IOS_BUNDLE, "data"], text=True
    ).strip()
    documents = Path(container) / "Documents"
    documents.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=documents) as remote:
        for name in FIXTURE_FILES:
            shutil.copyfile(dataset_dir / name, Path(remote) / name)
        command = [*simctl, "launch", "--console", "--terminate-running-process", args.device, IOS_BUNDLE]
        yield command, remote, remote


def result_rows(text):
    return [
        json.loads(line.split(PREFIX, 1)[1])
        for line in text.splitlines()
        if PREFIX in line
    ]


def run_phase(transport, launch, output, phase, index, trial=None, observe_storage=False):
    tag = f"{index}-{phase}" if trial is None else f"{index}-{phase}-{trial}"
    launcher, data, collections = launch
    command = [*launcher, phase, index, str(data), str(Path(collections) / index)]
    ios_result = None
    if transport == "ios":
        ios_result = Path(collections) / f"{tag}.result"
        command.append(str(ios_result))
    elif transport == "android":
        env = ["env", "ZVEC_BENCH_QUERY_IO=1"] if observe_storage else []
        command = [*command[:2], shlex.join([*env, *command[2:]])]
    elif observe_storage:
        command.append("--observe-storage")
    log = output / f"{tag}.log"
    print(f"Running {tag}", flush=True)
    ios_observation = None
    with log.open("w", encoding="utf-8") as stream:
        if ios_result and observe_storage:
            from storage_observation import run_ios_observed

            returncode, ios_observation = run_ios_observed(
                command, os.environ, stream, ios_result.with_suffix(".io"), ios_result
            )
        else:
            returncode = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=3600).returncode
    log_text = log.read_text(encoding="utf-8", errors="replace")
    rows = result_rows(log_text)
    launch_race = (
        transport == "ios" and returncode == 3
        and "FBSOpenApplicationServiceErrorDomain" in log_text
    )
    if ios_result and (returncode == 0 or launch_race) and (launch_race or not rows):
        deadline = time.monotonic() + 15
        while not ios_result.is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        if ios_result.is_file():
            recovered = ios_result.read_text(encoding="utf-8")
            rows = result_rows(recovered)
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"\nRecovered simulator result\n{recovered}")
        else:
            launch_race = False
    if (returncode and not launch_race) or len(rows) != 1:
        print(log_text[-6000:], flush=True)
        raise RuntimeError(f"{tag}: code={returncode}, results={len(rows)}. See {log}.")
    row = rows[0]
    if row["index"] != index or row["phase"] != phase:
        raise RuntimeError(f"Unexpected result for {tag}")
    if ios_observation is not None:
        row["storage_observation"] = ios_observation
    (output / f"{tag}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, separators=(",", ":")), flush=True)
    return row


def measure(args, launch, output, dataset):
    builds = {
        index: run_phase(args.transport, launch, output, "build", index)
        for index in INDEXES
    }
    if any(row["vectors"] != dataset["vectors"] or not row["index_complete"]
           for row in builds.values()):
        raise RuntimeError("A build did not verify the dataset and complete index")

    queries = {index: [] for index in INDEXES}
    observe = args.observe_storage and (args.transport != "native" or sys.platform in ("darwin", "win32"))
    for trial in range(1, args.trials + 1):
        order = INDEXES if trial % 2 else tuple(reversed(INDEXES))
        for index in order:
            row = run_phase(args.transport, launch, output, "query", index, trial,
                            observe_storage=observe and index == "diskann")
            if (row["vectors"] != dataset["vectors"] or row["dimension"] != dataset["dimension"]
                    or row["queries_per_pass"] != dataset["evaluation_queries"]):
                raise RuntimeError(f"{index} queried the wrong dataset")
            queries[index].append(row)
    return builds, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sdk = parser.add_mutually_exclusive_group(required=True)
    sdk.add_argument("--binary", type=Path, help="Native C++ benchmark executable or iOS app")
    sdk.add_argument("--python-sdk", action="store_true", help="Run against the installed Zvec wheel")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default=f"{platform.system().lower()}-{platform.machine().lower()}")
    parser.add_argument("--transport", choices=["native", "android", "ios"], default="native")
    parser.add_argument("--device", help="Simulator UDID when --transport=ios")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--observe-storage", action="store_true",
                        help="Collect paired I/O observations during non-Linux DiskANN queries")
    args = parser.parse_args()
    if args.trials < 1 or args.trials % 2 == 0:
        parser.error("--trials must be a positive odd number")
    if args.transport == "ios" and not args.device:
        parser.error("--transport=ios requires --device")
    if args.python_sdk and args.transport != "native":
        parser.error("--python-sdk requires native transport")
    if args.observe_storage and args.transport == "native" and not args.python_sdk:
        parser.error("Native storage observation requires --python-sdk")

    binary = args.binary.resolve(strict=True) if args.binary else None
    dataset_dir = args.data.resolve(strict=True)
    dataset = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    for name in FIXTURE_FILES:
        with (dataset_dir / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != dataset["sha256"][name]:
                raise ValueError(f"Dataset checksum mismatch: {name}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with prepare_transport(args, binary, dataset_dir, output) as launch:
        builds, queries = measure(args, launch, output, dataset)

    medians = {
        index: {
            key: median(trial[key] for trial in rows)
            for key in rows[0]
            if isinstance(rows[0][key], (int, float))
        }
        for index, rows in queries.items()
    }
    hnsw, diskann = (medians[index] for index in INDEXES)
    comparable = (
        min(hnsw["recall_at_10"], diskann["recall_at_10"]) >= 0.93
        and abs(hnsw["recall_at_10"] - diskann["recall_at_10"]) <= 0.02
    )
    report = {
        "version": "v0.8.1",
        "measurement": "Public Collection API; serial calls with one query thread",
        "sdk": "Python wheel" if args.python_sdk else "C++",
        "storage_observation_requested": args.observe_storage,
        "label": args.label,
        "transport": args.transport,
        "coordinator_platform": platform.platform(),
        "revision": os.environ.get("GITHUB_SHA"),
        "dataset": dataset,
        "builds": builds,
        "query_trials": queries,
        "query_medians": medians,
        "comparable_recall": comparable,
        "diskann_query_rss_delta_saved_bytes": (
            hnsw["query_rss_delta_bytes"] - diskann["query_rss_delta_bytes"]
        ),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not comparable:
        raise RuntimeError("Recall comparability gate did not pass; raw results retained")


if __name__ == "__main__":
    main()

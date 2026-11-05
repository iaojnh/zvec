"""Audit immutable CI artifacts and retain each platform's first qualified run."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

from benchmark_diskann_paced_io import aggregate, evaluate_probe, validate_query
from summarize_diskann_controlled_io import EXPECTED
from summarize_diskann_paced_io import build_report


def verify_clock(query):
    count = query["queries"] * query["repetitions"]
    wall = query["timed_block_wall_ms"]
    if wall <= 0 or not math.isclose(count * 1000 / wall, query["qps"], rel_tol=1e-9):
        raise ValueError("QPS does not match actual timed-block wall time")
    if wall + 0.001 < count * 1000 / query["search_only_qps"]:
        raise ValueError("Complete wall time is shorter than search-call time")
    pace = query.get("native_pacing")
    if pace and (pace["bytes"] != 4096 * pace["reads"] or
                 pace["actual_wait_ns"] > wall * 1e6 * 1.001):
        raise ValueError("Native read volume or actual wait is inconsistent with wall time")


def audit_report(report, expected_commit, source_run):
    if (report["label"] not in EXPECTED or report["kind"] != "paced-native-file-io" or
            report["host"]["commit"] != expected_commit or
            report["host"]["run_url"].rstrip("/").split("/")[-1] != str(source_run)):
        raise ValueError("Unexpected platform, benchmark commit or source run")
    if (report["probe_operations"] != 8192 or report["probe_warmup_operations"] != 8192 or
            report["common_iops"] != 2000 or report["queue_depth"] != 8 or
            report["trials"] != 3 or report["repetitions"] != 3):
        raise ValueError("Qualification/query protocol changed")
    if not report.get("publication_ready"):
        return None
    for rate, probe in report["probes"].items():
        if evaluate_probe(probe["raw"], report["label"], int(rate)) != probe["verification"]:
            raise ValueError("Recorded probe decision differs from raw measurements")
    if not report["profiles"]["2000"]["eligible"] or not report["profiles"]["0"]["eligible"]:
        raise ValueError("Missing common tier or unlimited native reference")
    count = 0
    for rate, profile in report["profiles"].items():
        if not profile["eligible"]:
            continue
        if int(rate) and not report["probes"][rate]["verification"]["eligible"]:
            raise ValueError("Query profile did not pass its I/O qualification")
        if set(profile["raw_trials"]) != {"hnsw", "diskann"}:
            raise ValueError("Incomplete HNSW/DiskANN pair")
        for algorithm, queries in profile["raw_trials"].items():
            if len(queries) != 3 or aggregate(queries) != profile["medians"][algorithm]:
                raise ValueError("Incomplete trials or incorrect median")
            for query in queries:
                if (query["queries"] != 1000 or query["repetitions"] != 3 or
                        not query["workload_warmup"]):
                    raise ValueError("Query workload/warmup changed")
                validate_query(query, algorithm, report["fixture"], int(rate), report["label"],
                               profile["bytes_per_second_limit"], report["queue_depth"])
                verify_clock(query)
                count += 1
    return count


def ordered_artifacts(artifacts):
    return sorted(artifacts, key=lambda a: (a["created_at"], a["id"]))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = os.environ.get("GITHUB_REPOSITORY", "iaojnh/zvec")
    base = "https://api.github.com/repos/" + repository
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Zvec-paced-results-audit",
               "Authorization": "Bearer " + os.environ["GH_TOKEN"]}

    def get(path):
        request = urllib.request.Request(base + path, headers=headers)
        try:
            return urllib.request.build_opener(NoRedirect).open(request, timeout=30).read()
        except urllib.error.HTTPError as error:
            if error.code != 302:
                raise
            # Never forward GitHub credentials to the signed blob destination.
            return urllib.request.urlopen(error.headers["Location"], timeout=30).read()

    run = json.loads(get(f"/actions/runs/{args.source_run_id}"))
    if run["head_sha"] != args.expected_commit:
        raise ValueError("Source workflow SHA does not match requested benchmark revision")
    artifacts = []
    page = 1
    while True:
        items = json.loads(get(f"/actions/runs/{args.source_run_id}/artifacts?per_page=100&page={page}"))["artifacts"]
        artifacts.extend(a for a in items if a["name"].startswith("paced-"))
        if len(items) < 100:
            break
        page += 1
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {}
    records = []
    process_trials = 0
    for artifact in ordered_artifacts(artifacts):
        if artifact["expired"]:
            raise ValueError("Source artifact expired")
        payload = get(f"/actions/artifacts/{artifact['id']}/zip")
        raw = args.output / "all-attempts" / str(artifact["id"])
        raw.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist():
                destination = (raw / info.filename).resolve()
                if not destination.is_relative_to(raw.resolve()):
                    raise ValueError("Unsafe artifact path")
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(info))
        paths = list(raw.rglob("paced-results.json"))
        record = {key: artifact[key] for key in ("id", "name", "created_at")}
        record["artifact_url"] = f"https://github.com/{repository}/actions/runs/{args.source_run_id}/artifacts/{artifact['id']}"
        record["selected"] = False
        if not paths:
            record["reason"] = "No completed measurement report"
            records.append(record)
            continue
        if len(paths) != 1:
            raise ValueError("Ambiguous artifact report")
        report_bytes = paths[0].read_bytes()
        report = json.loads(report_bytes)
        record["label"] = report["label"]
        record["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
        trials = audit_report(report, args.expected_commit, args.source_run_id)
        record["qualified"] = trials is not None
        if trials is not None and report["label"] not in selected:
            directory = args.output / "selected" / report["label"]
            directory.mkdir(parents=True)
            (directory / "paced-results.json").write_bytes(report_bytes)
            selected[report["label"]] = artifact["id"]
            process_trials += trials
            record["selected"] = True
        elif trials is None:
            record["reason"] = "Common tier did not qualify or measurement incomplete"
        else:
            record["reason"] = "An earlier complete qualified report already exists"
        records.append(record)
    manifest = {"source_run": run["html_url"], "benchmark_commit": args.expected_commit,
                "selection_rule": "Earliest artifact per platform with complete qualified trials; never rank by QPS",
                "selected_artifacts": selected, "query_process_trials": process_trials, "all_artifacts": records}
    (args.output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    markdown, reports, failures = build_report(args.output / "selected")
    if failures or set(selected) != EXPECTED:
        raise ValueError("Missing complete qualified platform: " + ", ".join(failures))
    markdown += "\n## 结果来源与重跑\n\n按每个平台的首份完整合格产物汇总，不按性能高低挑选。"
    markdown += "未达标及后续重跑结果全部保存在 `all-attempts/`，选择规则、产物编号与文件校验值见 `selection.json`。\n"
    (args.output / "publication-comparison.md").write_text(markdown)
    (args.output / "all-platform-results.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Audited {len(reports)} platforms and {process_trials} query-process trials; all raw attempts retained.")


if __name__ == "__main__":
    main()

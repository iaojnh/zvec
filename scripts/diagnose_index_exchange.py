"""Complete migration stages while retaining every strict numerical failure."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from pathlib import Path

from verify_windows_index_exchange import (
    ANN_RECALL_DROP, CASES, SCORE_ATOL, Case, compare, pack, runtime, save, unpack, zvec,
)


def differences(expected, actual, exact):
    result = {"recall_delta": actual["recall"]-expected["recall"]}
    for mode in ("ordinary", "exhaustive"):
        errors, changed, missing = [], 0, 0
        for left, right in zip(expected[mode], actual[mode]):
            a = dict(zip(left["ids"], left["scores"]))
            b = dict(zip(right["ids"], right["scores"]))
            errors.extend(abs(a[k]-b[k]) for k in a.keys() & b.keys())
            changed += a.keys() != b.keys()
            missing += len(a.keys()-b.keys())
        result[mode] = {"max_score_error": max(errors, default=0.0),
                        "changed_topk_queries": changed, "missing_topk_hits": missing,
                        "queries": len(expected[mode])}
    try:
        compare(expected, actual, exact)
        result["strict_pass"] = True
    except AssertionError as error:
        result["strict_pass"] = False
        result["strict_error"] = str(error)
    return result


def run(args):
    out = args.root/"out"
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    own = args.root/"own-native"
    unpack(next(args.bundles.rglob(f"native-{args.os}.zip")), own)

    def record(label, callback):
        try:
            item = {"label": label, "io_cud_reopen_pass": True, "metrics": callback()}
        except Exception:
            item = {"label": label, "io_cud_reopen_pass": False,
                    "traceback": traceback.format_exc()}
        reports.append(item)
        print(json.dumps(item), flush=True)
        save(out/f"diagnostic-{args.action}-{args.os}.json",
             {"runtime": runtime(), "original_run": 45774216017,
              "score_atol": SCORE_ATOL, "recall_drop": ANN_RECALL_DROP,
              "reports": reports})

    if args.action == "exchange":
        for archive in sorted(args.bundles.rglob("native-*.zip")):
            source = archive.stem.removeprefix("native-")
            if source == args.os and not args.allow_self:
                continue
            incoming = args.root/"incoming"/source
            unpack(archive, incoming)
            for backend in ("mmap", "buffer"):
                bundle = args.root/"modified"/f"{source}-{backend}"
                bundle.mkdir(parents=True, exist_ok=True)
                for definition in CASES:
                    def check(definition=definition):
                        case = Case(definition)
                        reference = json.loads((incoming/f"{case.name}.json").read_text())
                        path = bundle/case.name
                        shutil.copytree(incoming/case.name, path)
                        metrics = {}
                        for readonly in (True, False):
                            observed = case.observe(path, backend, False, readonly)
                            metrics[f"before_readonly_{readonly}"] = differences(
                                reference["baseline"], observed, case.flat)
                        modified = case.mutate(path, backend)
                        metrics["after_cud"] = differences(reference["mutated"], modified, case.flat)
                        save(bundle/f"{case.name}.json", modified)
                        return metrics
                    record(f"{source}->{args.os}/{definition[0]}/{backend}", check)
                pack(bundle, out/f"diagnostic-mutated-{source}-{args.os}-{backend}.zip")
    else:
        for archive in sorted(args.bundles.rglob(f"diagnostic-mutated-{args.os}-*.zip")):
            incoming = args.root/"returned"/archive.stem
            unpack(archive, incoming)
            for definition in CASES:
                def check_return(definition=definition):
                    case = Case(definition)
                    reference = json.loads((own/f"{case.name}.json").read_text())["mutated"]
                    sender = json.loads((incoming/f"{case.name}.json").read_text())
                    metrics = {}
                    for backend in ("mmap", "buffer"):
                        result = case.observe(incoming/case.name, backend, True, True)
                        metrics[f"{backend}_origin"] = differences(reference, result, case.flat)
                        metrics[f"{backend}_sender"] = differences(sender, result, case.flat)
                    return metrics
                record(f"{archive.stem}/{definition[0]}", check_return)

    io_failures = [r for r in reports if not r["io_cud_reopen_pass"]]
    strict_failures = [r for r in reports if r["io_cud_reopen_pass"] and
                       any(not m["strict_pass"] for m in r["metrics"].values())]
    summary = {"total": len(reports), "io_cud_reopen_failed": len(io_failures),
               "strict_numerical_failed": len(strict_failures)}
    print("SUMMARY "+json.dumps(summary), flush=True)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as stream:
            stream.write("## Diagnostic results\n\n"+chr(96)*3+"json\n"+json.dumps(summary, indent=2)+"\n"+chr(96)*3+"\n")
    assert reports and not io_failures and not strict_failures, "Original thresholds retained; see diagnostics"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("exchange", "return"))
    parser.add_argument("--os", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--allow-self", action="store_true", help="Local harness smoke test only")
    args = parser.parse_args()
    zvec.init(log_level=zvec.LogLevel.WARN, query_threads=2, optimize_threads=2)
    run(args)

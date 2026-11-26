"""Exchange actual indexes between native Windows, macOS and Linux runtimes.

The native phase exports pristine indexes and independent mutation references.
The exchange phase reads another OS's files, mutates them, and exports them.
The return phase reopens those mutations on the originating OS. No rebuild or
optimize is performed on a migrated index. Every failed assertion is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import platform
import shutil
import traceback
import zipfile
from pathlib import Path

import numpy as np
import zvec

SOURCE_SHA = "ae3c1afa7d9658debf6876bb159e15b442dfb35f"
SCORE_ATOL = 2e-5
ANN_RECALL_DROP = 0.02
CASES = [(f"{a}_{q}", a, q) for a in ("flat", "hnsw", "ivf", "vamana")
         for q in ("fp32", "fp16", "int8", "int4")]
CASES += [("diskann_fp32", "diskann", "fp32"),
          ("flat_sparse", "flat_sparse", "fp32"),
          ("hnsw_sparse", "hnsw_sparse", "fp32")]


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def pack(directory, archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        # Empty mutable-segment directories are part of a complete collection.
        for path in sorted(directory.rglob("*")):
            zipped.write(path, path.relative_to(directory).as_posix())


def unpack(archive, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(directory)


def runtime():
    return {"source_sha": SOURCE_SHA, "platform": platform.platform(),
            "machine": platform.machine(), "page_size": mmap.PAGESIZE,
            "mapping_allocation_granularity": mmap.ALLOCATIONGRANULARITY,
            "python": platform.python_version(), "numpy": np.__version__,
            "module": zvec.__file__}


def compare(expected, actual, exact):
    assert expected["count"] == actual["count"]
    assert expected["data_hash"] == actual["data_hash"]
    for mode in ("ordinary", "exhaustive"):
        left, right = expected[mode], actual[mode]
        assert len(left) == len(right)
        for e, a in zip(left, right):
            es, ac = dict(zip(e["ids"], e["scores"])), dict(zip(a["ids"], a["scores"]))
            for key in es.keys() & ac.keys():
                assert abs(es[key] - ac[key]) <= SCORE_ATOL, (mode, key, es[key], ac[key])
            if exact or mode == "exhaustive":
                missing, added = es.keys() - ac.keys(), ac.keys() - es.keys()
                assert len(missing) == len(added)
                # Equal-distance hits at the Top-K boundary can differ across
                # the STL implementations. Only tied substitutions are valid.
                for x, y in zip(sorted(es[k] for k in missing), sorted(ac[k] for k in added)):
                    assert abs(x-y) <= SCORE_ATOL, (mode, "different non-tied hits", x, y)
    if not exact:
        assert expected["recall"] - actual["recall"] <= ANN_RECALL_DROP + 1e-12, (
            "ANN recall drop", expected["recall"], actual["recall"])


class Case:
    def __init__(self, definition):
        self.name, self.algorithm, self.quant = definition
        self.sparse = self.algorithm.endswith("_sparse")
        self.flat = self.algorithm.startswith("flat")
        self.count = 20000 if self.algorithm == "hnsw_sparse" else 4096 if self.name == "hnsw_fp32" else 2048
        self.dim = 768 if self.sparse else 1024 if self.name == "hnsw_fp32" else 128
        rng = np.random.default_rng(827 + [x[0] for x in CASES].index(self.name))
        # Integer values scaled by a power of two are byte-identical on all OSes.
        self.vectors = rng.integers(-1024, 1025, (self.count+128, self.dim), dtype=np.int32).astype(np.float32) / 1024
        if self.sparse:
            self.positions = np.argsort(-np.abs(self.vectors), axis=1, kind="stable")[:, :16]
        self.data_hash = hashlib.sha256(self.vectors.tobytes()).hexdigest()
        options = {"metric_type": zvec.MetricType.IP if self.sparse else zvec.MetricType.COSINE}
        if self.quant != "fp32":
            options["quantize_type"] = getattr(zvec.QuantizeType, self.quant.upper())
        if self.flat:
            self.index_param, self.query_param, self.full_param = zvec.FlatIndexParam(**options), None, None
        elif self.algorithm.startswith("hnsw"):
            self.index_param = zvec.HnswIndexParam(m=36 if self.sparse else 16, ef_construction=100, **options)
            self.query_param = zvec.HnswQueryParam(ef=200)
            self.full_param = zvec.HnswQueryParam(ef=200, is_linear=True)
        elif self.algorithm == "ivf":
            self.index_param = zvec.IVFIndexParam(n_list=16, n_iters=5, **options)
            self.query_param, self.full_param = zvec.IVFQueryParam(nprobe=8), zvec.IVFQueryParam(nprobe=16)
        elif self.algorithm == "vamana":
            self.index_param = zvec.VamanaIndexParam(max_degree=32, search_list_size=80, **options)
            self.query_param = zvec.VamanaQueryParam(ef_search=200)
            self.full_param = zvec.VamanaQueryParam(ef_search=200, is_linear=True)
        else:
            self.index_param = zvec.DiskAnnIndexParam(max_degree=24, list_size=64, **options)
            self.query_param, self.full_param = zvec.DiskAnnQueryParam(list_size=150), None

    def value(self, row):
        if self.sparse:
            return {int(i): float(self.vectors[row, i]) for i in sorted(self.positions[row])}
        return self.vectors[row].tolist()

    def option(self, backend, readonly=False):
        return zvec.CollectionOption(enable_mmap=backend == "mmap", read_only=readonly)

    def observe(self, path, backend, mutated, readonly):
        collection = zvec.open(str(path), option=self.option(backend, readonly))
        try:
            n = self.count + (128 if mutated else 0)
            assert collection.stats.doc_count == n-int(mutated)
            rows = sorted(set(np.linspace(0, self.count-1, 64, dtype=int).tolist()
                              + list(range(self.count, n)) + [0, 2]))
            if mutated:
                rows.remove(2)
                assert not collection.fetch(["2"])
            for start in range(0, len(rows), 128):
                selected = rows[start:start+128]
                docs = collection.fetch([str(i) for i in selected])
                assert len(docs) == len(selected)
                for i in selected:
                    doc = docs[str(i)]
                    assert doc.fields["row"] == (999999 if mutated and i == 0 else i)
                    wanted = self.value(n-1 if mutated and i == 0 else i)
                    actual = doc.vectors["embedding"]
                    if self.sparse:
                        assert actual.keys() == wanted.keys()
                        actual, wanted = [actual[k] for k in wanted], list(wanted.values())
                    assert np.allclose(actual, wanted, rtol=2e-6, atol=2e-6), (self.name, i, "raw data")
            dense = np.array(self.vectors[:n], dtype=np.float64)
            if self.sparse:
                dense[:] = 0
                for i in range(n):
                    for j, value in self.value(i).items():
                        dense[i, j] = value
            if mutated:
                dense[0] = dense[-1]
            norms = np.linalg.norm(dense, axis=1) if not self.sparse else None
            ordinary, exhaustive, recalls = [], [], []
            query_rows = np.linspace(0, self.count-1, 20, dtype=int).tolist() + ([self.count, n-1] if mutated else [])
            for row in query_rows:
                q = np.zeros(self.dim, dtype=np.float64) if self.sparse else self.vectors[row].astype(np.float64)
                if self.sparse:
                    for j, value in self.value(row).items():
                        q[j] = value
                scores = dense @ q
                if not self.sparse:
                    scores = 1-scores/(norms*np.linalg.norm(q))
                if mutated:
                    scores[2] = -np.inf if self.sparse else np.inf
                truth = set(np.argsort(-scores if self.sparse else scores, kind="stable")[:10])

                def search(param):
                    hits = collection.query(zvec.Query("embedding", vector=self.value(row), param=param), topk=10, include_vector=False)
                    assert len(hits) == 10 and len({h.id for h in hits}) == 10
                    assert all(np.isfinite(h.score) for h in hits)
                    assert not mutated or "2" not in {h.id for h in hits}
                    if self.algorithm == "diskann":
                        for hit in hits:
                            assert abs(hit.score-scores[int(hit.id)]) <= SCORE_ATOL
                    return {"ids":[h.id for h in hits], "scores":[h.score for h in hits]}

                answer = search(self.query_param)
                ordinary.append(answer)
                recalls.append(len({int(k) for k in answer["ids"]} & truth)/10)
                if self.flat:
                    exhaustive.append(answer)
                elif self.full_param is not None:
                    exhaustive.append(search(self.full_param))
            return {"count":n-int(mutated), "data_hash":self.data_hash,
                    "fetched_documents":len(rows), "ordinary":ordinary,
                    "exhaustive":exhaustive, "recall":float(np.mean(recalls))}
        finally:
            collection.close()

    def mutate(self, path, backend):
        collection = zvec.open(str(path), option=self.option(backend))
        try:
            docs = [zvec.Doc(id=str(i), fields={"row":i}, vectors={"embedding":self.value(i)})
                    for i in range(self.count, self.count+128)]
            assert all(s.ok() for s in collection.insert(docs))
            assert collection.delete("2").ok()
            assert collection.update(zvec.Doc(id="0", fields={"row":999999},
                    vectors={"embedding":self.value(self.count+127)})).ok()
            collection.flush()
        finally:
            collection.close()
        answer = self.observe(path, backend, True, True)
        compare(answer, self.observe(path, backend, True, False), True)
        return answer

    def build(self, path):
        schema = zvec.CollectionSchema(name="index_exchange",
            fields=[zvec.FieldSchema("row", zvec.DataType.INT64)],
            vectors=[zvec.VectorSchema("embedding", zvec.DataType.SPARSE_VECTOR_FP32 if self.sparse else zvec.DataType.VECTOR_FP32,
                                      0 if self.sparse else self.dim, index_param=self.index_param)])
        collection = zvec.create_and_open(str(path), schema, option=self.option("mmap"))
        try:
            for start in range(0, self.count, 512):
                docs = [zvec.Doc(id=str(i), fields={"row":i}, vectors={"embedding":self.value(i)})
                        for i in range(start, min(start+512, self.count))]
                assert all(s.ok() for s in collection.insert(docs))
            collection.flush()
            collection.optimize()
        finally:
            collection.close()


def run(args):
    out = args.root / "out"
    out.mkdir(parents=True, exist_ok=True)
    reports = []

    def record(label, callback):
        try:
            result = callback()
            item = {"label":label, "success":True, "result":result}
        except Exception:
            item = {"label":label, "success":False, "traceback":traceback.format_exc()}
        reports.append(item)
        print(json.dumps({k:v for k,v in item.items() if k != "result"}), flush=True)
        save(out/f"{args.action}-{args.os}.json", {"runtime":runtime(), "reports":reports})

    definitions = [d for d in CASES if not args.case or d[0] in args.case]
    if args.action == "native":
        native = args.root / "native"
        native.mkdir(parents=True, exist_ok=True)
        for definition in definitions:
            def build(definition=definition):
                case = Case(definition)
                path = native/case.name
                case.build(path)
                baseline = case.observe(path, "mmap", False, True)
                compare(baseline, case.observe(path, "buffer", False, True), True)
                mutable = args.root/"native-mutations"/case.name
                mutable.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, mutable)
                mutated = case.mutate(mutable, "mmap")
                save(native/f"{case.name}.json", {"baseline":baseline, "mutated":mutated})
                return {"case":case.name, "data_hash":case.data_hash}
            record(definition[0], build)
        save(native/"runtime.json", runtime())
        pack(native, out/f"native-{args.os}.zip")
    else:
        native = args.root/"own-native"
        unpack(next(args.bundles.rglob(f"native-{args.os}.zip")), native)
        for archive in sorted(args.bundles.rglob("native-*.zip" if args.action == "exchange" else f"mutated-{args.os}-*.zip")):
            source = archive.stem.removeprefix("native-")
            if args.action == "exchange" and source == args.os and not args.allow_self:
                continue
            if args.action == "exchange":
                incoming = args.root/"incoming"/source
                unpack(archive, incoming)
                for backend in ("mmap", "buffer"):
                    bundle = args.root/"mutations"/f"{source}-{backend}"
                    bundle.mkdir(parents=True, exist_ok=True)
                    for definition in definitions:
                        def check(definition=definition):
                            case = Case(definition)
                            reference = json.loads((incoming/f"{case.name}.json").read_text())
                            path = bundle/case.name
                            shutil.copytree(incoming/case.name, path)
                            compare(reference["baseline"], case.observe(path, backend, False, True), case.flat)
                            compare(reference["baseline"], case.observe(path, backend, False, False), case.flat)
                            modified = case.mutate(path, backend)
                            compare(reference["mutated"], modified, case.flat)
                            save(bundle/f"{case.name}.json", modified)
                            return {"case":case.name, "source":source, "destination":args.os, "backend":backend}
                        record(f"{source}->{args.os}/{definition[0]}/{backend}", check)
                    pack(bundle, out/f"mutated-{source}-{args.os}-{backend}.zip")
            else:
                incoming = args.root/"returned"/archive.stem
                unpack(archive, incoming)
                for definition in definitions:
                    def check_return(definition=definition):
                        case = Case(definition)
                        reference = json.loads((native/f"{case.name}.json").read_text())["mutated"]
                        sent = json.loads((incoming/f"{case.name}.json").read_text())
                        for backend in ("mmap", "buffer"):
                            result = case.observe(incoming/case.name, backend, True, True)
                            compare(reference, result, case.flat)
                            compare(sent, result, case.flat)
                        return {"case":case.name, "archive":archive.name}
                    record(f"return/{archive.stem}/{definition[0]}", check_return)
    assert reports and all(item["success"] for item in reports), "See retained per-case failures"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["native", "exchange", "return"])
    parser.add_argument("--os", required=True, choices=["windows", "macos", "linux"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundles", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--allow-self", action="store_true", help="Local harness smoke test only")
    args = parser.parse_args()
    zvec.init(log_level=zvec.LogLevel.WARN, query_threads=2, optimize_threads=2)
    run(args)

#!/usr/bin/env python3
"""Run one matrix/fill experiment locally, without a batch scheduler."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["unsymmetric", "spd"])
    p.add_argument("phase", choices=["timing", "quality"])
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--k", type=int, choices=range(4), required=True)
    p.add_argument("--out", type=Path, required=True, help="new output directory; never overwrites an experiment")
    p.add_argument("--threads", default="2,4,8,16,32,64,128")
    p.add_argument("--sync-threads", type=int, default=128)
    p.add_argument("--solver-workers", type=int, default=1)
    a = p.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    if not threads or any(t not in [1,2,4,8,16,32,64,128] for t in threads):
        p.error("threads must be powers of two from 1 through 128")
    if a.phase == "quality" and 1 in threads:
        p.error("quality uses the sequential baseline in place of one-thread async")
    if a.sync_threads < 1 or a.solver_workers < 1:
        p.error("worker/thread counts must be positive")
    matrix = a.matrix.resolve(strict=True)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "config/experiments.json").read_text())
    env = dict(os.environ, OMP_PROC_BIND="close", OMP_PLACES="cores", OPENBLAS_NUM_THREADS="1")
    records = []
    metadata = {"matrix": str(matrix), "k": a.k, "kind": a.kind, "phase": a.phase,
                "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "threads": threads, "sync_threads": a.sync_threads,
                "environment": {k: v for k, v in env.items() if k.startswith(("OMP_", "OPENBLAS_"))},
                "commands": records, "config": config}
    with matrix.open("rb") as f:
        metadata["matrix_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    def run(name, executable, args):
        binary = ROOT / "c_ilu" / executable
        command = [str(binary), *map(str, args)]
        record = {"name": name, "command": command, "started": time.time(),
                  "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
        records.append(record)
        (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("+", " ".join(command), flush=True)
        with (out / f"{name}.csv").open("w") as stdout, (out / f"{name}.log").open("w") as stderr:
            result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
        record.update(returncode=result.returncode, finished=time.time())
        (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(f"{name} failed; see {out / (name + '.log')}")
    base = ["--matrix", matrix, "--k", a.k, "--symbolic", "levelset", "--preprocess",
            "diag-rcm" if a.kind == "spd" else "mc64-rcm", "--ats-scale", "folded"]
    methods = "ats_ic,par_ic" if a.kind == "spd" else "all"
    if a.phase == "quality":
        factors = out / "factors"
        run("producer", "dump_ilu_factors", base + ["--out-dir", factors, "--sweeps-list",
            "1,2,3" if a.kind == "spd" else "1,2,3,4,5", "--thread-list", a.threads,
            "--sync-threads", a.sync_threads, "--seq", "baseline", "--methods", methods,
            "--variants", "async" if a.kind == "spd" else "both", "--async-repeats", "3"])
        run("solver", "cg_from_factors" if a.kind == "spd" else "gmres_from_factors",
            ["--matrix", matrix, "--factor-index", factors / "factor_index.csv", "--out", out / "solver_results.csv",
             "--nrhs", 1, "--restart", 50, "--maxiter", 2000, "--tol", "1e-8", "--seed", 1337,
             "--workers", a.solver_workers])
    else:
        # bench_ilu constructs the shared symbolic pattern once per invocation.
        run("timing", "bench_ilu", base + ["--thread-list", a.threads, "--seq", "baseline",
            "--methods", methods, "--variants", "async" if a.kind == "spd" else "both",
            "--sweeps", 5, "--numeric-repeats", 3])
    print(f"Completed: {out}")


if __name__ == "__main__":
    main()

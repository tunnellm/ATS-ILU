#!/usr/bin/env python3
"""Small local build/producer/replay checks; no remote access or large inputs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expect-mc64", action="store_true")
    a = p.parse_args()
    subprocess.run(["make", "-C", str(ROOT / "c_ilu"), "check"], check=True)
    def run(binary, *args, expected=0):
        r = subprocess.run([str(ROOT / "c_ilu" / binary), *map(str, args)], text=True, capture_output=True)
        if r.returncode != expected:
            raise AssertionError(f"{binary}: returncode {r.returncode}\n{r.stdout}\n{r.stderr}")
        return r
    with tempfile.TemporaryDirectory(prefix="ats-small-test-") as directory:
        tmp = Path(directory)
        for kind, matrix, preprocess, methods, solver in (
            ("ilu", "tiny_general.mtx", "mc64-rcm", "all", "gmres_from_factors"),
            ("ic", "tiny_symmetric.mtx", "diag-rcm", "ats_ic,par_ic", "cg_from_factors")):
            source = ROOT / "c_ilu/tests/data" / matrix
            target = tmp / kind
            r = run("dump_ilu_factors", "--matrix", source, "--out-dir", target,
                    "--k", 1, "--sweeps-list", "1,2,3", "--thread-list", 2,
                    "--sync-threads", 2, "--seq", "baseline", "--methods", methods,
                    "--async-repeats", 3, "--ats-scale", "folded", "--preprocess", preprocess)
            rows = list(csv.DictReader((target / "factor_index.csv").open()))
            assert rows, "empty factor index"
            if kind == "ilu":
                assert all(row["preprocess_requested"] == "mc64-rcm" for row in rows)
                effective = "mc64-rcm" if a.expect_mc64 else "rcm"
                assert all(row["preprocess"] == effective for row in rows)
                assert ("WARNING: MC64" in r.stderr) == (not a.expect_mc64)
            assert (target / "structure.bin.preprocessing").is_file()
            run("dump_ilu_factors", "--matrix", source, "--out-dir", tmp / f"{kind}-mismatch",
                "--prepared-structure", target / "structure.bin", "--k", 2,
                "--thread-list", 2, "--preprocess", preprocess, expected=1)
            output = tmp / f"{kind}_solver.csv"
            run(solver, "--matrix", source, "--factor-index", target / "factor_index.csv",
                "--out", output, "--restart", 50, "--maxiter", 2000, "--tol", "1e-8", "--seed", 1337)
            solved = list(csv.DictReader(output.open()))
            assert solved, "empty solver replay"
            assert all(int(row["converged"]) == 1 for row in solved), solved
            print(f"{kind}: {len(rows)} factor states, {len(solved)} converged solver rows")
        result = run("bench_ilu", "--matrix", ROOT / "c_ilu/tests/data/tiny_general.mtx",
                     "--k", 1, "--threads", 2, "--sweeps", 3,
                     "--numeric-repeats", 3, "--preprocess", "mc64-rcm", "--ats-scale", "folded")
        rows = list(csv.DictReader(result.stdout.splitlines()))
        assert len([r for r in rows if r["method"] == "sequential_ilu"]) == 3
        assert all(float(r["t_factor"]) >= 0 for r in rows)
        print(f"timing: {len(rows)} rows; sequential baseline measured three times")
    print(json.dumps({"small_test": "passed", "external_mc64": a.expect_mc64}))


if __name__ == "__main__":
    main()

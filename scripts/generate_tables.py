#!/usr/bin/env python3
"""Generate the paper's accounting and objective-contraction tables."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "ilu": "results/analysis_2026-07-31/unsymmetric_gmres/combined_solver_outcomes.csv",
    "ic": "results/analysis_2026-08-02/spd_cg/combined_solver_outcomes.csv",
}
NAMES = {"ats_ilu": "ATS-ILU", "parilu": "ParILU", "ats_ic": "ATS-IC", "par_ic": "ParIC"}


def write_tex(path, columns, rows):
    escape = lambda value: str(value).replace("%", r"\%").replace("_", r"\_")
    lines = [r"\begin{tabular}{@{}" + "l" * len(columns) + "@{}}", r"\toprule",
             " & ".join(map(escape, columns)) + r" \\", r"\midrule"]
    lines += [" & ".join(map(escape, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    out = args.output / "tables"
    out.mkdir(parents=True, exist_ok=True)
    accounts, contractions = [], []
    for kind, filename in INPUTS.items():
        data = pd.read_csv(args.work / filename)
        data["pair"] = data["matrix_id"] if kind == "ic" else data["matrix"]
        data["factor_ok"] = data.factor_status.eq("ok") & np.isfinite(data.objective)
        baseline = data[data.method.str.startswith("sequential")][["pair", "k", "objective"]]
        data = data.merge(baseline.rename(columns={"objective": "seq_objective"}), on=["pair", "k"], validate="many_to_one")
        iterative = data[~data.method.str.startswith("sequential")]
        for (method, variant), group in iterative.groupby(["method", "variant"]):
            skipped = group.factor_status.eq("dependent_sweep_skipped")
            ran = group.solver_recorded.eq(1) if kind == "ic" else group.solver_status.notna()
            ok = group.factor_ok & group.converged.eq(1) & group.iterations.gt(0)
            accounts.append(dict(method=method, schedule=variant, requested=len(group),
                failed=int((~group.factor_ok & ~skipped).sum()), skipped=int(skipped.sum()),
                run=int(ran.sum()), nonconvergent=int((ran & ~ok).sum())))
        key = ["pair", "k", "method", "variant", "threads", "repeat"]
        transitions = []
        for before, after in ((1, 2), (2, 3)):
            cols = key + ["objective", "factor_ok", "seq_objective"]
            joined = iterative[iterative.sweeps.eq(before)][cols].merge(
                iterative[iterative.sweeps.eq(after)][cols], on=key,
                suffixes=("_before", "_after"), validate="one_to_one")
            joined = joined[joined.factor_ok_before & joined.factor_ok_after &
                            joined.objective_before.gt(0) & joined.objective_after.ge(0)].copy()
            joined["transition"] = f"{before}->{after}"
            joined["ratio"] = np.sqrt(joined.objective_after / joined.objective_before)
            joined["nonincreasing"] = joined.objective_after.le((1 + 1e-12) * joined.objective_before)
            at_floor = joined[["objective_before", "objective_after"]].max(axis=1).le(1.001 * joined.seq_objective_before)
            joined["floor_aware"] = joined.nonincreasing | at_floor
            transitions.append(joined)
        for (method, variant), group in pd.concat(transitions).groupby(["method", "variant"]):
            row = dict(method=method, schedule=variant, finite_transitions=len(group),
                       strict_nonincreasing_percent=100 * group.nonincreasing.mean(),
                       floor_aware_nonincreasing_percent=100 * group.floor_aware.mean())
            for transition, values in group.groupby("transition"):
                row[f"{transition}_median"] = values.ratio.median()
                row[f"{transition}_p95"] = values.ratio.quantile(.95)
            contractions.append(row)
    accounting = pd.DataFrame(accounts)
    pd.DataFrame(contractions).to_csv(out / "objective_contraction.csv", index=False)
    accounting.to_csv(out / "experiment_accounting.csv", index=False)
    schedule = lambda value: "Asynchronous" if value == "async" else "Synchronous"
    write_tex(out / "experiment_accounting.tex",
              ["Method", "Schedule", "Requested", "Failed", "Skipped", "Solver runs", "Nonconvergent"],
              [[NAMES[r.method], schedule(r.schedule), r.requested, r.failed, r.skipped, r.run, r.nonconvergent]
               for r in accounting.itertuples()])
    write_tex(out / "objective_contraction.tex",
              ["Method", "Schedule", "Nonincreasing", "1 to 2: median (P95)", "2 to 3: median (P95)"],
              [[NAMES[r["method"]], schedule(r["schedule"]), f"{r['strict_nonincreasing_percent']:.1f}%",
                f"{r['1->2_median']:.3f} ({r['1->2_p95']:.3g})",
                f"{r['2->3_median']:.3f} ({r['2->3_p95']:.3g})"] for r in contractions])
    marginal = args.work / "results/publication_timing_2026-08-04/table_01_marginal_sweeps"
    if (marginal / "table.tex").exists():
        shutil.copy2(marginal / "table.tex", out / "marginal_sweeps.tex")
        shutil.copy2(marginal / "marginal_sweep_summary.csv", out / "marginal_sweeps.csv")
    print(f"Tables: {out}")


if __name__ == "__main__":
    main()

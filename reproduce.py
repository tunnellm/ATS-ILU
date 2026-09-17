#!/usr/bin/env python3
"""Rebuild the manuscript outputs without rerunning large experiments."""
import argparse
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PUBLICATION = Path("results/publication_timing_2026-08-04")
QUALITY = Path("results/analysis_2026-08-02/combined_quality_figures")
DERIVED = [Path(p) for p in (
    "results/analysis_2026-07-31/unsymmetric_gmres",
    "results/analysis_2026-08-02/spd_cg",
    "results/analysis_2026-08-03/unsymmetric_timing",
    "results/analysis_2026-08-03/spd_timing")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="skip Julia rendering")
    parser.add_argument("--julia", default="julia +1.12", help="Julia command, or path to a Julia 1.12 executable")
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT.is_relative_to(out):
        parser.error("output must be a separate directory")
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    for directory in [Path("scripts"), Path("plotting"), Path("data"), PUBLICATION, *DERIVED]:
        shutil.copytree(ROOT / directory, work / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    def run(command):
        print("+", " ".join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), cwd=work, env=env, check=True)
    if not args.data_only:
        scripts = ["plotting/plot_combined_solver_quality_profiles.jl",
                   "plotting/plot_combined_objective_solver_case_studies.jl",
                   str(PUBLICATION / "figure_01_scaling/plot.jl"),
                   str(PUBLICATION / "figure_02_work_scaling/plot.jl"),
                   str(PUBLICATION / "figure_04_pareto/plot.jl"),
                   str(PUBLICATION / "table_01_marginal_sweeps/generate.jl")]
        for script in scripts:
            run([*shlex.split(args.julia), "--startup-file=no", f"--project={ROOT}", script])
        figures = [QUALITY / "solver_quality_relative_sequential_combined.pdf",
                   QUALITY / "objective_vs_solver_case_studies_combined.pdf",
                   PUBLICATION / "figure_01_scaling/strong_scaling_method_relative_median_iqr.pdf",
                   PUBLICATION / "figure_02_work_scaling/speedup_vs_pattern_width_64_threads.pdf",
                   PUBLICATION / "figure_04_pareto/construction_quality_pareto.pdf"]
        (out / "figures").mkdir(exist_ok=True)
        for figure in figures:
            for extension in ("pdf", "svg", "png"):
                source = (work / figure).with_suffix("." + extension)
                shutil.copy2(source, out / "figures" / source.name)
    run([sys.executable, ROOT / "scripts/generate_tables.py", "--work", work, "--output", out])
    print(f"Results: {out}")


if __name__ == "__main__":
    main()

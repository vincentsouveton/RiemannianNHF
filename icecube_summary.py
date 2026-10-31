"""Metrics of Section 4.4 for every seed of icecube_experiment.py, their means
and standard deviations, and the deviations from a long reference HMC run in
units of standard error. Writes results/<run>/summary.json and
figures/icecube_table.tex.

Usage:
    python icecube_summary.py --long-seeds 100 101
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

from icecube import angle
from icecube_experiment import localization
from runs import run_dir

ROWS = [  # (label, key, format)
    ("angle to source (deg)", "angle_to_source_deg", "{:.4f}"),
    ("90% radius (deg)", "r90_deg", "{:.4f}"),
    ("acceptance rate", "acceptance", "{:.3f}"),
    ("autocorrelation time of E", "tau_E", "{:.2f}"),
    ("effective sample size", "ess", "{:.0f}"),
    ("evaluations of E / ESS", "true_evaluations_per_ess", "{:.2f}"),
    ("wall time / ESS (s)", "wall_time_per_ess", "{:.4f}"),
]


def deviations(stats, ref):
    """Distance to the long reference in units of the combined standard error."""
    md = angle(torch.tensor(stats["mean_direction"]), torch.tensor(ref["mean_direction"])).item()
    z = lambda k: (stats[k + "_deg"] - ref[k + "_deg"]) / math.hypot(stats[k + "_se_deg"],
                                                                     ref[k + "_se_deg"])
    return {"mean_direction": math.degrees(md) / math.hypot(stats["mean_direction_se_deg"],
                                                            ref["mean_direction_se_deg"]),
            "angle_to_source": z("angle_to_source"), "r90": z("r90")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/icecube.yaml")
    ap.add_argument("--long-seeds", type=int, nargs="*", default=[],
                    help="seeds of the long reference runs (--reference-only), pooled")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = run_dir(args.config, cfg)
    reports = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, "seed*", "report.json")))]
    if not reports:
        raise SystemExit(f"no report under {root}: run icecube_experiment.py first")

    long = None
    if args.long_seeds:
        states = torch.cat([torch.load(os.path.join(root, f"seed{s}", "reference.pt"),
                                       weights_only=False)["states"] for s in args.long_seeds])
        src = torch.load(os.path.join(root, "dataset.pt"), weights_only=False)["source"]
        long = localization(states, src)
        long["n"] = len(states)
        for rep in reports:
            for method in ("hmc", "surrogate"):
                rep[method]["deviation_in_se"] = deviations(rep[method], long)

    seeds = [r["seed"] for r in reports]
    print(f"ns = {reports[0]['ns']}, M = {reports[0]['M']}, N = {reports[0]['N']}, "
          f"L = {reports[0]['L']}, epochs = {reports[0]['epochs']}, seeds {seeds}")
    if long:
        print(f"long reference ({long['n']} states): angle to source "
              f"{long['angle_to_source_deg']:.4f} +- {long['angle_to_source_se_deg']:.4f} deg, "
              f"90% radius {long['r90_deg']:.4f} +- {long['r90_se_deg']:.4f} deg")
    head = "".join(f"{'seed ' + str(s):>12s}" for s in seeds)
    for method in ("hmc", "surrogate"):
        print(f"\n--- {method}\n{'':30s}{head}{'mean +- std':>22s}")
        for label, key, fmt in ROWS:
            v = [r[method][key] for r in reports]
            print(f"{label:30s}" + "".join(f"{fmt.format(x):>12s}" for x in v)
                  + f"{np.mean(v):>13.4f} +- {np.std(v, ddof=1):.4f}")
        if long:
            for k in ("mean_direction", "angle_to_source", "r90"):
                v = [r[method]["deviation_in_se"][k] for r in reports]
                print(f"{'deviation, ' + k.replace('_', ' ') + ' (SE)':30s}"
                      + "".join(f"{x:>12.2f}" for x in v))
    print("\n--- surrogate")
    extra = [("speed-up, evaluations of E", lambda r: r["speedup"]["true_evaluations"]),
             ("speed-up, wall time", lambda r: r["speedup"]["wall_time"]),
             ("speed-up, with training", lambda r: r["speedup"]["wall_time_with_training"]),
             ("training + duration (s)", lambda r: r["surrogate"]["training_time"]),
             ("slope of -log U on E", lambda r: r["slope_minus_logU_on_E"]),
             ("corr. of -log U and E", lambda r: r["corr_minus_logU_on_E"])]
    for label, get in extra:
        v = [get(r) for r in reports]
        print(f"{label:30s}" + "".join(f"{x:>12.3f}" for x in v)
              + f"{np.mean(v):>13.3f} +- {np.std(v, ddof=1):.3f}")

    json.dump({"long_reference": long, "reports": reports},
              open(os.path.join(root, "summary.json"), "w"), indent=2)

    # LaTeX table: mean +- std over the seeds
    ms = lambda vals, f: f"${f.format(np.mean(vals))} \\pm {f.format(np.std(vals, ddof=1))}$"
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r" & HMC & Surrogate HMC \\", r"\midrule"]
    for label, key, fmt in ROWS:
        lines.append(f"{label} & " + " & ".join(ms([r[m][key] for r in reports], fmt)
                                                  for m in ("hmc", "surrogate")) + r" \\")
    lines.append(r"\midrule")
    for label, key, f in [("speed-up, evaluations of $E$", "true_evaluations", "{:.1f}"),
                          ("speed-up, wall time", "wall_time", "{:.1f}"),
                          ("speed-up, with training", "wall_time_with_training", "{:.1f}")]:
        lines.append(f"{label} & -- & " + ms([r["speedup"][key] for r in reports], f) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs("figures", exist_ok=True)
    with open("figures/icecube_table.tex", "w") as f:
        f.write("\n".join(lines).replace("%", r"\%") + "\n")
    print(f"\nsaved {root}/summary.json and figures/icecube_table.tex")


if __name__ == "__main__":
    main()

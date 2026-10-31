"""Evaluate the trained RNHF and RCNF of a config: sample figures, learned
potential, matched-potential scores of Section 4.3 and metrics of Table 1.
Outputs go to results/<run>/ (samples.png, potential.png,
matched_potential.png, metrics.json).

Usage:
    python evaluate.py --config configs/s2.yaml
"""

import argparse
import json
import os

import torch
import yaml

# Hyperbolic ambient coordinates grow like cosh(r): float64 avoids the
# cancellation of Minkowski products at large radii.
torch.set_default_dtype(torch.float64)

from geometry import make_manifold
from metrics import compute_metrics, kl_knn_suite
from models import build_model
from plotting import sample_figure, potential_figure, matched_potential_figure
from runs import run_dir
from targets import make_base, make_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-kl", action="store_true", help="skip the KL estimate")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    space = cfg["space"]
    out_dir = run_dir(args.config, cfg)
    manifold = make_manifold(space)
    base, target = make_base(cfg, manifold), make_target(cfg, manifold)

    # the training set of train.py (same seed, same generation)
    torch.manual_seed(cfg["seed"])
    data = target.sample(cfg["train"]["n_samples"])

    models = {}
    for kind in ["rnhf", "rcnf"]:
        m = build_model(kind, cfg, manifold, base)
        m.load_state_dict(torch.load(os.path.join(out_dir, f"{kind}.pt"), weights_only=True))
        models[kind] = m.eval()

    n = cfg["eval"]["n_samples"]
    torch.manual_seed(cfg["seed"] + 1)
    samples = {"truth": data[torch.randperm(len(data))[:n]],
               "RNHF": models["rnhf"].sample(n),
               "RCNF": models["rcnf"].sample(n)}

    rnhf = models["rnhf"]
    marks = torch.tensor(cfg["target"]["mus"]) if space == "s2" else None
    sample_figure(space, samples, os.path.join(out_dir, "samples.png"))
    potential_figure(space, rnhf.potential, os.path.join(out_dir, "potential.png"),
                     sigma_p2=(rnhf.sigma_p ** 2).item(), marks=marks)
    scores = matched_potential_figure(space, rnhf.potential, target,
                                      os.path.join(out_dir, "matched_potential.png"))
    print(f"[free energy] correlation of log U and log rho_1 {scores['corr']:.3f}, "
          f"slope {scores['slope']:.3f}   [learned sigma_p] {rnhf.sigma_p.item():.4f}")

    results = {kind: compute_metrics(manifold, samples["truth"], samples[name])
               for kind, name in [("rnhf", "RNHF"), ("rcnf", "RCNF")]}
    results["matched_potential"] = scores
    results["sigma_p"] = rnhf.sigma_p.item()
    if not args.no_kl:      # sample-only estimator, on fresh and larger draws
        kl = kl_knn_suite(manifold, target.sample,
                          {"rnhf": models["rnhf"].sample, "rcnf": models["rcnf"].sample},
                          seed=cfg["seed"] + 2)
        for kind in ("rnhf", "rcnf"):
            results[kind]["kl"] = kl[kind]["mean"]

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== {space} ===\n{'metric':<12}{'RNHF':>12}{'RCNF':>12}")
    for k in results["rnhf"]:
        print(f"{k:<12}{results['rnhf'][k]:>12.5f}{results['rcnf'][k]:>12.5f}")
    print(f"saved {out_dir}/metrics.json and figures")


if __name__ == "__main__":
    main()

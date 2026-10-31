"""Section 4.4: localizing a neutrino source, recycling HMC samples into a
surrogate sampler with RNHF.

Stages, saved under results/<run>/ and reused when the script is run again:

  dataset.pt            scrambled events, injected signal events, background
                        density and maximum of the likelihood (shared by all seeds)
  seed<s>/reference.pt  geodesic HMC with the true gradients from the maximum of
                        the likelihood: burn-in, then M states. The first N states
                        train RNHF, states N..M-1 are the reference second phase
  seed<s>/rnhf.pt       RNHF trained for E epochs on the first N states
  seed<s>/surrogate.pt  surrogate HMC with E_hat = -log U from the N-th state,
                        M - N transitions
  seed<s>/report.json   metrics of Section 4.4 for both methods

Usage:
    python icecube_experiment.py --seed 0
    python icecube_experiment.py --seed 100 --reference-only   # long reference runs
    python icecube_experiment.py --seed 101 --reference-only
    python icecube_summary.py --long-seeds 100 101
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

from geometry import Sphere2
from hmc import hmc, mean_laplacian, duration, autocorr_time
from icecube import (load_events, scramble, inject, background_logpdf, radec_to_xyz, angle,
                     mean_direction, PointSourceEnergy, maximize_likelihood, LocalChart,
                     WrappedGaussianS2)
from models import build_model
from runs import run_dir

FRAME = Sphere2().frame


# ------------------------------------------------------------------ dataset

def make_dataset(cfg, path):
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d, src = cfg["data"], cfg["source"]
    ra, dec, sigma = load_events(d["path"])
    sigma = sigma.clamp(min=math.radians(0.2))                          # floor of the release
    ra = scramble(ra, d["seed"])
    n_src = radec_to_xyz(math.radians(src["ra_deg"]), math.radians(src["dec_deg"]))
    xs, ss = inject(n_src, src["ns"], dec, sigma, seed=d["seed"] + 1)
    x, sigma = torch.cat([radec_to_xyz(ra, dec), xs]), torch.cat([sigma, ss])
    log_b = background_logpdf(torch.asin(x[:, 2]))
    energy = PointSourceEnergy(x, sigma, log_b, src["ns"])
    t0 = time.perf_counter()
    mle = maximize_likelihood(energy)
    print(f"[data] maximum of the likelihood at {math.degrees(angle(mle, n_src)):.3f} deg "
          f"from the source ({time.perf_counter() - t0:.0f} s)")
    ds = {"x": x, "sigma": sigma, "log_b": log_b, "ns": src["ns"], "source": n_src, "mle": mle}
    torch.save(ds, path)
    return ds


# ------------------------------------------------------------------ exact HMC

def reference_chain(cfg, energy, start, seed, n_samples):
    """Burn-in from `start`, re-estimating the duration (20) on each block, then
    n_samples states of geodesic HMC with the true gradients."""
    h = cfg["hmc"]
    ell = duration(mean_laplacian(energy.E, start[None], FRAME))
    q = start
    for b in range(h["n_burn"] // h["adapt_every"]):
        r = hmc(q, h["adapt_every"], ell, h["n_steps"], energy.grad, energy.E,
                seed=1000 * seed + b, label=f"burn-in {b + 1}")
        q = r["states"][-1]
        ell = duration(mean_laplacian(energy.E, r["states"][::5], FRAME))
    r = hmc(q, n_samples, ell, h["n_steps"], energy.grad, energy.E,
            seed=1000 * seed + 999, label="reference HMC")
    r["ell"] = ell
    return r


# ------------------------------------------------------------------ RNHF

def make_rnhf(cfg, data):
    """RNHF on a patch of S^2 around the training states: networks fed with
    tangent coordinates divided by the spread a of the states, a wrapped
    Gaussian base `base_factor` times broader than them, and an initial sigma_p
    (and encoder scale) equal to the quarter-turn clock (pi / 2) a / sqrt(2)."""
    c = mean_direction(data)
    a = angle(data, c[None]).pow(2).mean().sqrt().item()                # rms angular spread
    M = LocalChart(c, a)
    base = WrappedGaussianS2(M, c, cfg["base_factor"] * a / math.sqrt(2))
    sigma_p = 0.5 * math.pi * a / math.sqrt(2)
    model_cfg = dict(cfg["model"], sigma_p=sigma_p, momentum_scale=sigma_p)
    return build_model("rnhf", {"model": model_cfg}, M, base)


def train_rnhf(model, data, cfg, seed):
    tr = cfg["train"]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    model.train()
    for epoch in range(1, tr["epochs"] + 1):
        perm, total = torch.randperm(len(data)), 0.0
        for s in range(0, len(data), tr["batch"]):
            batch = data[perm[s:s + tr["batch"]]]
            loss = model.loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if epoch == 1 or epoch % 20 == 0:
            print(f"[rnhf] epoch {epoch}/{tr['epochs']}  loss {total / len(data):.4f}")
    model.eval()
    return model


# ------------------------------------------------------------------ metrics

def localization(X, source, n_batches=20):
    """Posterior mean direction, its angle to the source and the radius of the
    90% credible region around it, with batch-means standard errors."""
    def stats(Y):
        m = mean_direction(Y)
        return m, angle(m, source).item(), torch.quantile(angle(Y, m[None]), 0.9).item()

    m, d, r90 = stats(X)
    k = len(X) // n_batches
    B = [stats(X[i * k:(i + 1) * k]) for i in range(n_batches)]
    se = lambda v: math.degrees(np.std(v, ddof=1) / math.sqrt(n_batches))
    ra = math.degrees(math.atan2(m[1], m[0])) % 360
    return {"mean_ra_deg": ra, "mean_dec_deg": math.degrees(math.asin(m[2])),
            "mean_direction": m.tolist(),
            # rms angular error of the mean direction over its two tangent coordinates
            "mean_direction_se_deg": math.degrees(math.sqrt(
                sum(angle(b[0], m).item() ** 2 for b in B) / (n_batches * (n_batches - 1)))),
            "angle_to_source_deg": math.degrees(d), "angle_to_source_se_deg": se([b[1] for b in B]),
            "r90_deg": math.degrees(r90), "r90_se_deg": se([b[2] for b in B])}


def chain_metrics(states, E, accept, source, n_true_evals, wall_time):
    tau = autocorr_time(E.numpy())
    ess = len(E) / tau
    return {**localization(states, source), "acceptance": accept.double().mean().item(),
            "tau_E": tau, "ess": ess, "true_evaluations": n_true_evals, "wall_time": wall_time,
            "true_evaluations_per_ess": n_true_evals / ess, "wall_time_per_ess": wall_time / ess}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/icecube.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=None, help="HMC states after burn-in")
    ap.add_argument("--reference-only", action="store_true",
                    help="only run exact HMC (for the long reference run)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    M_states = args.M or cfg["hmc"]["n_samples"]
    N, L = cfg["train"]["n_train"], cfg["hmc"]["n_steps"]
    root = run_dir(args.config, cfg, create=True)
    out = os.path.join(root, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)

    ds = make_dataset(cfg, os.path.join(root, "dataset.pt"))
    energy = PointSourceEnergy(ds["x"], ds["sigma"], ds["log_b"], ds["ns"])

    # 1. exact geodesic HMC
    path = os.path.join(out, "reference.pt")
    if not os.path.exists(path):
        torch.save(reference_chain(cfg, energy, ds["mle"], args.seed, M_states), path)
    ref = torch.load(path, weights_only=False)
    if len(ref["states"]) != M_states:
        raise SystemExit(f"{path} has {len(ref['states'])} states, not {M_states}")
    if args.reference_only:
        return

    # 2. RNHF on the first N states (never overwritten)
    data = ref["states"][:N]
    model = make_rnhf(cfg, data)
    path = os.path.join(out, "rnhf.pt")
    t0 = time.perf_counter()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        info = json.load(open(os.path.join(out, "rnhf.json")))
    else:
        train_rnhf(model, data, cfg, args.seed)
        info = {"train_time": time.perf_counter() - t0}
        torch.save(model.state_dict(), path)

    # 3. surrogate HMC with E_hat = -log U (19) and the duration (20) of E_hat
    log_u = model.potential.log_u
    e_hat = lambda q: -log_u(q.reshape(-1, 3)).reshape(q.shape[:-1])

    def grad_hat(q):
        with torch.enable_grad():
            qq = q.detach().requires_grad_(True)
            g, = torch.autograd.grad(e_hat(qq), qq)
        return g - (g @ q) * q

    t0 = time.perf_counter()
    ell = duration(mean_laplacian(e_hat, data, FRAME))
    info["duration_time"] = time.perf_counter() - t0
    json.dump(info, open(os.path.join(out, "rnhf.json"), "w"), indent=2)

    path = os.path.join(out, "surrogate.pt")
    if not os.path.exists(path):
        sur = hmc(ref["states"][N - 1], M_states - N, ell, L, grad_hat, energy.E,
                  seed=1000 * args.seed + 500, label="surrogate HMC")
        sur["ell"] = ell
        torch.save(sur, path)
    sur = torch.load(path, weights_only=False)

    # 4. metrics of Section 4.4 on the second phase (M - N samples per method)
    n2 = M_states - N
    r = chain_metrics(ref["states"][N:], ref["E"][N:], ref["accept"][N:], ds["source"],
                      n_true_evals=(L + 1) * n2, wall_time=ref["time"] * n2 / M_states)
    s = chain_metrics(sur["states"], sur["E"], sur["accept"], ds["source"],
                      n_true_evals=n2, wall_time=sur["time"])
    setup_time = info["train_time"] + info["duration_time"]
    s["training_time"] = setup_time
    s["wall_time_per_ess_with_training"] = (sur["time"] + setup_time) / s["ess"]
    with torch.no_grad():
        u, e = -log_u(data), energy.E(data)
    du, de = u - u.mean(), e - e.mean()
    report = {
        "seed": args.seed, "ns": ds["ns"], "M": M_states, "N": N, "L": L,
        "epochs": cfg["train"]["epochs"],
        "mle_angle_to_source_deg": math.degrees(angle(ds["mle"], ds["source"]).item()),
        "hmc": dict(r, duration=ref["ell"]), "surrogate": dict(s, duration=sur["ell"]),
        "speedup": {"true_evaluations": r["true_evaluations_per_ess"] / s["true_evaluations_per_ess"],
                    "wall_time": r["wall_time_per_ess"] / s["wall_time_per_ess"],
                    "wall_time_with_training": r["wall_time_per_ess"]
                    / s["wall_time_per_ess_with_training"]},
        # regression of -log U on E over the training states: depth and shape
        "slope_minus_logU_on_E": ((du * de).sum() / (de * de).sum()).item(),
        "corr_minus_logU_on_E": ((du * de).sum() / (du.norm() * de.norm())).item(),
    }
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

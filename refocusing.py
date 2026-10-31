"""Figure 6: transport with the free energy V = -sigma_p^2 log rho_1 as a
function of the momentum scale sigma_p.

For each sigma_p on a grid, positions are drawn from the base of each config,
momenta from N_g(0, sigma_p^2 I), the geodesic leapfrog integrates the flow of
the free energy for T = 1, and the k-NN estimate of KL(rho_1 || output) is
recorded. The grid is refined around the minimum. The red line marks the
sigma_p learned by the trained RNHF (results/<space>/rnhf.pt).

Usage:
    python refocusing.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

from geometry import make_manifold
from metrics import kl_knn
from models import build_model
from targets import make_base, make_target

SPACES = [("r2", r"$\mathbb{R}^2$"), ("h2", r"$\mathbb{H}^2$"), ("s2", r"$\mathbb{S}^2$")]


def free_energy_flow(manifold, base, target, sigma_p, n_steps, n_samples, T=1.0, seed=0):
    """Output positions of one pass of the flow of V = -sigma_p^2 log rho_1."""
    def force(q):                                         # Riemannian gradient of V
        q = q.detach().requires_grad_(True)
        g = torch.autograd.grad(target.log_prob(q).sum(), q)[0]
        return -sigma_p ** 2 * manifold.egrad_to_rgrad(q, g).detach()

    torch.manual_seed(seed)
    q = base.sample(n_samples)
    p = manifold.tangent_from_coords(q, sigma_p * torch.randn(n_samples, manifold.dim))
    dt = T / n_steps
    for _ in range(n_steps):
        p = p - 0.5 * dt * force(q)
        q_new = manifold.projx(manifold.exp(q, dt * p))
        p = manifold.proj(q_new, manifold.transport(q, q_new, p))
        q = q_new
        p = p - 0.5 * dt * force(q)
    return q


def kl_curve(space, sigmas, n_steps, n_samples, seed=99):
    cfg = yaml.safe_load(open(f"configs/{space}.yaml"))
    M = make_manifold(space)
    base, target = make_base(cfg, M), make_target(cfg, M)
    torch.manual_seed(seed)
    reference = target.sample(n_samples)
    return np.array([kl_knn(M, reference, free_energy_flow(M, base, target, float(s),
                                                            n_steps, n_samples))
                     for s in sigmas])


def learned_sigma_p(space):
    cfg = yaml.safe_load(open(f"configs/{space}.yaml"))
    M = make_manifold(space)
    model = build_model("rnhf", cfg, M, make_base(cfg, M))
    model.load_state_dict(torch.load(f"results/{space}/rnhf.pt", weights_only=True))
    return model.sigma_p.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma-min", type=float, default=0.1)
    ap.add_argument("--sigma-max", type=float, default=2.0)
    ap.add_argument("--n-grid", type=int, default=26)
    ap.add_argument("--refine", type=int, default=5, help="extra points around the minimum")
    ap.add_argument("--n-steps", type=int, default=30, help="leapfrog steps over T = 1")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--out", default="figures/refocusing_sigma_p.png")
    args = ap.parse_args()

    grid = np.linspace(args.sigma_min, args.sigma_max, args.n_grid)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, (space, label) in zip(axes, SPACES):
        kl = kl_curve(space, grid, args.n_steps, args.n_samples)
        g = grid
        if args.refine:
            centre = grid[np.argmin(kl)]
            extra = centre + (grid[1] - grid[0]) * np.linspace(-0.75, 0.75, args.refine + 2)[1:-1]
            g = np.concatenate([grid, extra])
            kl = np.concatenate([kl, kl_curve(space, extra, args.n_steps, args.n_samples)])
            order = np.argsort(g)
            g, kl = g[order], kl[order]
        kl = np.clip(kl, 0.0, None)
        s_learned = learned_sigma_p(space)
        ax.plot(g, kl, color="tab:blue", lw=2.2)
        ax.axvline(g[np.argmin(kl)], color="tab:blue", ls=":", lw=1.4, alpha=0.7)
        ax.axvline(s_learned, color="crimson", ls="--", lw=1.8,
                   label=fr"learned $\sigma_p={s_learned:.2f}$")
        ax.scatter([s_learned], [np.interp(s_learned, g, kl)], color="crimson", s=55, zorder=5)
        ax.set_title(label, fontsize=15)
        ax.set_xlabel(r"momentum scale $\sigma_p$", fontsize=12)
        ax.set_xlim(g.min(), g.max()); ax.set_ylim(0.0, 1.8)
        ax.grid(alpha=0.25); ax.legend(fontsize=11, loc="upper right")
        print(f"{space}: optimum sigma_p = {g[np.argmin(kl)]:.2f}, learned = {s_learned:.2f}")
    axes[0].set_ylabel(r"$\widehat{\mathrm{KL}}(\rho_1\,\|\,p_{\sigma_p})$", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

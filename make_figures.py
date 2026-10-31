"""Paper figures, rendered from the saved results.

Usage:
    python make_figures.py samples      # Figure 3, from results/{r2,h2,s2}
    python make_figures.py lens         # Figure 4, from results/r2_gauss
    python make_figures.py potentials   # Figure 5, from results/{r2,h2,s2}
    python make_figures.py icecube      # Section 4.4, from results/icecube
    python make_figures.py icecube RUN  # same, from results/RUN
    python make_figures.py all

Figures are written to figures/. Figure 6 has its own script, refocusing.py,
since it runs the flow of the free energy rather than rendering trained models.
"""

import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.gridspec import GridSpec

from geometry import make_manifold
from models import build_model
from plotting import sphere_grid, poincare_to_hyperboloid
from targets import make_base, make_target

torch.set_default_dtype(torch.float64)

SPACES = [("r2", r"$\mathbb{R}^2$"), ("h2", r"$\mathbb{H}^2$"), ("s2", r"$\mathbb{S}^2$")]
CMAP = "viridis"
os.makedirs("figures", exist_ok=True)


def load(run):
    """Config, manifold, base, target and trained models of results/<run>."""
    cfg = yaml.safe_load(open(f"configs/{run}.yaml"))
    M = make_manifold(cfg["space"])
    base, target = make_base(cfg, M), make_target(cfg, M)
    models = {}
    for name in ("rnhf", "rcnf"):
        path = f"results/{run}/{name}.pt"
        if os.path.exists(path):
            m = build_model(name, cfg, M, base)
            m.load_state_dict(torch.load(path, weights_only=True))
            models[name] = m.eval()
    return cfg, M, base, target, models


def save(fig, name):
    fig.savefig(f"figures/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figures/{name}")


# ------------------------------------------------------------------ Figure 3

def geodesic_kde(M, samples, grid, bw):
    """Gaussian kernel density estimate with geodesic distances (unnormalized)."""
    dens = torch.zeros(len(grid))
    for a in range(0, len(grid), 2000):
        d = M.dist(grid[a:a + 2000].unsqueeze(1), samples.unsqueeze(0))
        dens[a:a + 2000] = torch.exp(-0.5 * (d / bw) ** 2).mean(1)
    return dens


def fig_samples(n=4000, res=120):
    torch.manual_seed(0)
    cols = ["truth", "RNHF", "RCNF"]
    fig = plt.figure(figsize=(10.8, 10.5))
    gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.06, hspace=0.06)
    for i, (sp, label) in enumerate(SPACES):
        cfg, M, base, target, mdl = load(sp)
        with torch.no_grad():
            S = {"truth": target.sample(n), "RNHF": mdl["rnhf"].sample(n), "RCNF": mdl["rcnf"].sample(n)}
        inside = None
        if sp == "r2":
            allS = torch.cat(list(S.values()))
            lim = 1.1 * max(allS[:, 0].abs().quantile(0.996), allS[:, 1].abs().quantile(0.996)).item()
            u = torch.linspace(-lim, lim, res)
            gx, gy = torch.meshgrid(u, u, indexing="xy")
            grid, bw, extent = torch.stack([gx, gy], -1).reshape(-1, 2), 0.35, (-lim, lim, -lim, lim)
        elif sp == "h2":
            u = torch.linspace(-1.0, 1.0, res)
            gx, gy = torch.meshgrid(u, u, indexing="xy")
            inside = (gx ** 2 + gy ** 2).reshape(-1) < 0.995
            r = (gx ** 2 + gy ** 2).clamp(max=0.999).sqrt() / (gx ** 2 + gy ** 2).sqrt().clamp(min=1e-12)
            grid, bw, extent = poincare_to_hyperboloid(gx * r, gy * r), 0.18, (-1, 1, -1, 1)
        else:
            XYZ, grid = sphere_grid(res)
            bw = 0.22
        dens = {k: geodesic_kde(M, S[k], grid, bw) for k in cols}
        vmax = max(d.max().item() for d in dens.values())
        norm = plt.Normalize(0.0, vmax)
        for j, col in enumerate(cols):
            if sp == "s2":
                ax = fig.add_subplot(gs[i, j], projection="3d")
                ax.plot_surface(*(c.numpy() for c in XYZ), rstride=1, cstride=1,
                                facecolors=plt.cm.viridis(norm(dens[col].reshape(res, res).numpy())),
                                shade=False, linewidth=0, antialiased=False)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
                ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-72); ax.set_axis_off()
                ax.text2D(-0.10, 0.5, label if j == 0 else "", transform=ax.transAxes,
                          fontsize=16, rotation=90, va="center", ha="center")
            else:
                d = dens[col].clone()
                if inside is not None:
                    d[~inside] = np.nan
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(d.reshape(res, res).numpy(), origin="lower", extent=extent,
                          aspect="equal", cmap=CMAP, norm=norm)
                if sp == "h2":
                    th = np.linspace(0, 2 * np.pi, 200)
                    ax.plot(np.cos(th), np.sin(th), color="gray", lw=0.7)
                    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.text(-0.10, 0.5, label, transform=ax.transAxes, fontsize=16,
                            rotation=90, va="center", ha="center")
            if i == 0:
                ax.set_title(col, fontsize=15)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=fig.add_subplot(gs[i, 3]))
    save(fig, "samples_all.png")


# ------------------------------------------------------------------ Figure 4

def fig_lens(n=2500, L=60, T=1.0):
    """Phase-space rotation learned on the Gaussian target of configs/r2_gauss.yaml,
    shown along the first coordinate, in normalized coordinates (q - m) / a and p / sigma_p."""
    cfg, M, base, target, mdl = load("r2_gauss")
    model, a = mdl["rnhf"], cfg["target"]["sigma"]
    sigma = model.sigma_p.item()
    tau, dt = sigma ** 2, T / L
    torch.manual_seed(3)
    q = base.sample(n)
    p = M.tangent_from_coords(q, sigma * torch.randn(n, M.dim))
    snaps = [(q, p)]
    for k in range(1, L + 1):                             # kick-drift-kick (Eq. 3)
        p = p - 0.5 * dt * model.potential.rgrad(q, tau).detach()
        q_new = M.projx(M.exp(q, dt * p))
        p = M.proj(q_new, M.transport(q, q_new, p))
        q = q_new
        p = p - 0.5 * dt * model.potential.rgrad(q, tau).detach()
        if k in (L // 2, L):
            snaps.append((q, p))
    xy = [((qq / a)[:, 0].numpy(), (M.coords_from_tangent(qq, pp) / sigma)[:, 0].numpy())
          for qq, pp in snaps]
    x0 = xy[0][0]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    titles = ["(a) Initial lift: broad position base",
              "(b) Intermediate phase: information exchange",
              "(c) Quarter period: target position marginal"]
    for ax, (x, y), title in zip(axes.flat[:3], xy, titles):
        ax.scatter(x, y, c=x0, cmap=CMAP, s=7, alpha=0.55, edgecolors="none")
        ax.axhline(0, color="steelblue", lw=0.7, alpha=0.6)
        ax.axvline(0, color="steelblue", lw=0.7, alpha=0.6)
        ax.set_xlim(-7.2, 7.2); ax.set_ylim(-7.2, 7.2); ax.set_aspect("equal", "box")
        ax.set_xlabel(r"normalized position $(q-m)/a$", fontsize=11)
        ax.set_ylabel(r"normalized momentum $p/\sigma_p$", fontsize=11)
        ax.set_title(title, fontsize=12)
    ax = axes.flat[3]
    xs, sb = np.linspace(-7, 7, 400), cfg["base"]["sigma"]
    ax.plot(xs, np.exp(-0.5 * (xs * a / sb) ** 2) / (math.sqrt(2 * math.pi) * sb / a),
            label=r"position base $\rho_0$", lw=2)
    ax.plot(xs, np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi), label=r"target $\rho_1$", lw=2)
    ax.hist(xy[2][0], bins=70, density=True, histtype="step", color="green", lw=1.6,
            label=r"RNHF position at $T_\star$")
    ax.set_xlim(-7, 7); ax.set_xlabel(r"normalized position $(q-m)/a$", fontsize=11)
    ax.set_ylabel("density", fontsize=11); ax.legend(fontsize=10)
    ax.set_title("(d) The broad base is forgotten in position", fontsize=12)
    fig.tight_layout()
    save(fig, "hamiltonian_lens_learned.png")
    print(f"  sigma_p = {sigma:.4f}, corr(x_T, x_0) = {np.corrcoef(xy[2][0], x0)[0, 1]:+.3f}")


# ------------------------------------------------------------------ Figure 5

@torch.no_grad()
def fig_potentials(res=140):
    fig = plt.figure(figsize=(15, 4.6))
    gs = GridSpec(1, 6, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05], wspace=0.30)
    cross = dict(marker="x", c="red", s=90, linewidths=2.2, zorder=10)
    for i, (sp, label) in enumerate(SPACES):
        cfg, M, base, target, mdl = load(sp)
        m = mdl["rnhf"]
        tau = (m.sigma_p ** 2).item()
        if sp == "s2":
            XYZ, Q = sphere_grid(res)
            V = (-tau * m.potential.log_u(Q)).reshape(res, res).numpy()
            norm = matplotlib.colors.Normalize(V.min(), V.max())
            ax = fig.add_subplot(gs[2 * i], projection="3d")
            ax.plot_surface(*(c.numpy() for c in XYZ), rstride=1, cstride=1,
                            facecolors=plt.cm.viridis(norm(V)), shade=False, linewidth=0,
                            antialiased=False)
            mus = 1.04 * torch.tensor(cfg["target"]["mus"])
            ax.scatter(mus[:, 0], mus[:, 1], mus[:, 2], marker="x", c="red", s=90,
                       linewidths=2.5, depthshade=False, zorder=10)
            ax.set_xlim(-0.72, 0.72); ax.set_ylim(-0.72, 0.72); ax.set_zlim(-0.72, 0.72)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-72); ax.set_axis_off()
            im = matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP)
        else:
            lim = 4.5 if sp == "r2" else 0.6
            u = torch.linspace(-lim, lim, res)
            gx, gy = torch.meshgrid(u, u, indexing="xy")
            if sp == "r2":
                Q = torch.stack([gx, gy], -1).reshape(-1, 2)
                modes = torch.tensor(cfg["target"]["means"])
            else:
                Q = poincare_to_hyperboloid(gx, gy)
                o = M.origin().expand(2, -1)
                mo = M.exp(o, M.tangent_from_coords(o, torch.tensor(cfg["target"]["centers"])))
                modes = mo[:, 1:] / (1 + mo[:, :1])             # Poincare disk
            V = -tau * m.potential.log_u(Q)
            if sp == "h2":
                V = torch.where((gx ** 2 + gy ** 2).reshape(-1) < lim ** 2, V, torch.nan)
            ax = fig.add_subplot(gs[2 * i])
            im = ax.imshow(V.reshape(res, res).numpy(), origin="lower",
                           extent=(-lim, lim, -lim, lim), aspect="equal", cmap=CMAP)
            ax.scatter(modes[:, 0], modes[:, 1], **cross)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=15)
        fig.colorbar(im, cax=fig.add_subplot(gs[2 * i + 1]))
    save(fig, "potentials_all.png")


# ------------------------------------------------------------------ Section 4.4

def fig_icecube(run="icecube", long_seeds=(100, 101), n_max=4000):
    """Section 4.4. (a) 50% and 90% regions (kernel density estimates) of the
    second-phase samples of exact and surrogate HMC, in a gnomonic chart around
    the posterior mean; grey dots are exact HMC samples. (b) Distribution of the
    angular distance to the posterior mean, with the long reference runs (seeds
    `long_seeds`, --reference-only) in grey. (c) Wall-clock cost of N_eff
    effective samples: c_H N_eff for exact HMC and T_0 + c_S N_eff for the
    surrogate, whose fixed cost T_0 is the training of RNHF and the computation
    of the duration; the lines cross at N* = T_0 / (c_H - c_S), computed from the
    costs averaged over the seeds (thin lines: individual seeds; grey band: the
    range of their crossings)."""
    from icecube import angle, mean_direction
    root = f"results/{run}"
    seeds = sorted(d for d in glob.glob(f"{root}/seed*")
                   if os.path.exists(f"{d}/report.json"))
    if not seeds:
        raise SystemExit(f"no finished run under {root}: run icecube_experiment.py first")
    ds = torch.load(f"{root}/dataset.pt", weights_only=False)
    chains = {"exact HMC": [], "surrogate HMC": []}
    reports = []
    for d in seeds:
        chains["exact HMC"].append(torch.load(f"{d}/reference.pt", weights_only=False))
        chains["surrogate HMC"].append(torch.load(f"{d}/surrogate.pt", weights_only=False))
        reports.append(json.load(open(f"{d}/report.json")))
    N = reports[0]["N"]
    second = {"exact HMC": [c["states"][N:] for c in chains["exact HMC"]],
              "surrogate HMC": [c["states"] for c in chains["surrogate HMC"]]}
    long = [torch.load(f"{root}/seed{s}/reference.pt", weights_only=False)["states"]
            for s in long_seeds if os.path.exists(f"{root}/seed{s}/reference.pt")]
    long = torch.cat(long) if long else None
    ref = long if long is not None else torch.cat(second["exact HMC"])
    c = mean_direction(ref)

    # gnomonic chart around c, in degrees, east to the left as on the sky
    north = torch.tensor([0.0, 0.0, 1.0])
    e_ra = torch.linalg.cross(north, c); e_ra = e_ra / e_ra.norm()
    e_dec = torch.linalg.cross(c, e_ra)
    chart = lambda X: np.degrees(torch.stack([-(X @ e_ra), X @ e_dec], -1).numpy()
                                 / (X @ c).numpy()[:, None])
    colors = {"exact HMC": "tab:blue", "surrogate HMC": "tab:orange"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    from scipy.stats import gaussian_kde
    P = chart(torch.cat(second["exact HMC"]))
    ax.scatter(P[::3, 0], P[::3, 1], s=2, alpha=0.25, color="0.6")
    lim = 1.6 * np.abs(P).max(0).max()
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 150), np.linspace(-lim, lim, 150))
    for name, ls in (("exact HMC", "-"), ("surrogate HMC", "--")):
        P = chart(torch.cat(second[name]))
        kde = gaussian_kde(P.T)
        levels = np.quantile(kde(P.T), [0.1, 0.5])        # contours of the 90% and 50% regions
        ax.contour(gx, gy, kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape),
                   levels=levels, colors=colors[name], linestyles=ls, linewidths=1.8)
        ax.plot([], [], color=colors[name], ls=ls, label=f"{name}, 50% and 90% regions")
    s = chart(ds["source"][None])[0]
    ax.plot(*s, marker="*", color="red", ms=12, ls="none", label="injected source")
    ax.set_xlim(-lim / 1.6 * 1.1, lim / 1.6 * 1.1); ax.set_ylim(-lim / 1.6 * 1.1, lim / 1.6 * 1.1)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel(r"$\Delta\alpha\cos\delta$ (deg)"); ax.set_ylabel(r"$\Delta\delta$ (deg)")
    ax.set_title("(a) posterior of the source direction")
    r90 = torch.quantile(angle(ref, c[None]), 0.9).item()

    ax = axes[1]
    bins = np.linspace(0, 2.2 * math.degrees(r90), 40)
    if long is not None:
        ax.hist(np.degrees(angle(long, c[None]).numpy()), bins, density=True, color="0.8",
                label="long reference")
    for name in ("exact HMC", "surrogate HMC"):
        ax.hist(np.degrees(angle(torch.cat(second[name]), c[None]).numpy()), bins,
                density=True, histtype="step", lw=1.8, color=colors[name], label=name)
    ax.set_xlabel("angle to the posterior mean (deg)"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.set_title("(b) radial distribution")

    ax = axes[2]
    T0 = np.array([r["surrogate"]["training_time"] for r in reports])
    cH = np.array([r["hmc"]["wall_time_per_ess"] for r in reports])
    cS = np.array([r["surrogate"]["wall_time_per_ess"] for r in reports])
    n = np.linspace(0, n_max, 400)
    for k in range(len(reports)):                         # individual seeds
        ax.plot(n, cH[k] * n / 60, color=colors["exact HMC"], lw=0.8, alpha=0.35)
        ax.plot(n, (T0[k] + cS[k] * n) / 60, color=colors["surrogate HMC"], lw=0.8, alpha=0.35)
    ax.plot(n, cH.mean() * n / 60, color=colors["exact HMC"], lw=2.2, label="exact HMC")
    ax.plot(n, (T0.mean() + cS.mean() * n) / 60, color=colors["surrogate HMC"], lw=2.2,
            label="surrogate HMC, training included")
    n_star = T0.mean() / (cH.mean() - cS.mean())
    n_seeds = T0 / (cH - cS)
    ax.axvspan(n_seeds.min(), n_seeds.max(), color="0.9", zorder=0, label="crossings of the seeds")
    ax.axvline(n_star, color="k", ls="--", lw=1)
    ax.annotate(f"$N^\\star \\approx {n_star:.0f}$", (n_star, 0.08 * cH.mean() * n_max / 60),
                xytext=(6, 0), textcoords="offset points", fontsize=11)
    ax.set_xlim(0, n_max); ax.set_ylim(0, None)
    ax.set_xlabel(r"effective samples $N_{\mathrm{eff}}$")
    ax.set_ylabel("wall-clock time (min)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(c) cost of $N_{\\mathrm{eff}}$ effective samples")
    fig.tight_layout()
    save(fig, "icecube_posterior.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    figures = {"samples": fig_samples, "lens": fig_lens, "potentials": fig_potentials,
               "icecube": fig_icecube}
    for name, make in figures.items():
        if what in (name, "all"):
            make(*sys.argv[2:])

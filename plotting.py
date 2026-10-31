"""Diagnostic figures of evaluate.py: samples, learned potential, and the
comparison of the learned landscape log U with the free energy log rho_1
(Section 4.3)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry import make_manifold


def sphere_grid(res):
    """Longitude-latitude grid of S^2: the coordinate arrays and the points (res^2, 3)."""
    lon = torch.linspace(-np.pi, np.pi, res)
    lat = torch.linspace(-np.pi / 2, np.pi / 2, res)
    glon, glat = torch.meshgrid(lon, lat, indexing="ij")
    X, Y, Z = torch.cos(glat) * torch.cos(glon), torch.cos(glat) * torch.sin(glon), torch.sin(glat)
    return (X, Y, Z), torch.stack([X, Y, Z], -1).reshape(-1, 3)


def poincare_to_hyperboloid(gx, gy):
    z2 = gx ** 2 + gy ** 2
    return torch.stack([(1 + z2) / (1 - z2), 2 * gx / (1 - z2), 2 * gy / (1 - z2)], -1).reshape(-1, 3)


def sphere_surface(ax, XYZ, values, title):
    norm = matplotlib.colors.Normalize(np.nanmin(values), np.nanmax(values))
    ax.plot_surface(*(c.numpy() for c in XYZ), rstride=1, cstride=1,
                    facecolors=plt.cm.viridis(norm(values)), shade=False, linewidth=0,
                    antialiased=False)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.set_title(title)
    return matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")


def sample_figure(space, samples, path):
    """Scatter plots of a dict name -> samples (ambient coordinates)."""
    kw = {"projection": "3d"} if space == "s2" else {}
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4), subplot_kw=kw)
    for ax, (name, q) in zip(axes, samples.items()):
        if space == "s2":
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=3, alpha=0.5)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        else:
            z = q if space == "r2" else q[:, 1:] / (1 + q[:, :1])        # Poincare disk
            ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.5)
            lim = 5 if space == "r2" else 1.05
            if space == "h2":
                ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="gray"))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


@torch.no_grad()
def potential_figure(space, potential, path, sigma_p2=1.0, res=100, marks=None):
    """Learned potential V = -sigma_p^2 log U; `marks` (S^2 only) are drawn as stars."""
    if space == "s2":
        XYZ, Q = sphere_grid(res)
        V = (-sigma_p2 * potential.log_u(Q)).reshape(res, res).numpy()
        fig = plt.figure(figsize=(5.5, 5))
        ax = fig.add_subplot(111, projection="3d")
        sm = sphere_surface(ax, XYZ, V, "V(q) = -sigma_p^2 log U(q)")
        if marks is not None:
            m = 1.02 * marks.numpy()
            ax.scatter(m[:, 0], m[:, 1], m[:, 2], color="red", marker="*", s=120, depthshade=False)
        ax.view_init(elev=25, azim=-45)
        fig.colorbar(sm, ax=ax, shrink=0.7)
    else:
        lim = 5 if space == "r2" else 0.999
        u = torch.linspace(-lim, lim, res)
        gx, gy = torch.meshgrid(u, u, indexing="xy")
        Q = torch.stack([gx, gy], -1).reshape(-1, 2) if space == "r2" else poincare_to_hyperboloid(gx, gy)
        V = -sigma_p2 * potential.log_u(Q)
        if space == "h2":
            V = torch.where((gx ** 2 + gy ** 2).reshape(-1) < 0.995, V, torch.nan)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(V.reshape(res, res), origin="lower", extent=(-lim, lim, -lim, lim), aspect="auto")
        ax.set_title("V(q) = -sigma_p^2 log U(q)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def free_energy_scores(logU, logrho):
    """Pearson correlation and regression slope of log U on log rho_1 (Section 4.3);
    both are insensitive to the additive constants."""
    a, b = logU - logU.mean(), logrho - logrho.mean()
    return {"corr": ((a * b).sum() / (a.norm() * b.norm())).item(),
            "slope": ((a * b).sum() / (b * b).sum()).item()}


@torch.no_grad()
def matched_potential_figure(space, potential, target, path, res=120):
    """Compare the learned landscape log U with log rho_1 on a grid of each space:
    [-5, 5]^2 on R^2; the longitude-latitude grid on S^2; on H^2, the Poincare
    disk restricted to |z|^2 < 0.95 and to 1.3 times the geodesic radius that
    contains 95% of the target mass (the far tails, where the volume factor
    dominates log rho_1, carry no mass). Returns the scores of free_energy_scores."""
    if space == "s2":
        XYZ, Q = sphere_grid(res)
        keep = torch.ones(len(Q), dtype=torch.bool)
    else:
        lim = 5 if space == "r2" else 0.985
        u = torch.linspace(-lim, lim, res)
        gx, gy = torch.meshgrid(u, u, indexing="xy")
        if space == "r2":
            Q = torch.stack([gx, gy], -1).reshape(-1, 2)
            keep = torch.ones(len(Q), dtype=torch.bool)
        else:
            Q = poincare_to_hyperboloid(gx, gy)
            M = make_manifold("h2")
            o = M.origin()
            r_target = M.dist(o, target.sample(20000)).quantile(0.95).item()
            keep = ((gx ** 2 + gy ** 2).reshape(-1) < 0.95) & (M.dist(o, Q) < 1.3 * r_target)
    logU, logrho = potential.log_u(Q), target.log_prob(Q)
    scores = free_energy_scores(logU[keep], logrho[keep])

    fields = [("learned log U", logU - logU[keep].mean()),
              ("target log rho_1", logrho - logrho[keep].mean())]
    fields.append(("difference (flat = free energy)", fields[0][1] - fields[1][1]))
    title = f"{space.upper()}: corr = {scores['corr']:.3f}, slope = {scores['slope']:.3f}"
    if space == "s2":
        fig = plt.figure(figsize=(15, 5))
        for i, (name, f) in enumerate(fields, 1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            sm = sphere_surface(ax, XYZ, f.reshape(res, res).numpy(), name)
            ax.view_init(elev=20, azim=-60)
            fig.colorbar(sm, ax=ax, shrink=0.6)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (name, f) in zip(axes, fields):
            im = ax.imshow(torch.where(keep, f, torch.nan).reshape(res, res), origin="lower",
                           extent=(-lim, lim, -lim, lim), aspect="auto")
            ax.set_title(name)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return scores

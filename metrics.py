"""Sample-based metrics of Table 1 (Appendix I).

X are target (training) samples and Y generated samples, in ambient
coordinates. Distances are geodesic: the MMD uses a Laplace kernel, the energy
distance and precision / recall use the distances themselves, and the moments
are taken in ambient coordinates. The Kullback-Leibler divergence is estimated
from samples alone with the k-nearest-neighbour estimator of Perez-Cruz (2008).
"""

import math

import torch


def pairwise(manifold, A, B):
    """Matrix of geodesic distances between the rows of A and B."""
    return manifold.dist(A.unsqueeze(1), B.unsqueeze(0))


@torch.no_grad()
def compute_metrics(manifold, X, Y, k=5):
    Dxx, Dyy, Dxy = pairwise(manifold, X, X), pairwise(manifold, Y, Y), pairwise(manifold, X, Y)
    n, m = len(X), len(Y)

    s = torch.cat([Dxx.flatten(), Dyy.flatten(), Dxy.flatten()]).median()   # kernel width
    K = lambda D: torch.exp(-D / s)
    Kxx, Kyy = K(Dxx), K(Dyy)
    mmd2 = ((Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1))        # unbiased estimate
            + (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1))
            - 2 * K(Dxy).mean()).clamp(min=0.0)

    r = Dxx.sort(dim=1).values[:, k].median()     # training radius (column 0 is the point itself)
    cx, cy = X - X.mean(0), Y - Y.mean(0)
    return {
        "mean_l2": (X.mean(0) - Y.mean(0)).norm().item(),
        "cov_fro": (cx.T @ cx / (n - 1) - cy.T @ cy / (m - 1)).norm().item(),
        "mmd2": mmd2.item(),
        "energy": (2 * Dxy.mean() - Dxx.mean() - Dyy.mean()).item(),
        "precision": (Dxy.min(dim=0).values <= r).double().mean().item(),
        "recall": (Dxy.min(dim=1).values <= r).double().mean().item(),
    }


# ------------------------------------------------- Kullback-Leibler divergence

def kth_nn_dist(manifold, A, B, k, exclude_self, chunk=256):
    """Geodesic distance from each point of A to its k-th nearest neighbour in B
    (among the other points when B is A itself)."""
    out = torch.empty(len(A))
    take = k + 1 if exclude_self else k
    for s in range(0, len(A), chunk):
        D = pairwise(manifold, A[s:s + chunk], B)
        out[s:s + chunk] = D.topk(take, dim=1, largest=False).values[:, -1]
    return out.clamp(min=1e-12)


@torch.no_grad()
def kl_knn(manifold, X, Y, k=1):
    """Perez-Cruz (2008) estimate of KL(P || Q) from X ~ P (n points) and
    Y ~ Q (m points) on a d-dimensional manifold:

        KL = (d / n) sum_i log(nu_k(x_i) / rho_k(x_i)) + log(m / (n - 1)),

    with rho_k(x_i) and nu_k(x_i) the distances from x_i to its k-th nearest
    neighbour among the other X and among Y."""
    n, m, d = len(X), len(Y), manifold.dim
    rho = kth_nn_dist(manifold, X, X, k, exclude_self=True)
    nu = kth_nn_dist(manifold, X, Y, k, exclude_self=False)
    return ((d / n) * torch.log(nu / rho).sum() + math.log(m / (n - 1))).item()


@torch.no_grad()
def kl_knn_suite(manifold, sample_target, samplers, n=4000, k=1, n_draws=3, seed=0):
    """KL(target || model) of each model, averaged over independent draws of
    n target and n model samples."""
    vals = {name: [] for name in samplers}
    for i in range(n_draws):
        torch.manual_seed(seed + i)
        X = sample_target(n)
        for name, sampler in samplers.items():
            vals[name].append(kl_knn(manifold, X, sampler(n), k=k))
    return {name: {"mean": sum(v) / len(v), "draws": v} for name, v in vals.items()}

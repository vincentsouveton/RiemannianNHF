"""Target distributions (bimodal, one per model space) and base distributions.

Each class samples and evaluates its log-density with respect to the Riemannian
volume. To add a target or a base, add a class and register it in make_target
or make_base.
"""

import math
import torch


# ---------------------------------------------------------------- targets

class GaussianMixture:
    """Mixture of isotropic Gaussians on R^d."""

    def __init__(self, means, sigma, weights=None):
        self.means = means          # (K, d)
        self.sigma = sigma
        if weights is None:
            weights = torch.full((len(means),), 1.0 / len(means))
        self.weights = torch.as_tensor(weights, dtype=means.dtype)

    def sample(self, n):
        k = torch.multinomial(self.weights, n, replacement=True)
        return self.means[k] + self.sigma * torch.randn(n, self.means.shape[1])

    def log_prob(self, q):
        """Log-density of the isotropic Gaussian mixture at q, with respect to the
        Lebesgue measure on R^d. Mixture weights may be unequal."""
        d = self.means.shape[1]
        # (..., K) squared distances to each mean
        diff = q.unsqueeze(-2) - self.means                      # (..., K, d)
        sq = (diff ** 2).sum(-1)                                 # (..., K)
        log_comp = -0.5 * sq / self.sigma**2 \
                   - 0.5 * d * math.log(2 * math.pi * self.sigma**2)
        logw = torch.log(self.weights)
        return torch.logsumexp(log_comp + logw, dim=-1)


class VMFMixture:
    """Mixture of von Mises-Fisher distributions on S^2."""

    def __init__(self, mus, kappa, weights=None):
        self.mus = mus              # (K, 3), unit vectors
        self.kappa = kappa
        if weights is None:
            weights = torch.full((len(mus),), 1.0 / len(mus))
        self.weights = torch.as_tensor(weights, dtype=mus.dtype)

    def sample(self, n):
        k = torch.multinomial(self.weights, n, replacement=True)
        mu = self.mus[k]
        # sample around the north pole: w by inverse CDF, uniform angle in the plane
        u = torch.rand(n)
        w = 1.0 + torch.log(u + (1.0 - u) * math.exp(-2.0 * self.kappa)) / self.kappa
        phi = 2.0 * math.pi * torch.rand(n)
        s = (1.0 - w**2).clamp(min=0.0).sqrt()
        x = torch.stack([s * torch.cos(phi), s * torch.sin(phi), w], dim=-1)
        # Householder reflection mapping the north pole e3 to mu
        e3 = torch.tensor([0.0, 0.0, 1.0]).expand_as(mu)
        v = e3 - mu
        nrm2 = (v**2).sum(-1, keepdim=True).clamp(min=1e-12)
        return x - 2.0 * (x * v).sum(-1, keepdim=True) / nrm2 * v

    def log_prob(self, q):
        """Log-density of the vMF mixture on S^2 w.r.t. the surface measure.
        A single vMF(mu, kappa) has density
        kappa / (4 pi sinh kappa) * exp(kappa <mu, q>)."""
        kappa = self.kappa
        # log C_2(kappa) = log kappa - log(4 pi sinh kappa), with a stable log sinh
        log_sinh = kappa + math.log1p(-math.exp(-2 * kappa)) - math.log(2.0)
        log_norm = math.log(kappa) - math.log(4 * math.pi) - log_sinh
        dots = q.unsqueeze(-2) * self.mus                        # (..., K, 3)
        log_comp = kappa * dots.sum(-1) + log_norm               # (..., K)
        logw = torch.log(self.weights)
        return torch.logsumexp(log_comp + logw, dim=-1)


class WrappedGaussianMixture:
    """Mixture of wrapped Gaussians on H^2: Gaussians in the tangent space at
    the origin, pushed through the exponential map."""

    def __init__(self, manifold, centers, sigma, weights=None):
        self.M = manifold
        self.centers = centers      # (K, 2), tangent coordinates at the origin
        self.sigma = sigma
        if weights is None:
            weights = torch.full((len(centers),), 1.0 / len(centers))
        self.weights = torch.as_tensor(weights, dtype=centers.dtype)

    def sample(self, n):
        k = torch.multinomial(self.weights, n, replacement=True)
        x = self.centers[k] + self.sigma * torch.randn(n, 2)
        o = self.M.origin().expand(n, -1)
        return self.M.exp(o, self.M.tangent_from_coords(o, x))

    def log_prob(self, q):
        """Log-density of the wrapped-Gaussian mixture w.r.t. the Riemannian
        volume on H^2. Each component is N(x; center, sigma^2 I) in the tangent
        coordinates x = log_o(q) at the origin, divided by the exp-map Jacobian
        J(r) = sinh(r)/r with r = |x| = d_g(o, q)."""
        o = self.M.origin().to(q.device).expand(*q.shape[:-1], -1)
        u = self.M.log(o, q)                              # tangent vector at o
        x = self.M.coords_from_tangent(o, u)              # (..., 2) coordinates
        r = x.norm(dim=-1).clamp(min=1e-9)                # geodesic radius
        log_jac = torch.log(torch.sinh(r) / r)            # log J(r), d = 2
        d = self.M.dim
        diff = x.unsqueeze(-2) - self.centers             # (..., K, 2)
        sq = (diff ** 2).sum(-1)
        log_comp = -0.5 * sq / self.sigma**2 \
                   - 0.5 * d * math.log(2 * math.pi * self.sigma**2)
        logw = torch.log(self.weights)
        return torch.logsumexp(log_comp + logw, dim=-1) - log_jac


def make_target(cfg, manifold):
    c = cfg["target"]
    if c["kind"] == "gaussian_mixture":
        return GaussianMixture(torch.tensor(c["means"]), c["sigma"], c.get("weights"))
    if c["kind"] == "vmf_mixture":
        return VMFMixture(torch.tensor(c["mus"]), c["kappa"], c.get("weights"))
    if c["kind"] == "wrapped_mixture":
        return WrappedGaussianMixture(manifold, torch.tensor(c["centers"]), c["sigma"], c.get("weights"))
    raise ValueError(c["kind"])


# ---------------------------------------------------------------- bases

class GaussianBase:
    """Isotropic Gaussian at the origin of R^d."""

    def __init__(self, manifold, sigma):
        self.M = manifold
        self.sigma = sigma

    def sample(self, n):
        return self.sigma * torch.randn(n, self.M.dim)

    def log_prob(self, q):
        d = self.M.dim
        return -(q**2).sum(-1) / (2 * self.sigma**2) - 0.5 * d * math.log(2 * math.pi * self.sigma**2)


class UniformSphereBase:
    """Uniform distribution on S^2 (density 1 / 4 pi w.r.t. the area)."""

    def __init__(self, manifold):
        self.M = manifold

    def sample(self, n):
        q = torch.randn(n, 3)
        return q / q.norm(dim=-1, keepdim=True)

    def log_prob(self, q):
        return torch.full(q.shape[:-1], -math.log(4 * math.pi), device=q.device)


class WrappedGaussianBase:
    """Wrapped Gaussian at the origin of H^2. Its density with respect to the
    Riemannian volume is N(x; 0, sigma^2 I) / (sinh r / r), where x are the
    tangent coordinates of the point and r = |x| (Jacobian of exp on H^2)."""

    def __init__(self, manifold, sigma):
        self.M = manifold
        self.sigma = sigma

    def sample(self, n):
        x = self.sigma * torch.randn(n, 2)
        o = self.M.origin().expand(n, -1)
        return self.M.exp(o, self.M.tangent_from_coords(o, x))

    def log_prob(self, q):
        o = self.M.origin(q.device).expand_as(q)
        x = self.M.coords_from_tangent(o, self.M.log(o, q))
        r = ((x**2).sum(-1) + 1e-14).sqrt()
        log_gauss = -(x**2).sum(-1) / (2 * self.sigma**2) - math.log(2 * math.pi * self.sigma**2)
        # log(sinh r / r), numerically stable
        log_sinhc = torch.where(r < 1e-4, r**2 / 6.0,
                                r + torch.log1p(-torch.exp(-2 * r)) - math.log(2.0) - torch.log(r.clamp(min=1e-12)))
        return log_gauss - log_sinhc


def make_base(cfg, manifold):
    c = cfg["base"]
    if c["kind"] == "gaussian":
        return GaussianBase(manifold, c["sigma"])
    if c["kind"] == "uniform_sphere":
        return UniformSphereBase(manifold)
    if c["kind"] == "wrapped_gaussian":
        return WrappedGaussianBase(manifold, c["sigma"])
    raise ValueError(c["kind"])

"""Localization of a neutrino point source on the sky (Section 4.4): data,
energy and maximum of the likelihood.

Data: the public IceCube point-source sample 2008-2018 (IceCube Collaboration,
2021, doi:10.7910/DVN/VKL316). Each event has a reconstructed direction x_i on
S^2 and an angular uncertainty sigma_i >= 0.2 deg.

Injection test: the right ascensions are scrambled, which removes any real
source while preserving the acceptance of the detector, and ns signal events
are added around the source direction, with uncertainties drawn from those of
the data at the same declination.

Energy (Eq. 21): with f = ns / M, kappa_i = sigma_i^-2 and B the background
density, a function of the declination only,

    E(n) = - sum_i log[ f vMF(x_i; n, kappa_i) + (1 - f) B(x_i) ].

The vMF density on S^2 is c_i exp(kappa_i (n.x_i - 1)), c_i = kappa_i / (2 pi (1 - e^{-2 kappa_i})).
Dropping the constant - sum_i log((1 - f) B(x_i)), we use

    E(n) = - sum_i softplus(t_i(n)),   t_i(n) = log(f c_i / ((1 - f) B(x_i))) + kappa_i (n.x_i - 1),

whose ambient gradient is - sum_i sigmoid(t_i(n)) kappa_i x_i. Every evaluation
is a sum over all events.
"""

import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from geometry import Sphere2


# ------------------------------------------------------------------ geometry

def radec_to_xyz(ra, dec):
    """Equatorial angles (radians) to unit vectors of R^3."""
    ra, dec = torch.as_tensor(ra), torch.as_tensor(dec)
    return torch.stack([torch.cos(dec) * torch.cos(ra),
                        torch.cos(dec) * torch.sin(ra), torch.sin(dec)], -1)


def angle(a, b):
    """Angular distance between unit vectors (radians), accurate at small angles."""
    return 2 * torch.asin(((a - b).norm(dim=-1) / 2).clamp(max=1.0))


def mean_direction(X):
    m = X.mean(0)
    return m / m.norm()


# ------------------------------------------------------------------ data

def load_events(path):
    """RA, Dec and angular uncertainty (radians) of all events of the release,
    read from <path>/events/*.tab (or *.csv). Columns: MJD, log10(E/GeV),
    AngErr[deg], RA[deg], Dec[deg], Azimuth[deg], Zenith[deg]."""
    files = sorted(glob.glob(os.path.join(path, "events", "*.tab")) +
                   glob.glob(os.path.join(path, "events", "*.csv")))
    if not files:
        raise FileNotFoundError(f"no event file in {path}/events")
    rows = [np.loadtxt((line.replace('"', "") for line in open(f)), comments="#", ndmin=2)
            for f in files]
    a = np.radians(np.concatenate(rows)[:, 2:5])
    print(f"[data] {len(a)} events from {len(files)} files in {path}")
    return torch.tensor(a[:, 1]), torch.tensor(a[:, 2]), torch.tensor(a[:, 0])


def scramble(ra, seed):
    """Uniform right ascensions."""
    g = torch.Generator().manual_seed(seed)
    return 2 * math.pi * torch.rand(ra.shape, generator=g)


def inject(src, ns, dec, sigma, seed, band=0.05):
    """ns signal events around the source direction src: uncertainties drawn from
    the events with |sin(dec) - sin(dec_src)| < band, directions drawn from
    vMF(src, sigma^-2), the point-spread function of the likelihood."""
    g = torch.Generator().manual_seed(seed)
    pool = sigma[(torch.sin(dec) - src[2]).abs() < band]
    s = pool[torch.randint(len(pool), (ns,), generator=g)]
    kappa = s ** -2
    u = torch.rand(ns, generator=g)
    w = 1 + torch.log(u + (1 - u) * torch.exp(-2 * kappa)) / kappa       # cos(angle to src)
    phi = 2 * math.pi * torch.rand(ns, generator=g)
    e1, e2 = Sphere2().frame(src[None])[0]
    r = torch.sqrt((1 - w ** 2).clamp(min=0))
    x = w[:, None] * src + r[:, None] * (torch.cos(phi)[:, None] * e1 + torch.sin(phi)[:, None] * e2)
    return x / x.norm(dim=-1, keepdim=True), s


def background_logpdf(dec, n_bins=500):
    """log B at each event: B(x) = h(sin dec) / (2 pi), with h the histogram of
    sin(dec) over all events, smoothed by a Gaussian kernel of one bin."""
    from scipy.ndimage import gaussian_filter1d
    s = torch.sin(dec).numpy()
    h, edges = np.histogram(s, bins=n_bins, range=(-1, 1), density=True)
    h = gaussian_filter1d(h, 1.0, mode="nearest")
    h = h / (h.sum() * (edges[1] - edges[0]))
    hs = np.interp(s, 0.5 * (edges[1:] + edges[:-1]), h)
    return torch.tensor(np.log(np.maximum(hs, 1e-12) / (2 * math.pi)))


# ------------------------------------------------------------------ energy

class PointSourceEnergy:
    """Energy (21) of the source direction, up to an additive constant."""

    def __init__(self, x, sigma, log_b, ns):
        f = ns / x.shape[0]
        kappa = sigma ** -2
        log_c = torch.log(kappa) - math.log(2 * math.pi) - torch.log1p(-torch.exp(-2 * kappa))
        self.x, self.kappa = x, kappa
        self.a = math.log(f) - math.log1p(-f) + log_c - log_b - kappa    # t_i = a_i + kappa_i n.x_i

    def E(self, n):
        """Energy at one direction (3,) or a batch of directions (k, 3)."""
        batch = n.reshape(-1, 3)
        out = torch.cat([-F.softplus(self.a + self.kappa * (batch[i:i + 8] @ self.x.T)).sum(-1)
                         for i in range(0, batch.shape[0], 8)])
        return out if n.dim() > 1 else out[0]

    def grad(self, n):
        """Riemannian gradient of E at one direction n (3,)."""
        t = self.a + self.kappa * (self.x @ n)
        g = -(torch.sigmoid(t) * self.kappa) @ self.x
        return g - (g @ n) * n


def fibonacci_sphere(k):
    """k nearly uniform points on S^2."""
    i = torch.arange(k) + 0.5
    z = 1 - 2 * i / k
    phi = math.pi * (3 - math.sqrt(5)) * i
    r = torch.sqrt(1 - z ** 2)
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], -1)


def maximize_likelihood(energy, n_grid=20000, n_starts=10, iters=200, max_dec_deg=85.0):
    """Minimum of E over the declinations |dec| < max_dec_deg: E on a Fibonacci
    grid, then gradient descent with an adaptive step from the best grid points.
    The polar caps are left out: scrambling the right ascensions does not spread
    the events that the reconstruction piles up at the poles, so that the
    background model, uniform in right ascension, underestimates the density
    there by two orders of magnitude and creates spurious maxima."""
    G = fibonacci_sphere(n_grid)
    G = G[G[:, 2].abs() < math.sin(math.radians(max_dec_deg))]
    Eg = torch.cat([energy.E(G[i:i + 64]) for i in range(0, len(G), 64)])
    best = []
    for k in torch.argsort(Eg)[:n_starts]:
        n, e, step = G[k], Eg[k].item(), 1e-3
        for _ in range(iters):
            if step <= 1e-12:                                            # converged
                break
            g = energy.grad(n)
            while step > 1e-12:
                m = n - step * g / g.norm()
                m = m / m.norm()
                em = energy.E(m).item()
                if em < e:
                    n, e, step = m, em, 1.5 * step
                    break
                step /= 2
        best.append((e, n))
    return min(best, key=lambda b: b[0])[1]


# ------------------------------------------------------------------ RNHF on a small patch

class LocalChart(Sphere2):
    """S^2 whose network input is log_c(q) / s, the tangent coordinates at the
    centre c of the samples divided by their angular spread s. The geometry is
    that of S^2; only the input of the networks is rescaled, so that they see
    O(1) coordinates although the posterior is a fraction of a degree wide."""

    def __init__(self, center, scale):
        self.c, self.s = center, scale
        self.Ec = Sphere2.frame(self, center[None])[0]                   # (2, 3)

    def embed(self, q):
        return (self.log(self.c.expand_as(q), q) @ self.Ec.T) / self.s

    @property
    def embed_dim(self):
        return 2


class WrappedGaussianS2:
    """Wrapped Gaussian on S^2 at c with scale b: X ~ N(0, b^2 I) in the tangent
    plane at c, Q = exp_c(X), of density N(x; 0, b^2 I) / (sin r / r), r = |x| < pi."""

    def __init__(self, manifold, center, scale):
        self.M, self.c, self.b = manifold, center, scale
        self.Ec = Sphere2.frame(manifold, center[None])[0]

    def sample(self, n):
        return self.M.exp(self.c.expand(n, -1), self.b * torch.randn(n, 2) @ self.Ec)

    def log_prob(self, q):
        x = self.M.log(self.c.expand_as(q), q) @ self.Ec.T
        r2 = (x ** 2).sum(-1)
        r = r2.clamp(min=1e-30).sqrt()
        log_sinc = torch.where(r < 1e-4, -r2 / 6, torch.log(torch.sin(r) / r))
        return -r2 / (2 * self.b ** 2) - math.log(2 * math.pi * self.b ** 2) - log_sinc

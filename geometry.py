"""Riemannian geometry of the model spaces.

Points and tangent vectors are stored in ambient coordinates:
  - Euclidean space  : R^d,
  - sphere S^2       : unit vectors of R^3,
  - hyperbolic plane : hyperboloid {x : <x,x>_L = -1, x0 > 0} in R^3,
                       with the Minkowski form <x,y>_L = -x0 y0 + x1 y1 + x2 y2.

To add a new model space, subclass Manifold and register it in make_manifold.
"""

import torch


class Manifold:
    name = None
    dim = None          # intrinsic dimension d
    ambient_dim = None  # dimension D of the ambient representation

    # ---- metric ----
    def inner(self, q, u, v):
        """Riemannian inner product of tangent vectors u, v at q."""
        raise NotImplementedError

    def norm2(self, q, u):
        return self.inner(q, u, u)

    def proj(self, q, u):
        """Orthogonal projection of an ambient vector u onto the tangent space at q."""
        raise NotImplementedError

    def egrad_to_rgrad(self, q, g):
        """Riemannian gradient from the ambient (Euclidean) gradient g."""
        raise NotImplementedError

    # ---- geodesics ----
    def exp(self, q, v):
        """Exponential map: time-one position of the geodesic from q with velocity v."""
        raise NotImplementedError

    def log(self, q, r):
        """Inverse of exp: tangent vector v at q such that exp(q, v) = r."""
        raise NotImplementedError

    def transport(self, q, r, u):
        """Parallel transport of the tangent vector u from q to r along the geodesic."""
        raise NotImplementedError

    def dist(self, q, r):
        """Geodesic distance (supports broadcasting)."""
        raise NotImplementedError

    def projx(self, q):
        """Re-projection onto the manifold (numerical safety)."""
        raise NotImplementedError

    # ---- frames ----
    def origin(self, device=None):
        """A reference point o (used for base distributions and chart metrics)."""
        raise NotImplementedError

    def frame(self, q):
        """Orthonormal basis of the tangent space at q, shape (..., d, D)."""
        raise NotImplementedError

    def embed(self, q):
        """Bounded input featurization of the point for neural networks
        (identity when the ambient coordinates are already bounded)."""
        return q

    @property
    def embed_dim(self):
        return self.ambient_dim

    def tangent_from_coords(self, q, c):
        """Tangent vector at q from its coordinates c (..., d) in the frame."""
        return torch.einsum("...i,...id->...d", c, self.frame(q))

    def coords_from_tangent(self, q, u):
        """Coordinates (..., d) of the tangent vector u in the frame at q."""
        return self.inner(q.unsqueeze(-2), self.frame(q), u.unsqueeze(-2))


class Euclidean(Manifold):
    name = "r2"

    def __init__(self, d=2):
        self.dim = d
        self.ambient_dim = d

    def inner(self, q, u, v):
        return (u * v).sum(-1)

    def proj(self, q, u):
        return u

    def egrad_to_rgrad(self, q, g):
        return g

    def exp(self, q, v):
        return q + v

    def log(self, q, r):
        return r - q

    def transport(self, q, r, u):
        return u

    def dist(self, q, r):
        return (q - r).norm(dim=-1)

    def projx(self, q):
        return q

    def origin(self, device=None):
        return torch.zeros(self.dim, device=device)

    def frame(self, q):
        e = torch.eye(self.dim, device=q.device, dtype=q.dtype)
        return e.expand(*q.shape[:-1], self.dim, self.dim)


class Sphere2(Manifold):
    """Unit sphere S^2 embedded in R^3."""
    name = "s2"
    dim = 2
    ambient_dim = 3
    eps = 1e-7

    def inner(self, q, u, v):
        return (u * v).sum(-1)

    def proj(self, q, u):
        return u - (q * u).sum(-1, keepdim=True) * q

    def egrad_to_rgrad(self, q, g):
        return self.proj(q, g)

    def exp(self, q, v):
        t = ((v * v).sum(-1, keepdim=True) + 1e-14).sqrt()
        sinc = torch.where(t < self.eps, 1.0 - t**2 / 6.0, torch.sin(t) / t.clamp(min=self.eps))
        return torch.cos(t) * q + sinc * v

    def log(self, q, r):
        c = (q * r).sum(-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        t = torch.acos(c)
        u = r - c * q                                # norm sin(t)
        fac = t / u.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.where(t < self.eps, u, fac * u)

    def transport(self, q, r, u):
        c = (q * r).sum(-1, keepdim=True)
        return u - (r * u).sum(-1, keepdim=True) / (1.0 + c) * (q + r)

    def dist(self, q, r):
        return torch.acos((q * r).sum(-1).clamp(-1.0, 1.0))

    def projx(self, q):
        return q / q.norm(dim=-1, keepdim=True)

    def origin(self, device=None):
        return torch.tensor([0.0, 0.0, 1.0], device=device)

    def frame(self, q):
        # reference axis least aligned with q, projected and normalized
        a = torch.zeros_like(q)
        a.scatter_(-1, q.abs().argmin(-1, keepdim=True), 1.0)
        e1 = self.proj(q, a)
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = torch.cross(q, e1, dim=-1)
        return torch.stack([e1, e2], dim=-2)


class Hyperboloid2(Manifold):
    """Hyperbolic plane H^2, hyperboloid model in R^3 with the Minkowski form."""
    name = "h2"
    dim = 2
    ambient_dim = 3
    eps = 1e-7

    def _J(self, x):
        return torch.tensor([-1.0, 1.0, 1.0], device=x.device, dtype=x.dtype)

    def minkowski(self, u, v):
        return (u * v * self._J(u)).sum(-1)

    def inner(self, q, u, v):
        return self.minkowski(u, v)     # positive definite on tangent spaces

    def proj(self, q, u):
        return u + self.minkowski(q, u).unsqueeze(-1) * q

    def egrad_to_rgrad(self, q, g):
        return self.proj(q, self._J(g) * g)

    def exp(self, q, v):
        t = (self.minkowski(v, v).clamp(min=0.0) + 1e-14).sqrt().unsqueeze(-1)
        sinhc = torch.where(t < self.eps, 1.0 + t**2 / 6.0, torch.sinh(t) / t.clamp(min=self.eps))
        return self.projx(torch.cosh(t) * q + sinhc * v)

    def log(self, q, r):
        a = (-self.minkowski(q, r)).clamp(min=1.0 + 1e-7).unsqueeze(-1)
        t = torch.acosh(a)
        u = r - a * q                                # Minkowski norm sinh(t)
        n = (self.minkowski(u, u).clamp(min=0.0) + 1e-14).sqrt().unsqueeze(-1)
        fac = t / n.clamp(min=self.eps)
        return torch.where(t < self.eps, u, fac * u)

    def transport(self, q, r, u):
        a = -self.minkowski(q, r).unsqueeze(-1)
        return u + self.minkowski(r, u).unsqueeze(-1) / (1.0 + a) * (q + r)

    def dist(self, q, r):
        return torch.acosh((-self.minkowski(q, r)).clamp(min=1.0))

    def projx(self, q):
        x = q[..., 1:]
        x0 = (1.0 + (x**2).sum(-1, keepdim=True)).sqrt()
        return torch.cat([x0, x], dim=-1)

    def origin(self, device=None):
        return torch.tensor([1.0, 0.0, 0.0], device=device)

    def embed(self, q):
        return q[..., 1:] / (1.0 + q[..., :1])      # Poincare-disk coordinates

    @property
    def embed_dim(self):
        return 2

    def frame(self, q):
        a1 = torch.zeros_like(q); a1[..., 1] = 1.0
        a2 = torch.zeros_like(q); a2[..., 2] = 1.0
        e1 = self.proj(q, a1)
        e1 = e1 / self.minkowski(e1, e1).clamp(min=1e-14).sqrt().unsqueeze(-1)
        e2 = self.proj(q, a2)
        e2 = e2 - self.minkowski(e2, e1).unsqueeze(-1) * e1
        e2 = e2 / self.minkowski(e2, e2).clamp(min=1e-14).sqrt().unsqueeze(-1)
        return torch.stack([e1, e2], dim=-2)


def make_manifold(space):
    return {"r2": Euclidean, "s2": Sphere2, "h2": Hyperboloid2}[space]()

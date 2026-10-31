"""RNHF and RCNF models (Section 2.3 and Appendix H).

RNHF: Hamiltonian H(q, p) = |p|_g^2 / 2 + V_theta(q) with the kinetic energy
fixed by the metric and the potential V_theta = -sigma_p^2 log U_theta (Eq. 7).
The base is rho_0(q) N_g(p; 0, sigma_p^2 I), the flow is the geodesic
kick-drift-kick integrator (Eq. 3), which preserves phase-space volume exactly,
and training maximizes the ELBO (Eq. 8) with a Gaussian momentum encoder
q_phi(p | q). Momenta are stored as tangent vectors in ambient coordinates,
identified with covectors through the metric.

RCNF: a neural vector field integrated by a re-anchored geodesic midpoint
rule, trained by maximum likelihood through the Riemannian divergence, which
is evaluated as the trace of the derivative of the field in an orthonormal
frame.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(din, hidden, depth, dout):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers += [nn.Linear(d, dout)]
    return nn.Sequential(*layers)


class Potential(nn.Module):
    """Potential V(q) = -sigma_p^2 log U(q) (Eq. 7), with log U = softplus(net),
    so that U > 0. The free energy of the data is V = -sigma_p^2 log rho_1
    (Eq. 11), for which log U equals log rho_1 up to a constant. The factor
    sigma_p^2 is supplied by the model when the force is computed."""

    def __init__(self, manifold, hidden, depth):
        super().__init__()
        self.M = manifold
        self.net = mlp(manifold.embed_dim, hidden, depth, 1)

    def log_u(self, q):
        """log U(q), the dimensionless landscape."""
        raw = self.net(self.M.embed(q)).squeeze(-1)
        return F.softplus(raw)                    # U = exp(softplus) > 0

    def rgrad(self, q, sigma_p2):
        """Riemannian gradient of V = -sigma_p^2 log U at q."""
        cg = torch.is_grad_enabled() and self.training
        with torch.enable_grad():
            qq = q if q.requires_grad else q.detach().requires_grad_(True)
            g = torch.autograd.grad(self.log_u(qq).sum(), qq, create_graph=cg)[0]
        return -sigma_p2 * self.M.egrad_to_rgrad(q, g)


class Encoder(nn.Module):
    """Gaussian momentum encoder q_phi(p | q) = N(mu(q), L(q) L(q)^T), expressed
    in an orthonormal frame of the tangent space at q. `scale` multiplies the
    momentum (default 1): for data concentrated at an angular scale a, setting
    it to the expected momentum scale lets the network work with O(1) outputs.
    The log-density carries the corresponding -d log(scale)."""

    def __init__(self, manifold, hidden, depth, scale=1.0):
        super().__init__()
        self.M = manifold
        self.scale = float(scale)
        d = manifold.dim
        self.net = mlp(manifold.embed_dim, hidden, depth, d + d * (d + 1) // 2)
        self.tril = torch.tril_indices(d, d)

    def sample_and_logprob(self, q):
        d = self.M.dim
        out = self.net(self.M.embed(q))
        mu, l = out[..., :d], out[..., d:]
        L = torch.zeros(*q.shape[:-1], d, d, dtype=q.dtype, device=q.device)
        L[..., self.tril[0], self.tril[1]] = l
        diag = L.diagonal(dim1=-2, dim2=-1)
        eps = torch.randn_like(mu)
        c = mu + torch.einsum("...ij,...j->...i", L - torch.diag_embed(diag) + torch.diag_embed(diag.exp()), eps)
        logq = -0.5 * (eps**2).sum(-1) - diag.sum(-1) - 0.5 * d * math.log(2 * math.pi)
        if self.scale != 1.0:
            c, logq = self.scale * c, logq - d * math.log(self.scale)
        return self.M.tangent_from_coords(q, c), logq


class RNHF(nn.Module):
    """Riemannian Neural Hamiltonian Flow with fixed kinetic energy."""

    def __init__(self, manifold, base, hidden=128, depth=3, n_steps=15, sigma_p=1.0,
                 momentum_scale=1.0):
        super().__init__()
        self.M, self.base = manifold, base
        self.n_steps = n_steps
        # sigma_p is a learned global scalar, parametrized in log-space to stay positive
        self.log_sigma_p = nn.Parameter(torch.tensor(math.log(float(sigma_p))))
        self.potential = Potential(manifold, hidden, depth)
        self.encoder = Encoder(manifold, hidden, depth, momentum_scale)

    @property
    def sigma_p(self):
        return self.log_sigma_p.exp()

    def integrate(self, q, p, forward=True):
        """Geodesic kick-drift-kick leapfrog. The backward pass applies the
        inverse substeps in reverse time order: it is the exact inverse map.
        The drift is the exact geodesic flow of the fixed kinetic energy; the
        momentum, being parallel to the drift direction, is transported along
        the drift geodesic itself (for which the endpoint transport formula
        below is exact at our step sizes)."""
        M, dt = self.M, 1.0 / self.n_steps
        sp2 = self.sigma_p ** 2
        steps = range(self.n_steps) if forward else reversed(range(self.n_steps))
        for _ in steps:
            if forward:
                p = p - 0.5 * dt * self.potential.rgrad(q, sp2)
                q_new = M.projx(M.exp(q, dt * p))
                p = M.proj(q_new, M.transport(q, q_new, p))
                q = q_new
                p = p - 0.5 * dt * self.potential.rgrad(q, sp2)
            else:
                p = p + 0.5 * dt * self.potential.rgrad(q, sp2)
                q_new = M.projx(M.exp(q, -dt * p))
                p = M.proj(q_new, M.transport(q, q_new, p))
                q = q_new
                p = p + 0.5 * dt * self.potential.rgrad(q, sp2)
        return q, p

    def elbo(self, q):
        """Phase-space evidence lower bound at data points q (no Jacobian term:
        the leapfrog preserves the Liouville volume exactly)."""
        d = self.M.dim
        p, logq = self.encoder.sample_and_logprob(q)
        q0, p0 = self.integrate(q, p, forward=False)
        lp_pos = self.base.log_prob(q0)
        lp_mom = -self.M.norm2(q0, p0) / (2 * self.sigma_p**2) \
                 - 0.5 * d * torch.log(2 * math.pi * self.sigma_p**2)
        return lp_pos + lp_mom - logq

    def loss(self, q):
        return -self.elbo(q).mean()

    @torch.no_grad()
    def sample(self, n):
        q0 = self.base.sample(n)
        eps = self.sigma_p * torch.randn(n, self.M.dim)
        p0 = self.M.tangent_from_coords(q0, eps)
        q1, _ = self.integrate(q0, p0, forward=True)
        return q1


class RCNF(nn.Module):
    """Riemannian Continuous Normalizing Flow baseline (same geometric pipeline
    as RNHF: re-anchored geodesic integration, second-order midpoint rule)."""

    def __init__(self, manifold, base, hidden=128, depth=3, n_steps=15, reg=0.0, fmax=10.0):
        super().__init__()
        self.M, self.base, self.n_steps = manifold, base, n_steps
        self.reg, self.fmax = reg, fmax
        self.net = mlp(manifold.embed_dim + 1, hidden, depth, manifold.ambient_dim)

    def field(self, t, q):
        tt = torch.full((*q.shape[:-1], 1), t, device=q.device, dtype=q.dtype)
        f = self.M.proj(q, self.net(torch.cat([self.M.embed(q), tt], dim=-1)))
        n = (self.M.norm2(q, f).clamp(min=0.0) + 1e-14).sqrt().unsqueeze(-1)
        return f * (self.fmax / n.clamp(min=self.fmax))   # cap |f|_g at fmax

    def divergence(self, t, q):
        """Riemannian divergence of the field at q, as the frame trace of its
        derivative (the metric correction vanishes in re-anchored coordinates).
        Also returns the squared Frobenius norm of the frame Jacobian, used to
        regularize the field (Finlay et al., 2020)."""
        cg = torch.is_grad_enabled() and self.training
        with torch.enable_grad():
            qq = q if q.requires_grad else q.detach().requires_grad_(True)
            e = self.M.frame(qq).detach()                       # (..., d, D)
            f = self.field(t, qq)
            div, jac2 = 0.0, 0.0
            for j in range(self.M.dim):
                h = self.M.inner(qq, f, e[..., j, :]).sum()
                g = torch.autograd.grad(h, qq, create_graph=cg, retain_graph=True)[0]
                div = div + (g * e[..., j, :]).sum(-1)
                for i in range(self.M.dim):                     # entries <e_j, d_{e_i} f>
                    jac2 = jac2 + (g * e[..., i, :]).sum(-1) ** 2
        return div, jac2

    def flow(self, q, forward=True, with_div=False):
        """Geodesic midpoint integration; optionally accumulates int_0^1 div dt."""
        M, dt = self.M, 1.0 / self.n_steps
        sign = 1.0 if forward else -1.0
        steps = range(self.n_steps) if forward else reversed(range(self.n_steps))
        intdiv, penalty = 0.0, 0.0
        for n in steps:
            t_here = n * dt if forward else (n + 1) * dt
            tm = (n + 0.5) * dt
            qm = M.projx(M.exp(q, sign * 0.5 * dt * self.field(t_here, q)))
            fm = self.field(tm, qm)
            if with_div:
                div, jac2 = self.divergence(tm, qm)
                intdiv = intdiv + dt * div
                penalty = penalty + dt * (M.norm2(qm, fm) + jac2)
            k = M.transport(qm, q, fm)
            q = M.projx(M.exp(q, sign * dt * k))
        return q, intdiv, penalty

    def log_prob(self, q):
        q0, intdiv, _ = self.flow(q, forward=False, with_div=True)
        return self.base.log_prob(q0) - intdiv

    def loss(self, q):
        """Negative log-likelihood plus a regularization of the field norm and
        of its Jacobian, which keeps the fixed-step discretization faithful
        (Finlay et al., 2020)."""
        q0, intdiv, penalty = self.flow(q, forward=False, with_div=True)
        logp = self.base.log_prob(q0) - intdiv
        return -logp.mean() + self.reg * penalty.mean()

    @torch.no_grad()
    def sample(self, n):
        q1, _, _ = self.flow(self.base.sample(n), forward=True)
        return q1


def build_model(kind, cfg, manifold, base):
    m = cfg["model"]
    if kind == "rnhf":
        return RNHF(manifold, base, m["hidden"], m["depth"], m["n_steps"],
                    m["sigma_p"], m.get("momentum_scale", 1.0))
    if kind == "rcnf":
        return RCNF(manifold, base, m.get("rcnf_hidden", m["hidden"]), m["depth"], m["n_steps"],
                    m.get("rcnf_reg", 0.0), m.get("rcnf_fmax", 10.0))
    raise ValueError(kind)

"""Geodesic Hamiltonian Monte Carlo on the sphere S^2, exact or with a surrogate
energy (Section 3.3), and the statistics of the resulting chains.

One transition from q draws a tangent momentum p ~ N(0, I), runs L
kick-drift-kick steps of size h = ell / L with the gradient of an energy E_kick,
and accepts the end point (q', p') with probability min(1, exp(-dH)), where

    dH = E(q') + |p'|^2 / 2 - E(q) - |p|^2 / 2                       (Eq. 17)

uses the true energy E. The drift is the exact geodesic flow of the sphere,

    q(h) = cos(|p| h) q + sin(|p| h) p / |p|,    p(h) = cos(|p| h) p - |p| sin(|p| h) q.

Exact HMC kicks with E itself. Surrogate HMC kicks with E_hat = -log U_theta
(Eq. 19): the proposal still preserves volume and is reversible, so the chain
still leaves pi ~ exp(-E) invariant, and E is evaluated once per transition.
The gradient at the current state is kept from the previous transition, so a
transition costs exactly L gradients of E_kick and one evaluation of E.
"""

import math
import time

import numpy as np
import torch


def proj(q, u):
    """Projection of u onto the tangent plane of the unit sphere at q."""
    return u - (q * u).sum(-1, keepdim=True) * q


def geodesic_flow(q, p, h):
    """Exact flow of the kinetic energy |p|^2 / 2 on the unit sphere for a time h."""
    a = p.norm(dim=-1, keepdim=True).clamp(min=1e-300)
    q1 = torch.cos(a * h) * q + torch.sin(a * h) * p / a
    p1 = torch.cos(a * h) * p - a * torch.sin(a * h) * q
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    return q1, proj(q1, p1)


@torch.no_grad()
def hmc(q, n_transitions, ell, n_steps, grad_kick, energy, seed=0, label="HMC"):
    """Chain of n_transitions states started at q (3,).

    grad_kick(q): Riemannian gradient of the energy used in the kicks;
    energy(q):    the true energy E, used in the Metropolis test.
    """
    gen = torch.Generator().manual_seed(seed)
    h = ell / n_steps
    g, e = grad_kick(q), energy(q)
    states = torch.empty(n_transitions, 3)
    energies = torch.empty(n_transitions)
    accept = torch.zeros(n_transitions, dtype=torch.bool)
    t0 = time.perf_counter()
    for t in range(n_transitions):
        p = proj(q, torch.randn(3, generator=gen))
        h0 = e + 0.5 * (p @ p)
        q1, p1 = q, p - 0.5 * h * g
        for i in range(n_steps):
            q1, p1 = geodesic_flow(q1, p1, h)
            g1 = grad_kick(q1)
            p1 = p1 - (0.5 * h if i == n_steps - 1 else h) * g1
        e1 = energy(q1)
        dh = (e1 + 0.5 * (p1 @ p1) - h0).item()
        if math.isfinite(dh) and torch.rand((), generator=gen).item() < math.exp(min(0.0, -dh)):
            q, g, e = q1, g1, e1
            accept[t] = True
        states[t], energies[t] = q, e
        if (t + 1) % max(1, n_transitions // 10) == 0:
            print(f"[{label}] {t + 1}/{n_transitions}  acceptance "
                  f"{accept[:t + 1].double().mean():.3f}  ({time.perf_counter() - t0:.0f} s)")
    return {"states": states, "E": energies, "accept": accept,
            "time": time.perf_counter() - t0}


def mean_laplacian(f, X, frame):
    """Average over the points X of the Laplace-Beltrami operator of f (Eq. 20):
    the sum, over an orthonormal tangent frame (e_1, e_2) at x, of the second
    derivatives of f along the great circles cos(t) x + sin(t) e_i at t = 0."""
    E = frame(X)                                          # (n, 2, 3)
    total = 0.0
    with torch.enable_grad():
        for i in range(2):
            t = torch.zeros(X.shape[0], requires_grad=True)
            q = torch.cos(t)[:, None] * X + torch.sin(t)[:, None] * E[:, i]
            d1, = torch.autograd.grad(f(q).sum(), t, create_graph=True)
            d2, = torch.autograd.grad(d1.sum(), t)
            total = total + d2.mean().item()
    return total


def duration(mean_lap, dim=2):
    """Duration (20): a quarter period in a well of curvature E[Laplacian] / d."""
    if not mean_lap > 0:
        raise ValueError(f"mean Laplacian {mean_lap:.3g} <= 0: no quarter turn")
    return 0.5 * math.pi * math.sqrt(dim / mean_lap)


def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time, with Sokal's automatic window
    (the smallest W such that W >= c tau(W)); tau = 1 for independent draws."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if np.ptp(x) == 0.0:                                  # the chain never moved
        return float(n)
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    taus = 2.0 * np.cumsum(acf / acf[0]) - 1.0
    w = np.argmax(np.arange(n) >= c * taus)
    return float(taus[w] if w > 0 else taus[-1])

"""Tau-leap stochastic simulation of the deleterious-mutation SIR."""

from __future__ import annotations

import math

import numpy as np


def model_base_rate_fun(
    y: np.ndarray,
    beta_0: float,
    s_d: float,
    lamb: float,
    mu: float,
    gamma: float,
    N: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rates, stoichiometry)`` for the deleterious SIR.

    Events (rows of ``stoich``): birth into S; death of S; infection into each
    class ``I_k``; recovery of each ``I_k``; death of each ``I_k``; death of R.
    """
    S = y[0]
    Ik = y[1:-1]
    R = y[-1]
    max_k = len(Ik)
    beta = beta_0 * (1 - s_d) ** np.arange(max_k)

    stoich = np.zeros((3 * max_k + 3, max_k + 2))
    rates = np.zeros(3 + 3 * max_k)

    # birth S
    rates[0] = mu * N
    stoich[0, 0] = 1
    # death S
    rates[1] = mu * S
    stoich[1, 0] = -1

    # infection into each class
    eq2 = np.zeros_like(Ik)
    for k in range(max_k):
        stoich[2 + k, 0] = -1
        stoich[2 + k, k + 1] = 1
        for j in range(k + 1):
            eq2[k] += (
                (S / N)
                * beta[k - j]
                * np.exp(-lamb)
                * ((lamb**j) / math.factorial(j))
                * Ik[k - j]
            )
    rates[2 : 2 + max_k] = eq2

    # recovery I_k -> R
    for k in range(max_k):
        rates[2 + max_k + k] = gamma * Ik[k]
        stoich[max_k + 2 + k, k + 1] = -1
        stoich[max_k + 2 + k, -1] = 1

    # death I_k
    for k in range(max_k):
        rates[2 + 2 * max_k + k] = mu * Ik[k]
        stoich[(2 * max_k) + 2 + k, k + 1] = -1

    # death R
    rates[-1] = mu * R
    stoich[-1, -1] = -1
    return rates, stoich


def integrate_tau_leap(
    model_rate_fun,
    y0: np.ndarray,
    t_end: float,
    tau: float,
    beta_0: float,
    s_d: float,
    lamb: float,
    mu: float,
    gamma: float,
    N: float,
    rng: np.random.Generator | None = None,
    record_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a tau-leap process; return ``(ts, Y)`` with ``Y`` shape ``(n, state)``."""
    if rng is None:
        rng = np.random.default_rng()
    y = np.array(y0, dtype=float)
    n_steps = int(np.ceil(t_end / tau))
    ts: list[float] = [0.0]
    Y: list[np.ndarray] = [y.copy()]
    for i in range(n_steps):
        rates, stoich = model_rate_fun(y, beta_0, s_d, lamb, mu, gamma, N)
        events = rng.poisson(np.clip(rates, 0, None) * tau)
        y = y + events @ stoich
        if np.any(y < 0):
            y = np.clip(y, 0, None)
        if (i + 1) % record_every == 0:
            ts.append((i + 1) * tau)
            Y.append(y.copy())
    return np.array(ts), np.array(Y)

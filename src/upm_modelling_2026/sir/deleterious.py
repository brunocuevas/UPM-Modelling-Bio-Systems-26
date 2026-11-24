"""Deterministic SIR with Poisson deleterious mutations.

Each infection transmits with a Poisson number of deleterious mutations
(rate ``lamb``). Fitness declines multiplicatively: ``beta_k = beta_0 *
(1 - s_d)**k``. State vector layout is ``[S, I_0, ..., I_{K-1}, R]``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import factorial


def prepare_population(k_max: int, I0: float, N: float) -> np.ndarray:
    """Initial state with ``I0`` in class 0 and the rest susceptible."""
    y = np.zeros(k_max + 2)
    y[0] = N - I0
    y[1] = I0
    return y


def model_base(
    t: float,
    y: np.ndarray,
    beta_0: float,
    s_d: float,
    lamb: float,
    mu: float,
    gamma: float,
    N: float,
) -> np.ndarray:
    """ODE right-hand side for deleterious-mutation SIR."""
    S = y[0]
    I = y[1:-1]
    R = y[-1]
    max_k = len(I)
    beta = beta_0 * (1 - s_d) ** np.arange(max_k)
    dS = N * mu - S * mu - (S / N) * (beta * I).sum()
    dI = -I * (mu + gamma)
    for k in range(max_k):
        for j in range(k + 1):
            dI[k] += (
                (S / N)
                * beta[k - j]
                * np.exp(-lamb)
                * ((lamb**j) / math.factorial(j))
                * I[k - j]
            )
    dR = gamma * I.sum() - R * mu
    dy = np.zeros_like(y)
    dy[0] = dS
    dy[1:-1] = dI
    dy[-1] = dR
    return dy


def compute_equilibrium_population(
    k_max: int,
    beta_0: float,
    s_d: float,
    lamb: float,
    mu: float,
    gamma: float,
    N: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mutation–selection equilibrium total I and class probabilities.

    Returns ``(I_tot, p, k)`` where ``p`` is the Poisson(``lamb / s_d``) mass
    over classes ``k = 0..k_max-1``.
    """
    k = np.arange(k_max)
    theta = lamb / s_d
    I_tot = ((mu * N) / (mu + gamma)) * (1 - (mu + gamma) / (beta_0 * np.exp(-lamb)))
    p = np.exp(-theta) * ((theta**k) / factorial(k))
    return float(I_tot), p, k


class DeleteriousResult(NamedTuple):
    t: np.ndarray
    y: np.ndarray  # shape (k_max + 2, n_times)


def simulate_deleterious(
    *,
    k_max: int = 20,
    beta_0: float = 0.5625,
    s_d: float = 0.08,
    lamb: float = 0.10,
    mu: float = 1.0 / (30 * 365),
    gamma: float = 0.0,
    N: float = 1000.0,
    I0: float = 1.0,
    t0: float = 0.0,
    tfinal: float = 100.0,
    intervals: int = 1000,
    y0: np.ndarray | None = None,
) -> DeleteriousResult:
    """Integrate the deleterious-mutation SIR with dense output sampling."""
    y0 = prepare_population(k_max, I0, N) if y0 is None else np.asarray(y0, dtype=float)
    sol = solve_ivp(
        model_base,
        y0=y0,
        t_span=(t0, tfinal),
        args=(beta_0, s_d, lamb, mu, gamma, N),
        dense_output=True,
    )
    if sol.sol is None:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    t = np.linspace(t0, tfinal, intervals)
    return DeleteriousResult(t=t, y=sol.sol(t))

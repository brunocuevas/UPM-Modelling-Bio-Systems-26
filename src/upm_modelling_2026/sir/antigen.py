"""SIR with life-history compartments and two antigenic types.

State layout (``K`` mutation classes)::

    [S0,
     I0r[0..K), I0m[0..K),   # naive hosts: resident / mutant
     Imr[0..K), Irm[0..K),   # cross-infection after prior exposure
     Sr, Sm, Srm]

``sigma`` scales transmission into hosts with heterologous history
(partial immune escape). Deleterious load still follows Poisson ``lamb``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp


def model_antigen(
    t: float,
    y: np.ndarray,
    beta_0: float,
    s_d: float,
    lamb: float,
    mu: float,
    gamma: float,
    sigma: float,
    N: float,
) -> np.ndarray:
    """ODE RHS for two-strain antigen + deleterious-load dynamics."""
    K = (len(y) - 4) // 4
    S0 = y[0]
    I0r = y[1 : K + 1]
    I0m = y[K + 1 : 2 * K + 1]
    Imr = y[2 * K + 1 : 3 * K + 1]
    Irm = y[3 * K + 1 : 4 * K + 1]
    Sr = y[4 * K + 1]
    Sm = y[4 * K + 2]
    Srm = y[-1]

    beta = beta_0 * (1 - s_d) ** np.arange(K)
    Ir_k = I0r + Imr
    Im_k = I0m + Irm
    Fr = (beta * Ir_k).sum()
    Fm = (beta * Im_k).sum()

    dS0 = N * mu - S0 * mu - (S0 / N) * (Fr + Fm)
    dSr = gamma * I0r.sum() - Sr * mu - sigma * (Sr / N) * Fm
    dSm = gamma * I0m.sum() - Sm * mu - sigma * (Sm / N) * Fr
    dSrm = gamma * (Imr.sum() + Irm.sum()) - Srm * mu

    dI0r = -I0r * (mu + gamma)
    dI0m = -I0m * (mu + gamma)
    dImr = -Imr * (mu + gamma)
    dIrm = -Irm * (mu + gamma)
    for k in range(K):
        for j in range(k + 1):
            pj = np.exp(-lamb) * (lamb**j) / math.factorial(j)
            dI0r[k] += (S0 / N) * beta[k - j] * pj * Ir_k[k - j]
            dI0m[k] += (S0 / N) * beta[k - j] * pj * Im_k[k - j]
            dImr[k] += sigma * (Sm / N) * beta[k - j] * pj * Ir_k[k - j]
            dIrm[k] += sigma * (Sr / N) * beta[k - j] * pj * Im_k[k - j]

    dy = np.zeros_like(y)
    dy[0] = dS0
    dy[1 : K + 1] = dI0r
    dy[K + 1 : 2 * K + 1] = dI0m
    dy[2 * K + 1 : 3 * K + 1] = dImr
    dy[3 * K + 1 : 4 * K + 1] = dIrm
    dy[4 * K + 1] = dSr
    dy[4 * K + 2] = dSm
    dy[-1] = dSrm
    return dy


def mutant_initial_state(
    K: int,
    i_bg: int,
    count: float,
    beta_0: float,
    s_d: float,
    lamb: float,
    mu: float,
    gamma: float,
    N: float,
) -> np.ndarray:
    """Resident at mutation–selection equilibrium, plus ``count`` mutants in class ``i_bg``."""
    y = np.zeros(4 * K + 4)
    S_hat = (mu + gamma) / (beta_0 * np.exp(-lamb)) * N
    I_tot = mu * N * (1 - S_hat / N) / (mu + gamma)
    p = (
        np.exp(-lamb / s_d)
        * (lamb / s_d) ** np.arange(K)
        / np.array([math.factorial(k) for k in range(K)], dtype=float)
    )
    p /= p.sum()
    y[0] = S_hat
    y[1 : K + 1] = I_tot * p
    y[4 * K + 1] = gamma * I_tot / mu  # Sr
    y[K + 1 + i_bg] = count
    y[0] -= count
    return y


def unpack_antigen_state(y: np.ndarray, K: int) -> dict[str, np.ndarray]:
    """Split a state vector (or trajectory ``y[state, t]``) into named compartments."""
    return {
        "S0": y[0],
        "I0r": y[1 : K + 1],
        "I0m": y[K + 1 : 2 * K + 1],
        "Imr": y[2 * K + 1 : 3 * K + 1],
        "Irm": y[3 * K + 1 : 4 * K + 1],
        "Sr": y[4 * K + 1],
        "Sm": y[4 * K + 2],
        "Srm": y[-1],
    }


class AntigenResult(NamedTuple):
    t: np.ndarray
    y: np.ndarray  # shape (4K+4, n_times)
    K: int


def simulate_antigen(
    y0: np.ndarray,
    *,
    beta_0: float = 0.5625,
    s_d: float = 0.008,
    lamb: float = 0.10,
    mu: float = 1.0 / (30 * 365),
    gamma: float = 0.25,
    sigma: float = 0.045,
    N: float = 4e9,
    t0: float = 0.0,
    tfinal: float = 1500.0,
    intervals: int = 1500,
    method: str = "LSODA",
) -> AntigenResult:
    """Integrate the antigen-evolution SIR; returns dense samples on ``[t0, tfinal]``."""
    K = (len(y0) - 4) // 4
    sol = solve_ivp(
        model_antigen,
        y0=np.asarray(y0, dtype=float),
        t_span=(t0, tfinal),
        args=(beta_0, s_d, lamb, mu, gamma, sigma, N),
        method=method,
        dense_output=True,
    )
    if sol.sol is None:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    t = np.linspace(t0, tfinal, intervals)
    return AntigenResult(t=t, y=sol.sol(t), K=K)

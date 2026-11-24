"""Helpers for MutAntiGen timeseries / mutation-series plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .tree_io import read_timeseries


def read_mutation_series(path: str | Path) -> pd.DataFrame:
    """Parse comma-separated integer rows from ``out.mutationSeries``."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        vals = [int(x.strip()) for x in line.split(",") if x.strip()]
        if vals:
            rows.append(vals)
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    arr = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return pd.DataFrame(arr, columns=[f"class_{j}" for j in range(width)])


def plot_timeseries_columns(
    path: str | Path,
    columns: list[str],
    out_path: str | Path,
    *,
    logy: bool = False,
    figsize=(8, 4),
) -> Path:
    """Line-plot selected columns from ``out.timeseries`` against ``date``."""
    df = read_timeseries(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing}; available={list(df.columns)}")
    fig, ax = plt.subplots(figsize=figsize)
    for c in columns:
        ax.plot(df["date"], df[c], label=c)
    ax.set_xlabel("date")
    ax.set_ylabel(", ".join(columns) if len(columns) == 1 else "value")
    if logy:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mutation_series(
    path: str | Path,
    out_path: str | Path,
    *,
    n_classes: int = 10,
    figsize=(8, 4),
) -> Path:
    """Plot the first ``n_classes`` mutation-load classes over time index."""
    df = read_mutation_series(path)
    if df.empty:
        raise ValueError(f"no rows in {path}")
    cols = list(df.columns[:n_classes])
    fig, ax = plt.subplots(figsize=figsize)
    for c in cols:
        ax.plot(df.index, df[c], label=c)
    ax.set_xlabel("time index")
    ax.set_ylabel("count")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

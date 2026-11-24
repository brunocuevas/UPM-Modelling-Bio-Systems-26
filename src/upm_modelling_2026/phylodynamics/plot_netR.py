"""Color a MutAntiGen phylogeny by net reproductive number (Net R)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection

from .tree_io import branch_polyline_segments, read_branches, read_timeseries, timeseries_interpolator


def netR_interpolator(fitness_path: Path | str):
    """Build ``f(year) -> exp(meanR)`` from ``out.viralFitnessSeries``."""
    fitness_path = Path(fitness_path)
    df = read_timeseries(fitness_path)
    if "meanR" not in df.columns:
        raise KeyError(f"meanR not in {fitness_path}; columns={list(df.columns)}")
    df = df.copy()
    df["netR"] = np.exp(df["meanR"].astype(float))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "netR"])
    if df.empty:
        raise ValueError(f"no finite netR values in {fitness_path}")
    return timeseries_interpolator(df, "netR")


def plot_tree_netR(
    branches_path: Path | str,
    fitness_path: Path | str,
    out_path: Path | str,
    *,
    n_sub: int = 4,
    linewidth: float = 1.0,
    cmap_name: str = "coolwarm",
    figsize=(11, 6),
    orientation: str = "horizontal",
    pct_clip=(5, 95),
    vcenter: float | None = 1.0,
) -> Path:
    """Paint branch mid-times by ``netR(t) = exp(meanR(t))``."""
    branches_path = Path(branches_path)
    fitness_path = Path(fitness_path)
    out_path = Path(out_path)

    edges = read_branches(branches_path)
    if not edges:
        raise ValueError(f"no edges parsed from {branches_path}")

    f = netR_interpolator(fitness_path)
    starts, ends, years = branch_polyline_segments(edges, n_sub=n_sub)
    values = np.array([f(y) for y in years], dtype=float)

    lo = float(np.percentile(values, pct_clip[0]))
    hi = float(np.percentile(values, pct_clip[1]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if hi <= lo:
            hi = lo + 1e-9

    try:
        import seaborn as sns

        cmap = sns.color_palette(cmap_name, as_cmap=True)
    except Exception:
        cmap = cm.get_cmap(cmap_name)

    if vcenter is not None and lo < vcenter < hi:
        try:
            from matplotlib.colors import TwoSlopeNorm

            norm = TwoSlopeNorm(vmin=lo, vcenter=vcenter, vmax=hi)
        except Exception:
            norm = plt.Normalize(lo, hi)
    else:
        norm = plt.Normalize(lo, hi)

    if orientation == "horizontal":
        segs = [((s[1], s[0]), (e[1], e[0])) for s, e in zip(starts, ends)]
        xlabel, ylabel = "Year", "Layout"
    else:
        segs = list(zip(starts, ends))
        xlabel, ylabel = "Layout", "Year"
        if figsize == (11, 6):
            figsize = (10, 7)

    colors = [cmap(norm(v)) for v in values]
    lc = LineCollection(segs, colors=colors, linewidths=linewidth, capstyle="round")

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if orientation == "vertical":
        ax.invert_yaxis()
    ax.set_title(r"Phylogeny colored by Net $R$ ($\exp(\mathrm{meanR})$)")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"Net $R$")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

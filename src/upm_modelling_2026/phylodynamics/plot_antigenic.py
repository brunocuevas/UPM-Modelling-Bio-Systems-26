"""Color a MutAntiGen phylogeny by antigenic type (SIMMAP)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

from .tree_io import layout_tree, read_simmap, read_tips, simmap_segments, simmap_segments_proportional


def _palette(n: int, name: str = "tab20") -> ListedColormap:
    try:
        import seaborn as sns

        cols = sns.color_palette(name if n <= 20 else "husl", n_colors=max(n, 1))
        return ListedColormap(cols)
    except Exception:
        base = cm.get_cmap("tab20" if n <= 20 else "hsv", max(n, 1))
        return ListedColormap(base(np.linspace(0, 1, max(n, 1))))


def plot_antigenic_tree(
    simmap_path: Path | str,
    tips_path: Path | str | None,
    out_path: Path | str,
    *,
    linewidth: float = 1.2,
    figsize=(10, 7),
    orientation: str = "horizontal",
    proportional: bool = True,
) -> Path:
    """Draw a square phylogeny painted by antigenic SIMMAP state.

    ``orientation``: ``'horizontal'`` = time left→right; ``'vertical'`` = top→bottom.
    """
    simmap_path = Path(simmap_path)
    tips_path = Path(tips_path) if tips_path else None
    out_path = Path(out_path)

    root = read_simmap(simmap_path)
    tips = read_tips(tips_path) if tips_path and tips_path.exists() else None
    layout_tree(root, tips)
    seg_fn = simmap_segments_proportional if proportional else simmap_segments
    geoms, states = seg_fn(root)
    if not geoms:
        raise ValueError("no segments to plot — empty tree?")

    states_arr = np.asarray(states, dtype=float)
    finite = states_arr[np.isfinite(states_arr)]
    if finite.size == 0:
        raise ValueError("no finite antigenic states on tree")
    uniq = np.unique(np.round(finite).astype(int))
    state_to_i = {int(s): i for i, s in enumerate(uniq)}
    color_idx = np.array(
        [state_to_i.get(int(round(s)), 0) if np.isfinite(s) else 0 for s in states_arr],
        dtype=int,
    )
    cmap = _palette(len(uniq))
    bounds = np.arange(len(uniq) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    if orientation == "horizontal":
        segs = [((y0, x0), (y1, x1)) for x0, y0, x1, y1 in geoms]
        xlabel, ylabel = "Year", "Layout"
        if figsize == (10, 7):
            figsize = (11, 6)
    else:
        segs = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in geoms]
        xlabel, ylabel = "Layout", "Year"

    lc = LineCollection(
        segs,
        colors=[cmap(norm(i)) for i in color_idx],
        linewidths=linewidth,
        capstyle="round",
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if orientation == "vertical":
        ax.invert_yaxis()
    ax.set_title("Phylogeny colored by antigenic type")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=range(len(uniq)))
    cbar.ax.set_yticklabels([str(s) for s in uniq])
    cbar.set_label("Antigenic type (SIMMAP state)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

"""I/O helpers for MutAntiGen tree plotting (SIMMAP, branches, tips, timeseries)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _f(x: str) -> float:
    """Float that accepts '1.5' or locale '1,5'."""
    return float(str(x).strip().replace(",", "."))


# ---------------------------------------------------------------------------
# Timeseries
# ---------------------------------------------------------------------------

def read_timeseries(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # MutAntiGen often writes comma decimals on macOS locales
    try:
        df = pd.read_csv(path, sep="\t", decimal=",")
        if "date" in df.columns and pd.api.types.is_numeric_dtype(df["date"]):
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep="\t", decimal=".")


def timeseries_interpolator(df: pd.DataFrame, column: str = "netau"):
    """Return f(year) -> value, with Inf/NaN cleaned."""
    if column not in df.columns:
        raise KeyError(f"{column} not in timeseries columns: {list(df.columns)}")
    t = df["date"].to_numpy(dtype=float)
    y = df[column].to_numpy(dtype=float)
    y = np.where(np.isfinite(y), y, np.nan)
    if np.all(np.isnan(y)):
        raise ValueError(f"all values in column {column!r} are non-finite")
    # fill NaN by interpolation then edge values
    nans = np.isnan(y)
    if nans.any():
        y[nans] = np.interp(t[nans], t[~nans], y[~nans])
    lo, hi = float(np.min(y)), float(np.max(y))

    def f(year: float) -> float:
        return float(np.interp(year, t, y))

    f.t = t  # type: ignore[attr-defined]
    f.y = y  # type: ignore[attr-defined]
    f.lo = lo  # type: ignore[attr-defined]
    f.hi = hi  # type: ignore[attr-defined]
    return f


# ---------------------------------------------------------------------------
# Tips / branches (Java out.tips / out.branches; locale-safe)
# ---------------------------------------------------------------------------

_TIP_FLOAT = r"[0-9]+(?:[.,][0-9]+)?"
_BRANCH_NODE = re.compile(
    rf'\{{"([^"]+)",({_TIP_FLOAT}),([01]),([01]),([01]),([0-9]+),({_TIP_FLOAT}),({_TIP_FLOAT}),({_TIP_FLOAT})\}}'
)


def read_tips(path: str | Path) -> pd.DataFrame:
    """Read out.tips; handles EU comma decimals inside CSV fields."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        return pd.DataFrame()
    rows = []
    # header is standard quoted CSV with names
    for line in text[1:]:
        if not line.strip():
            continue
        # "name",year,trunk,tip,mark,location,layout,ag1,ag2  with possible comma decimals
        m = re.match(
            rf'^"([^"]+)",({_TIP_FLOAT}),([01]),([01]),([01]),([0-9]+),({_TIP_FLOAT}),({_TIP_FLOAT}),({_TIP_FLOAT})\s*$',
            line,
        )
        if not m:
            # fallback: pandas with decimal=","
            continue
        rows.append(
            dict(
                name=m.group(1),
                year=_f(m.group(2)),
                trunk=int(m.group(3)),
                tip=int(m.group(4)),
                mark=int(m.group(5)),
                location=int(m.group(6)),
                layout=_f(m.group(7)),
                ag1=_f(m.group(8)),
                ag2=_f(m.group(9)),
            )
        )
    if rows:
        return pd.DataFrame(rows)
    return pd.read_csv(path, decimal=",")


def read_branches(path: str | Path) -> List[dict]:
    """Parse out.branches into edge dicts with child/parent geometry."""
    path = Path(path)
    edges = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        mc, mp = _BRANCH_NODE.match(parts[0]), _BRANCH_NODE.match(parts[1])
        if not mc or not mp:
            continue
        cov = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 0

        def node(m):
            return dict(
                name=m.group(1),
                year=_f(m.group(2)),
                trunk=int(m.group(3)),
                tip=int(m.group(4)),
                mark=int(m.group(5)),
                deme=int(m.group(6)),
                layout=_f(m.group(7)),
            )

        edges.append(dict(child=node(mc), parent=node(mp), coverage=cov))
    return edges


def branch_polyline_segments(
    edges: List[dict],
    *,
    n_sub: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Square-tree segments from out.branches.
    Returns (start_xy[N,2], end_xy[N,2], mid_year[N]) for coloring by time.
    """
    starts, ends, years = [], [], []
    for e in edges:
        c, p = e["child"], e["parent"]
        # vertical at parent layout: parent year -> child year
        x0, y0 = p["layout"], p["year"]
        x1, y1 = p["layout"], c["year"]
        # horizontal at child year: parent layout -> child layout
        x2, y2 = c["layout"], c["year"]
        for (ax, ay, bx, by) in ((x0, y0, x1, y1), (x1, y1, x2, y2)):
            if n_sub <= 1:
                starts.append((ax, ay))
                ends.append((bx, by))
                years.append(0.5 * (ay + by))
            else:
                for i in range(n_sub):
                    t0, t1 = i / n_sub, (i + 1) / n_sub
                    starts.append((ax + (bx - ax) * t0, ay + (by - ay) * t0))
                    ends.append((ax + (bx - ax) * t1, ay + (by - ay) * t1))
                    years.append(ay + (by - ay) * (0.5 * (t0 + t1)))
    return np.asarray(starts), np.asarray(ends), np.asarray(years, dtype=float)


# ---------------------------------------------------------------------------
# SIMMAP Newick (out.simmapAntigenic / out.simmapLoad)
# ---------------------------------------------------------------------------

@dataclass
class Seg:
    state: float
    length: float


@dataclass
class TreeNode:
    name: str = ""
    children: List["TreeNode"] = field(default_factory=list)
    # segments on the edge FROM parent TO this node (root has none)
    segs: List[Seg] = field(default_factory=list)
    # filled by layout
    x: float = 0.0
    y: float = 0.0  # time (years); larger = more recent
    post_id: int = -1


def _parse_simmap_annot(s: str, i: int) -> Tuple[List[Seg], int]:
    """Parse {state,len:state,len:...} starting at s[i]=='{'."""
    assert s[i] == "{"
    i += 1
    depth = 1
    j = i
    while j < len(s) and depth:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    body = s[i : j - 1]
    segs: List[Seg] = []
    if body.strip():
        for part in body.split(":"):
            a, b = part.split(",", 1)
            segs.append(Seg(state=float(a), length=float(b)))
    return segs, j


def parse_simmap(text: str) -> TreeNode:
    """Parse a one-line SIMMAP Newick string into a TreeNode tree."""
    s = text.strip().rstrip(";").strip()
    i = 0
    n = len(s)

    def skip_ws():
        nonlocal i
        while i < n and s[i].isspace():
            i += 1

    def parse_name() -> str:
        nonlocal i
        start = i
        while i < n and s[i] not in "(),:":
            i += 1
        return s[start:i].strip()

    def parse_node() -> TreeNode:
        nonlocal i
        skip_ws()
        node = TreeNode()
        if i < n and s[i] == "(":
            i += 1  # (
            node.children.append(parse_node())
            skip_ws()
            while i < n and s[i] == ",":
                i += 1
                node.children.append(parse_node())
                skip_ws()
            if i >= n or s[i] != ")":
                raise ValueError(f"expected ')' at {i}")
            i += 1
            skip_ws()
            # optional internal label (usually empty)
            if i < n and s[i] not in ":,)":
                node.name = parse_name()
        else:
            node.name = parse_name()
        skip_ws()
        if i < n and s[i] == ":":
            i += 1
            skip_ws()
            if i < n and s[i] == "{":
                node.segs, i = _parse_simmap_annot(s, i)
            else:
                # plain length
                start = i
                while i < n and s[i] not in ",)":
                    i += 1
                length = float(s[start:i].replace(",", "."))
                node.segs = [Seg(state=np.nan, length=length)]
        return node

    root = parse_node()
    return root


def read_simmap(path: str | Path) -> TreeNode:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # first non-empty line
    for line in text.splitlines():
        if line.strip():
            return parse_simmap(line)
    raise ValueError(f"no tree in {path}")


def branch_length(node: TreeNode) -> float:
    return float(sum(seg.length for seg in node.segs))


def layout_tree(
    root: TreeNode,
    tips: Optional[pd.DataFrame] = None,
) -> None:
    """
    Assign x (layout) and y (year) on nodes.
    Prefer out.tips year/layout for leaves when names match; else sequential x
    and y = cumulative time from root (root y=0, tips at clade depths).
    """
    tip_map: Dict[str, dict] = {}
    if tips is not None and len(tips):
        for r in tips.itertuples(index=False):
            tip_map[str(r.name)] = dict(year=float(r.year), layout=float(r.layout))

    leaves: List[TreeNode] = []

    def collect(n: TreeNode):
        if not n.children:
            leaves.append(n)
        for c in n.children:
            collect(c)

    collect(root)

    # x coordinates
    for i, leaf in enumerate(leaves):
        if leaf.name in tip_map:
            leaf.x = tip_map[leaf.name]["layout"]
        else:
            leaf.x = float(i)

    def set_internal_x(n: TreeNode) -> float:
        if not n.children:
            return n.x
        xs = [set_internal_x(c) for c in n.children]
        n.x = float(sum(xs) / len(xs))
        return n.x

    set_internal_x(root)

    # y: if tip years known, set leaves then parents = child.y - bl(child)
    use_abs = any(leaf.name in tip_map for leaf in leaves)
    if use_abs:
        for leaf in leaves:
            if leaf.name in tip_map:
                leaf.y = tip_map[leaf.name]["year"]
            else:
                leaf.y = 0.0

        def set_parents(n: TreeNode) -> None:
            for c in n.children:
                set_parents(c)
                # parent time from this child edge
                py = c.y - branch_length(c)
                # if multiple children, take min (oldest consistent)
                if n.children and c is n.children[0]:
                    n.y = py
                else:
                    n.y = min(n.y, py)

        set_parents(root)
        # fix root: recompute as min over children
        if root.children:
            root.y = min(c.y - branch_length(c) for c in root.children)
    else:
        # relative times from root
        root.y = 0.0

        def push(n: TreeNode):
            for c in n.children:
                c.y = n.y + branch_length(c)
                push(c)

        push(root)


def simmap_segments(
    root: TreeNode,
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """
    Expand each parent→child edge into geometric subsegments colored by state.
    Returns list of (x0,y0,x1,y1) and parallel list of states.
    Uses square-tree drawing (vertical then horizontal).
    """
    geoms: List[Tuple[float, float, float, float]] = []
    states: List[float] = []

    def walk(parent: TreeNode, child: TreeNode):
        # path: (parent.x, parent.y) --vert--> (parent.x, child.y) --horiz--> (child.x, child.y)
        # SIMMAP segments run along the full edge length in time (y).
        segs = child.segs or [Seg(state=np.nan, length=max(child.y - parent.y, 1e-12))]
        total = sum(s.length for s in segs) or 1e-12
        # parameterize full L-shaped path length in drawing space
        # Use time-only vertical coloring: distribute segments along y from parent.y to child.y
        y = parent.y
        x_vert = parent.x
        for s in segs:
            y2 = y + s.length
            # vertical piece for this time span
            geoms.append((x_vert, y, x_vert, y2))
            states.append(s.state)
            y = y2
        # horizontal connector at child time (state = last)
        last_state = segs[-1].state if segs else np.nan
        geoms.append((parent.x, child.y, child.x, child.y))
        states.append(last_state)
        for g in child.children:
            walk(child, g)

    for c in root.children:
        walk(root, c)
    return geoms, states


def simmap_segments_proportional(
    root: TreeNode,
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """
    Like simmap_segments, but L-path is colored with states proportional to
    segment lengths along the whole polyline (closer to MATLAB lineage paint).
    """
    geoms: List[Tuple[float, float, float, float]] = []
    states: List[float] = []

    def walk(parent: TreeNode, child: TreeNode):
        segs = child.segs or [Seg(state=np.nan, length=1.0)]
        pts = [
            (parent.x, parent.y),
            (parent.x, child.y),
            (child.x, child.y),
        ]
        # drop zero-length legs (same x,y) to avoid infinite subdivision
        legs: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        for a, b in zip(pts, pts[1:]):
            ll = abs(b[0] - a[0]) + abs(b[1] - a[1])
            if ll > 1e-15:
                legs.append((a, b, ll))
        if not legs:
            # degenerate edge: still record a point marker as tiny stub
            geoms.append((parent.x, parent.y, child.x, child.y))
            states.append(segs[-1].state if segs else np.nan)
        else:
            path_len = sum(ll for _, _, ll in legs)
            seg_total = sum(max(s.length, 0.0) for s in segs) or 1e-12
            # cumulative leg ends in path distance
            cum = []
            acc = 0.0
            for _, _, ll in legs:
                acc += ll
                cum.append(acc)

            def point_at(dist: float) -> Tuple[float, float]:
                dist = min(max(dist, 0.0), path_len)
                prev = 0.0
                for (p0, p1, ll), end in zip(legs, cum):
                    if dist <= end + 1e-15:
                        t = (dist - prev) / ll if ll else 0.0
                        return (
                            p0[0] + (p1[0] - p0[0]) * t,
                            p0[1] + (p1[1] - p0[1]) * t,
                        )
                    prev = end
                return legs[-1][1]

            path_pos = 0.0
            for s in segs:
                span = (max(s.length, 0.0) / seg_total) * path_len
                target = min(path_pos + span, path_len)
                if target - path_pos < 1e-15:
                    path_pos = target
                    continue
                # split only at leg boundaries inside (path_pos, target)
                cuts = [path_pos]
                for end in cum:
                    if path_pos < end < target:
                        cuts.append(end)
                cuts.append(target)
                for d0, d1 in zip(cuts, cuts[1:]):
                    if d1 - d0 < 1e-15:
                        continue
                    x0, y0 = point_at(d0)
                    x1, y1 = point_at(d1)
                    geoms.append((x0, y0, x1, y1))
                    states.append(s.state)
                path_pos = target

        for g in child.children:
            walk(child, g)

    for c in root.children:
        walk(root, c)
    return geoms, states

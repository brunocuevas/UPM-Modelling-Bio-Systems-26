"""Command-line entry points for UPM modelling 2026."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _add_common_sir_params(p: argparse.ArgumentParser) -> None:
    p.add_argument("--beta0", type=float, default=0.5625)
    p.add_argument("--sd", type=float, default=0.08, help="deleterious fitness cost s_d")
    p.add_argument("--lamb", type=float, default=0.10, help="Poisson mutation rate λ")
    p.add_argument("--mu", type=float, default=1.0 / (30 * 365))
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--N", type=float, default=1000.0)
    p.add_argument("--tfinal", type=float, default=100.0)
    p.add_argument("--out", type=Path, default=Path("sir_out.png"))


def cmd_sir_deleterious(args: argparse.Namespace) -> None:
    from .sir import simulate_deleterious

    res = simulate_deleterious(
        k_max=args.k_max,
        beta_0=args.beta0,
        s_d=args.sd,
        lamb=args.lamb,
        mu=args.mu,
        gamma=args.gamma,
        N=args.N,
        I0=args.I0,
        tfinal=args.tfinal,
        intervals=args.intervals,
    )
    t, y = res.t, res.y
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(t, y[0], label="S", color="black")
    ax.plot(t, y[1], label="I0", color="red")
    if y.shape[0] > 3:
        ax.plot(t, y[2:-1].sum(0), label="I variants", color="pink")
    ax.plot(t, y[-1], label="R", color="gray", linestyle="--")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


def cmd_sir_stochastic(args: argparse.Namespace) -> None:
    from .sir import integrate_tau_leap, model_base_rate_fun, prepare_population

    y0 = prepare_population(args.k_max, args.I0, args.N)
    rng = np.random.default_rng(args.seed)
    ts, Y = integrate_tau_leap(
        model_base_rate_fun,
        y0,
        args.tfinal,
        args.tau,
        args.beta0,
        args.sd,
        args.lamb,
        args.mu,
        args.gamma,
        args.N,
        rng=rng,
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ts, Y[:, 0], label="S", color="black")
    ax.plot(ts, Y[:, 1], label="I0", color="red")
    if Y.shape[1] > 3:
        ax.plot(ts, Y[:, 2:-1].sum(1), label="I variants", color="pink")
    ax.plot(ts, Y[:, -1], label="R", color="gray", linestyle="--")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


def cmd_sir_antigen(args: argparse.Namespace) -> None:
    from .sir import mutant_initial_state, simulate_antigen

    y0 = mutant_initial_state(
        args.K,
        args.i_bg,
        args.count,
        args.beta0,
        args.sd,
        args.lamb,
        args.mu,
        args.gamma,
        args.N,
    )
    res = simulate_antigen(
        y0,
        beta_0=args.beta0,
        s_d=args.sd,
        lamb=args.lamb,
        mu=args.mu,
        gamma=args.gamma,
        sigma=args.sigma,
        N=args.N,
        tfinal=args.tfinal,
        intervals=args.intervals,
    )
    t, y, K = res.t, res.y, res.K
    res_I = y[1 : 1 + K].sum(0) + y[1 + 2 * K : 1 + 3 * K].sum(0)
    mut_I = y[1 + K : 1 + 2 * K].sum(0) + y[1 + 3 * K : 1 + 4 * K].sum(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y[0], label="S0")
    ax.plot(t, res_I, label="resident I", color="black")
    ax.plot(t, mut_I, label="mutant I", color="black", linestyle="--")
    ax.plot(t, y[4 * K + 1], label="Sr")
    ax.plot(t, y[4 * K + 2], label="Sm")
    ax.plot(t, y[4 * K + 3], label="Srm")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


def cmd_plot_tree_antigenic(args: argparse.Namespace) -> None:
    from .phylodynamics import plot_antigenic_tree

    if not args.simmap.exists():
        raise SystemExit(f"missing simmap file: {args.simmap}")
    tips = args.tips if args.tips.exists() else None
    out = plot_antigenic_tree(
        args.simmap,
        tips,
        args.out,
        linewidth=args.lw,
        orientation="vertical" if args.vertical else "horizontal",
    )
    print(f"wrote {out}")


def cmd_plot_tree_netr(args: argparse.Namespace) -> None:
    from .phylodynamics import plot_tree_netR

    if not args.branches.exists():
        raise SystemExit(f"missing {args.branches}")
    if not args.fitness.exists():
        raise SystemExit(f"missing {args.fitness}")
    out = plot_tree_netR(
        args.branches,
        args.fitness,
        args.out,
        n_sub=args.n_sub,
        linewidth=args.lw,
        cmap_name=args.cmap,
        orientation="vertical" if args.vertical else "horizontal",
        vcenter=None if args.no_center else 1.0,
    )
    print(f"wrote {out}")


def cmd_plot_timeseries(args: argparse.Namespace) -> None:
    from .phylodynamics import plot_timeseries_columns

    if not args.timeseries.exists():
        raise SystemExit(f"missing {args.timeseries}")
    out = plot_timeseries_columns(
        args.timeseries,
        args.columns,
        args.out,
        logy=args.logy,
    )
    print(f"wrote {out}")


def cmd_plot_mutation_series(args: argparse.Namespace) -> None:
    from .phylodynamics import plot_mutation_series

    if not args.series.exists():
        raise SystemExit(f"missing {args.series}")
    out = plot_mutation_series(args.series, args.out, n_classes=args.n_classes)
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upm-modelling",
        description="UPM modelling 2026: SIR evolution models and phylodynamic plots",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("sir-deleterious", help="ODE SIR with deleterious mutations")
    _add_common_sir_params(d)
    d.add_argument("--k-max", type=int, default=20)
    d.add_argument("--I0", type=float, default=1.0)
    d.add_argument("--intervals", type=int, default=1000)
    d.set_defaults(func=cmd_sir_deleterious)

    s = sub.add_parser("sir-stochastic", help="Tau-leap deleterious SIR")
    _add_common_sir_params(s)
    s.add_argument("--k-max", type=int, default=5)
    s.add_argument("--I0", type=float, default=1.0)
    s.add_argument("--tau", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=32)
    s.set_defaults(gamma=0.25, N=100.0, tfinal=100.0, func=cmd_sir_stochastic)

    a = sub.add_parser("sir-antigen", help="Life-history + antigen evolution SIR")
    a.add_argument("--beta0", type=float, default=0.5625)
    a.add_argument("--sd", type=float, default=0.008)
    a.add_argument("--lamb", type=float, default=0.10)
    a.add_argument("--mu", type=float, default=1.0 / (30 * 365))
    a.add_argument("--gamma", type=float, default=0.25)
    a.add_argument("--sigma", type=float, default=0.045, help="cross-immunity escape")
    a.add_argument("--N", type=float, default=4e9)
    a.add_argument("--K", type=int, default=45)
    a.add_argument("--i-bg", type=int, default=3, help="mutant background class")
    a.add_argument("--count", type=float, default=1.0, help="initial mutant count")
    a.add_argument("--tfinal", type=float, default=1500.0)
    a.add_argument("--intervals", type=int, default=1500)
    a.add_argument("--out", type=Path, default=Path("sir_antigen.png"))
    a.set_defaults(func=cmd_sir_antigen)

    ta = sub.add_parser("plot-tree-antigenic", help="Phylogeny colored by antigenic SIMMAP")
    ta.add_argument("--simmap", type=Path, default=Path("out.simmapAntigenic"))
    ta.add_argument("--tips", type=Path, default=Path("out.tips"))
    ta.add_argument("--out", type=Path, default=Path("tree_antigenic.png"))
    ta.add_argument("--lw", type=float, default=1.2)
    ta.add_argument("--vertical", action="store_true")
    ta.set_defaults(func=cmd_plot_tree_antigenic)

    tn = sub.add_parser("plot-tree-netr", help="Phylogeny colored by Net R")
    tn.add_argument("--branches", type=Path, default=Path("out.branches"))
    tn.add_argument("--fitness", type=Path, default=Path("out.viralFitnessSeries"))
    tn.add_argument("--out", type=Path, default=Path("tree_netR.png"))
    tn.add_argument("--n-sub", type=int, default=4)
    tn.add_argument("--lw", type=float, default=1.0)
    tn.add_argument("--cmap", default="coolwarm")
    tn.add_argument("--vertical", action="store_true")
    tn.add_argument("--no-center", action="store_true")
    tn.set_defaults(func=cmd_plot_tree_netr)

    ts = sub.add_parser("plot-timeseries", help="Plot columns from out.timeseries")
    ts.add_argument("--timeseries", type=Path, default=Path("out.timeseries"))
    ts.add_argument("--columns", nargs="+", required=True)
    ts.add_argument("--out", type=Path, default=Path("timeseries.png"))
    ts.add_argument("--logy", action="store_true")
    ts.set_defaults(func=cmd_plot_timeseries)

    ms = sub.add_parser("plot-mutation-series", help="Plot out.mutationSeries")
    ms.add_argument("--series", type=Path, default=Path("out.mutationSeries"))
    ms.add_argument("--out", type=Path, default=Path("mutation_series.png"))
    ms.add_argument("--n-classes", type=int, default=10)
    ms.set_defaults(func=cmd_plot_mutation_series)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


# Thin dedicated entry points used by pyproject scripts
def main_sir_deleterious() -> None:
    main(["sir-deleterious", *(__import__("sys").argv[1:])])


def main_sir_antigen() -> None:
    main(["sir-antigen", *(__import__("sys").argv[1:])])


def main_plot_tree_antigenic() -> None:
    main(["plot-tree-antigenic", *(__import__("sys").argv[1:])])


def main_plot_tree_netr() -> None:
    main(["plot-tree-netr", *(__import__("sys").argv[1:])])

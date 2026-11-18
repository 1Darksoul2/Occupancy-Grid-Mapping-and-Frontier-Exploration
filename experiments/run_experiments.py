"""Batch experiments: maps x strategies x start poses x noise seeds.

Outputs (default results/offline/):
  summary.csv                    one row per run
  summary.md                     mean +- std table per map and strategy (paste into report)
  coverage_vs_distance.png       coverage curves, one panel per map
  coverage_vs_time.png
  bars_distance95.png            distance travelled to reach 95 % coverage
  final_maps.png                 final occupancy grids + trajectories
  lambda_sweep.png / .csv        (with --lambda-sweep) effect of the cost weight lambda

Usage:
  python experiments/run_experiments.py                    # full comparison (~5 min on 8 cores)
  python experiments/run_experiments.py --quick            # 1 start, 1 seed
  python experiments/run_experiments.py --lambda-sweep 0.05 0.15 0.3 0.6 1.0
"""

import argparse
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from aisr_explore import STRATEGIES, make_config          # noqa: E402
from aisr_explore.maps import MAPS                        # noqa: E402
from aisr_explore.sim2d import run_episode                # noqa: E402

# Reference categorical palette, fixed order (blue, orange, aqua, ...);
# line style + marker give a second, colour-independent encoding for print.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STYLES = ["-", "--", "-."]
MARKERS = ["o", "s", "^", "D", "v"]
LABELS = {"nearest": "Nearest frontier", "utility": "Utility (IG·e^(−λd))",
          "utility_commit": "Utility + commitment (ours)"}


def _job(args):
    map_name, strategy, start, seed, max_time, overrides = args
    cfg = make_config(strategy, **overrides)
    summary, trace, traj, state, truth = run_episode(map_name, cfg, start, seed=seed, max_time=max_time)
    summary["strategy"] = strategy
    summary.update({k: v for k, v in overrides.items()})
    return summary, trace, traj, state


def run_all(jobs, workers):
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            out = []
            for i, r in enumerate(pool.imap(_job, jobs), 1):
                out.append(r)
                print(f"  [{i}/{len(jobs)}] {r[0]['map']:8s} {r[0]['strategy']:15s} start={r[0]['start']} "
                      f"seed={r[0]['seed']} cov={r[0]['coverage']:.3f} t={r[0]['time']:.0f}s", flush=True)
    else:
        out = [_job(j) for j in jobs]
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    return out


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ms(vals, fmt="{:.1f}"):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if v.size == 0:
        return "n/a"
    return (fmt + " ± " + fmt).format(v.mean(), v.std(ddof=1) if v.size > 1 else 0.0)


def summary_table(rows, maps, strategies):
    head = ("| Map | Strategy | Runs | Time to 95 % [s] | Distance to 95 % [m] | Total distance [m] | "
            "Final coverage [%] | Map accuracy [%] | Goals | Escapes | Plan time [ms] |")
    lines = [head, "|" + "---|" * 11]
    for m in maps:
        for s in strategies:
            r = [x for x in rows if x["map"] == m and x["strategy"] == s]
            if not r:
                continue
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                m, LABELS.get(s, s), len(r),
                ms([x["t95"] for x in r]), ms([x["d95"] for x in r]), ms([x["distance"] for x in r]),
                ms([100 * x["coverage"] for x in r]), ms([100 * x["accuracy"] for x in r], "{:.2f}"),
                ms([x["n_goals"] for x in r]), ms([x["n_escapes"] for x in r]),
                ms([x["mean_plan_ms"] for x in r], "{:.0f}")))
    return "\n".join(lines)


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a96")
    ax.tick_params(colors="#55554f", labelsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(results, maps, strategies, xcol, xlabel, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, maps):
        for i, s in enumerate(strategies):
            traces = [tr for (sm, tr, _, _) in results if sm["map"] == m and sm["strategy"] == s]
            if not traces:
                continue
            xmax = max(tr[-1, xcol] for tr in traces)
            grid = np.linspace(0, xmax, 300)
            ys = np.array([np.interp(grid, tr[:, xcol], 100 * tr[:, 2], right=100 * tr[-1, 2]) for tr in traces])
            ax.fill_between(grid, ys.min(0), ys.max(0), color=COLORS[i], alpha=0.12, linewidth=0)
            ax.plot(grid, ys.mean(0), color=COLORS[i], linestyle=STYLES[i % 3], linewidth=2,
                    label=LABELS.get(s, s))
        ax.set_title(m.capitalize(), fontsize=11, color="#1d1d1b")
        ax.set_xlabel(xlabel, fontsize=10, color="#3d3d3a")
        _style_axes(ax)
    axes[0].set_ylabel("Coverage of free space [%]", fontsize=10, color="#3d3d3a")
    axes[0].set_ylim(0, 102)
    axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bars(rows, maps, strategies, key, ylabel, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.8 / len(strategies)
    x = np.arange(len(maps))
    for i, s in enumerate(strategies):
        means, stds = [], []
        for m in maps:
            v = np.array([r[key] for r in rows if r["map"] == m and r["strategy"] == s], float)
            v = v[~np.isnan(v)]
            means.append(v.mean() if v.size else 0)
            stds.append(v.std(ddof=1) if v.size > 1 else 0)
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, means, width * 0.92, yerr=stds, color=COLORS[i], label=LABELS.get(s, s),
               error_kw=dict(ecolor="#55554f", elinewidth=1, capsize=3), hatch=[None, "//", ".."][i % 3],
               edgecolor="white", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in maps])
    ax.set_ylabel(ylabel, fontsize=10, color="#3d3d3a")
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=len(strategies), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_final_maps(results, maps, strategies, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#b9b8b2", "#ffffff", "#1d1d1b"])   # unknown, free, occupied
    fig, axes = plt.subplots(len(maps), len(strategies),
                             figsize=(3.3 * len(strategies), 3.3 * len(maps)), squeeze=False)
    for r, m in enumerate(maps):
        for c, s in enumerate(strategies):
            ax = axes[r, c]
            sel = [x for x in results if x[0]["map"] == m and x[0]["strategy"] == s
                   and x[0]["start"] == 0 and x[0]["seed"] == 0]
            ax.set_xticks([]); ax.set_yticks([])
            if not sel:
                continue
            sm, tr, traj, state = sel[0]
            ax.imshow(state + 1, cmap=cmap, vmin=0, vmax=2, origin="lower", extent=[-6, 6, -6, 6],
                      interpolation="nearest")
            ax.plot(traj[:, 0], traj[:, 1], color=COLORS[c], linewidth=1.6)
            ax.plot(*traj[0], marker="o", color=COLORS[c], markersize=7, markeredgecolor="white")
            ax.set_xlim(-5.4, 5.4); ax.set_ylim(-5.4, 5.4)
            ax.set_title(f"{m} · {LABELS.get(s, s)}\n{sm['distance']:.0f} m, {sm['time']:.0f} s",
                         fontsize=8.5, color="#1d1d1b")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=list(MAPS))
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--starts", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-time", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "offline"))
    ap.add_argument("--quick", action="store_true", help="one start pose and one seed")
    ap.add_argument("--lambda-sweep", nargs="*", type=float, default=None,
                    help="also sweep lambda for the utility_commit strategy")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # Windows consoles default to cp1252
    except Exception:
        pass
    if a.quick:
        a.starts, a.seeds = [0], [0]
    os.makedirs(a.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    jobs = [(m, s, st, sd, a.max_time, {}) for m in a.maps for s in a.strategies
            for st in a.starts for sd in a.seeds]
    print(f"Running {len(jobs)} episodes on {a.workers} workers ...")
    results = run_all(jobs, a.workers)
    rows = [r[0] for r in results]
    write_csv(rows, os.path.join(a.out, "summary.csv"))
    table = summary_table(rows, a.maps, a.strategies)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)

    plot_curves(results, a.maps, a.strategies, 1, "Distance travelled [m]",
                os.path.join(a.out, "coverage_vs_distance.png"))
    plot_curves(results, a.maps, a.strategies, 0, "Time [s]",
                os.path.join(a.out, "coverage_vs_time.png"))
    plot_bars(rows, a.maps, a.strategies, "d95", "Distance to 95 % coverage [m]",
              os.path.join(a.out, "bars_distance95.png"))
    plot_bars(rows, a.maps, a.strategies, "t95", "Time to 95 % coverage [s]",
              os.path.join(a.out, "bars_time95.png"))
    plot_final_maps(results, a.maps, a.strategies, os.path.join(a.out, "final_maps.png"))

    if a.lambda_sweep:
        jobs = [(m, "utility_commit", st, sd, a.max_time, {"lambda_cost": lam})
                for lam in a.lambda_sweep for m in a.maps for st in a.starts for sd in a.seeds[:1]]
        print(f"Lambda sweep: {len(jobs)} episodes ...")
        sweep = [r[0] for r in run_all(jobs, a.workers)]
        write_csv(sweep, os.path.join(a.out, "lambda_sweep.csv"))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for i, m in enumerate(a.maps):
            mean = [np.nanmean([r["d95"] for r in sweep if r["map"] == m and r["lambda_cost"] == lam])
                    for lam in a.lambda_sweep]
            ax.plot(a.lambda_sweep, mean, color=COLORS[i], linestyle=STYLES[i % 3], marker=MARKERS[i],
                    markersize=8, markeredgecolor="white", linewidth=2, label=m.capitalize())
        ax.set_xlabel("Cost weight λ [1/m]", fontsize=10, color="#3d3d3a")
        ax.set_ylabel("Distance to 95 % coverage [m]", fontsize=10, color="#3d3d3a")
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "lambda_sweep.png"), dpi=200)
        plt.close(fig)
    print("Results written to", a.out)


if __name__ == "__main__":
    main()

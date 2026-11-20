"""Meeting figures as PNGs: uv run --group plots python scripts/figures.py runs/train/s3-n10

Writes to runs/figures/<run>/ (or --out):
  1-accuracy-by-certainty.png   accuracy and number of decisions at each certainty level
  2-threshold.png               human calls per episode against wrong actions caught
  3-episode.png                 episode-level certainty against episode failure, four ways
--audit keeps only the wrong actions that annotator blames on the agent.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calibrate import episode_scores, labelled
from pilot import decisions, episodes

from swarm_uncertainty.calibration import (
    AGGREGATORS,
    accuracy_by_level,
    certainty,
    threshold_sweep,
)
from swarm_uncertainty.metrics import auroc, bootstrap_ci

BLUE, LIGHT, GREY = "#1f4e8c", "#9fb3cf", "#777777"
KINDS = ("action", "plan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--audit")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path("runs/figures") / args.run.name
    out.mkdir(parents=True, exist_ok=True)
    records = decisions(args.run, args.audit)
    n_episodes = len(episodes(args.run))
    which = "agent's own errors" if args.audit else "all labelled errors"
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    data = {}
    for kind in KINDS:
        chosen = labelled(records, kind)
        data[kind] = ([certainty(r) for r in chosen], [r.step_label for r in chosen])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, height_ratios=[3, 2])
    for col, kind in enumerate(KINDS):
        levels = accuracy_by_level(*data[kind])
        xs = [lv.certainty for lv in levels]
        top, bottom = axes[0][col], axes[1][col]
        top.plot([0.3, 1], [0.3, 1], "--", color=GREY, linewidth=1, label="perfectly calibrated")
        top.plot(xs, [lv.accuracy for lv in levels], "o-", color=BLUE, label="observed")
        for lv in levels:
            top.annotate(
                f"{lv.accuracy:.0%}",
                (lv.certainty, lv.accuracy),
                (0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        wrong = sum(lv.errors for lv in levels)
        top.set(
            title=f"At the {kind}  ({wrong} wrong of {len(data[kind][0])})",
            ylabel="accuracy",
            ylim=(-0.05, 1.12),
        )
        top.legend(loc="lower left", fontsize=9, frameon=False)
        bars = bottom.bar(xs, [lv.count for lv in levels], width=0.07, color=LIGHT)
        bottom.bar_label(bars, fontsize=10)
        bottom.set(
            xlabel="certainty (share of 10 samples that agree)",
            ylabel="decisions",
            xticks=[i / 10 for i in range(3, 11)],
        )
    fig.suptitle(f"Certainty against accuracy, {which}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "1-accuracy-by-certainty.png", dpi=180)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for kind, style in zip(KINDS, ("o-", "s--"), strict=True):
        points = [p for p in threshold_sweep(*data[kind]) if p.threshold != float("inf")]
        total = points[0].missed + points[0].caught
        xs = [p.interventions / n_episodes for p in points]
        ax.plot(
            xs,
            [p.caught for p in points],
            style,
            color=BLUE if kind == "action" else GREY,
            label=f"at the {kind} ({total} wrong)",
        )
        for p, x in zip(points[1:], xs[1:], strict=True):
            ax.annotate(
                f"<{p.threshold:.1f}",
                (x, p.caught),
                (6, -12),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set(
        xlabel="human calls per episode",
        ylabel="wrong actions caught",
        title=f"What each threshold buys, {which}\n(label: call a human when certainty is below)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "2-threshold.png", dpi=180)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.38
    for i, kind in enumerate(KINDS):
        for j, how in enumerate(AGGREGATORS):
            groups = list(episode_scores({str(args.run): records}, how, kind).values())
            pairs = [p for g in groups for p in g]

            def stat(ps: list) -> float:
                return auroc([s for s, _ in ps], [y for _, y in ps])

            value = stat(pairs)
            lo, hi = bootstrap_ci(groups, stat)
            x = j + (i - 0.5) * width
            ax.bar(
                x,
                value - 0.5,
                width,
                bottom=0.5,
                color=BLUE if kind == "action" else LIGHT,
                label=f"{kind} decisions" if j == 0 else None,
            )
            ax.errorbar(x, value, [[value - lo], [hi - value]], color="black", capsize=4)
            ax.annotate(
                f"{value:.2f}",
                (x, hi),
                (0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=1)
    ax.set(
        xticks=range(len(AGGREGATORS)),
        xticklabels=list(AGGREGATORS),
        ylim=(0.3, 1.0),
        ylabel="AUROC against the episode failing (0.5 = chance)",
        title="Certainty combined over the episode, with 95% intervals",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "3-episode.png", dpi=180)
    print(f"wrote 3 figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

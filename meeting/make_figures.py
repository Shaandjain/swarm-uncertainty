"""Meeting figures from the s3-n10 train run (72 tasks, 143 episodes, 10 samples per decision).
Numbers are the printed output of scripts/calibrate.py and scripts/pilot.py on that run,
with the Sonnet 5.5 audit (agent's own errors only) unless marked otherwise.
Run: uv run --group plots python meeting/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
BLUE, LIGHT, GREY = "#1f4e8c", "#9fb3cf", "#777777"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# (certainty, decisions, accuracy)
LEVELS = {
    "action (7 wrong of 212)": [
        (1.0, 201, 0.980),
        (0.9, 4, 0.75),
        (0.8, 1, 0.0),
        (0.7, 4, 0.75),
        (0.6, 2, 1.0),
    ],
    "plan (6 wrong of 168)": [
        (1.0, 110, 0.973),
        (0.9, 18, 0.944),
        (0.8, 6, 1.0),
        (0.7, 15, 1.0),
        (0.6, 8, 0.875),
        (0.5, 9, 0.889),
        (0.4, 2, 1.0),
    ],
}
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, height_ratios=[3, 2])
for col, (name, rows) in enumerate(LEVELS.items()):
    rows = sorted(rows)
    xs, ns, acc = zip(*rows)
    top, bottom = axes[0][col], axes[1][col]
    top.plot([0.3, 1], [0.3, 1], "--", color=GREY, lw=1, label="perfectly calibrated")
    top.plot(xs, acc, "o-", color=BLUE, label="observed")
    for x, a in zip(xs, acc):
        top.annotate(
            f"{a:.0%}", (x, a), (0, 8), textcoords="offset points", ha="center", fontsize=10
        )
    top.set(title=f"At the {name}", ylabel="accuracy", ylim=(-0.05, 1.14))
    top.legend(loc="lower left", fontsize=9, frameon=False)
    bars = bottom.bar(xs, ns, width=0.07, color=LIGHT)
    bottom.bar_label(bars, fontsize=10)
    bottom.set(
        xlabel="certainty (share of 10 samples that agree)",
        ylabel="decisions",
        xticks=[i / 10 for i in range(3, 11)],
    )
fig.suptitle("Certainty against accuracy, agent's own errors", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "1-accuracy-by-certainty.png", dpi=180)

# (call a human when certainty is below, calls, caught) over 143 episodes
SWEEP = {
    "action (7 wrong)": [(0.6, 0, 0), (0.7, 2, 0), (0.8, 6, 1), (0.9, 7, 2), (1.0, 11, 3)],
    "plan (6 wrong)": [
        (0.4, 0, 0),
        (0.5, 2, 0),
        (0.6, 11, 1),
        (0.7, 19, 2),
        (0.8, 34, 2),
        (0.9, 40, 2),
        (1.0, 58, 3),
    ],
}
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for (name, pts), style, color in zip(SWEEP.items(), ("o-", "s--"), (BLUE, GREY)):
    xs = [c / 143 for _, c, _ in pts]
    ax.plot(xs, [k for _, _, k in pts], style, color=color, label=f"at the {name}")
    for (t, _, k), x in zip(pts[1:], xs[1:]):
        ax.annotate(
            f"<{t:.1f}", (x, k), (5, -13), textcoords="offset points", fontsize=9, color=color
        )
ax.axvline(0.25, color="black", lw=1, ls=":")
ax.annotate("budget: 0.25 calls", (0.25, 3.2), (5, 0), textcoords="offset points", fontsize=10)
ax.set(
    xlabel="human calls per episode",
    ylabel="agent's own errors caught",
    ylim=(-0.2, 3.6),
    title="What each threshold buys\n(label = call a human when certainty is below this)",
)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "2-threshold.png", dpi=180)

# episode-level AUROC against episode failure, all labelled errors: (value, low, high)
EPISODE = {
    "action decisions\n(136 episodes, 20 failed)": {
        "mean": (0.643, 0.530, 0.758),
        "product": (0.639, 0.528, 0.752),
        "min": (0.639, 0.528, 0.752),
        "max": (0.550, 0.500, 0.621),
    },
    "plan decisions\n(127 episodes, 18 failed)": {
        "mean": (0.589, 0.439, 0.742),
        "product": (0.593, 0.445, 0.738),
        "min": (0.594, 0.440, 0.748),
        "max": (0.538, 0.395, 0.694),
    },
}
fig, ax = plt.subplots(figsize=(9, 5.2))
w = 0.38
for i, ((name, vals), color) in enumerate(zip(EPISODE.items(), (BLUE, LIGHT))):
    for j, (how, (v, lo, hi)) in enumerate(vals.items()):
        x = j + (i - 0.5) * w
        ax.bar(x, v - 0.5, w, bottom=0.5, color=color, label=name if j == 0 else None)
        ax.errorbar(x, v, [[v - lo], [hi - v]], color="black", capsize=4)
        ax.annotate(
            f"{v:.2f}", (x, hi), (0, 4), textcoords="offset points", ha="center", fontsize=10
        )
ax.axhline(0.5, color=GREY, ls="--", lw=1)
ax.annotate(
    "chance", (3.45, 0.5), (0, 4), textcoords="offset points", ha="right", fontsize=10, color=GREY
)
ax.set(
    xticks=range(4),
    xticklabels=["mean", "product", "min", "max"],
    ylim=(0.3, 0.95),
    ylabel="AUROC against the episode failing",
    title="Certainty combined over the whole episode (95% intervals)",
)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "3-episode.png", dpi=180)

# 5 vs 10 samples, action, agent's own errors (Sonnet audit both)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
axes[0].bar(
    ["5 samples\n(3 of 14)", "10 samples\n(3 of 7)"],
    [3 / 14, 3 / 7],
    color=[LIGHT, BLUE],
    width=0.5,
)
axes[0].set(
    ylim=(0, 1),
    ylabel="share of agent's own errors flagged",
    title="Errors flagged by any disagreement",
)
for x, v in enumerate((3 / 14, 3 / 7)):
    axes[0].annotate(f"{v:.0%}", (x, v), (0, 5), textcoords="offset points", ha="center")
for x, (v, lo, hi), c in zip((0, 1), ((0.596, 0.485, 0.703), (0.693, 0.480, 0.981)), (LIGHT, BLUE)):
    axes[1].bar(x, v - 0.5, 0.5, bottom=0.5, color=c)
    axes[1].errorbar(x, v, [[v - lo], [hi - v]], color="black", capsize=5)
    axes[1].annotate(f"{v:.2f}", (x, hi), (0, 4), textcoords="offset points", ha="center")
axes[1].axhline(0.5, color=GREY, ls="--", lw=1)
axes[1].set(
    xticks=[0, 1],
    xticklabels=["5 samples", "10 samples"],
    ylim=(0.3, 1.05),
    ylabel="AUROC (0.5 = chance)",
    title="Does certainty rank wrong actions?",
)
fig.suptitle("Five samples against ten, at the action", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "4-five-vs-ten.png", dpi=180)
print("wrote 4 figures to", OUT)

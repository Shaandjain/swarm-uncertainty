"""Figures for the Nov 27 meeting, drawn from the runs under runs/ (not hard-coded).
Run: uv run --group plots python meeting/nov27_figures.py
Needs the audits (scripts/audit.py), reviews (scripts/review.py) and replays
(scripts/intervene.py) of the train and test runs. Writes meeting/nov27-*.png.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from calibrate import labelled
from pilot import decisions
from signals import rows_of, signals

from swarm_uncertainty.calibration import certainty
from swarm_uncertainty.signals import AGENT_KINDS

for font in (HERE / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update(
    {
        "font.family": ["EB Garamond", "serif"],
        "font.size": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)
NAVY, LIGHT, GREY, RED = "#1f3a5f", "#a9bcd6", "#8a8a8a", "#a8503a"
ANNOTATOR = "claude-sonnet-5-5"
RUNS = Path("runs")
UNSELECTED = [RUNS / "train/s3-n10", RUNS / "test/s3-n10-test"]
ALL = [*UNSELECTED, RUNS / "target/s3-target", RUNS / "target/s4-target"]


def bucket(c: float) -> str:
    return "all 10 agree" if c >= 1 else "7 to 9 agree" if c >= 0.7 else "6 or fewer agree"


BUCKETS = ("all 10 agree", "7 to 9 agree", "6 or fewer agree")


def certainty_figure() -> None:
    """Share of decisions that are wrong at each level of agreement."""
    records = [r for run in UNSELECTED for r in decisions(run, ANNOTATOR)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind in zip(axes, ("action", "plan"), strict=True):
        wrong, total = Counter(), Counter()
        for r in labelled(records, kind):
            total[bucket(certainty(r))] += 1
            wrong[bucket(certainty(r))] += r.step_label
        rates = [wrong[b] / total[b] for b in BUCKETS]
        bars = ax.bar(BUCKETS, rates, color=[NAVY, LIGHT, LIGHT], width=0.6)
        for bar, b in zip(bars, BUCKETS, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{wrong[b]} of {total[b]}",
                ha="center",
            )
        ax.set(title=f"At the {kind}", ylim=(0, 0.45))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_ylabel("decisions that were wrong")
    fig.supxlabel("how many of the 10 samples chose the same action", fontsize=15)
    fig.tight_layout()
    fig.savefig(HERE / "nov27-1-certainty.png")


def sources_figure() -> None:
    """Wrong actions by where the error came from."""
    groups = {
        "slip": "Agent: right product,\nwrong item ID",
        "interpretation": "Agent: misread the\nrequest or policy",
        "policy": "Agent: misread the\nrequest or policy",
    }
    other = "Not the agent:\nbenchmark error"
    counts: Counter = Counter()
    for run in ALL:
        with (run / f"audit-{ANNOTATOR}.jsonl").open() as handle:
            for a in map(json.loads, handle):
                counts[groups.get(a["category"], other)] += 1
    names = [*dict.fromkeys(groups.values()), other]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(names, [counts[n] for n in names], color=[NAVY, NAVY, GREY], width=0.6)
    ax.bar_label(bars, padding=3)
    ax.set(
        ylabel="wrong actions",
        title=f"Where the {sum(counts.values())} wrong actions came from",
        ylim=(0, max(counts.values()) * 1.15),
    )
    fig.savefig(HERE / "nov27-2-sources.png")


def signals_figure() -> None:
    """How many of the agent's own errors each signal catches, and its false alarms."""
    rows = [r for run in UNSELECTED for r in rows_of(run, ANNOTATOR)]
    rows = [r for r in rows if len(r["reviews"]) == 2]
    kinds = [r["kind"] for r in rows]
    shown = {
        "action split": "Action samples disagree",
        "plan split": "Plan samples disagree",
        "plan != call": "Call differs from plan samples",
        "review wrong (haiku-5-5)": "Review by the same model",
        "review wrong (sonnet-5-5)": "Review by a stronger model",
    }
    errors = sum(k in AGENT_KINDS for k in kinds)
    right = kinds.count("right")
    flags = signals(rows)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, label) in enumerate(shown.items()):
        caught = sum(f and k in AGENT_KINDS for f, k in zip(flags[name], kinds, strict=True))
        alarms = sum(f and k == "right" for f, k in zip(flags[name], kinds, strict=True))
        ax.barh(i, caught, color=RED if name.startswith("review") else NAVY, height=0.6)
        ax.text(caught + 0.12, i, f"{caught} of {errors}", va="center")
        ax.text(errors + 1.6, i, f"{alarms / right:.1%}", va="center", ha="right", color=GREY)
    ax.text(errors + 1.6, -0.75, "right actions\nflagged", ha="right", color=GREY, fontsize=13)
    ax.set_yticks(range(len(shown)), list(shown.values()))
    ax.invert_yaxis()
    ax.set(
        xlim=(0, errors + 1.7),
        xticks=range(0, errors + 1, 2),
        xlabel="agent's own errors caught",
    )
    ax.set_title(f"Errors each signal catches ({errors} errors in {len(rows)} actions)", loc="left")
    fig.savefig(HERE / "nov27-3-signals.png")


def replay_figure() -> None:
    """Share of fresh samples that make the right call at the wrong-ID states."""
    arms = {
        "control": "Nothing added",
        "verify": "Told to check\nthe call",
        "feedback-claude-haiku-5-5": "Shown the\nreviewer's concern",
    }
    shares, states = [], 0
    for arm in arms:
        gold = n = 0
        for run in UNSELECTED:
            with (run / f"intervene-{arm}.jsonl").open() as handle:
                for row in map(json.loads, handle):
                    if row["kind"] == "slip":
                        n += 1
                        gold += row["after"].count("gold") / len(row["after"])
        shares.append(gold / n)
        states = n
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(list(arms.values()), shares, color=[GREY, LIGHT, NAVY], width=0.6)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in shares], padding=3)
    ax.set(
        ylim=(0, 1),
        ylabel="samples that make the right call",
        title=f"Fixing a wrong item ID ({states} cases, 10 samples each)",
    )
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.savefig(HERE / "nov27-4-replay.png")


def box(ax, x, y, w, h, text, *, fill="white", dashed=False, bold=False):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fill,
            ec=NAVY,
            lw=1.3,
            ls="--" if dashed else "-",
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=13, fontweight="bold" if bold else None)


def arrow(ax, start, end, label="", *, offset=(0, 0.14), curve=0.0, dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": NAVY,
            "lw": 1.3,
            "ls": "--" if dashed else "-",
            "connectionstyle": f"arc3,rad={curve}",
        },
    )
    if label:
        mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=12, color=GREY)


def architecture_figure() -> None:
    """The first setup (agreement between samples) and the current one (review of the call)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6))
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 3.7))
        ax.axis("off")
    w, h = 1.9, 0.9

    ax = axes[0]
    ax.set_title("First setup: agreement between samples", loc="left")
    box(ax, 1.1, 2.4, w, h, "Customer")
    box(ax, 3.6, 2.4, w, h, "Agent", fill=LIGHT)
    box(ax, 6.1, 2.4, w, h, "Sample the\ndecision 10 times")
    box(ax, 8.6, 2.4, w, h, "Uncertainty U\nfrom agreement")
    box(ax, 8.6, 0.6, w, h, "Act on the\nmajority")
    box(ax, 6.1, 0.6, w, h, "Ask the customer\none question")
    arrow(ax, (2.05, 2.5), (2.65, 2.5))
    arrow(ax, (2.65, 2.3), (2.05, 2.3))
    arrow(ax, (4.55, 2.4), (5.15, 2.4))
    arrow(ax, (7.05, 2.4), (7.65, 2.4))
    arrow(ax, (8.6, 1.95), (8.6, 1.05))
    ax.text(8.75, 1.5, r"$U < \tau$", va="center", fontsize=14)
    arrow(ax, (8.0, 1.95), (6.7, 1.05))
    ax.text(7.1, 1.6, r"$U \geq \tau$", va="center", ha="right", fontsize=14)
    ax.plot([5.15, 1.1], [0.6, 0.6], color=NAVY, lw=1.3)
    arrow(ax, (1.1, 0.6), (1.1, 1.95))

    ax = axes[1]
    ax.set_title("Current setup: review of the call", loc="left")
    box(ax, 1.1, 2.2, w, h, "Customer")
    box(ax, 3.6, 2.2, w, h, "Agent", fill=LIGHT)
    box(ax, 6.1, 2.2, w, h, "Proposed call")
    box(ax, 8.6, 2.2, w, h, "Reviewer\n(policy, conversation,\ncall)", fill=LIGHT)
    box(ax, 8.6, 0.6, w, h, "Act")
    box(ax, 3.6, 0.6, w, h, "Human", dashed=True)
    arrow(ax, (2.05, 2.3), (2.65, 2.3))
    arrow(ax, (2.65, 2.1), (2.05, 2.1))
    arrow(ax, (4.55, 2.2), (5.15, 2.2))
    arrow(ax, (7.05, 2.2), (7.65, 2.2))
    arrow(ax, (8.6, 1.75), (8.6, 1.05))
    ax.text(8.75, 1.4, "sound", va="center", fontsize=13, color=GREY)
    arrow(ax, (8.2, 2.68), (4.0, 2.68), curve=0.25)
    ax.text(
        6.1, 3.45, "wrong: the concern goes back to the agent", ha="center", fontsize=13, color=GREY
    )
    arrow(ax, (3.6, 1.75), (3.6, 1.05), dashed=True)
    ax.text(3.75, 1.4, "still disagree (planned)", va="center", fontsize=13, color=GREY)
    fig.tight_layout()
    fig.savefig(HERE / "nov27-0-architecture.png")


def equations() -> None:
    """The uncertainty measure, the threshold fit and the intervention rule, as images."""
    lines = {
        "uncertainty": r"$U(s) \;=\; 1 \;-\; \frac{1}{N}\,\max_{a}\; n_a(s), \qquad N = 10$",
        "threshold": (
            r"$\tau^{*} \;=\; \arg\max_{\tau}\; \mathrm{caught}(\tau)"
            r"\quad \mathrm{subject\ to} \quad "
            r"\frac{\mathrm{calls}(\tau)}{\mathrm{episodes}} \;\leq\; B$"
        ),
        "rule": (
            r"$\mathrm{intervene\ when} \quad "
            r"P(\mathrm{error} \mid \mathrm{flag}) \,\cdot\, q \;>\; c$"
        ),
    }
    with plt.rc_context({"mathtext.fontset": "stix"}):
        for name, text in lines.items():
            fig = plt.figure(figsize=(6, 0.7))
            fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=19)
            fig.savefig(HERE / f"nov27-eq-{name}.png", pad_inches=0.05)
            plt.close(fig)


if __name__ == "__main__":
    architecture_figure()
    equations()
    certainty_figure()
    sources_figure()
    signals_figure()
    replay_figure()

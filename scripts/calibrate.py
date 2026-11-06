"""Certainty against accuracy: uv run python scripts/calibrate.py runs/train/s1 runs/train/s3 --pool

Prints, at the action and at the plan:
  - accuracy and number of decisions at each certainty level (1 - U), with the calibration error;
  - a threshold sweep: a human is called whenever certainty < t, and each row shows how many
    decisions that is, how many wrong actions it catches, and the error rate left over;
  - episode-level certainty (mean, product, min, max over the episode's decisions) scored
    against whether the episode failed, by AUROC with a task-bootstrap interval.

Decisions follow pilot.py: labelled, before the first wrong write, plans scored apart from
actions; --audit keeps only the wrong actions that annotator blames on the agent; --pool
scores the runs as one. --figures DIR also writes PNGs (needs: uv sync --group plots).
--budget B prints the threshold that catches the most wrong actions with at most B human calls
per episode, as the `threshold` value a config takes.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from pilot import decisions, episodes

from swarm_uncertainty.calibration import (
    AGGREGATORS,
    Level,
    SweepPoint,
    accuracy_by_level,
    as_u_threshold,
    calibration_error,
    certainty,
    episode_certainty,
    fit_threshold,
    threshold_sweep,
)
from swarm_uncertainty.metrics import auroc, bootstrap_ci
from swarm_uncertainty.records import DecisionRecord

KINDS = ("action", "plan")


def labelled(records: list[DecisionRecord], kind: str) -> list[DecisionRecord]:
    return [
        r
        for r in records
        if (r.decision_type == "plan") == (kind == "plan")
        and r.step_label is not None
        and not r.extra.get("after_divergence")
    ]


def print_levels(levels: list[Level]) -> None:
    print("    certainty  decisions   share  accuracy  errors")
    for lv in levels:
        print(
            f"    {lv.certainty:9.2f}  {lv.count:9d}  {lv.share:6.1%}  {lv.accuracy:8.1%}"
            f"  {lv.errors:6d}"
        )
    print(f"    calibration error {calibration_error(levels):.3f}")


def print_sweep(points: list[SweepPoint], episodes_seen: int) -> None:
    print("    call human if certainty <   calls  per episode  caught  missed  error left")
    for p in points:
        t = "everything" if p.threshold == float("inf") else f"{p.threshold:.2f}"
        per_episode = p.interventions / episodes_seen if episodes_seen else 0.0
        print(
            f"    {t:>24}  {p.interventions:6d}  {per_episode:11.2f}  {p.caught:6d}"
            f"  {p.missed:6d}  {p.residual_error_rate:10.1%}"
        )


def episode_scores(
    runs: dict[str, list[DecisionRecord]], how: str, kind: str
) -> dict[str, list[tuple[float, int]]]:
    """(uncertainty, failed) per episode, grouped by task for the bootstrap."""
    by_task: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for run, records in runs.items():
        outcome = {(e["task_id"], e["trial"]): int(e["reward"] < 1.0) for e in episodes(Path(run))}
        for (task, trial), c in episode_certainty(records, how, kind).items():
            if (task, trial) in outcome:
                by_task[f"{run}:{task}"].append((1.0 - c, outcome[(task, trial)]))
    return by_task


def print_episodes(runs: dict[str, list[DecisionRecord]], kind: str) -> None:
    print(f"  episode level, {kind} decisions vs. episode failure:")
    for how in AGGREGATORS:
        groups = list(episode_scores(runs, how, kind).values())
        pairs = [p for g in groups for p in g]
        if not pairs:
            return
        scores, failed = zip(*pairs, strict=True)

        def stat(ps: list[tuple[float, int]]) -> float:
            return auroc([s for s, _ in ps], [y for _, y in ps])

        lo, hi = bootstrap_ci(groups, stat)
        print(
            f"    {how:8} episodes {len(pairs):4d}  failed {sum(failed):3d}"
            f"  AUROC {auroc(scores, failed):.3f} ({lo:.3f} to {hi:.3f})"
        )


def figures(out: Path, name: str, by_kind: dict[str, list[Level]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(by_kind), figsize=(5 * len(by_kind), 6), sharex=True)
    for col, (kind, levels) in enumerate(by_kind.items()):
        top, bottom = axes[0][col], axes[1][col]
        xs = [lv.certainty for lv in levels]
        top.plot(xs, [lv.accuracy for lv in levels], "o-", color="#1f4e8c")
        top.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        for lv in levels:
            top.annotate(
                f"n={lv.count}",
                (lv.certainty, lv.accuracy),
                (4, -14),
                "data",
                "offset points",
                fontsize=8,
                color="#3d3d3d",
            )
        top.set(title=f"at the {kind}", ylabel="accuracy", ylim=(0, 1.02))
        bottom.bar(xs, [lv.count for lv in levels], width=0.08, color="#9fb3cf")
        bottom.set(xlabel="certainty (share of samples agreeing)", ylabel="decisions")
    fig.tight_layout()
    path = out / f"{name}-accuracy-by-certainty.png"
    fig.savefig(path, dpi=200)
    print(f"  wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--audit", help="annotator whose audit decides which errors count")
    parser.add_argument("--pool", action="store_true", help="score the runs as one")
    parser.add_argument("--figures", type=Path, help="directory for PNGs")
    parser.add_argument(
        "--budget",
        type=float,
        help="human calls per episode allowed; prints the threshold that catches the most errors",
    )
    args = parser.parse_args()

    loaded = {str(run): decisions(run, args.audit) for run in args.runs}
    batches = {"pooled": loaded} if args.pool else {run: {run: rs} for run, rs in loaded.items()}
    for name, runs in batches.items():
        print(name if not args.pool else f"pooled: {', '.join(runs)}")
        if args.audit:
            print(f"  wrong actions: only those audit-{args.audit} calls the agent's own")
        n_episodes = sum(len(episodes(Path(run))) for run in runs)
        records = [r for rs in runs.values() for r in rs]
        by_kind = {}
        for kind in KINDS:
            chosen = labelled(records, kind)
            if not chosen:
                continue
            cs = [certainty(r) for r in chosen]
            ys = [r.step_label for r in chosen]
            print(f"  at the {kind}: {len(chosen)} labelled decisions, {sum(ys)} wrong")
            by_kind[kind] = accuracy_by_level(cs, ys)
            print_levels(by_kind[kind])
            points = threshold_sweep(cs, ys)
            print_sweep(points, n_episodes)
            if args.budget is not None:
                best = fit_threshold(points, n_episodes, args.budget)
                if best is None:
                    print(f"    no threshold calls a human within {args.budget} per episode")
                else:
                    print(
                        f"    fit for <= {args.budget} calls per episode: threshold "
                        f"{as_u_threshold(best, cs)} (escalate when U >= it), catches "
                        f"{best.caught} of {best.caught + best.missed} with "
                        f"{best.interventions} calls"
                    )
            print_episodes(runs, kind)
        if args.figures and by_kind:
            figures(args.figures, Path(name).name, by_kind)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

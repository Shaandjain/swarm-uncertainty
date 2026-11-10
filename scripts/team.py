"""Does a team of models see what one model misses? uv run python scripts/team.py runs/target/s3-target

Reads the run's replays at one cut point (--at, default plan), one file per model, made by
scripts/replay.py with the same --correct so every model saw the same decisions. The run's
own worker is the actor. For each decision it scores three signals (see team.py) and prints:
  - AUROC of each signal against the action being wrong, with a task-bootstrap interval;
  - the confident errors (actor's samples all agree, action wrong) and how many the team flags;
  - false alarms: right actions the team flags;
  - how often the team's majority is the answer key's action when the actor's is not;
  - what the extra models cost per decision.
--audit keeps only the wrong actions that annotator blames on the agent.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from swarm_uncertainty.actions import action_key
from swarm_uncertainty.labels import AGENT_ERRORS
from swarm_uncertainty.metrics import auroc, bootstrap_ci
from swarm_uncertainty.team import TeamSignal, team_signal

Decision = tuple[str, int, int]


def load(run: Path, at: str) -> tuple[dict[Decision, dict[str, list[dict]]], dict[str, float]]:
    """Replay rows per decision and model, and each model's cost per decision."""
    by_decision: dict[Decision, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    cost: dict[str, float] = defaultdict(float)
    for path in sorted(run.glob("replay-*-default*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for row in map(json.loads, handle):
                if row.get("at", "action") != at:
                    continue
                key = (row["task_id"], row["trial"], row["decision_index"])
                by_decision[key][row["model"]].append(row)
                cost[row["model"]] += row["cost_usd"]
    return by_decision, {m: c / len(by_decision) for m, c in cost.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--at", choices=("action", "plan", "proposal"), default="plan")
    parser.add_argument("--audit", help="annotator whose audit decides which errors count")
    args = parser.parse_args()

    actor = json.loads((args.run / "config.json").read_text())["worker_models"][0]
    by_decision, cost = load(args.run, args.at)
    models = sorted({m for rows in by_decision.values() for m in rows})
    blame = {}
    if args.audit:
        with (args.run / f"audit-{args.audit}.jsonl").open(encoding="utf-8") as handle:
            blame = {
                (r["task_id"], r["trial"], r["decision_index"]): r["category"]
                for r in map(json.loads, handle)
            }

    scored: list[tuple[Decision, int, TeamSignal, bool, bool]] = []
    for key, rows in by_decision.items():
        if actor not in rows or len(rows) < len(models):
            continue
        label = rows[actor][0]["original_label"]
        if label == 1 and args.audit and blame.get(key) not in AGENT_ERRORS | {None}:
            continue
        keys = {m: [action_key(r["tool"], r["args"]) for r in rs] for m, rs in rows.items()}
        signal = team_signal(keys, actor)
        gold = {
            action_key(r["tool"], r["args"])
            for rs in rows.values()
            for r in rs
            if r["matches_gold"]
        }
        scored.append(
            (key, label, signal, signal.actor_majority in gold, signal.team_majority in gold)
        )

    wrong = [s for s in scored if s[1] == 1]
    right = [s for s in scored if s[1] == 0]
    print(f"{args.run} at the {args.at}: {len(wrong)} wrong + {len(right)} right decisions")
    print(f"  team: {', '.join(models)}; actor {actor}")
    if args.audit:
        print(f"  wrong actions: only those audit-{args.audit} calls the agent's own")

    by_task: dict[str, list[tuple[int, TeamSignal]]] = defaultdict(list)
    for key, label, signal, _, _ in scored:
        by_task[key[0]].append((label, signal))
    for name in ("self_uncertainty", "disagreement", "pooled"):

        def stat(items: list[tuple[int, TeamSignal]], field: str = name) -> float:
            return auroc([getattr(s, field) for _, s in items], [y for y, _ in items])

        lo, hi = bootstrap_ci(list(by_task.values()), stat)
        everything = [item for items in by_task.values() for item in items]
        print(f"  AUROC {name:17} {stat(everything):.3f} ({lo:.3f} to {hi:.3f})")

    def flagged(items: list, field: str) -> int:
        return sum(getattr(s[2], field) > 0 for s in items)

    confident = [s for s in wrong if s[2].self_uncertainty == 0]
    print(
        f"  wrong actions flagged: self {flagged(wrong, 'self_uncertainty')}, "
        f"team disagreement {flagged(wrong, 'disagreement')}, of {len(wrong)}"
    )
    print(
        f"  confident errors (actor unanimous and wrong): {len(confident)}, "
        f"team disagreement flags {flagged(confident, 'disagreement')}"
    )
    print(
        f"  false alarms on right actions: self {flagged(right, 'self_uncertainty')}, "
        f"team disagreement {flagged(right, 'disagreement')}, of {len(right)}"
    )
    fixable = [s for s in wrong if not s[3]]
    print(
        f"  wrong where the actor's plan is not the answer key's: {len(fixable)}, "
        f"team majority is the answer key's on {sum(s[4] for s in fixable)}"
    )
    broken = sum(s[3] and not s[4] for s in right)
    print(f"  right actions where following the team majority would leave the key: {broken}")
    extra = sum(c for m, c in cost.items() if m != actor)
    print(f"  cost per decision: actor ${cost.get(actor, 0):.3f}, other models ${extra:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

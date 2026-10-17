"""Pilot numbers for RQ1: uv run python scripts/pilot.py runs/s1 [runs/s0]

Prints to the terminal. Decisions after the first wrong write in an episode are left out of
the RQ1 numbers (first-divergence rule).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from swarm_uncertainty.metrics import auroc, bootstrap_ci, mean_pass_at_k
from swarm_uncertainty.records import read_records


def episodes(run: Path) -> list[dict]:
    with (run / "episodes.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome_summary(run: Path) -> None:
    eps = episodes(run)
    by_task: dict[str, list[int]] = defaultdict(list)
    for e in eps:
        by_task[e["task_id"]].append(int(e["reward"] == 1.0))
    trials = min(len(v) for v in by_task.values())
    db = [e["db_reward"] for e in eps if e["db_reward"] is not None]
    cost = [e["agent_cost_usd"] + e["user_cost_usd"] + e["sample_cost_usd"] for e in eps]
    print(f"{run.name}: {len(eps)} episodes over {len(by_task)} tasks")
    print(f"  reward {sum(e['reward'] for e in eps) / len(eps):.3f}", end="")
    print(f"   db reward {sum(db) / len(db):.3f}" if db else "", end="")
    for k in range(1, trials + 1):
        print(f"   pass^{k} {mean_pass_at_k(by_task, k):.3f}", end="")
    print(f"\n  decisions per episode {sum(e['decisions'] for e in eps) / len(eps):.2f}")
    print(f"  cost per episode ${sum(cost) / len(cost):.4f}, total ${sum(cost):.2f}")


def rq1(run: Path) -> None:
    records = list(read_records(run / "decisions.jsonl"))
    labelled = [
        r for r in records if r.step_label is not None and not r.extra.get("after_divergence")
    ]
    print(f"  decision points {len(records)}, labelled before divergence {len(labelled)}")
    if not labelled:
        return
    wrong = [r for r in labelled if r.step_label == 1]
    print(f"  wrong-write rate {len(wrong) / len(labelled):.3f} ({len(wrong)} wrong)")
    unanimous = sum(r.pooled_uncertainty == 0 for r in labelled) / len(labelled)
    print(f"  U = 0 on {unanimous:.3f} of labelled decisions", end="")
    if wrong:
        missed = sum(r.pooled_uncertainty == 0 for r in wrong) / len(wrong)
        print(f", on {missed:.3f} of wrong ones", end="")
    print()

    by_u: dict[float, list[int]] = defaultdict(list)
    for r in labelled:
        by_u[round(r.pooled_uncertainty, 3)].append(r.step_label)
    print("  error rate by U:")
    for u in sorted(by_u):
        labels = by_u[u]
        print(f"    U={u:.1f}  n={len(labels):4d}  error {sum(labels) / len(labels):.3f}")

    pairs_by_task: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in labelled:
        pairs_by_task[r.task_id].append((r.pooled_uncertainty, r.step_label))

    def stat(pairs: list[tuple[float, int]]) -> float:
        return auroc([u for u, _ in pairs], [y for _, y in pairs])

    point = stat([p for ps in pairs_by_task.values() for p in ps])
    low, high = bootstrap_ci(list(pairs_by_task.values()), stat)
    print(f"  AUROC {point:.3f}  (95% task-bootstrap CI {low:.3f} to {high:.3f})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for run in args.runs:
        outcome_summary(run)
        rq1(run)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

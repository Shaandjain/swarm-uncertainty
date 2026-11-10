"""What intervening bought: uv run python scripts/compare.py runs/test/s3-n10-test runs/test/s2-test ...

The first run is the baseline; each later run is compared with it on the episodes (task, trial)
both have. Prints pass rate, human calls per episode (oracle escalations plus questions asked
at the plan), the change in pass rate with a task-bootstrap interval, and that change per call.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from pilot import episodes

from swarm_uncertainty.metrics import bootstrap_ci
from swarm_uncertainty.records import read_records

Episode = tuple[str, int]


def outcomes(run: Path) -> tuple[dict[Episode, float], dict[Episode, int]]:
    """Reward and number of human calls for each episode of a run."""
    eps = episodes(run)
    reward = {(e["task_id"], e["trial"]): e["reward"] for e in eps}
    calls: dict[Episode, int] = defaultdict(int)
    for e in eps:
        calls[(e["task_id"], e["trial"])] += e["escalations"]
    for r in read_records(run / "decisions.jsonl"):
        if r.extra.get("asked"):
            calls[(r.task_id, r.trial)] += 1
    return reward, calls


def mean_difference(pairs: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in pairs) / len(pairs) if pairs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    base_reward, _ = outcomes(args.baseline)
    print(
        f"baseline {args.baseline}: {len(base_reward)} episodes, pass rate "
        f"{sum(base_reward.values()) / len(base_reward):.3f}"
    )
    for run in args.runs:
        reward, calls = outcomes(run)
        shared = sorted(set(reward) & set(base_reward))
        by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for task, trial in shared:
            by_task[task].append((base_reward[(task, trial)], reward[(task, trial)]))
        pairs = [p for ps in by_task.values() for p in ps]
        gain = mean_difference(pairs)
        lo, hi = bootstrap_ci(list(by_task.values()), mean_difference)
        per_episode = sum(calls[e] for e in shared) / len(shared)
        print(f"{run}: {len(shared)} shared episodes")
        print(
            f"  pass rate {sum(b for _, b in pairs) / len(pairs):.3f}"
            f"   change {gain:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})"
        )
        print(f"  human calls per episode {per_episode:.2f}", end="")
        print(f"   change per call {gain / per_episode:+.3f}" if per_episode else "")
        fixed = sum(a < 1 <= b for a, b in pairs)
        broken = sum(b < 1 <= a for a, b in pairs)
        print(f"  episodes fixed {fixed}, broken {broken}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

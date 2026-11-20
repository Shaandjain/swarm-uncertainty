"""Which signal sees which kind of error: uv run python scripts/signals.py runs/train/s3-n10 ...

For every labelled action (before the first wrong one) of the runs, pooled, prints one row per
signal: how often it fires, how often on right actions, and what share it catches of slips,
misreadings and benchmark errors (kinds from audit-<annotator>.jsonl). `precision` is the
share of its flags that are the agent's own errors, which is also the break-even price of a
human call in units of one wrong action when the call always fixes it.

Signals from the run itself:
  action split      the action's samples did not all agree
  plan split        the samples of the plan before it did not all agree; `plan <= 0.6` is the
                    stricter cut
  plan != call      the call differs from the action most plan samples went on to make
From scripts/review.py (review-<model>.jsonl, when present):
  mismatch          the call differs from the proposal the customer agreed to
  review flags      the reviewer answered unsure or wrong; `review wrong` is the stricter cut
"""

import argparse
import json
from pathlib import Path

from swarm_uncertainty.actions import action_key
from swarm_uncertainty.calibration import certainty
from swarm_uncertainty.records import read_records
from swarm_uncertainty.signals import KINDS, kind_of, score
from swarm_uncertainty.trajectory import labelled_actions


def rows_of(run: Path, annotator: str) -> list[dict]:
    records = list(read_records(run / "decisions.jsonl"))
    plans: dict[tuple[str, int], list] = {}
    for r in records:
        if r.decision_type == "plan":
            plans.setdefault((r.task_id, r.trial), []).append(r)
    audit = {}
    path = run / f"audit-{annotator}.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for a in map(json.loads, handle):
                audit[(a["task_id"], a["trial"], a["decision_index"])] = a["category"]
    reviews: dict[str, dict] = {}
    for path in sorted(run.glob("review-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            reviews[path.stem.removeprefix("review-")] = {
                (v["task_id"], v["trial"], v["decision_index"]): v for v in map(json.loads, handle)
            }
    rows = []
    last_action: dict[tuple[str, int], int] = {}
    for d, rank in labelled_actions(records):
        key = (d.task_id, d.trial, d.decision_index)
        # The plan for this action is the last one made since the previous executed action;
        # an older plan was for something else.
        since = last_action.get(key[:2], -1)
        last_action[key[:2]] = d.decision_index
        before = [p for p in plans.get(key[:2], []) if since < p.decision_index < d.decision_index]
        plan = max(before, key=lambda p: p.decision_index, default=None)
        rows.append(
            {
                "run": str(run),
                "decision": d,
                "rank": rank,
                "plan": plan,
                "kind": kind_of(d.step_label, audit.get(key)),
                "action_certainty": certainty(d),
                "plan_certainty": certainty(plan) if plan else None,
                "plan_differs": plan is not None
                and action_key(plan.chosen_tool, plan.chosen_args)
                != action_key(d.chosen_tool, d.chosen_args),
                "reviews": {m: v[key] for m, v in reviews.items() if key in v},
            }
        )
    return rows


def signals(rows: list[dict]) -> dict[str, list[bool]]:
    out = {
        "action split": [r["action_certainty"] < 1 for r in rows],
        "plan split": [r["plan_certainty"] is not None and r["plan_certainty"] < 1 for r in rows],
        "plan <= 0.6": [
            r["plan_certainty"] is not None and r["plan_certainty"] <= 0.6 for r in rows
        ],
        "plan != call": [r["plan_differs"] for r in rows],
    }
    models = sorted({m for r in rows for m in r["reviews"]})
    for m in models:
        # A reviewer that has not seen every action is scored by report() on those it has.
        if not all(m in r["reviews"] for r in rows):
            continue
        short = m.removeprefix("claude-")
        mismatch = [r["reviews"][m]["match"] == "mismatch" for r in rows]
        flags = [r["reviews"][m]["review"] != "sound" for r in rows]
        out[f"mismatch ({short})"] = mismatch
        out[f"review flags ({short})"] = flags
        out[f"review wrong ({short})"] = [r["reviews"][m]["review"] == "wrong" for r in rows]
        out[f"plan split or review flags ({short})"] = [
            a or b for a, b in zip(out["plan split"], flags, strict=True)
        ]
        out[f"mismatch or review wrong ({short})"] = [
            a or b for a, b in zip(mismatch, out[f"review wrong ({short})"], strict=True)
        ]
    return out


def report(title: str, rows: list[dict]) -> None:
    models = {m for r in rows for m in r["reviews"]}
    partial = sorted(m for m in models if not all(m in r["reviews"] for r in rows))
    kinds = [r["kind"] for r in rows]
    counts = {k: kinds.count(k) for k in KINDS}
    print(f"{title}: {len(rows)} actions: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
    print(
        f"  {'signal':44} fires  on right   slips  misread   bench  precision"
        "   (precision = break-even call price)"
    )
    for name, flags in signals(rows).items():
        s = score(name, flags, kinds)

        def cell(kind: str, s=s) -> str:
            return f"{s.flagged_by_kind[kind]}/{s.total_by_kind[kind]}"

        print(
            f"  {name:44} {s.flag_rate:5.1%}  {s.recall('right'):8.1%}  {cell('slip'):>6}"
            f"  {cell('misreading'):>7}  {cell('benchmark'):>6}  {s.precision:9.2f}"
        )
    base = score("", [True] * len(rows), kinds)
    print(f"  calling on every action: precision {base.precision:.2f}\n")
    for m in partial:
        seen = [{**r, "reviews": {m: r["reviews"][m]}} for r in rows if m in r["reviews"]]
        report(f"{title}, only the actions {m} reviewed", seen)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--annotator", default="claude-sonnet-5-5")
    args = parser.parse_args()
    per_run = {str(run): rows_of(run, args.annotator) for run in args.runs}
    for run, rows in per_run.items():
        report(run, rows)
    if len(per_run) > 1:
        report("pooled", [r for rows in per_run.values() for r in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

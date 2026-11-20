"""What an intervention does at the decisions a signal flags:
uv run python scripts/intervene.py runs/train/s3-n10 --intervention verify

Comparing whole episodes with and without an intervention is too noisy: two attempts at the
same task under the same condition already differ in outcome about one time in five. This
holds the state fixed instead. For every labelled action the signal flags, right or wrong, it
rebuilds the conversation just before the call and samples the agent's next step N times with
the intervention. The run's own samples at that decision are the comparison, so each state is
its own control.

  verify    signal: the call differs from the action most plan samples made (plan != call).
            The agent is told to check its call against the agreed proposal and the lookups.
  feedback  signal: the reviewer (review-<model>.jsonl, --reviewer) answered unsure or wrong.
            The agent is shown the call it was about to make and the reviewer's reason.

  control   the verify states again with nothing added. The run's own samples are the ones
            that got the action labelled wrong, so they understate how often a fresh sample is
            right; compare an intervention's "after" with the control's "after".

None of these uses a person. Prints, by kind of error, the share of samples that make a gold action
before and after, and writes <run>/intervene-<intervention>.jsonl.
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from signals import rows_of

from swarm_uncertainty.config import sanitize_no_proxy, use_own_anthropic_key
from swarm_uncertainty.signals import KINDS
from swarm_uncertainty.trajectory import action_index

VERIFY = (
    "\n\nBefore any call that changes the database, re-read the proposal the customer agreed "
    "to and check every identifier and value in your call against it and against your lookups."
)
FEEDBACK = (
    "\n\nA reviewer looked at the call you were about to make next, {call}, and said: "
    "{reason} Take this into account in your next step."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--intervention", choices=("control", "verify", "feedback"), required=True)
    parser.add_argument("--reviewer", default="claude-haiku-5-5")
    parser.add_argument("--annotator", default="claude-sonnet-5-5")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    sanitize_no_proxy()
    use_own_anthropic_key()
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import SystemMessage
    from tau2.data_model.simulation import SimulationRun
    from tau2.registry import registry
    from tau2.utils.llm_utils import generate

    from swarm_uncertainty.actions import (
        READ_PREFIX,
        RESPOND_TOOL,
        action_key,
        proposal_from_message,
    )
    from swarm_uncertainty.orchestrator import write_tool_names
    from swarm_uncertainty.tasks import gold_actions, load_tasks

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    spent, failed = 0.0, 0

    def one(run: Path) -> tuple[list[dict], int]:
        failures = 0
        cfg = json.loads((run / "config.json").read_text())
        model = cfg["worker_models"][0]
        env = registry.get_env_constructor(cfg["domain"])()
        tools = env.get_tools()
        write_tools = write_tool_names(tools)
        base = LLMAgent(tools=tools, domain_policy=env.get_policy(), llm=model).get_init_state()
        prompt = base.system_messages[0].content
        tasks = {t["id"]: t for t in load_tasks(cfg["domain"], cfg["split"])}

        def outcome(key: str, gold: set[str], original: str, tool: str) -> str:
            if key in gold:
                return "gold"
            if tool == RESPOND_TOOL or tool.startswith(READ_PREFIX):
                return "holds"
            return "same wrong" if key == original else "other wrong"

        def flagged(row: dict) -> str | None:
            """The note to add for this state, or None when the signal does not fire."""
            if args.intervention in ("control", "verify"):
                note = VERIFY if args.intervention == "verify" else ""
                return note if row["plan_differs"] else None
            review = row["reviews"].get(args.reviewer)
            if review is None or review["review"] == "sound":
                return None
            d = row["decision"]
            call = f"{d.chosen_tool}({json.dumps(d.chosen_args)})"
            return FEEDBACK.format(call=call, reason=review["review_reason"])

        def replay(row: dict, note: str) -> dict:
            d = row["decision"]
            sim = SimulationRun.model_validate_json(
                (run / "sims" / f"{d.task_id}_{d.trial}.json").read_text()
            )
            history = sim.messages[: action_index(sim.messages, row["rank"], write_tools)]
            gold = {
                action_key(a["name"], a.get("arguments")) for a in gold_actions(tasks[d.task_id])
            }
            original = action_key(d.chosen_tool, d.chosen_args)
            before = [outcome(p.key, gold, original, p.tool) for p in d.proposals]
            after, cost = [], 0.0
            for _ in range(args.samples):
                message = generate(
                    model=model,
                    messages=[SystemMessage(role="system", content=prompt + note), *history],
                    tools=tools,
                    call_name="intervene",
                    temperature=cfg["temperature"],
                )
                p = proposal_from_message(message, write_tools, "w0")
                after.append(outcome(p.key, gold, original, p.tool))
                cost += message.cost or 0.0
            return {
                "run": str(run),
                "task_id": d.task_id,
                "trial": d.trial,
                "decision_index": d.decision_index,
                "intervention": args.intervention,
                "kind": row["kind"],
                "before": before,
                "after": after,
                "cost_usd": cost,
            }

        jobs = [(r, note) for r in rows_of(run, args.annotator) if (note := flagged(r)) is not None]
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for (row, _), future in [(j, pool.submit(replay, *j)) for j in jobs]:
                try:
                    rows.append(future.result())
                except Exception as error:  # noqa: BLE001  one failed state must not end it
                    failures += 1
                    d = row["decision"]
                    print(f"task {d.task_id} trial {d.trial}: failed: {error}", file=sys.stderr)
        suffix = f"-{args.reviewer}" if args.intervention == "feedback" else ""
        out = run / f"intervene-{args.intervention}{suffix}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"{run}: {len(rows)} flagged states -> {out}")
        return rows, failures

    for run in args.runs:
        rows, failures = one(run)
        failed += failures
        for row in rows:
            t = totals[row["kind"]]
            t["states"] += 1
            for arm in ("before", "after"):
                for name in ("gold", "holds", "same wrong", "other wrong"):
                    t[f"{arm} {name}"] += row[arm].count(name) / len(row[arm])
        spent += sum(r["cost_usd"] for r in rows)

    print(f"\n{args.intervention}: mean share of samples per state, before -> after")
    print(f"  {'kind':11} states  gold action     holds (asks/looks up)  same wrong   other wrong")
    for kind in KINDS:
        t = totals.get(kind)
        if not t:
            continue
        n = t["states"]

        def pair(name: str, t=t, n=n) -> str:
            return f"{t[f'before {name}'] / n:.2f} -> {t[f'after {name}'] / n:.2f}"

        print(
            f"  {kind:11} {int(n):6d}  {pair('gold'):14}  {pair('holds'):21}"
            f"  {pair('same wrong'):11}  {pair('other wrong')}"
        )
    print(f"${spent:.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

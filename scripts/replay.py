"""Replay decisions and read the reasoning: uv run python scripts/replay.py runs/train/s1

Diagnostic only, not a condition. Rebuilds the agent's view of each first wrong action in a
run (plus --correct right ones drawn at random) from sims/, asks a model (--model, default
the run's worker) N more times, and records what it does and a summary of its thinking.
--effort high makes the model think on steps where it would otherwise skip thinking.

--at sets where the conversation is cut:
  action    just before the action, as the run saw it.
  plan      before the agent's proposal to the user. Each sample writes the proposal afresh,
            hears "Yes, please go ahead.", and its next call is its planned action.
  proposal  after the agent's actual proposal, answered with the same generic yes.
Comparing the three separates what writing the plan does from what the user's reply does.

Writes <run>/replay-<model>-<effort>[-plan|-proposal].jsonl, one line per sample;
scripts/collect.py loads them into the replays table.
"""

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swarm_uncertainty.config import sanitize_no_proxy, use_own_anthropic_key
from swarm_uncertainty.records import read_records
from swarm_uncertainty.trajectory import action_index, labelled_actions, proposal_index

MAX_TOKENS = 16000
CONFIRMATION = "Yes, please go ahead."


def outcome_of(row: dict) -> str:
    from swarm_uncertainty.actions import READ_PREFIX, RESPOND_TOOL

    if row["same_as_original"]:
        return "same action as the run"
    if row["matches_gold"]:
        return "gold action"
    if row["tool"] == RESPOND_TOOL or row["tool"].startswith(READ_PREFIX):
        return "asks or looks up first"
    return "another action"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--model", help="model to ask; default the run's first worker")
    parser.add_argument("--correct", type=int, default=0, help="right decisions to add")
    parser.add_argument("--at", choices=("action", "plan", "proposal"), default="action")
    parser.add_argument(
        "--effort", default="default", help="low, medium, high, ...; default as run"
    )
    parser.add_argument("--workers", type=int, default=4, help="model calls in parallel")
    args = parser.parse_args()

    sanitize_no_proxy()
    use_own_anthropic_key()
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import UserMessage
    from tau2.data_model.simulation import SimulationRun
    from tau2.registry import registry
    from tau2.utils.llm_utils import generate

    from swarm_uncertainty.actions import action_key, proposal_from_message
    from swarm_uncertainty.orchestrator import write_tool_names
    from swarm_uncertainty.tasks import gold_actions, load_tasks

    cfg = json.loads((args.run / "config.json").read_text())
    model = args.model or cfg["worker_models"][0]
    env = registry.get_env_constructor(cfg["domain"])()
    tools = env.get_tools()
    write_tools = write_tool_names(tools)
    system = LLMAgent(tools=tools, domain_policy=env.get_policy(), llm=model).get_init_state()
    tasks = {t["id"]: t for t in load_tasks(cfg["domain"], cfg["split"])}

    extra: dict = {}
    if model.startswith("claude"):
        extra["thinking"] = {"type": "adaptive", "display": "summarized"}
    if args.effort != "default":
        extra["output_config"] = {"effort": args.effort}

    def ask(history: list):
        return generate(
            model=model,
            messages=system.system_messages + history,
            tools=tools,
            call_name="replay",
            temperature=1.0,
            max_tokens=MAX_TOKENS,
            **extra,
        )

    yes = UserMessage(role="user", content=CONFIRMATION)

    def sample(decision, history: list, index: int) -> dict:
        plan, cost = "", 0.0
        message = ask([*history, yes] if args.at == "proposal" else history)
        if args.at == "plan" and not message.tool_calls:
            plan, cost = message.content or "", message.cost or 0.0
            message = ask([*history, message, yes])
        raw = (message.raw_data or {}).get("choices", [{}])[0].get("message", {})
        proposal = proposal_from_message(message, write_tools, "w0")
        gold = {
            action_key(a["name"], a.get("arguments")) for a in gold_actions(tasks[decision.task_id])
        }
        return {
            "task_id": decision.task_id,
            "trial": decision.trial,
            "decision_index": decision.decision_index,
            "sample_index": index,
            "model": model,
            "effort": args.effort,
            "at": args.at,
            "plan": plan,
            "original_label": decision.step_label,
            "original_tool": decision.chosen_tool,
            "original_args": decision.chosen_args,
            "thinking": raw.get("reasoning_content") or "",
            "content": message.content or "",
            "tool": proposal.tool,
            "args": proposal.args,
            "same_as_original": proposal.key
            == action_key(decision.chosen_tool, decision.chosen_args),
            "matches_gold": proposal.key in gold,
            "cost_usd": cost + (message.cost or 0.0),
        }

    labelled = labelled_actions(read_records(args.run / "decisions.jsonl"))
    right = [p for p in labelled if p[0].step_label == 0]
    chosen = [p for p in labelled if p[0].step_label == 1]
    chosen += random.Random(0).sample(right, min(args.correct, len(right)))
    jobs, skipped = [], 0
    for d, rank in chosen:
        sim = SimulationRun.model_validate_json(
            (args.run / "sims" / f"{d.task_id}_{d.trial}.json").read_text()
        )
        end = action_index(sim.messages, rank, write_tools)
        if args.at != "action":
            start = proposal_index(sim.messages, end)
            if start is None:
                skipped += 1
                continue
            end = start if args.at == "plan" else start + 1
        jobs += [(d, sim.messages[:end], i) for i in range(args.samples)]
    if skipped:
        print(f"{skipped} decisions skipped: no reply put the action to the user first")
    decisions = [d for d, _ in chosen]
    wrong = sum(d.step_label for d in decisions)
    print(
        f"{args.run}: {wrong} wrong + {len(decisions) - wrong} right decisions"
        f" x {args.samples} samples, {model}, effort {args.effort}"
    )

    suffix = "" if args.at == "action" else f"-{args.at}"
    out = args.run / f"replay-{model}-{args.effort}{suffix}.jsonl"
    out.unlink(missing_ok=True)
    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job, future in [(j, pool.submit(sample, *j)) for j in jobs]:
            try:
                rows.append(future.result())
            except Exception as error:  # noqa: BLE001  one failed call must not end the replay
                failed += 1
                print(
                    f"task {job[0].task_id} trial {job[0].trial}: failed: {error}",
                    file=sys.stderr,
                )
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    for label, name in ((1, "wrong"), (0, "right")):
        outcome = Counter(outcome_of(r) for r in rows if r["original_label"] == label)
        if outcome:
            print(
                f"on {name} decisions: " + ", ".join(f"{n} {k}" for k, n in outcome.most_common())
            )
    print(f"{len(rows)} samples -> {out}, ${sum(r['cost_usd'] for r in rows):.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

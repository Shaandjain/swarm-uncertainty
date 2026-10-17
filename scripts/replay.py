"""Replay decisions and read the reasoning: uv run python scripts/replay.py runs/train/s1

Diagnostic only, not a condition. For each first wrong write in a run (the decisions RQ1
counts), plus --correct right ones drawn at random, rebuild the agent's view at that moment
from sims/, ask a model (--model, default the run's worker) again N times, and record what
it does and a summary of its thinking. Another model here previews a K=2 team offline.

--at plan rewinds further, to just before the agent's reply that proposed the action and
asked the user to confirm. Each sample writes that reply afresh, the user answers "Yes,
please go ahead.", and the call the agent then makes is the sample's planned action. This
asks whether the plans disagree before the user has agreed to one. The model thinks by default
(adaptive) in the runs too; display "summarized" only makes that thinking visible, so the
replay asks the same question the run did. At default effort the model often skips thinking
on a step, so --effort high is the arm to read; it also shows whether more thought fixes
the error. Writes <run>/replay-<model>-<effort>[-plan].jsonl, one line per sample; scripts/collect.py
loads them into the replays table.
"""

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swarm_uncertainty.config import sanitize_no_proxy, use_own_anthropic_key

MAX_TOKENS = 16000


def labelled_decisions(run: Path) -> list[dict]:
    """Executed decisions up to and including each episode's first wrong one, with their rank
    among executed ones: the decisions RQ1 counts."""
    by_episode: dict[tuple[str, int], list[dict]] = {}
    with (run / "decisions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                by_episode.setdefault((r["task_id"], r["trial"]), []).append(r)
    labelled = []
    for records in by_episode.values():
        executed = [
            r
            for r in sorted(records, key=lambda r: r["decision_index"])
            if r["step_label"] is not None
        ]
        for rank, r in enumerate(executed):
            if not r["extra"].get("after_divergence"):
                labelled.append({**r, "rank": rank})
    return sorted(labelled, key=lambda r: (int(r["task_id"]), r["trial"], r["decision_index"]))


def outcome_of(row: dict) -> str:
    from swarm_uncertainty.actions import READ_PREFIX, RESPOND_TOOL

    if row["same_as_original"]:
        return "same action as the run"
    if row["matches_gold"]:
        return "gold action"
    if row["tool"] == RESPOND_TOOL or row["tool"].startswith(READ_PREFIX):
        return "asks or looks up first"
    return "another action"


def history_before(messages: list, rank: int, write_tools: frozenset[str]) -> list:
    """Messages before the rank-th executed write or transfer: what the agent saw then."""
    from swarm_uncertainty.actions import TRANSFER_TOOLS

    acting = write_tools | TRANSFER_TOOLS
    seen = 0
    for i, m in enumerate(messages):
        if m.role == "assistant" and any(c.name in acting for c in m.tool_calls or []):
            if seen == rank:
                return messages[:i]
            seen += 1
    raise ValueError(f"no executed write number {rank} in the trajectory")


def plan_history(history: list) -> list | None:
    """History before the agent's last plain reply with a user turn after it, or None if the
    write was not preceded by one (nothing was put to the user first)."""
    for j in range(len(history) - 1, -1, -1):
        m = history[j]
        if m.role == "assistant" and m.content and not m.tool_calls:
            if any(n.role == "user" for n in history[j + 1 :]):
                return history[:j]
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--model", help="model to ask; default the run's first worker")
    parser.add_argument("--correct", type=int, default=0, help="right decisions to add")
    parser.add_argument("--at", choices=("action", "plan"), default="action")
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

    def sample(decision: dict, history: list, index: int) -> dict:
        plan, cost = "", 0.0
        message = ask(history)
        if args.at == "plan" and not message.tool_calls:
            plan, cost = message.content or "", message.cost or 0.0
            yes = UserMessage(role="user", content="Yes, please go ahead.")
            message = ask([*history, message, yes])
        raw = (message.raw_data or {}).get("choices", [{}])[0].get("message", {})
        proposal = proposal_from_message(message, write_tools, "w0")
        gold = {
            action_key(a["name"], a.get("arguments"))
            for a in gold_actions(tasks[decision["task_id"]])
        }
        return {
            "task_id": decision["task_id"],
            "trial": decision["trial"],
            "decision_index": decision["decision_index"],
            "sample_index": index,
            "model": model,
            "effort": args.effort,
            "at": args.at,
            "plan": plan,
            "original_label": decision["step_label"],
            "original_tool": decision["chosen_tool"],
            "original_args": decision["chosen_args"],
            "thinking": raw.get("reasoning_content") or "",
            "content": message.content or "",
            "tool": proposal.tool,
            "args": proposal.args,
            "same_as_original": proposal.key
            == action_key(decision["chosen_tool"], decision["chosen_args"]),
            "matches_gold": proposal.key in gold,
            "cost_usd": cost + (message.cost or 0.0),
        }

    labelled = labelled_decisions(args.run)
    right = [d for d in labelled if d["step_label"] == 0]
    decisions = [d for d in labelled if d["step_label"] == 1]
    decisions += random.Random(0).sample(right, min(args.correct, len(right)))
    jobs, skipped = [], []
    for d in decisions:
        sim = SimulationRun.model_validate_json(
            (args.run / "sims" / f"{d['task_id']}_{d['trial']}.json").read_text()
        )
        history = history_before(sim.messages, d["rank"], write_tools)
        if args.at == "plan":
            history = plan_history(history)
            if history is None:
                skipped.append(d)
                continue
        jobs += [(d, history, i) for i in range(args.samples)]
    if skipped:
        print(f"{len(skipped)} decisions skipped: no reply put the action to the user first")
    wrong = sum(d["step_label"] for d in decisions)
    print(
        f"{args.run}: {wrong} wrong + {len(decisions) - wrong} right decisions"
        f" x {args.samples} samples, {model}, effort {args.effort}"
    )

    suffix = "-plan" if args.at == "plan" else ""
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
                    f"task {job[0]['task_id']} trial {job[0]['trial']}: failed: {error}",
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

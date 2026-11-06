"""Second annotator for wrong actions: uv run python scripts/audit.py runs/train/s1 [runs/...]

Each run's first wrong action per episode (the decisions RQ1 counts) is classified by a
model that sees the simulated customer's instructions, the answer key, the store policy and
the conversation, and nobody else's labels. The rubric was fixed before any model label was
read. Writes <run>/audit-<model>.jsonl, then prints agreement with every other annotator's
file in the run (audit-*.jsonl, e.g. audit-manual.jsonl from a hand audit). collect.py loads
them into the audits table; pilot.py --audit keeps only the genuine agent errors.
"""

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swarm_uncertainty.config import sanitize_no_proxy, use_own_anthropic_key
from swarm_uncertainty.labels import AGENT_ERRORS, AUDIT_CATEGORIES
from swarm_uncertainty.metrics import cohen_kappa
from swarm_uncertainty.records import read_records
from swarm_uncertainty.trajectory import action_index, labelled_actions

# Fixed before any model labels were read, so the categories could not drift toward the
# result we expected.
RUBRIC = """You are auditing one action that an automatic grader labelled wrong in a customer-service
benchmark. A simulated customer, following written instructions, talks to an agent that reads
and changes a store database through tools under a written policy. An action is labelled wrong
when it does not match the task's answer key. Your job is to decide where the error came from.

Choose exactly one category:

- slip: the agent's stated plan to the customer was right for the request and the policy, but
  the call it made differs from what it stated (a wrong item ID, order, amount or field).
- interpretation: the agent acted on a reading of the request that differs from what the
  customer's instructions want, or filled in a value the customer never gave, and the mistake
  started in the agent's reasoning, not in the customer's messages.
- policy: the action breaks the written policy, or the agent acted where the policy says it
  should not, whether or not the system blocked the call.
- user_simulator: the simulated customer's messages departed from its own instructions, and
  the agent reasonably did what the customer asked.
- answer_key: the expected actions conflict with the policy or depend on a choice the
  customer's instructions never specify, and the agent's action is a reasonable reading.
- label_artifact: the action is consistent with the instructions, the policy and the episode's
  outcome, and is counted wrong only because it is missing from the answer key's list.

Decide by the earliest point where the conversation went off the customer's instructions.
Prefer user_simulator, answer_key or label_artifact only when the agent behaved reasonably.

Answer with one JSON object and nothing else:
{"category": "<one of the six>", "reason": "<one or two sentences citing the evidence>"}"""

TOOL_RESULT_CHARS = 800


def transcript(messages: list, marked: int) -> str:
    lines = []
    for i, m in enumerate(messages):
        mark = ">>> LABELLED WRONG: " if i == marked else ""
        if m.role == "tool":
            lines.append(f"[{i}] TOOL RESULT: {(m.content or '')[:TOOL_RESULT_CHARS]}")
        elif m.tool_calls:
            calls = "; ".join(f"{c.name}({json.dumps(c.arguments)})" for c in m.tool_calls)
            lines.append(f"[{i}] {mark}AGENT CALLS: {calls}")
        else:
            lines.append(f"[{i}] {m.role.upper()}: {m.content or ''}")
    return "\n".join(lines)


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in the reply: {text[:200]!r}")
    answer = json.loads(match.group(0))
    if answer.get("category") not in AUDIT_CATEGORIES:
        raise ValueError(f"unknown category: {answer.get('category')!r}")
    return answer


def labels(path: Path) -> dict[tuple[str, int, int], str]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(r["task_id"], r["trial"], r["decision_index"]): r["category"] for r in rows}


def agreement(run: Path, mine: Path) -> None:
    ours = labels(mine)
    for other in sorted(run.glob("audit-*.jsonl")):
        if other == mine:
            continue
        theirs = labels(other)
        keys = sorted(ours.keys() & theirs.keys())
        if not keys:
            continue
        a, b = [ours[k] for k in keys], [theirs[k] for k in keys]
        agent = [c in AGENT_ERRORS for c in a], [c in AGENT_ERRORS for c in b]
        print(
            f"  vs {other.stem.removeprefix('audit-')} on {len(keys)}: "
            f"category agreement {sum(x == y for x, y in zip(a, b)) / len(keys):.2f} "
            f"(kappa {cohen_kappa(a, b):.2f}), "
            f"agent-or-not agreement {sum(x == y for x, y in zip(*agent)) / len(keys):.2f} "
            f"(kappa {cohen_kappa(*agent):.2f})"
        )
        name = other.stem.removeprefix("audit-")
        for k, x, y in zip(keys, a, b):
            if x != y:
                print(f"    task {k[0]} trial {k[1]}: {x} here, {y} by {name}")


def audit_run(run: Path, model: str, workers: int) -> int:
    """Classify a run's wrong actions; returns the number of failed calls."""
    import litellm
    from tau2.data_model.simulation import SimulationRun
    from tau2.registry import registry

    from swarm_uncertainty.orchestrator import write_tool_names
    from swarm_uncertainty.tasks import data_dir, gold_actions, load_tasks

    cfg = json.loads((run / "config.json").read_text())
    write_tools = write_tool_names(registry.get_env_constructor(cfg["domain"])().get_tools())
    policy = (data_dir() / "tau2" / "domains" / cfg["domain"] / "policy.md").read_text()
    tasks = {t["id"]: t for t in load_tasks(cfg["domain"], cfg["split"])}
    wrong = [
        (d, rank)
        for d, rank in labelled_actions(read_records(run / "decisions.jsonl"))
        if d.step_label == 1
    ]

    def judge(d, rank: int) -> dict:
        sim = SimulationRun.model_validate_json(
            (run / "sims" / f"{d.task_id}_{d.trial}.json").read_text()
        )
        task = tasks[d.task_id]
        dossier = (
            f"STORE POLICY\n{policy}\n\n"
            "CUSTOMER'S INSTRUCTIONS\n"
            f"{json.dumps(task['user_scenario']['instructions'], indent=2)}\n\n"
            f"ANSWER KEY (expected actions)\n{json.dumps(gold_actions(task), indent=2)}\n\n"
            f"EPISODE OUTCOME: reward {sim.reward_info.reward if sim.reward_info else 0}\n\n"
            "CONVERSATION\n"
            f"{transcript(sim.messages, action_index(sim.messages, rank, write_tools))}"
        )
        response = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": dossier}],
            max_tokens=16000,
        )
        answer = parse(response.choices[0].message.content or "")
        return {
            "task_id": d.task_id,
            "trial": d.trial,
            "decision_index": d.decision_index,
            "annotator": model,
            "category": answer["category"],
            "agent_error": answer["category"] in AGENT_ERRORS,
            "reason": answer.get("reason", ""),
            "cost_usd": litellm.completion_cost(response) or 0.0,
        }

    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (d, _), future in [(w, pool.submit(judge, *w)) for w in wrong]:
            try:
                rows.append(future.result())
            except Exception as error:  # noqa: BLE001  one failed call must not end the audit
                failed += 1
                print(f"task {d.task_id} trial {d.trial}: failed: {error}", file=sys.stderr)
    out = run / f"audit-{model}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    counts = Counter(r["category"] for r in rows)
    print(f"{run}: {len(rows)} wrong actions -> {out}, ${sum(r['cost_usd'] for r in rows):.2f}")
    print("  " + ", ".join(f"{n} {c}" for c, n in counts.most_common()))
    agreement(run, out)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--workers", type=int, default=4, help="model calls in parallel")
    args = parser.parse_args()

    sanitize_no_proxy()
    use_own_anthropic_key()
    failed = sum(audit_run(run, args.model, args.workers) for run in args.runs)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

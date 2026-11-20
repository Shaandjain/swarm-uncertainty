"""Two other signals for the same decisions: uv run python scripts/review.py runs/train/s3-n10

Self-consistency measures how much the agent's answer varies, so it cannot see a mistake the
agent makes the same way every time. For every labelled action of a run (before the first
wrong one), this asks a model two separate questions, neither of which sees the answer key
or the customer's hidden instructions:

  match   given only the agent's last proposal, the customer's reply and the call: does the
          call do what was proposed and agreed? (match / mismatch / no_plan). Aimed at slips.
  review  given the policy, the conversation so far and the call: is the call right for what
          the customer asked and the policy allows? (sound / unsure / wrong). Aimed at
          misread requests and policy breaks.

Both prompts were fixed before any of their answers were read. Writes
<run>/review-<model>.jsonl, one line per action, keeping the lines already there and asking
only about the actions that are missing; scripts/signals.py scores it.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swarm_uncertainty.config import sanitize_no_proxy, use_own_anthropic_key
from swarm_uncertainty.records import read_records
from swarm_uncertainty.trajectory import action_index, labelled_actions, proposal_index

MATCH_PROMPT = """A customer-service agent proposed an action to a customer, the customer replied, and the
agent then made a tool call. Decide whether the call does exactly what was proposed and agreed:
the same order, items, replacement items, payment method, address and amounts.

Answer with one JSON object and nothing else:
{"verdict": "match" | "mismatch" | "no_plan", "reason": "<one sentence>"}

- match: every identifier and value in the call is the one proposed and agreed.
- mismatch: the call differs from what was proposed and agreed in any identifier or value, or
  the customer did not agree.
- no_plan: the agent's message did not put this action to the customer."""

REVIEW_PROMPT = """You are reviewing one tool call a customer-service agent is about to make. The call changes
the store's database. You see the store policy, the conversation so far including the agent's
lookups, and the call. Decide whether this exact call is the right thing to do now: it does
what the customer asked for in this conversation, every identifier and value is correct given
the lookups, and the policy allows it.

Answer with one JSON object and nothing else:
{"verdict": "sound" | "unsure" | "wrong", "reason": "<one or two sentences citing the evidence>"}

- sound: you checked the request, the values and the policy, and found nothing wrong.
- unsure: something the call depends on is ambiguous or was never established.
- wrong: you can point to a specific request, value or policy rule the call gets wrong."""

VERDICTS = {
    "match": ("match", "mismatch", "no_plan"),
    "review": ("sound", "unsure", "wrong"),
}
TOOL_RESULT_CHARS = 3000


def call_text(message) -> str:
    return "; ".join(f"{c.name}({json.dumps(c.arguments)})" for c in message.tool_calls or [])


def transcript(messages: list) -> str:
    lines = []
    for m in messages:
        if m.role == "tool":
            lines.append(f"TOOL RESULT: {(m.content or '')[:TOOL_RESULT_CHARS]}")
        elif m.tool_calls:
            lines.append(f"AGENT CALLS: {call_text(m)}")
        else:
            lines.append(f"{m.role.upper()}: {m.content or ''}")
    return "\n".join(lines)


def parse(text: str, kind: str) -> dict:
    found = re.search(r"\{.*\}", text, re.DOTALL)
    if not found:
        raise ValueError(f"no JSON object in the reply: {text[:200]!r}")
    try:
        answer = json.loads(found.group(0))
    except json.JSONDecodeError:
        # An unescaped quote inside the reason breaks the JSON; the verdict is still readable.
        verdict = re.search(r'"verdict"\s*:\s*"(\w+)"', found.group(0))
        reason = re.search(r'"reason"\s*:\s*"(.*)"\s*\}', found.group(0), re.DOTALL)
        answer = {
            "verdict": verdict.group(1) if verdict else None,
            "reason": reason.group(1) if reason else "",
        }
    if answer.get("verdict") not in VERDICTS[kind]:
        raise ValueError(f"unknown verdict: {answer.get('verdict')!r}")
    return answer


def review_run(run: Path, model: str, workers: int) -> int:
    """Ask both questions about each labelled action; returns the number of failed actions."""
    import litellm
    from tau2.data_model.simulation import SimulationRun
    from tau2.registry import registry

    from swarm_uncertainty.orchestrator import write_tool_names
    from swarm_uncertainty.tasks import data_dir

    cfg = json.loads((run / "config.json").read_text())
    write_tools = write_tool_names(registry.get_env_constructor(cfg["domain"])().get_tools())
    policy = (data_dir() / "tau2" / "domains" / cfg["domain"] / "policy.md").read_text()

    def ask(system: str, user: str, kind: str) -> tuple[dict, float]:
        response = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=4000,
        )
        answer = parse(response.choices[0].message.content or "", kind)
        return answer, litellm.completion_cost(response) or 0.0

    def judge(d, rank: int) -> dict:
        sim = SimulationRun.model_validate_json(
            (run / "sims" / f"{d.task_id}_{d.trial}.json").read_text()
        )
        at = action_index(sim.messages, rank, write_tools)
        call = call_text(sim.messages[at])
        start = proposal_index(sim.messages, at)
        if start is None:
            match, match_cost = {"verdict": "no_plan", "reason": "no proposal was answered"}, 0.0
        else:
            said = "\n".join(
                f"{m.role.upper()}: {m.content}"
                for m in sim.messages[start:at]
                if m.role in ("assistant", "user") and m.content and not m.tool_calls
            )
            match, match_cost = ask(MATCH_PROMPT, f"{said}\n\nCALL: {call}", "match")
        review, review_cost = ask(
            REVIEW_PROMPT,
            f"STORE POLICY\n{policy}\n\nCONVERSATION\n{transcript(sim.messages[1:at])}"
            f"\n\nCALL UNDER REVIEW: {call}",
            "review",
        )
        return {
            "task_id": d.task_id,
            "trial": d.trial,
            "decision_index": d.decision_index,
            "model": model,
            "label": d.step_label,
            "match": match["verdict"],
            "match_reason": match.get("reason", ""),
            "review": review["verdict"],
            "review_reason": review.get("reason", ""),
            "cost_usd": match_cost + review_cost,
        }

    out = run / f"review-{model}.jsonl"
    rows, failed = [], 0
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    done = {(r["task_id"], r["trial"], r["decision_index"]) for r in rows}
    actions = [
        a
        for a in labelled_actions(read_records(run / "decisions.jsonl"))
        if (a[0].task_id, a[0].trial, a[0].decision_index) not in done
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (d, _), future in [(a, pool.submit(judge, *a)) for a in actions]:
            try:
                rows.append(future.result())
            except Exception as error:  # noqa: BLE001  one failed call must not end the review
                failed += 1
                print(f"task {d.task_id} trial {d.trial}: failed: {error}", file=sys.stderr)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"{run}: {len(rows)} actions ({len(actions)} new) -> {out}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--workers", type=int, default=6, help="model calls in parallel")
    args = parser.parse_args()

    sanitize_no_proxy()
    use_own_anthropic_key()
    failed = sum(review_run(run, args.model, args.workers) for run in args.runs)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

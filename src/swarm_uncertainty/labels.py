"""Right/wrong labels for decision points, from the task's gold actions.

Every executed write goes through the orchestrator, so one episode's decision records, in
order, are its history of writes. A decision is labelled only when its chosen action was
executed as a write or transfer: right (0) if it matches a gold action not yet used, wrong (1)
otherwise. A decision whose majority chose to reply or look something up gets no label.
Decisions after the first wrong one are flagged: the trajectory has left the gold path.
A plan decision takes the label of the next executed write or transfer after it, the action
it was planning; none follows when the user did not go ahead.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from swarm_uncertainty.actions import READ_PREFIX, RESPOND_TOOL, TRANSFER_TOOLS, action_key
from swarm_uncertainty.records import DecisionRecord


def label_episode(
    records: Sequence[DecisionRecord],
    gold_actions: Iterable[dict[str, Any]],
    write_tools: frozenset[str],
) -> None:
    """Set `step_label` and `extra["after_divergence"]` on one episode's records, in place."""
    remaining = Counter(
        action_key(a["name"], a.get("arguments"))
        for a in gold_actions
        if a["name"] in write_tools | TRANSFER_TOOLS
    )
    ordered = sorted(records, key=lambda r: r.decision_index)
    acted = [r for r in ordered if r.decision_type != "plan"]
    diverged = False
    for record in acted:
        record.extra["after_divergence"] = diverged
        if record.chosen_tool == RESPOND_TOOL or record.chosen_tool.startswith(READ_PREFIX):
            record.step_label = None
            continue
        key = action_key(record.chosen_tool, record.chosen_args)
        if remaining[key] > 0:
            remaining[key] -= 1
            record.step_label = 0
        else:
            record.step_label = 1
            diverged = True
    for plan in (r for r in ordered if r.decision_type == "plan"):
        before = [r for r in acted if r.decision_index < plan.decision_index]
        after = [
            r for r in acted if r.decision_index > plan.decision_index and r.step_label is not None
        ]
        plan.extra["after_divergence"] = any(r.step_label == 1 for r in before)
        plan.step_label = after[0].step_label if after else None

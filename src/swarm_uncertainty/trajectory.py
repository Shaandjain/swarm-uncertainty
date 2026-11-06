"""Locate a decision inside a saved tau2 trajectory, to rebuild what the agent saw then.

Decision records and trajectories meet at executed actions: the n-th executed write or
transfer among an episode's records is the n-th assistant message that calls one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from swarm_uncertainty.actions import TRANSFER_TOOLS
from swarm_uncertainty.records import DecisionRecord


def labelled_actions(records: Iterable[DecisionRecord]) -> list[tuple[DecisionRecord, int]]:
    """Each episode's executed actions up to and including its first wrong one (the decisions
    RQ1 counts), each with its rank among the episode's executed actions."""
    by_episode: dict[tuple[str, int], list[DecisionRecord]] = {}
    for r in records:
        if r.decision_type != "plan":
            by_episode.setdefault((r.task_id, r.trial), []).append(r)
    labelled: list[tuple[DecisionRecord, int]] = []
    for episode in by_episode.values():
        executed = [
            r for r in sorted(episode, key=lambda r: r.decision_index) if r.step_label is not None
        ]
        labelled += [
            (r, rank) for rank, r in enumerate(executed) if not r.extra.get("after_divergence")
        ]
    return sorted(labelled, key=lambda p: (int(p[0].task_id), p[0].trial, p[0].decision_index))


def action_index(messages: Sequence[Any], rank: int, write_tools: frozenset[str]) -> int:
    """Index of the message that made the rank-th executed write or transfer."""
    acting = write_tools | TRANSFER_TOOLS
    calls = [
        i
        for i, m in enumerate(messages)
        if m.role == "assistant" and any(c.name in acting for c in m.tool_calls or [])
    ]
    if rank >= len(calls):
        raise ValueError(f"no executed action number {rank} in the trajectory")
    return calls[rank]


def proposal_index(messages: Sequence[Any], before: int) -> int | None:
    """Index of the agent's last plain reply before message `before` that the user answered,
    or None if the action was not put to the user first."""
    for j in range(before - 1, -1, -1):
        m = messages[j]
        if m.role == "assistant" and m.content and not m.tool_calls:
            answered = any(n.role == "user" for n in messages[j + 1 : before])
            return j if answered else None
    return None

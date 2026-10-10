"""Oracles: what the orchestrator consults when its uncertainty is above threshold.

An oracle may choose one of the proposed clusters, veto them all (the orchestrator then
re-plans once), or add a one-line hint. It never executes anything itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from swarm_uncertainty.actions import Proposal, action_key
from swarm_uncertainty.llm import DecisionPacket
from swarm_uncertainty.measures import Cluster

Verdict = Literal["choose", "veto", "hint"]


@dataclass(frozen=True)
class OracleAnswer:
    verdict: Verdict
    chosen: Proposal | None = None
    hint: str = ""
    note: str = ""

    def __str__(self) -> str:
        if self.verdict == "choose" and self.chosen is not None:
            return f"choose {self.chosen.key}"
        if self.verdict == "hint":
            return f"hint: {self.hint}"
        return f"veto: {self.note}"


class Oracle(Protocol):
    name: str

    def consult(
        self, packet: DecisionPacket, clusters: Sequence[Cluster], pooled_uncertainty: float
    ) -> OracleAnswer: ...


class NoOracle:
    """Escalation is off. The orchestrator always proceeds with its majority action."""

    name = "none"

    def consult(
        self, packet: DecisionPacket, clusters: Sequence[Cluster], pooled_uncertainty: float
    ) -> OracleAnswer:
        top = clusters[0].example if clusters else None
        return OracleAnswer("choose", chosen=top, note="no oracle")


class GoldVetoOracle:
    """Upper bound: sees the task's gold actions. Chooses the largest matching cluster,
    otherwise vetoes. A simulated ceiling, never a headline result."""

    name = "gold_veto"

    def __init__(self, gold_actions: Iterable[dict[str, Any]]) -> None:
        self.gold_keys = {action_key(a["name"], a.get("arguments")) for a in gold_actions}

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> GoldVetoOracle:
        actions = (task.get("evaluation_criteria") or {}).get("actions") or []
        return cls(actions)

    def consult(
        self, packet: DecisionPacket, clusters: Sequence[Cluster], pooled_uncertainty: float
    ) -> OracleAnswer:
        for c in clusters:  # largest first
            if c.key in self.gold_keys:
                return OracleAnswer("choose", chosen=c.example, note="matches gold")
        return OracleAnswer("veto", note="no proposed action matches gold")


def make_oracle(name: str, task: dict[str, Any] | None = None) -> Oracle:
    if name == "none":
        return NoOracle()
    if name == "gold_veto":
        if task is None:
            raise ValueError("gold_veto oracle needs the task")
        return GoldVetoOracle.from_task(task)
    raise ValueError(f"unknown oracle: {name}")

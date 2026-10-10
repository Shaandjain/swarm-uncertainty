"""The one interface the orchestrator and workers use to talk to a model.

Real clients wrap LiteLLM (which tau2 already uses). The fake client replays scripted
proposals so the whole decision loop can be exercised offline, with no API key.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from swarm_uncertainty.actions import Proposal


@dataclass(frozen=True)
class DecisionPacket:
    """Everything a worker gets to see when asked "what is the next action?"."""

    policy: str
    conversation: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...] = ()
    question: str = "Propose the single next action."


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.cost_usd += other.cost_usd


@dataclass
class ProposalResponse:
    proposal: Proposal
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    """Anything that can turn a decision packet into one proposed action."""

    model: str

    def propose(
        self, packet: DecisionPacket, *, temperature: float, worker_id: str
    ) -> ProposalResponse: ...


class FakeLLM:
    """Replays a fixed sequence of proposals. Raises when the script runs out.

    Use one FakeLLM per worker so each worker's script is independent.
    """

    def __init__(self, script: Iterable[Proposal], model: str = "fake") -> None:
        self.model = model
        self._script: deque[Proposal] = deque(script)
        self.calls: list[tuple[DecisionPacket, float, str]] = []

    def propose(
        self, packet: DecisionPacket, *, temperature: float, worker_id: str
    ) -> ProposalResponse:
        self.calls.append((packet, temperature, worker_id))
        if not self._script:
            raise RuntimeError("FakeLLM script exhausted")
        proposal = self._script.popleft()
        proposal = Proposal(proposal.tool, proposal.args, proposal.reason, worker_id)
        return ProposalResponse(proposal, Usage(tokens_in=10, tokens_out=5))

    @classmethod
    def repeating(cls, proposals: Sequence[Proposal], times: int, model: str = "fake") -> FakeLLM:
        """Script that cycles through `proposals` until `times` proposals have been served."""
        cycle = [proposals[i % len(proposals)] for i in range(times)]
        return cls(cycle, model=model)

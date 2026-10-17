"""The one interface the orchestrator and workers use to talk to a model.

Real clients wrap LiteLLM (which tau2 already uses). The fake client replays scripted
proposals so the whole decision loop can be exercised offline, with no API key.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from swarm_uncertainty.actions import Proposal, proposal_from_message


@dataclass(frozen=True)
class DecisionPacket:
    """Everything a worker gets to see when asked "what is the next action?"."""

    policy: str
    conversation: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...] = ()
    question: str = "Propose the single next action."
    messages: tuple[Any, ...] = ()
    """tau2 messages (system prompt and history) for live clients."""
    tools: tuple[Any, ...] = ()
    write_tools: frozenset[str] = frozenset()


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
        proposal = replace(self._script.popleft(), worker_id=worker_id)
        return ProposalResponse(proposal, Usage(tokens_in=10, tokens_out=5))

    @classmethod
    def repeating(cls, proposals: Sequence[Proposal], times: int, model: str = "fake") -> FakeLLM:
        """Script that cycles through `proposals` until `times` proposals have been served."""
        cycle = [proposals[i % len(proposals)] for i in range(times)]
        return cls(cycle, model=model)


class LiteLLMClient:
    """A live worker: one tau2 `generate` call (LiteLLM underneath) per proposal."""

    def __init__(self, model: str, llm_args: dict[str, Any] | None = None) -> None:
        self.model = model
        self.llm_args = dict(llm_args or {})

    def propose(
        self, packet: DecisionPacket, *, temperature: float, worker_id: str
    ) -> ProposalResponse:
        from tau2.utils.llm_utils import generate  # late: tau2 reads env at import time

        message = generate(
            model=self.model,
            messages=list(packet.messages),
            tools=list(packet.tools),
            call_name="worker_sample",
            **{**self.llm_args, "temperature": temperature},
        )
        usage = message.usage or {}
        return ProposalResponse(
            proposal_from_message(message, packet.write_tools, worker_id),
            Usage(
                tokens_in=usage.get("prompt_tokens") or 0,
                tokens_out=usage.get("completion_tokens") or 0,
                cost_usd=message.cost or 0.0,
            ),
        )

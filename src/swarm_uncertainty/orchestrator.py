"""The orchestrator's decision loop: fan out, cluster, measure, escalate or act.

A decision point is a state-changing tool call, a transfer to a human agent, or ending the
task. Reads and plain replies are not. The write set comes from tau2's own
`__mutates_state__` marker, so the benchmark defines "write".

The loop knows nothing about tau2 messages; an adapter calls `decide()` whenever the
conversation reaches a decision point. With one worker and one sample it is a plain single
agent, which is how the baselines share this code path.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from swarm_uncertainty.actions import STOP_TOOL, TRANSFER_TOOLS, Proposal
from swarm_uncertainty.llm import DecisionPacket, LLMClient, Usage
from swarm_uncertainty.measures import (
    Cluster,
    cluster,
    disagreement,
    normalized_entropy,
    pooled_uncertainty,
)
from swarm_uncertainty.oracle import NoOracle, Oracle, OracleAnswer
from swarm_uncertainty.records import DecisionRecord, DecisionType, ProposalRecord

MUTATES_STATE_ATTR = "__mutates_state__"
TOOL_TYPE_ATTR = "__tool_type__"


def tool_mutates_state(tool: Any) -> bool:
    """True if a tau2 Tool (or decorated function) is marked as mutating state."""
    func = getattr(tool, "_func", tool)
    flag = getattr(func, MUTATES_STATE_ATTR, None)
    if flag is not None:
        return bool(flag)
    tool_type = getattr(func, TOOL_TYPE_ATTR, None)
    return str(getattr(tool_type, "value", tool_type)).lower() == "write"


def write_tool_names(tools: Iterable[Any]) -> frozenset[str]:
    return frozenset(
        getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools if tool_mutates_state(t)
    )


def decision_type(tool_name: str, write_tools: frozenset[str]) -> DecisionType | None:
    """Classify a proposed tool call; None means it is not a decision point."""
    if tool_name == STOP_TOOL:
        return "end"
    if tool_name in TRANSFER_TOOLS:
        return "transfer"
    if tool_name in write_tools:
        return "write"
    return None


@dataclass(frozen=True)
class Decision:
    action: Proposal
    clusters: tuple[Cluster, ...]
    pooled_uncertainty: float
    disagreement: float
    escalated: bool
    oracle_answer: OracleAnswer | None
    record: DecisionRecord


class Orchestrator:
    def __init__(
        self,
        workers: Sequence[LLMClient],
        *,
        n_samples: int,
        threshold: float,
        oracle: Oracle | None = None,
        escalate: bool = False,
        temperature: float = 0.7,
        max_escalations_per_task: int = 3,
        orchestrator_model: str = "",
    ) -> None:
        if not workers or n_samples < 1:
            raise ValueError("need at least one worker and one sample")
        self.workers = list(workers)
        self.n_samples = n_samples
        self.threshold = threshold
        self.oracle: Oracle = oracle or NoOracle()
        self.escalate = escalate
        self.temperature = temperature
        self.max_escalations = max_escalations_per_task
        self.orchestrator_model = orchestrator_model
        self.usage = Usage()
        self._escalations = 0
        self._index = 0

    def reset_task(self) -> None:
        self._escalations = 0
        self._index = 0

    def decide(
        self,
        packet: DecisionPacket,
        decision_type: DecisionType,
        *,
        task_id: str,
        trial: int,
        given: Sequence[Proposal] = (),
        shadow: bool = False,
    ) -> Decision:
        """`given` are proposals already drawn for worker 0; they count toward its N.
        A shadow decision is measured and recorded but never escalated."""
        started = time.monotonic()
        usage = Usage()
        by_worker = self._sample(packet, usage, given)
        clusters = cluster([p for ps in by_worker for p in ps])
        u = pooled_uncertainty(by_worker)
        d = disagreement(by_worker)
        would_escalate = u >= self.threshold
        can_escalate = not shadow and self.escalate and self._escalations < self.max_escalations

        answer: OracleAnswer | None = None
        escalated = False
        action = clusters[0].example
        if would_escalate and can_escalate:
            escalated = True
            self._escalations += 1
            answer = self.oracle.consult(packet, clusters, u)
            if answer.verdict == "choose" and answer.chosen is not None:
                action = answer.chosen
            elif answer.verdict == "veto":
                # Re-plan once: resample and take the new majority.
                by_worker = self._sample(packet, usage)
                clusters = cluster([p for ps in by_worker for p in ps])
                action = clusters[0].example
            # A hint is recorded but, in this version, leaves the majority action in place.

        self.usage.add(usage)
        record = DecisionRecord(
            task_id=task_id,
            trial=trial,
            decision_index=self._index,
            decision_type=decision_type,
            k_workers=len(self.workers),
            n_samples=self.n_samples,
            models={
                "orchestrator": self.orchestrator_model,
                **{f"worker_{i}": w.model for i, w in enumerate(self.workers)},
            },
            temperature=self.temperature,
            proposals=[
                ProposalRecord(p.tool, p.args, p.reason, p.worker_id, p.key)
                for ps in by_worker
                for p in ps
            ],
            cluster_sizes=[c.size for c in clusters],
            pooled_uncertainty=u,
            disagreement=d,
            entropy=normalized_entropy([p for ps in by_worker for p in ps]),
            threshold=self.threshold,
            chosen_tool=action.tool,
            chosen_args=action.args,
            would_escalate=would_escalate,
            escalated=escalated,
            oracle_type=self.oracle.name if escalated else None,
            oracle_answer=str(answer) if answer else None,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=usage.cost_usd,
            wall_ms=int((time.monotonic() - started) * 1000),
        )
        self._index += 1
        return Decision(action, tuple(clusters), u, d, escalated, answer, record)

    def _sample(
        self, packet: DecisionPacket, usage: Usage, given: Sequence[Proposal] = ()
    ) -> list[list[Proposal]]:
        by_worker: list[list[Proposal]] = []
        for i, worker in enumerate(self.workers):
            proposals = list(given) if i == 0 else []
            while len(proposals) < self.n_samples:
                response = worker.propose(packet, temperature=self.temperature, worker_id=f"w{i}")
                usage.add(response.usage)
                proposals.append(response.proposal)
            by_worker.append(proposals)
        return by_worker

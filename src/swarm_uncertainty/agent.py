"""The tau2 agent that routes decision points through the orchestrator.

Every turn is one ordinary model call, as in tau2's own LLMAgent. When that call proposes a
write or a transfer, the turn becomes a decision point: the orchestrator samples, clusters,
scores and may escalate, and the agent sends whichever proposal it chose. With one worker
and one sample, the agent's own call is the only sample, so the agent is tau2's plain agent.
"""

from __future__ import annotations

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage
from tau2.environment.tool import Tool
from tau2.utils.utils import get_now

from swarm_uncertainty.actions import Proposal, proposal_from_message
from swarm_uncertainty.llm import DecisionPacket
from swarm_uncertainty.orchestrator import Orchestrator, decision_type, write_tool_names
from swarm_uncertainty.records import DecisionRecord, DecisionType


class SamplingAgent(LLMAgent[LLMAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        orchestrator: Orchestrator,
        *,
        task_id: str,
        trial: int,
        llm_args: dict | None = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.orchestrator = orchestrator
        self.write_tools = write_tool_names(tools)
        self.task_id = task_id
        self.trial = trial
        self.records: list[DecisionRecord] = []

    def _generate_next_message(self, message, state: LLMAgentState) -> AssistantMessage:
        first = super()._generate_next_message(message, state)
        kind = self._decision_type(first)
        if kind is None:
            return first

        packet = DecisionPacket(
            policy=self.domain_policy,
            conversation=(),
            messages=tuple(state.system_messages + state.messages),
            tools=tuple(self.tools),
            write_tools=self.write_tools,
        )
        given: list[Proposal] = []
        if self.orchestrator.n_samples == 1 and len(self.orchestrator.workers) == 1:
            given.append(proposal_from_message(first, self.write_tools, "w0"))
        decision = self.orchestrator.decide(
            packet, kind, task_id=self.task_id, trial=self.trial, given=given
        )
        self.records.append(decision.record)
        # tau2 orders the trajectory by timestamp, so stamp the chosen sample as sent now
        return decision.action.message.model_copy(update={"timestamp": get_now()})

    def _decision_type(self, message: AssistantMessage) -> DecisionType | None:
        kinds = [decision_type(c.name, self.write_tools) for c in message.tool_calls or []]
        if "write" in kinds:
            return "write"
        if "transfer" in kinds:
            return "transfer"
        return None

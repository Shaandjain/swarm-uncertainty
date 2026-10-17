"""The tau2 agent that routes decision points through the orchestrator.

Every turn is one ordinary model call, as in tau2's own LLMAgent. When that call proposes a
write or a transfer, the turn becomes a decision point: the orchestrator samples, clusters,
scores and may escalate, and the agent sends whichever proposal it chose. With one worker
and one sample, the agent's own call is the only sample, so the agent is tau2's plain agent.

With plan probing on, a plain reply is also checked: if the user answering "yes" would lead
to a write or transfer, the reply put an action to the user. N samples then write the reply
afresh, each hears "yes", and the call each makes next is its planned action. The plans are
scored and recorded as a "plan" decision, but the agent's own reply is what gets sent.
"""

from __future__ import annotations

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.environment.tool import Tool
from tau2.utils.utils import get_now

from swarm_uncertainty.actions import RESPOND_TOOL, TRANSFER_TOOLS, Proposal, proposal_from_message
from swarm_uncertainty.llm import DecisionPacket, Usage
from swarm_uncertainty.orchestrator import Orchestrator, decision_type, write_tool_names
from swarm_uncertainty.records import DecisionRecord, DecisionType

CONFIRMATION = "Yes, please go ahead."


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
        probe_plans: bool = False,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.orchestrator = orchestrator
        self.write_tools = write_tool_names(tools)
        self.task_id = task_id
        self.trial = trial
        self.records: list[DecisionRecord] = []
        self.probe_plans = probe_plans

    def _generate_next_message(self, message, state: LLMAgentState) -> AssistantMessage:
        first = super()._generate_next_message(message, state)
        kind = self._decision_type(first)
        if kind is None:
            if self.probe_plans and not first.tool_calls:
                self._probe_plan(first, state.system_messages + state.messages)
            return first

        packet = self._packet(state.system_messages + state.messages)
        given: list[Proposal] = []
        if self.orchestrator.n_samples == 1 and len(self.orchestrator.workers) == 1:
            given.append(proposal_from_message(first, self.write_tools, "w0"))
        decision = self.orchestrator.decide(
            packet, kind, task_id=self.task_id, trial=self.trial, given=given
        )
        self.records.append(decision.record)
        # tau2 orders the trajectory by timestamp, so stamp the chosen sample as sent now
        return decision.action.message.model_copy(update={"timestamp": get_now()})

    def _packet(self, messages: list) -> DecisionPacket:
        return DecisionPacket(
            policy=self.domain_policy,
            conversation=(),
            messages=tuple(messages),
            tools=tuple(self.tools),
            write_tools=self.write_tools,
        )

    def _sample(self, messages: list, usage: Usage) -> Proposal:
        response = self.orchestrator.workers[0].propose(
            self._packet(messages), temperature=self.orchestrator.temperature, worker_id="w0"
        )
        usage.add(response.usage)
        return response.proposal

    def _after_yes(self, history: list, reply: AssistantMessage, usage: Usage) -> Proposal:
        """What the agent does once the user agrees to `reply`."""
        yes = UserMessage(role="user", content=CONFIRMATION)
        return self._sample([*history, reply, yes], usage)

    def _probe_plan(self, reply: AssistantMessage, history: list) -> None:
        usage = Usage()
        acting = self.write_tools | TRANSFER_TOOLS
        if self._after_yes(history, reply, usage).tool not in acting:
            self.orchestrator.usage.add(usage)
            return
        plans = []
        for _ in range(self.orchestrator.n_samples):
            sample = self._sample(history, usage)
            if sample.tool == RESPOND_TOOL:
                sample = self._after_yes(history, sample.message, usage)
            plans.append(sample)
        decision = self.orchestrator.decide(
            self._packet(history),
            "plan",
            task_id=self.task_id,
            trial=self.trial,
            given=plans,
            shadow=True,
        )
        self.orchestrator.usage.add(usage)
        record = decision.record
        record.tokens_in, record.tokens_out = usage.tokens_in, usage.tokens_out
        record.cost_usd = usage.cost_usd
        self.records.append(record)

    def _decision_type(self, message: AssistantMessage) -> DecisionType | None:
        kinds = [decision_type(c.name, self.write_tools) for c in message.tool_calls or []]
        if "write" in kinds:
            return "write"
        if "transfer" in kinds:
            return "transfer"
        return None

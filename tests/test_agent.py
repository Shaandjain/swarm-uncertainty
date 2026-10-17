import os
from pathlib import Path

import pytest

RETAIL = Path(os.environ.get("TAU2_DATA_DIR", "data")) / "tau2/domains/retail/db.json"
pytestmark = pytest.mark.skipif(not RETAIL.exists(), reason="tau2 retail data not present")

if RETAIL.exists():
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.registry import registry

    from swarm_uncertainty.actions import RESPOND_TOOL, proposal_from_message
    from swarm_uncertainty.agent import SamplingAgent
    from swarm_uncertainty.llm import FakeLLM
    from swarm_uncertainty.orchestrator import Orchestrator, write_tool_names


def call(name, **args):
    return AssistantMessage(
        role="assistant", content=None, tool_calls=[ToolCall(id=name, name=name, arguments=args)]
    )


def reply(text):
    return AssistantMessage(role="assistant", content=text)


CANCEL = {"order_id": "#W1", "reason": "no longer needed"}


@pytest.fixture
def env():
    return registry.get_env_constructor("retail")()


def agent_with(env, monkeypatch, first, samples, **orch):
    monkeypatch.setattr("tau2.agent.llm_agent.generate", lambda **_: first)
    writes = write_tool_names(env.get_tools())
    worker = FakeLLM([proposal_from_message(m, writes, "w0") for m in samples])
    orchestrator = Orchestrator([worker], **{"threshold": 0.5, **orch})
    agent = SamplingAgent(
        env.get_tools(), env.get_policy(), "fake", orchestrator, task_id="t", trial=0
    )
    return agent, worker


def step(agent):
    state = agent.get_init_state()
    return agent.generate_next_message(UserMessage(role="user", content="cancel it"), state)[0]


class TestProposalFromMessage:
    def test_reply_read_and_write(self, env):
        writes = frozenset({"cancel_pending_order"})
        assert proposal_from_message(reply("hi"), writes, "w0").key.startswith(RESPOND_TOOL)
        assert (
            proposal_from_message(reply("a"), writes, "w0").key
            == proposal_from_message(reply("b"), writes, "w0").key
        )
        read = proposal_from_message(call("get_order_details", order_id="#W1"), writes, "w0")
        assert read.tool == "read:get_order_details" and read.args == {}
        write = proposal_from_message(call("cancel_pending_order", **CANCEL), writes, "w0")
        assert write.tool == "cancel_pending_order" and write.args == CANCEL


class TestSamplingAgent:
    def test_reads_and_replies_skip_the_orchestrator(self, env, monkeypatch):
        agent, worker = agent_with(env, monkeypatch, reply("which order?"), [], n_samples=5)
        assert step(agent).content == "which order?"
        assert agent.records == [] and worker.calls == []

    def test_n1_uses_the_agents_own_call(self, env, monkeypatch):
        first = call("cancel_pending_order", **CANCEL)
        agent, worker = agent_with(env, monkeypatch, first, [], n_samples=1)
        assert step(agent).tool_calls == first.tool_calls
        assert worker.calls == []
        assert agent.records[0].pooled_uncertainty == 0.0

    def test_write_is_sampled_and_majority_sent(self, env, monkeypatch):
        first = call("cancel_pending_order", **CANCEL)
        ask = reply("Can you confirm you want to cancel #W1?")
        samples = [ask, ask, ask, first, first]
        agent, worker = agent_with(env, monkeypatch, first, samples, n_samples=5)
        sent = step(agent)
        assert len(worker.calls) == 5
        assert sent.content == ask.content and not sent.tool_calls
        record = agent.records[0]
        assert record.decision_type == "write"
        assert record.chosen_tool == RESPOND_TOOL
        assert record.pooled_uncertainty == pytest.approx(0.4)

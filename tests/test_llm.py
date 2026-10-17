import pytest

from swarm_uncertainty.actions import Proposal
from swarm_uncertainty.llm import DecisionPacket, FakeLLM, LiteLLMClient

PACKET = DecisionPacket(policy="p", conversation=({"role": "user", "content": "hi"},))


def test_fake_llm_replays_in_order_and_tags_worker():
    llm = FakeLLM([Proposal("a"), Proposal("b")])
    first = llm.propose(PACKET, temperature=0.7, worker_id="w1").proposal
    second = llm.propose(PACKET, temperature=0.7, worker_id="w1").proposal
    assert (first.tool, second.tool) == ("a", "b")
    assert first.worker_id == "w1"
    assert len(llm.calls) == 2


def test_fake_llm_exhausts():
    llm = FakeLLM([Proposal("a")])
    llm.propose(PACKET, temperature=0.0, worker_id="w0")
    with pytest.raises(RuntimeError):
        llm.propose(PACKET, temperature=0.0, worker_id="w0")


def test_repeating_cycles():
    llm = FakeLLM.repeating([Proposal("a"), Proposal("b")], times=5)
    tools = [llm.propose(PACKET, temperature=0.7, worker_id="w0").proposal.tool for _ in range(5)]
    assert tools == ["a", "b", "a", "b", "a"]


def test_live_client_passes_temperature_and_counts_usage(monkeypatch):
    from tau2.data_model.message import AssistantMessage, ToolCall

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="1", name="cancel_pending_order", arguments={"order_id": "#W1"})
            ],
            cost=0.002,
            usage={"prompt_tokens": 1200, "completion_tokens": 40},
        )

    monkeypatch.setattr("tau2.utils.llm_utils.generate", fake_generate)
    client = LiteLLMClient("some-model", {"temperature": 0.0, "max_tokens": 512})
    packet = DecisionPacket(
        policy="p", conversation=(), write_tools=frozenset({"cancel_pending_order"})
    )
    response = client.propose(packet, temperature=1.0, worker_id="w2")

    assert seen["temperature"] == 1.0 and seen["max_tokens"] == 512
    assert response.proposal.tool == "cancel_pending_order"
    assert response.proposal.worker_id == "w2"
    assert (response.usage.tokens_in, response.usage.tokens_out) == (1200, 40)
    assert response.usage.cost_usd == 0.002

import pytest

from swarm_uncertainty.actions import Proposal
from swarm_uncertainty.llm import DecisionPacket, FakeLLM

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

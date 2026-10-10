from swarm_uncertainty.actions import Proposal
from swarm_uncertainty.llm import DecisionPacket
from swarm_uncertainty.measures import cluster
from swarm_uncertainty.oracle import GoldVetoOracle, NoOracle

PACKET = DecisionPacket(policy="p", conversation=())
TASK = {
    "id": "task_x",
    "evaluation_criteria": {
        "actions": [
            {"name": "get_order_details", "arguments": {"order_id": "#W1"}},
            {
                "name": "cancel_pending_order",
                "arguments": {"order_id": "#W1", "reason": "no longer needed"},
            },
        ]
    },
}


def test_gold_veto_chooses_matching_cluster_even_if_minority():
    props = [Proposal("transfer_to_human_agents", {"summary": "x"})] * 3 + [
        Proposal("cancel_pending_order", {"order_id": "#w1", "reason": "No longer needed"})
    ]
    answer = GoldVetoOracle.from_task(TASK).consult(PACKET, cluster(props), 0.25)
    assert answer.verdict == "choose"
    assert answer.chosen.tool == "cancel_pending_order"


def test_gold_veto_vetoes_when_nothing_matches():
    props = [Proposal("cancel_pending_order", {"order_id": "#W9", "reason": "x"})]
    answer = GoldVetoOracle.from_task(TASK).consult(PACKET, cluster(props), 0.0)
    assert answer.verdict == "veto"


def test_no_oracle_picks_majority():
    props = [Proposal("a")] * 2 + [Proposal("b")]
    answer = NoOracle().consult(PACKET, cluster(props), 1 / 3)
    assert answer.verdict == "choose" and answer.chosen.tool == "a"

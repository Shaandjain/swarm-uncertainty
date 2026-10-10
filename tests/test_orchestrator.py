import pytest
from tau2.environment.toolkit import ToolType, is_tool

from swarm_uncertainty.actions import STOP_TOOL, Proposal
from swarm_uncertainty.llm import DecisionPacket, FakeLLM
from swarm_uncertainty.oracle import GoldVetoOracle
from swarm_uncertainty.orchestrator import Orchestrator, decision_type, write_tool_names

PACKET = DecisionPacket(policy="p", conversation=())
CANCEL = Proposal("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})
RESPOND = Proposal("respond", {"text": "this order can't be cancelled"})
TRANSFER = Proposal("transfer_to_human_agents", {"summary": "x"})


def worker(proposals, n):
    return FakeLLM.repeating(proposals, times=n)


class TestDecisionPoints:
    def test_write_tools_come_from_tau2_marker(self):
        @is_tool(ToolType.READ)
        def get_order_details(order_id: str) -> str:
            return ""

        @is_tool(ToolType.WRITE)
        def cancel_pending_order(order_id: str) -> str:
            return ""

        writes = write_tool_names([get_order_details, cancel_pending_order])
        assert writes == {"cancel_pending_order"}
        assert decision_type("cancel_pending_order", writes) == "write"
        assert decision_type("get_order_details", writes) is None
        assert decision_type("transfer_to_human_agents", writes) == "transfer"
        assert decision_type(STOP_TOOL, writes) == "end"


class TestSingleAgent:
    def test_k1_n1_is_a_plain_agent(self):
        orch = Orchestrator([worker([CANCEL], 1)], n_samples=1, threshold=0.5)
        d = orch.decide(PACKET, "write", task_id="t", trial=0)
        assert d.action.tool == "cancel_pending_order"
        assert d.pooled_uncertainty == 0.0
        assert not d.escalated
        assert d.record.k_workers == 1 and d.record.n_samples == 1

    def test_k1_n5_self_consistency(self):
        w = worker([CANCEL, CANCEL, CANCEL, CANCEL, TRANSFER], 5)
        orch = Orchestrator([w], n_samples=5, threshold=0.5)
        d = orch.decide(PACKET, "write", task_id="t", trial=0)
        assert d.pooled_uncertainty == pytest.approx(0.2)
        assert d.record.would_escalate is False
        assert len(w.calls) == 5


class TestTeam:
    def worked_example(self, **kw):
        w1 = worker([CANCEL, CANCEL, CANCEL, CANCEL, TRANSFER], 5)
        w2 = worker([CANCEL, CANCEL, CANCEL, RESPOND, RESPOND], 5)
        w3 = worker([RESPOND, RESPOND, RESPOND, RESPOND, TRANSFER], 5)
        return Orchestrator([w1, w2, w3], n_samples=5, **kw)

    def test_measures_match_design_doc(self):
        d = self.worked_example(threshold=0.4).decide(PACKET, "write", task_id="t", trial=0)
        assert d.pooled_uncertainty == pytest.approx(1 - 7 / 15)
        assert d.disagreement == pytest.approx(1 / 3)
        assert [c.size for c in d.clusters] == [7, 6, 2]
        assert d.record.would_escalate is True

    def test_escalation_off_takes_majority_and_logs_intent(self):
        d = self.worked_example(threshold=0.4, escalate=False).decide(
            PACKET, "write", task_id="t", trial=0
        )
        assert d.action.tool == "cancel_pending_order"
        assert d.record.would_escalate and not d.record.escalated

    def test_gold_oracle_overrides_majority(self):
        gold = GoldVetoOracle([{"name": "respond", "arguments": RESPOND.args}])
        d = self.worked_example(threshold=0.4, escalate=True, oracle=gold).decide(
            PACKET, "write", task_id="t", trial=0
        )
        assert d.escalated
        assert d.action.tool == "respond"
        assert d.record.oracle_type == "gold_veto"

    def test_veto_resamples_once(self):
        gold = GoldVetoOracle([{"name": "exchange_delivered_order_items", "arguments": {}}])
        w1 = worker([CANCEL, TRANSFER] * 5, 10)
        w2 = worker([RESPOND, TRANSFER] * 5, 10)
        orch = Orchestrator([w1, w2], n_samples=5, threshold=0.1, escalate=True, oracle=gold)
        d = orch.decide(PACKET, "write", task_id="t", trial=0)
        assert d.oracle_answer.verdict == "veto"
        assert len(w1.calls) == 10  # first round + one re-plan
        assert d.action.tool in {"cancel_pending_order", "transfer_to_human_agents"}

    def test_escalation_budget_per_task(self):
        gold = GoldVetoOracle([{"name": "respond", "arguments": RESPOND.args}])
        orch = self.worked_example(
            threshold=0.0, escalate=True, oracle=gold, max_escalations_per_task=1
        )
        # Reload scripts so each decide() has proposals to draw on.
        for w in orch.workers:
            w._script.extend(list(w._script))
        first = orch.decide(PACKET, "write", task_id="t", trial=0)
        second = orch.decide(PACKET, "write", task_id="t", trial=0)
        assert first.escalated and not second.escalated
        assert second.record.decision_index == 1
        orch.reset_task()
        assert orch._escalations == 0 and orch._index == 0

import json
import os
from pathlib import Path

import pytest

from swarm_uncertainty.actions import STOP_TOOL, Proposal, action_key
from swarm_uncertainty.measures import (
    agreement,
    cluster,
    disagreement,
    majority,
    normalized_entropy,
    pooled_uncertainty,
    uncertainty,
)


def cancel(order="#W123", reason="no longer needed", worker="w0"):
    return Proposal("cancel_pending_order", {"order_id": order, "reason": reason}, worker_id=worker)


def transfer(worker="w0"):
    return Proposal("transfer_to_human_agents", {"summary": "cannot cancel"}, worker_id=worker)


class TestActionKey:
    def test_same_tool_and_args_match(self):
        assert cancel().key == cancel().key

    def test_different_args_do_not_match(self):
        assert cancel("#W123").key != cancel("#W999").key

    def test_case_and_whitespace_are_ignored(self):
        a = action_key("Cancel_Pending_Order", {"order_id": " #W123 "})
        b = action_key("cancel_pending_order", {"order_id": "#w123"})
        assert a == b

    def test_numeric_spellings_agree(self):
        assert action_key("t", {"x": 25}) == action_key("t", {"x": 25.0})
        assert action_key("t", {"x": 25}) == action_key("t", {"x": "25"})
        assert action_key("t", {"x": 25}) != action_key("t", {"x": 25.5})

    def test_leading_zero_ids_stay_strings(self):
        assert action_key("t", {"zip": "02134"}) != action_key("t", {"zip": 2134})

    def test_list_order_is_ignored(self):
        a = action_key("return_delivered_order_items", {"item_ids": ["b", "a"]})
        b = action_key("return_delivered_order_items", {"item_ids": ["a", "b"]})
        assert a == b

    def test_arg_order_is_ignored(self):
        assert action_key("t", {"a": 1, "b": 2}) == action_key("t", {"b": 2, "a": 1})

    def test_transfer_summary_is_ignored(self):
        a = action_key("transfer_to_human_agents", {"summary": "user wants a refund"})
        b = action_key("transfer_to_human_agents", {"summary": "Refund request, escalating."})
        assert a == b

    def test_reason_and_worker_do_not_affect_key(self):
        a = Proposal("t", {"x": 1}, reason="because", worker_id="w1")
        b = Proposal("t", {"x": 1}, reason="other", worker_id="w2")
        assert a.key == b.key


class TestMeasures:
    def test_unanimous(self):
        props = [cancel() for _ in range(5)]
        assert agreement(props) == 1.0
        assert uncertainty(props) == 0.0
        assert normalized_entropy(props) == 0.0

    def test_four_to_one(self):
        props = [cancel() for _ in range(4)] + [transfer()]
        assert agreement(props) == pytest.approx(0.8)
        assert uncertainty(props) == pytest.approx(0.2)
        assert majority(props).tool == "cancel_pending_order"

    def test_all_different(self):
        props = [Proposal("t", {"i": i}) for i in range(5)]
        assert agreement(props) == pytest.approx(0.2)
        assert normalized_entropy(props) == pytest.approx(1.0)

    def test_cluster_order_largest_first(self):
        props = [transfer(), cancel(), cancel()]
        sizes = [c.size for c in cluster(props)]
        assert sizes == [2, 1]

    def test_empty(self):
        assert agreement([]) == 0.0
        assert uncertainty([]) == 1.0
        assert majority([]) is None
        assert normalized_entropy([]) == 0.0

    def test_stop_is_its_own_action(self):
        props = [Proposal(STOP_TOOL), cancel()]
        assert agreement(props) == pytest.approx(0.5)


class TestTeam:
    def test_worked_example_from_design_doc(self):
        respond = Proposal("respond", {"text": "this order can't be cancelled"})
        w1 = [cancel(worker="w1")] * 4 + [transfer("w1")]
        w2 = [cancel(worker="w2")] * 3 + [respond] * 2
        w3 = [respond] * 4 + [transfer("w3")]
        assert uncertainty(w1) == pytest.approx(0.2)
        assert uncertainty(w2) == pytest.approx(0.4)
        assert uncertainty(w3) == pytest.approx(0.2)
        assert pooled_uncertainty([w1, w2, w3]) == pytest.approx(1 - 7 / 15)
        assert disagreement([w1, w2, w3]) == pytest.approx(1 / 3)

    def test_single_worker_reduces_to_self_consistency(self):
        w = [cancel()] * 4 + [transfer()]
        assert pooled_uncertainty([w]) == uncertainty(w)
        assert disagreement([w]) == 0.0

    def test_disagreement_ignores_empty_workers(self):
        assert disagreement([[cancel()], [], [transfer()]]) == pytest.approx(0.5)


RETAIL_TASKS = Path(os.environ.get("TAU2_DATA_DIR", "data")) / "tau2/domains/retail/tasks.json"


@pytest.mark.skipif(not RETAIL_TASKS.exists(), reason="tau2 retail data not present")
def test_gold_actions_are_self_consistent():
    """Every gold action must equal itself after normalization, and distinct gold actions
    within a task must stay distinct. Guards against a normalization that is too aggressive."""
    tasks = json.loads(RETAIL_TASKS.read_text())
    for task in tasks:
        actions = (task.get("evaluation_criteria") or {}).get("actions") or []
        keys = [action_key(a["name"], a.get("arguments")) for a in actions]
        for a, k in zip(actions, keys, strict=True):
            assert action_key(a["name"], a.get("arguments")) == k
        raw = {json.dumps([a["name"], a.get("arguments")], sort_keys=True) for a in actions}
        assert len(set(keys)) >= len(raw) - _duplicates_allowed(actions)


def _duplicates_allowed(actions):
    """Gold lists may repeat an identical action (e.g. two reads); those may collapse."""
    seen = set()
    dupes = 0
    for a in actions:
        k = json.dumps([a["name"], a.get("arguments")], sort_keys=True)
        if k in seen:
            dupes += 1
        seen.add(k)
    return dupes

from swarm_uncertainty.actions import RESPOND_TOOL
from swarm_uncertainty.labels import label_episode
from swarm_uncertainty.records import DecisionRecord

WRITES = frozenset({"cancel_pending_order", "modify_pending_order_address"})
CANCEL_W1 = {"order_id": "#W1", "reason": "no longer needed"}
GOLD = [
    {"name": "get_order_details", "arguments": {"order_id": "#W1"}},
    {"name": "cancel_pending_order", "arguments": CANCEL_W1},
]


def decision(i, tool, args=None, kind="write"):
    return DecisionRecord(
        task_id="t",
        trial=0,
        decision_index=i,
        decision_type=kind,
        k_workers=1,
        n_samples=5,
        models={},
        temperature=1.0,
        proposals=[],
        cluster_sizes=[5],
        pooled_uncertainty=0.0,
        disagreement=0.0,
        entropy=0.0,
        threshold=0.4,
        chosen_tool=tool,
        chosen_args=args or {},
        would_escalate=False,
        escalated=False,
    )


def labels(records):
    label_episode(records, GOLD, WRITES)
    return [(r.step_label, r.extra["after_divergence"]) for r in records]


def test_matching_write_is_right_and_spelling_does_not_matter():
    spelled = {"order_id": " #w1 ", "reason": "No longer needed"}
    assert labels([decision(0, "cancel_pending_order", spelled)]) == [(0, False)]


def test_wrong_arguments_are_wrong():
    wrong = {"order_id": "#W2", "reason": "no longer needed"}
    assert labels([decision(0, "cancel_pending_order", wrong)]) == [(1, False)]


def test_a_gold_action_can_be_used_once():
    records = [
        decision(0, "cancel_pending_order", CANCEL_W1),
        decision(1, "cancel_pending_order", CANCEL_W1),
    ]
    assert labels(records) == [(0, False), (1, False)]


def test_reply_or_read_majority_is_unlabelled():
    records = [decision(0, RESPOND_TOOL), decision(1, "read:get_order_details")]
    assert labels(records) == [(None, False), (None, False)]


def test_decisions_after_the_first_wrong_one_are_flagged():
    records = [
        decision(0, "modify_pending_order_address", {"order_id": "#W1"}),
        decision(1, "cancel_pending_order", CANCEL_W1),
        decision(2, RESPOND_TOOL),
    ]
    assert labels(records) == [(1, False), (0, True), (None, True)]


def test_transfer_not_in_gold_is_wrong():
    assert labels([decision(0, "transfer_to_human_agents", {"summary": "x"})]) == [(1, False)]


def test_a_plan_takes_the_label_of_the_action_that_follows():
    wrong = {"order_id": "#W2", "reason": "no longer needed"}
    records = [
        decision(0, "cancel_pending_order", CANCEL_W1, kind="plan"),
        decision(1, "cancel_pending_order", wrong),
        decision(2, "cancel_pending_order", CANCEL_W1, kind="plan"),
    ]
    assert labels(records) == [(1, False), (1, False), (None, True)]

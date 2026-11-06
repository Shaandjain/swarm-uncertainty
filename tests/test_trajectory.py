from types import SimpleNamespace

import pytest

from swarm_uncertainty.records import DecisionRecord
from swarm_uncertainty.trajectory import action_index, labelled_actions, proposal_index

WRITES = frozenset({"cancel_pending_order"})


def msg(role, content=None, calls=()):
    tool_calls = [SimpleNamespace(name=c) for c in calls] or None
    return SimpleNamespace(role=role, content=content, tool_calls=tool_calls)


CONVERSATION = [
    msg("user", "cancel my order"),
    msg("assistant", calls=["get_order_details"]),
    msg("tool", "{...}"),
    msg("assistant", "I'll cancel #W1. Confirm?"),
    msg("user", "yes"),
    msg("assistant", calls=["cancel_pending_order"]),
    msg("tool", "{...}"),
    msg("assistant", calls=["transfer_to_human_agents"]),
]


def record(i, label, kind="write", diverged=False):
    return DecisionRecord(
        task_id="7",
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
        chosen_tool="cancel_pending_order",
        chosen_args={},
        would_escalate=False,
        escalated=False,
        step_label=label,
        extra={"after_divergence": diverged},
    )


def test_actions_are_ranked_among_executed_ones_and_plans_are_skipped():
    records = [
        record(0, 0, kind="plan"),
        record(1, None),
        record(2, 0),
        record(3, 1),
        record(4, 0, diverged=True),
    ]
    assert [(r.decision_index, rank) for r, rank in labelled_actions(records)] == [(2, 0), (3, 1)]


def test_action_index_counts_writes_and_transfers_only():
    assert action_index(CONVERSATION, 0, WRITES) == 5
    assert action_index(CONVERSATION, 1, WRITES) == 7
    with pytest.raises(ValueError):
        action_index(CONVERSATION, 2, WRITES)


def test_proposal_is_the_last_reply_the_user_answered():
    assert proposal_index(CONVERSATION, 5) == 3
    assert proposal_index(CONVERSATION, 1) is None
    unanswered = [msg("user", "cancel it"), msg("assistant", "Cancelling now."), msg("tool", "")]
    assert proposal_index(unanswered, 2) is None

from swarm_uncertainty.records import DecisionRecord, ProposalRecord, read_records, write_records


def make_record(i=0):
    return DecisionRecord(
        task_id="task_001",
        trial=0,
        decision_index=i,
        decision_type="write",
        k_workers=3,
        n_samples=5,
        models={"orchestrator": "m-big", "worker": "m-small"},
        temperature=0.7,
        proposals=[ProposalRecord("cancel_pending_order", {"order_id": "#W1"}, "", "w1", "k")],
        cluster_sizes=[7, 6, 2],
        pooled_uncertainty=1 - 7 / 15,
        disagreement=1 / 3,
        entropy=0.9,
        threshold=0.4,
        chosen_tool="cancel_pending_order",
        chosen_args={"order_id": "#W1"},
        would_escalate=True,
        escalated=False,
    )


def test_round_trip(tmp_path):
    path = tmp_path / "run" / "decisions.jsonl"
    write_records(path, [make_record(0), make_record(1)])
    write_records(path, [make_record(2)])  # append
    back = list(read_records(path))
    assert [r.decision_index for r in back] == [0, 1, 2]
    assert back[0] == make_record(0)
    assert back[0].proposals[0].tool == "cancel_pending_order"

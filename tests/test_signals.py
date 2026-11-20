from swarm_uncertainty.signals import kind_of, score


def test_kind_of_splits_wrong_actions_by_audit_category():
    assert kind_of(0, None) == "right"
    assert kind_of(1, "slip") == "slip"
    assert kind_of(1, "interpretation") == "misreading"
    assert kind_of(1, "policy") == "misreading"
    assert kind_of(1, "user_simulator") == "benchmark"
    assert kind_of(1, None) == "benchmark"


def test_score_counts_flags_per_kind():
    kinds = ["right", "right", "right", "slip", "slip", "misreading", "benchmark"]
    flags = [False, True, False, True, True, False, True]
    s = score("x", flags, kinds)
    assert (s.flagged, s.total) == (4, 7)
    assert s.recall("slip") == 1.0
    assert s.recall("misreading") == 0.0
    assert s.recall("right") == 1 / 3
    # two of the four flagged actions are the agent's own errors
    assert s.precision == 0.5
    assert s.base_rate == 3 / 7
    assert s.break_even(0.5) == 0.25


def test_score_with_nothing_flagged():
    s = score("x", [False, False], ["right", "slip"])
    assert s.precision == 0.0
    assert s.flag_rate == 0.0

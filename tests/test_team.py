import pytest

from swarm_uncertainty.team import team_signal


class TestTeamSignal:
    def test_everyone_agrees(self):
        s = team_signal({"a": ["x"] * 4, "b": ["x"] * 2, "c": ["x"]}, actor="a")
        assert (s.self_uncertainty, s.disagreement, s.pooled) == (0.0, 0.0, 0.0)
        assert not s.actor_outvoted

    def test_confident_actor_outvoted_by_the_team(self):
        s = team_signal({"a": ["x"] * 10, "b": ["y", "y", "x"], "c": ["y"] * 3}, actor="a")
        assert s.self_uncertainty == 0.0
        assert s.disagreement == pytest.approx(1 / 3)
        assert s.team_majority == "y" and s.actor_outvoted
        # x gets 1, 1/3, 0 of the three models' votes; y gets 0, 2/3, 1
        assert s.pooled == pytest.approx(1 - (2 / 3 + 1) / 3)

    def test_three_way_split_keeps_the_actor(self):
        s = team_signal({"a": ["x", "x", "z"], "b": ["y"], "c": ["z"]}, actor="a")
        assert s.team_majority == "x" and not s.actor_outvoted
        assert s.disagreement == pytest.approx(2 / 3)
        assert s.self_uncertainty == pytest.approx(1 / 3)

    def test_model_without_samples_is_left_out(self):
        s = team_signal({"a": ["x"], "b": [], "c": ["y"]}, actor="a")
        assert s.disagreement == pytest.approx(0.5)

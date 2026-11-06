import math

import pytest

from swarm_uncertainty.calibration import (
    accuracy_by_level,
    as_u_threshold,
    calibration_error,
    episode_certainty,
    fit_threshold,
    threshold_sweep,
)
from swarm_uncertainty.records import DecisionRecord


def record(task: str, trial: int, index: int, u: float, kind: str = "write") -> DecisionRecord:
    return DecisionRecord(
        task_id=task,
        trial=trial,
        decision_index=index,
        decision_type=kind,
        k_workers=1,
        n_samples=5,
        models={},
        temperature=1.0,
        proposals=[],
        cluster_sizes=[],
        pooled_uncertainty=u,
        disagreement=0.0,
        entropy=0.0,
        threshold=0.4,
        chosen_tool="t",
        chosen_args={},
        would_escalate=False,
        escalated=False,
    )


class TestAccuracyByLevel:
    def test_counts_and_accuracy_per_exact_level(self):
        levels = accuracy_by_level([1.0, 1.0, 1.0, 0.8, 0.6], [0, 0, 1, 0, 1])
        assert [(lv.certainty, lv.count, lv.errors) for lv in levels] == [
            (1.0, 3, 1),
            (0.8, 1, 0),
            (0.6, 1, 1),
        ]
        assert levels[0].share == pytest.approx(0.6)
        assert levels[0].accuracy == pytest.approx(2 / 3)

    def test_calibration_error(self):
        levels = accuracy_by_level([1.0, 1.0, 0.5, 0.5], [0, 1, 0, 0])
        # 0.5 * |0.5 - 1.0| + 0.5 * |1.0 - 0.5|
        assert calibration_error(levels) == pytest.approx(0.5)


class TestThresholdSweep:
    def test_endpoints(self):
        points = threshold_sweep([1.0, 0.8, 0.6, 1.0], [0, 1, 1, 1])
        first, last = points[0], points[-1]
        assert first.interventions == 0 and first.missed == 3
        assert last.threshold == math.inf and last.interventions == 4 and last.caught == 3

    def test_middle_threshold(self):
        points = threshold_sweep([1.0, 0.8, 0.6, 1.0], [0, 1, 1, 1])
        at_one = next(p for p in points if p.threshold == 1.0)
        assert at_one.interventions == 2 and at_one.caught == 2
        assert at_one.residual_error_rate == pytest.approx(0.5)
        assert at_one.caught_per_intervention == 1.0


class TestFitThreshold:
    CERTAINTIES = (1.0, 1.0, 0.8, 0.8, 0.6, 0.4)
    LABELS = (0, 1, 0, 1, 1, 1)

    def test_most_errors_within_budget(self):
        points = threshold_sweep(self.CERTAINTIES, self.LABELS)
        # 6 decisions over 2 episodes: one call per episode allows 2 calls
        best = fit_threshold(points, episodes=2, budget=1.0)
        assert best.threshold == 0.8 and best.interventions == 2 and best.caught == 2

    def test_nothing_fits(self):
        points = threshold_sweep(self.CERTAINTIES, self.LABELS)
        assert fit_threshold(points, episodes=100, budget=0.001) is None

    def test_u_threshold_calls_the_same_decisions(self):
        points = threshold_sweep(self.CERTAINTIES, self.LABELS)
        best = fit_threshold(points, episodes=2, budget=1.0)
        u = as_u_threshold(best, self.CERTAINTIES)
        assert u == 0.4
        called = [c for c in self.CERTAINTIES if round(1 - c, 6) >= u]
        assert len(called) == best.interventions


class TestEpisodeCertainty:
    def test_aggregators_on_two_decisions(self):
        records = [record("a", 0, 0, 0.2), record("a", 0, 1, 0.4), record("b", 0, 0, 0.0)]
        assert episode_certainty(records, "product")[("a", 0)] == pytest.approx(0.48)
        assert episode_certainty(records, "min")[("a", 0)] == pytest.approx(0.6)
        assert episode_certainty(records, "mean")[("a", 0)] == pytest.approx(0.7)
        assert episode_certainty(records, "max")[("b", 0)] == 1.0

    def test_kind_separates_plans_from_actions(self):
        records = [record("a", 0, 0, 0.4, kind="plan"), record("a", 0, 1, 0.0)]
        assert episode_certainty(records, "min", kind="plan") == {("a", 0): pytest.approx(0.6)}
        assert episode_certainty(records, "min") == {("a", 0): 1.0}

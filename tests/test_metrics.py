import pytest

from swarm_uncertainty.metrics import (
    auroc,
    expected_calibration_error,
    mean_pass_at_k,
    pass_at_k,
    reliability_bins,
    risk_coverage,
)


class TestAuroc:
    def test_perfect_ranking(self):
        assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0

    def test_reversed_ranking(self):
        assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0

    def test_ties_count_half(self):
        assert auroc([0.5, 0.5], [0, 1]) == 0.5

    def test_single_class(self):
        assert auroc([0.1, 0.9], [1, 1]) == 0.5


class TestCalibration:
    def test_bins_and_ece(self):
        scores = [0.1, 0.1, 0.9, 0.9]
        labels = [0, 0, 1, 1]
        bins = reliability_bins(scores, labels, n_bins=2)
        assert [b.count for b in bins] == [2, 2]
        assert bins[0].error_rate == 0.0 and bins[1].error_rate == 1.0
        assert expected_calibration_error(scores, labels, n_bins=2) == pytest.approx(0.1)

    def test_score_of_one_lands_in_top_bin(self):
        bins = reliability_bins([1.0], [1], n_bins=5)
        assert bins[-1].count == 1


class TestRiskCoverage:
    def test_curve(self):
        pts = risk_coverage([0.0, 0.2, 0.6, 0.8], [0, 0, 1, 1])
        by_t = {round(p.threshold, 3): p for p in pts}
        assert by_t[0.6].coverage == 0.5 and by_t[0.6].risk == 0.0
        full = pts[-1]
        assert full.coverage == 1.0 and full.risk == 0.5


class TestPassAtK:
    def test_matches_tau_bench_definition(self):
        assert pass_at_k([1, 1, 1, 1], 4) == 1.0
        assert pass_at_k([1, 1, 0, 0], 1) == 0.5
        assert pass_at_k([1, 1, 0, 0], 2) == pytest.approx(1 / 6)
        assert pass_at_k([1, 1, 0, 0], 4) == 0.0

    def test_mean_over_tasks(self):
        assert mean_pass_at_k({"a": [1, 1], "b": [0, 0]}, 1) == 0.5

    def test_k_too_large(self):
        with pytest.raises(ValueError):
            pass_at_k([1], 2)

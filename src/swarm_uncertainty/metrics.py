"""Metrics for "does the uncertainty number predict mistakes" and "what does escalating buy".

Inputs are plain lists: a score per decision point (higher = more uncertain) and a label
(1 = the action was wrong). Pure Python so the analysis has no extra dependencies.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import comb


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Probability that a random wrong decision scored higher than a random right one.

    Ties count half. Returns 0.5 when either class is empty.
    """
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    mean_score: float
    error_rate: float


def reliability_bins(scores: Sequence[float], labels: Sequence[int], n_bins: int = 5) -> list[Bin]:
    """Group decisions by score into equal-width bins; report observed error rate per bin."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = []
    for lo, hi in pairwise(edges):
        members = [
            (s, y)
            for s, y in zip(scores, labels, strict=True)
            if lo <= s < hi or (hi == 1.0 and s == 1.0)
        ]
        if members:
            bins.append(
                Bin(
                    lo,
                    hi,
                    len(members),
                    sum(s for s, _ in members) / len(members),
                    sum(y for _, y in members) / len(members),
                )
            )
        else:
            bins.append(Bin(lo, hi, 0, 0.0, 0.0))
    return bins


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[int], n_bins: int = 5
) -> float:
    """Count-weighted mean |error rate - mean score| across bins."""
    total = len(scores)
    if total == 0:
        return 0.0
    return sum(
        b.count / total * abs(b.error_rate - b.mean_score)
        for b in reliability_bins(scores, labels, n_bins)
        if b.count
    )


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float


def risk_coverage(scores: Sequence[float], labels: Sequence[int]) -> list[RiskCoveragePoint]:
    """For each distinct threshold: coverage = share handled alone (score < threshold),
    risk = error rate among those. Includes threshold just above the max (full coverage)."""
    pairs = sorted(zip(scores, labels, strict=True))
    thresholds = sorted({s for s, _ in pairs} | {max(scores, default=0.0) + 1e-9})
    out = []
    for t in thresholds:
        kept = [y for s, y in pairs if s < t]
        coverage = len(kept) / len(pairs) if pairs else 0.0
        risk = sum(kept) / len(kept) if kept else 0.0
        out.append(RiskCoveragePoint(t, coverage, risk))
    return out


def pass_at_k(successes: Sequence[int], k: int) -> float:
    """tau-bench pass^k for one task: chance that k random trials all succeed."""
    n, c = len(successes), sum(successes)
    if k > n:
        raise ValueError("k larger than number of trials")
    return comb(c, k) / comb(n, k)


def mean_pass_at_k(trials_by_task: dict[str, Sequence[int]], k: int) -> float:
    if not trials_by_task:
        return 0.0
    return sum(pass_at_k(t, k) for t in trials_by_task.values()) / len(trials_by_task)


def bootstrap_ci[T](
    groups: Sequence[Sequence[T]],
    statistic: Callable[[list[T]], float],
    *,
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval from resampling whole groups (tasks) with replacement.

    Decisions within a task are correlated, so tasks, not decisions, are the unit resampled.
    """
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        pooled = [item for _ in groups for item in rng.choice(groups)]
        values.append(statistic(pooled))
    values.sort()
    tail = (1 - level) / 2
    return values[int(tail * (n - 1))], values[int((1 - tail) * (n - 1))]

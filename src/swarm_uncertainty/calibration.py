"""Certainty against accuracy, where to put the threshold, and certainty over a whole episode.

Certainty is the share of samples in the largest group (1 - U). Labels follow metrics.py:
1 = the action was wrong. With n samples certainty takes only n distinct values, so levels
are grouped by exact value rather than into equal-width bins.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from swarm_uncertainty.records import DecisionRecord


def certainty(record: DecisionRecord) -> float:
    """1 - U, rounded so 0.8 from 4/5 and 1 - 0.2 compare equal."""
    return round(1.0 - record.pooled_uncertainty, 6)


@dataclass(frozen=True)
class Level:
    certainty: float
    count: int
    share: float
    errors: int

    @property
    def accuracy(self) -> float:
        return 1.0 - self.errors / self.count


def accuracy_by_level(certainties: Sequence[float], labels: Sequence[int]) -> list[Level]:
    """One row per distinct certainty, most certain first."""
    groups: dict[float, list[int]] = defaultdict(list)
    for c, y in zip(certainties, labels, strict=True):
        groups[c].append(y)
    total = len(certainties)
    return [
        Level(c, len(ys), len(ys) / total, sum(ys))
        for c, ys in sorted(groups.items(), reverse=True)
    ]


def calibration_error(levels: Sequence[Level]) -> float:
    """Expected calibration error over exact levels: sum of share x |accuracy - certainty|."""
    return sum(level.share * abs(level.accuracy - level.certainty) for level in levels)


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    interventions: int
    intervention_rate: float
    caught: int
    missed: int
    residual_error_rate: float

    @property
    def caught_per_intervention(self) -> float:
        return self.caught / self.interventions if self.interventions else 0.0


# A human is called whenever certainty falls below the threshold. Raising the threshold one
# level at a time shows what each extra block of interventions buys: errors caught, and the
# error rate of the decisions the agent still makes alone.
def threshold_sweep(certainties: Sequence[float], labels: Sequence[int]) -> list[SweepPoint]:
    """One point per candidate threshold, from no interventions to intervening on everything."""
    pairs = list(zip(certainties, labels, strict=True))
    levels = sorted({c for c, _ in pairs})
    thresholds = [levels[0] if levels else 0.0, *levels[1:], math.inf]
    points = []
    for t in thresholds:
        called = [y for c, y in pairs if c < t]
        alone = [y for c, y in pairs if c >= t]
        points.append(
            SweepPoint(
                threshold=t,
                interventions=len(called),
                intervention_rate=len(called) / len(pairs) if pairs else 0.0,
                caught=sum(called),
                missed=sum(alone),
                residual_error_rate=sum(alone) / len(alone) if alone else 0.0,
            )
        )
    return points


AGGREGATORS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": lambda cs: sum(cs) / len(cs),
    "product": math.prod,
    "min": min,
    "max": max,
}
"""Ways to combine the certainties of an episode's decisions into one number. `min` is the
least certain decision (the maximum U); `product` treats the decisions as independent."""


def episode_certainty(
    records: Iterable[DecisionRecord], how: str, kind: str = "action"
) -> dict[tuple[str, int], float]:
    """Certainty of each episode that has at least one decision of `kind` ("action"/"plan").

    Uses every decision of that kind, labelled or not: the episode outcome is the label.
    """
    combine = AGGREGATORS[how]
    by_episode: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        if (r.decision_type == "plan") == (kind == "plan"):
            by_episode[(r.task_id, r.trial)].append(certainty(r))
    return {episode: combine(cs) for episode, cs in by_episode.items()}

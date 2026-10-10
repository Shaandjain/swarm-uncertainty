"""Self-consistency uncertainty over a set of proposed actions.

All measures are plain counts and ratios.

- agreement:            share of proposals in the largest cluster
- uncertainty:          1 - agreement
- pooled uncertainty:   the same, over every worker's proposals together (the orchestrator's U)
- disagreement:         1 - share of workers whose majority action matches the most common majority (D)
- normalized entropy:   spread of the vote across clusters, scaled to [0, 1] (H)
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from swarm_uncertainty.actions import Proposal


@dataclass(frozen=True)
class Cluster:
    key: str
    size: int
    members: tuple[Proposal, ...]

    @property
    def example(self) -> Proposal:
        return self.members[0]


def cluster(proposals: Sequence[Proposal]) -> list[Cluster]:
    """Group proposals by action key, largest cluster first. Ties keep first-seen order."""
    order: dict[str, list[Proposal]] = {}
    for proposal in proposals:
        order.setdefault(proposal.key, []).append(proposal)
    clusters = [Cluster(key, len(members), tuple(members)) for key, members in order.items()]
    clusters.sort(key=lambda c: -c.size)
    return clusters


def agreement(proposals: Sequence[Proposal]) -> float:
    """Share of proposals that chose the most common action. Empty input -> 0."""
    if not proposals:
        return 0.0
    return cluster(proposals)[0].size / len(proposals)


def uncertainty(proposals: Sequence[Proposal]) -> float:
    """1 - agreement. Empty input -> 1 (we know nothing)."""
    if not proposals:
        return 1.0
    return 1.0 - agreement(proposals)


def pooled_uncertainty(proposals_by_worker: Iterable[Sequence[Proposal]]) -> float:
    """U: uncertainty over every worker's proposals pooled together."""
    pooled: list[Proposal] = []
    for proposals in proposals_by_worker:
        pooled.extend(proposals)
    return uncertainty(pooled)


def majority(proposals: Sequence[Proposal]) -> Proposal | None:
    """The most common proposal, or None if there are none."""
    if not proposals:
        return None
    return cluster(proposals)[0].example


def disagreement(proposals_by_worker: Sequence[Sequence[Proposal]]) -> float:
    """D: share of workers whose majority action differs from the most common majority.

    Workers with no proposals are ignored. Fewer than two workers -> 0.
    """
    majorities = [majority(p) for p in proposals_by_worker]
    keys = [m.key for m in majorities if m is not None]
    if len(keys) < 2:
        return 0.0
    top = Counter(keys).most_common(1)[0][1]
    return 1.0 - top / len(keys)


def normalized_entropy(proposals: Sequence[Proposal]) -> float:
    """H: entropy of the cluster distribution divided by ln(N). 0 = unanimous, 1 = all different."""
    n = len(proposals)
    if n <= 1:
        return 0.0
    sizes = [c.size for c in cluster(proposals)]
    entropy = -sum((s / n) * math.log(s / n) for s in sizes)
    return entropy / math.log(n)

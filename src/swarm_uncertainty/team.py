"""Uncertainty from a team of different models looking at one decision.

Each model gives a few samples of the action it would take. Three signals come out:
  self         1 - agreement among the acting model's own samples (the single-agent U)
  disagreement share of models whose majority action is not the team's most common majority
  pooled       1 - the largest share an action gets when every model's vote counts equally
A team catches what self-consistency cannot: an error the acting model makes every time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TeamSignal:
    self_uncertainty: float
    disagreement: float
    pooled: float
    actor_majority: str
    team_majority: str

    @property
    def actor_outvoted(self) -> bool:
        return self.actor_majority != self.team_majority


def team_signal(samples: Mapping[str, Sequence[str]], actor: str) -> TeamSignal:
    """`samples` maps each model to the action keys it sampled; `actor` is the model that acts.
    Models with no samples are left out. Ties go to the actor's majority, then first seen."""
    votes = {m: Counter(keys) for m, keys in samples.items() if keys}
    majorities = {m: c.most_common(1)[0][0] for m, c in votes.items()}
    ranked = Counter(majorities.values()).most_common()
    top = [key for key, n in ranked if n == ranked[0][1]]
    team = majorities[actor] if majorities[actor] in top else top[0]
    shares: Counter[str] = Counter()
    for counts in votes.values():
        total = sum(counts.values())
        for key, n in counts.items():
            shares[key] += n / total / len(votes)
    own = votes[actor]
    return TeamSignal(
        self_uncertainty=1.0 - own.most_common(1)[0][1] / sum(own.values()),
        disagreement=sum(m != team for m in majorities.values()) / len(majorities),
        pooled=1.0 - max(shares.values()),
        actor_majority=majorities[actor],
        team_majority=team,
    )

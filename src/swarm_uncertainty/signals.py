"""Score a yes/no signal against wrong actions split by where the error came from.

An action is right, a slip (the call differs from the agreed plan), a misreading (the agent
misread the request or the policy) or a benchmark error (the audit blames the simulated
customer, the answer key or the labelling). A signal is useful for intervening only on the
first two, and a different signal may be needed for each.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

KINDS = ("right", "slip", "misreading", "benchmark")
AGENT_KINDS = frozenset({"slip", "misreading"})


def kind_of(label: int, category: str | None) -> str:
    """Error kind from the grader's label and the audit's category (None when not audited)."""
    if label == 0:
        return "right"
    if category == "slip":
        return "slip"
    if category in ("interpretation", "policy"):
        return "misreading"
    return "benchmark"


@dataclass(frozen=True)
class SignalScore:
    name: str
    flagged: int
    total: int
    flagged_by_kind: Mapping[str, int]
    total_by_kind: Mapping[str, int]

    @property
    def flag_rate(self) -> float:
        return self.flagged / self.total if self.total else 0.0

    def recall(self, kind: str) -> float:
        n = self.total_by_kind.get(kind, 0)
        return self.flagged_by_kind.get(kind, 0) / n if n else 0.0

    @property
    def precision(self) -> float:
        """Share of flagged actions that are the agent's own errors."""
        hits = sum(self.flagged_by_kind.get(k, 0) for k in AGENT_KINDS)
        return hits / self.flagged if self.flagged else 0.0

    @property
    def base_rate(self) -> float:
        errors = sum(self.total_by_kind.get(k, 0) for k in AGENT_KINDS)
        return errors / self.total if self.total else 0.0

    def break_even(self, fix_rate: float = 1.0) -> float:
        """Calling on a flag pays while a call costs less than this share of a wrong action:
        P(agent error | flag) x P(the call fixes it)."""
        return self.precision * fix_rate


def score(name: str, flags: Sequence[bool], kinds: Sequence[str]) -> SignalScore:
    flagged_by_kind = dict.fromkeys(KINDS, 0)
    total_by_kind = dict.fromkeys(KINDS, 0)
    for flag, kind in zip(flags, kinds, strict=True):
        total_by_kind[kind] += 1
        flagged_by_kind[kind] += bool(flag)
    return SignalScore(name, sum(map(bool, flags)), len(flags), flagged_by_kind, total_by_kind)

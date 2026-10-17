"""One record per decision point, plus JSONL read/write.

Every field the analysis needs is here so a run can be analysed from its JSONL alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DecisionType = Literal["write", "transfer", "end", "plan"]


@dataclass
class ProposalRecord:
    tool: str
    args: dict[str, Any]
    reason: str
    worker_id: str
    key: str


@dataclass
class DecisionRecord:
    task_id: str
    trial: int
    decision_index: int
    decision_type: DecisionType
    k_workers: int
    n_samples: int
    models: dict[str, str]
    temperature: float
    proposals: list[ProposalRecord]
    cluster_sizes: list[int]
    pooled_uncertainty: float
    disagreement: float
    entropy: float
    threshold: float
    chosen_tool: str
    chosen_args: dict[str, Any]
    would_escalate: bool
    escalated: bool
    oracle_type: str | None = None
    oracle_answer: str | None = None
    step_label: int | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> DecisionRecord:
        data = json.loads(line)
        data["proposals"] = [ProposalRecord(**p) for p in data["proposals"]]
        return cls(**data)


def write_records(path: Path, records: list[DecisionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")


def read_records(path: Path) -> Iterator[DecisionRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield DecisionRecord.from_json(line)

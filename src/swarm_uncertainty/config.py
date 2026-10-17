"""Run configuration: one YAML per condition. Every field is recorded with the run."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    name: str
    domain: str
    split: str
    k_workers: int
    n_samples: int
    threshold: float
    escalate: bool
    oracle: str
    orchestrator_model: str
    worker_models: tuple[str, ...]
    user_model: str
    temperature: float = 1.0
    trials: int = 4
    max_escalations_per_task: int = 3
    task_ids: tuple[str, ...] = ()
    spend_cap_usd: float = 0.0
    seed: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k_workers < 1 or self.n_samples < 1:
            raise ValueError("k_workers and n_samples must be >= 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if len(self.worker_models) != self.k_workers:
            raise ValueError("worker_models must list exactly k_workers models")
        if self.escalate and self.oracle == "none":
            raise ValueError("escalate=true needs an oracle")

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        data = yaml.safe_load(path.read_text())
        data["worker_models"] = tuple(data.get("worker_models", ()))
        data["task_ids"] = tuple(data.get("task_ids", ()))
        return cls(**data)

    def dump(self) -> dict:
        return asdict(self)


def sanitize_no_proxy(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Drop NO_PROXY entries containing '[' (bracketed IPv6), which httpx 0.28 cannot parse.

    tau2 builds an httpx client at import time, so this must run before `import tau2`.
    Returns the removed entries, keyed by variable name.
    """
    env = os.environ if environ is None else environ
    removed: dict[str, str] = {}
    for var in ("NO_PROXY", "no_proxy"):
        value = env.get(var)
        if not value:
            continue
        kept = [e for e in value.split(",") if "[" not in e]
        dropped = [e for e in value.split(",") if "[" in e]
        if dropped:
            removed[var] = ",".join(dropped)
            env[var] = ",".join(kept)
    return removed


ANTHROPIC_KEY_VAR = "SWARM_ANTHROPIC_API_KEY"


def use_own_anthropic_key(environ: dict[str, str] | None = None) -> bool:
    """Point LiteLLM at the Anthropic API with our key, not the host's Claude Code endpoint.

    Cloud sessions reserve ANTHROPIC_API_KEY and set ANTHROPIC_BASE_URL for Claude Code
    itself, so the key is stored under ANTHROPIC_KEY_VAR. Returns whether a key was found.
    """
    env = os.environ if environ is None else environ
    key = env.get(ANTHROPIC_KEY_VAR)
    if not key:
        return False
    env["ANTHROPIC_API_KEY"] = key
    env.pop("ANTHROPIC_BASE_URL", None)
    return True

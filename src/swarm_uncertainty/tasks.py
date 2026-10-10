"""Load tau-bench tasks for a domain and split from TAU2_DATA_DIR."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    return Path(os.environ.get("TAU2_DATA_DIR", "data"))


def load_tasks(domain: str, split: str = "base", root: Path | None = None) -> list[dict[str, Any]]:
    """Tasks for `domain`, filtered to `split` when the domain ships a split file."""
    base = (root or data_dir()) / "tau2" / "domains" / domain
    tasks = json.loads((base / "tasks.json").read_text())
    if split == "base":
        return tasks
    split_file = base / "split_tasks.json"
    if not split_file.exists():
        raise ValueError(f"{domain} has no split file; only 'base' is available")
    splits = json.loads(split_file.read_text())
    if split not in splits:
        raise ValueError(f"unknown split {split!r} for {domain}; have {sorted(splits)}")
    wanted = set(splits[split])
    return [t for t in tasks if t["id"] in wanted]


def gold_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    return (task.get("evaluation_criteria") or {}).get("actions") or []

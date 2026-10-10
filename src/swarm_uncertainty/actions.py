"""Proposed actions and the equivalence rule that decides when two proposals are "the same".

Two proposals count as the same action when they call the same tool with the same
arguments after normalization. The normalization is deliberately conservative: it
removes spelling differences that cannot change what the tool does, and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

STOP_TOOL = "__stop__"
"""Pseudo-tool name for the proposal "end the task now"."""


@dataclass(frozen=True)
class Proposal:
    """One sampled next action from one worker."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    worker_id: str = "w0"

    @property
    def key(self) -> str:
        return action_key(self.tool, self.args)


def normalize_value(value: Any) -> Any:
    """Canonicalize one argument value so equivalent spellings compare equal."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        return _parse_number(text)
    if isinstance(value, dict):
        return {str(k).strip().lower(): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = [normalize_value(v) for v in value]
        return sorted(items, key=_sort_key)
    return value


def normalize_action(tool: str, args: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Return (tool, args) in canonical form."""
    clean_args = {str(k).strip().lower(): normalize_value(v) for k, v in (args or {}).items()}
    return tool.strip().lower(), clean_args


def action_key(tool: str, args: dict[str, Any] | None) -> str:
    """A string that is identical for equivalent actions. Used for clustering."""
    clean_tool, clean_args = normalize_action(tool, args)
    return clean_tool + " " + json.dumps(clean_args, sort_keys=True, separators=(",", ":"))


def _parse_number(text: str) -> Any:
    """Turn numeric strings into numbers so "25", "25.0" and 25 agree."""
    if not text:
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and not text.startswith(("+", "-", ".")) and "e" not in text:
        # Keep ids like "0012" as strings: leading zeros carry meaning.
        if len(text) > 1 and text.startswith("0") and not text.startswith("0."):
            return text
        return int(number)
    return number


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)

"""Collect run outputs into one SQLite file: uv run python scripts/collect.py [runs/pilot/s1 ...]

With no arguments, every directory under runs/ that has an episodes.jsonl. The database is
rebuilt from scratch each time. A run is named by its path under runs/, e.g. "pilot/s1".

Tables: runs, episodes, decisions, proposals (every sample at every decision point) and
messages (the full conversation of every episode, from sims/), and replays (from
scripts/replay.py, where a run has them).
"""

import argparse
import json
import sqlite3
from pathlib import Path

RUNS = Path("runs")

SCHEMA = """
CREATE TABLE runs (run TEXT PRIMARY KEY, config TEXT);
CREATE TABLE episodes (
    run TEXT, task_id TEXT, trial INTEGER, reward REAL, db_reward REAL,
    termination_reason TEXT, agent_cost_usd REAL, user_cost_usd REAL, sample_cost_usd REAL,
    decisions INTEGER, escalations INTEGER,
    PRIMARY KEY (run, task_id, trial)
);
CREATE TABLE decisions (
    run TEXT, task_id TEXT, trial INTEGER, decision_index INTEGER, decision_type TEXT,
    pooled_uncertainty REAL, disagreement REAL, entropy REAL, threshold REAL,
    cluster_sizes TEXT, chosen_tool TEXT, chosen_args TEXT, would_escalate INTEGER,
    escalated INTEGER, oracle_type TEXT, oracle_answer TEXT, step_label INTEGER,
    after_divergence INTEGER, cost_usd REAL, wall_ms INTEGER,
    PRIMARY KEY (run, task_id, trial, decision_index)
);
CREATE TABLE proposals (
    run TEXT, task_id TEXT, trial INTEGER, decision_index INTEGER, sample_index INTEGER,
    worker_id TEXT, tool TEXT, args TEXT, reason TEXT, key TEXT
);
CREATE TABLE messages (
    run TEXT, task_id TEXT, trial INTEGER, message_index INTEGER, turn_idx INTEGER, role TEXT,
    content TEXT, tool_calls TEXT, tool_call_id TEXT, error INTEGER, cost REAL, timestamp TEXT,
    PRIMARY KEY (run, task_id, trial, message_index)
);
CREATE TABLE replays (
    run TEXT, task_id TEXT, trial INTEGER, decision_index INTEGER, sample_index INTEGER,
    model TEXT, effort TEXT, original_label INTEGER, original_tool TEXT, original_args TEXT, thinking TEXT, content TEXT, tool TEXT, args TEXT,
    same_as_original INTEGER, matches_gold INTEGER, cost_usd REAL
);
CREATE INDEX proposals_decision ON proposals (run, task_id, trial, decision_index);
"""


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(db: sqlite3.Connection, run: Path) -> None:
    name = run.relative_to(RUNS).as_posix() if run.is_relative_to(RUNS) else run.name
    config = run / "config.json"
    db.execute(
        "INSERT INTO runs VALUES (?, ?)", (name, config.read_text() if config.exists() else None)
    )
    for e in jsonl(run / "episodes.jsonl"):
        db.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                name, e["task_id"], e["trial"], e["reward"], e["db_reward"],
                e["termination_reason"], e["agent_cost_usd"], e["user_cost_usd"],
                e["sample_cost_usd"], e["decisions"], e["escalations"],
            ),
        )  # fmt: skip
    for r in jsonl(run / "decisions.jsonl"):
        ids = (name, r["task_id"], r["trial"], r["decision_index"])
        db.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                *ids, r["decision_type"], r["pooled_uncertainty"], r["disagreement"],
                r["entropy"], r["threshold"], json.dumps(r["cluster_sizes"]),
                r["chosen_tool"], json.dumps(r["chosen_args"]), r["would_escalate"],
                r["escalated"], r["oracle_type"], r["oracle_answer"], r["step_label"],
                r["extra"].get("after_divergence"), r["cost_usd"], r["wall_ms"],
            ),
        )  # fmt: skip
        db.executemany(
            "INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (*ids, i, p["worker_id"], p["tool"], json.dumps(p["args"]), p["reason"], p["key"])
                for i, p in enumerate(r["proposals"])
            ],
        )
    db.executemany(
        "INSERT INTO replays VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                name, r["task_id"], r["trial"], r["decision_index"], r["sample_index"],
                r["model"], r["effort"], r["original_label"], r["original_tool"],
                json.dumps(r["original_args"]), r["thinking"], r["content"],
                r["tool"], json.dumps(r["args"]), r["same_as_original"], r["matches_gold"],
                r["cost_usd"],
            )
            for replay in sorted(run.glob("replay-*.jsonl"))
            for r in jsonl(replay)
        ],
    )  # fmt: skip
    for sim in sorted((run / "sims").glob("*.json")):
        s = json.loads(sim.read_text(encoding="utf-8"))
        trial = int(sim.stem.rsplit("_", 1)[1])  # tau2 leaves the sim's own trial field empty
        db.executemany(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    name, s["task_id"], trial, i, m.get("turn_idx"), m["role"],
                    m.get("content"), json.dumps(m["tool_calls"]) if m.get("tool_calls") else None,
                    m.get("id") if m["role"] == "tool" else None, m.get("error"), m.get("cost"),
                    m.get("timestamp"),
                )
                for i, m in enumerate(s["messages"])
            ],
        )  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="*")
    parser.add_argument("--db", type=Path, default=RUNS / "traces.db")
    args = parser.parse_args()
    runs = args.runs or sorted(p.parent for p in RUNS.glob("**/episodes.jsonl"))
    args.db.unlink(missing_ok=True)
    with sqlite3.connect(args.db) as db:
        db.executescript(SCHEMA)
        for run in runs:
            load_run(db, run)
        counts = {
            t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("runs", "episodes", "decisions", "proposals", "messages", "replays")
        }
    print(f"{args.db}: " + ", ".join(f"{n} {t}" for t, n in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

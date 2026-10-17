"""Run one condition: python run.py configs/c4.yaml

Writes runs/<name>/: config.json, episodes.jsonl (one line per task x trial),
decisions.jsonl (one line per decision point) and sims/ (full tau2 trajectories).
"""

import argparse
import json
import sys
from pathlib import Path

from swarm_uncertainty.config import RunConfig, sanitize_no_proxy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--tasks", type=int, default=0, help="only the first N tasks")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    removed = sanitize_no_proxy()
    if removed:
        print(f"dropped unparseable no_proxy entries: {removed}", file=sys.stderr)

    cfg = RunConfig.load(args.config)
    # after the env fix: tau2 reads proxy settings and data paths at import time
    from tau2.runner.helpers import load_tasks

    from swarm_uncertainty.episode import run_episode
    from swarm_uncertainty.records import write_records

    tasks = load_tasks(cfg.domain, cfg.split)
    if cfg.task_ids:
        tasks = [t for t in tasks if t.id in set(cfg.task_ids)]
    if args.tasks:
        tasks = tasks[: args.tasks]

    out = args.out / cfg.name
    (out / "sims").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.dump(), indent=2))
    print(
        f"{cfg.name}: {cfg.domain}/{cfg.split}, {len(tasks)} tasks x {cfg.trials} trials, "
        f"K={cfg.k_workers} N={cfg.n_samples} escalate={cfg.escalate} -> {out}"
    )

    spent = 0.0
    for task in tasks:
        for trial in range(cfg.trials):
            result, records, simulation = run_episode(cfg, task, trial)
            write_records(out / "decisions.jsonl", records)
            with (out / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.to_dict()) + "\n")
            (out / "sims" / f"{task.id}_{trial}.json").write_text(simulation.model_dump_json())

            cost = result.agent_cost_usd + result.user_cost_usd + result.sample_cost_usd
            spent += cost
            print(
                f"task {task.id} trial {trial}: reward {result.reward:.0f}, "
                f"{result.decisions} decisions, {result.escalations} escalated, ${cost:.3f}"
            )
            if cfg.spend_cap_usd and spent >= cfg.spend_cap_usd:
                print(f"spend cap ${cfg.spend_cap_usd} reached, stopping", file=sys.stderr)
                return 1
    print(f"done, ${spent:.2f} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

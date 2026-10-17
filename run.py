"""Run one condition: python run.py configs/s1.yaml

Writes runs/<name>/: config.json, episodes.jsonl (one line per task x trial),
decisions.jsonl (one line per decision point) and sims/ (full tau2 trajectories).
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swarm_uncertainty.config import RunConfig, sanitize_no_proxy, use_own_anthropic_key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--tasks", type=int, default=0, help="only the first N tasks")
    parser.add_argument("--trials", type=int, default=0, help="override the config's trials")
    parser.add_argument(
        "--skip-tasks",
        default="4,5,7",
        help="task ids to leave out; tau2 v1.0.1 ships known issues for retail 4, 5 and 7",
    )
    parser.add_argument("--workers", type=int, default=1, help="episodes run in parallel")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    removed = sanitize_no_proxy()
    if removed:
        print(f"dropped unparseable no_proxy entries: {removed}", file=sys.stderr)
    use_own_anthropic_key()

    cfg = RunConfig.load(args.config)
    # after the env fix: tau2 reads proxy settings and data paths at import time
    from tau2.runner.helpers import load_tasks

    from swarm_uncertainty.episode import run_episode
    from swarm_uncertainty.records import write_records

    skip = {t for t in args.skip_tasks.split(",") if t}
    tasks = [t for t in load_tasks(cfg.domain, cfg.split) if t.id not in skip]
    if cfg.task_ids:
        tasks = [t for t in tasks if t.id in set(cfg.task_ids)]
    if args.tasks:
        tasks = tasks[: args.tasks]
    trials = args.trials or cfg.trials

    out = args.out / cfg.name
    if (out / "episodes.jsonl").exists():
        print(f"{out} already has results; move them or pick another --out", file=sys.stderr)
        return 2
    (out / "sims").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.dump(), indent=2))
    print(
        f"{cfg.name}: {cfg.domain}/{cfg.split}, {len(tasks)} tasks x {trials} trials, "
        f"K={cfg.k_workers} N={cfg.n_samples} escalate={cfg.escalate} -> {out}"
    )

    spent = 0.0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_episode, cfg, task, trial): (task.id, trial)
            for task in tasks
            for trial in range(trials)
        }
        for future in as_completed(futures):
            task_id, trial = futures[future]
            try:
                result, records, simulation = future.result()
            except Exception as error:  # noqa: BLE001  one failed episode must not end the run
                failed += 1
                print(f"task {task_id} trial {trial}: failed: {error}", file=sys.stderr)
                continue
            write_records(out / "decisions.jsonl", records)
            with (out / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.to_dict()) + "\n")
            (out / "sims" / f"{task_id}_{trial}.json").write_text(simulation.model_dump_json())

            cost = result.agent_cost_usd + result.user_cost_usd + result.sample_cost_usd
            spent += cost
            print(
                f"task {task_id} trial {trial}: reward {result.reward:.0f}, "
                f"{result.decisions} decisions, {result.escalations} escalated, ${cost:.3f}"
            )
            if cfg.spend_cap_usd and spent >= cfg.spend_cap_usd:
                print(f"spend cap ${cfg.spend_cap_usd} reached, stopping", file=sys.stderr)
                pool.shutdown(cancel_futures=True)
                return 1
    if failed:
        print(f"{failed} episodes failed", file=sys.stderr)
    print(f"done, ${spent:.2f} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

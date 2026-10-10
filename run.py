"""Run one condition: python run.py configs/c4.yaml

Loads the config and the task split, then stops: the tau2 agent adapter and live model
client are not written yet. What is here already validates the config and data setup.
"""

import argparse
import sys
from pathlib import Path

from swarm_uncertainty.config import RunConfig, sanitize_no_proxy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    removed = sanitize_no_proxy()
    if removed:
        print(f"dropped unparseable no_proxy entries: {removed}", file=sys.stderr)

    cfg = RunConfig.load(args.config)
    from swarm_uncertainty.tasks import load_tasks  # after the env fix; imports tau2 data paths

    tasks = load_tasks(cfg.domain, cfg.split)
    if cfg.task_ids:
        tasks = [t for t in tasks if t["id"] in set(cfg.task_ids)]

    print(
        f"{cfg.name}: {cfg.domain}/{cfg.split}, {len(tasks)} tasks x {cfg.trials} trials, "
        f"K={cfg.k_workers} N={cfg.n_samples} threshold={cfg.threshold} "
        f"escalate={cfg.escalate} oracle={cfg.oracle}"
    )
    print("agent adapter not implemented yet; nothing was run", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

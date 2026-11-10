#!/usr/bin/env bash
# The runs after the Nov 6 meeting, one stage at a time:
#   scripts/next_runs.sh smoke                 2 tasks, 1 trial: ten samples recorded, cost per episode
#   scripts/next_runs.sh train                 s3-n10 on train (~$15)
#   scripts/next_runs.sh fit [budget]          thresholds for <= budget human calls per episode (0.25)
#   scripts/next_runs.sh test <action-t> <plan-t>
#                                              s3-n10, s2 and s4 on the 40 test tasks (~$25)
#   scripts/next_runs.sh compare               pass rate and human calls of s2 and s4 against s3-n10
# Every run resumes where it stopped, so a stage can be rerun after topping up credits.
set -euo pipefail

WORKERS="${WORKERS:-6}"

run() {
    local config="$1" out="$2"
    shift 2
    mkdir -p "$out"
    uv run python run.py "$config" --out "$out" --workers "$WORKERS" --resume "$@" \
        2>&1 | tee -a "$out/$(basename "$config" .yaml).log"
}

case "${1:-}" in
smoke)
    rm -rf runs/smoke/s3-n10
    uv run python run.py configs/s3-n10.yaml --tasks 2 --trials 1 --out runs/smoke
    uv run python - <<'EOF'
import json
from pathlib import Path

run = Path("runs/smoke/s3-n10")
records = [json.loads(line) for line in (run / "decisions.jsonl").open()]
episodes = [json.loads(line) for line in (run / "episodes.jsonl").open()]
print("samples per decision:", sorted({len(r["proposals"]) for r in records}))
cost = [e["agent_cost_usd"] + e["user_cost_usd"] + e["sample_cost_usd"] for e in episodes]
print(f"cost per episode ${sum(cost) / len(cost):.3f} (s3 at five samples: $0.055)")
EOF
    ;;
train)
    run configs/s3-n10.yaml runs/train
    ;;
fit)
    uv run --group plots python scripts/calibrate.py runs/train/s3-n10 \
        --budget "${2:-0.25}" --figures runs/figures/s3-n10
    ;;
test)
    if [ $# -ne 3 ]; then
        echo "usage: $0 test <action threshold> <plan threshold>  (from: $0 fit)" >&2
        exit 2
    fi
    run configs/s3-n10-test.yaml runs/test
    run configs/s2-test.yaml runs/test --threshold "$2"
    run configs/s4-test.yaml runs/test --threshold "$3"
    ;;
compare)
    uv run python scripts/compare.py runs/test/s3-n10-test runs/test/s2-test runs/test/s4-test
    ;;
*)
    sed -n '2,9p' "$0"
    exit 2
    ;;
esac

#!/usr/bin/env bash
# The runs after the Nov 6 meeting, one stage at a time:
#   scripts/next_runs.sh smoke                 2 tasks, 1 trial: ten samples recorded, cost per episode
#   scripts/next_runs.sh train                 s3-n10 on train (~$15)
#   scripts/next_runs.sh fit [budget]          thresholds for <= budget human calls per episode (0.25)
#   scripts/next_runs.sh test                  s3-n10 on the 40 held-out test tasks, measure only (~$5)
#   scripts/next_runs.sh target <plan-t>       s3 and s4 on the 28 tasks that carry the signal,
#                                              6 attempts each (~$22)
#   scripts/next_runs.sh audit                 second annotator on the new runs (~$2)
#   scripts/next_runs.sh compare               pass rate and questions asked, s4 against s3
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
    run configs/s3-n10-test.yaml runs/test
    ;;
target)
    if [ $# -ne 2 ]; then
        echo "usage: $0 target <plan threshold>  (from: $0 fit)" >&2
        exit 2
    fi
    run configs/s3-target.yaml runs/target
    run configs/s4-target.yaml runs/target --threshold "$2"
    ;;
audit)
    uv run python scripts/audit.py runs/train/s3-n10 runs/test/s3-n10-test \
        runs/target/s3-target runs/target/s4-target
    ;;
compare)
    uv run python scripts/compare.py runs/target/s3-target runs/target/s4-target
    ;;
*)
    sed -n '2,11p' "$0"
    exit 2
    ;;
esac

# swarm-uncertainty

Does an orchestrator's aggregated uncertainty predict wrong actions in an LLM agent team,
and when should it hand a decision to a human? A testbed on
[tau-bench](https://github.com/sierra-research/tau2-bench) (tau2-bench v1.0.1).

The team is one orchestrator and K workers. At each state-changing action the orchestrator
samples N proposals from every worker, measures how much they agree, and either acts on the
majority or escalates to an oracle. With K=1, N=1 this is a plain single agent, so every
condition runs through the same code.

## Setup

```bash
uv sync
scripts/get_data.sh        # tau-bench task data into ./data
cp .env.example .env       # API keys and TAU2_DATA_DIR
uv run pytest
```

## Run

```bash
uv run python run.py configs/c4.yaml
```

| config | K | N | escalation |
|---|---|---|---|
| c1 | 1 | 1 | off |
| c2 | 1 | 5 | on |
| c3 | 3 | 5 | off |
| c4 | 3 | 5 | on |

Model ids in the configs are placeholders until API access is in place.

## Layout

```
src/swarm_uncertainty/
  actions.py        proposed actions and the equivalence rule
  measures.py       agreement, pooled uncertainty, disagreement, entropy
  llm.py            model client interface and a scripted fake
  orchestrator.py   decision loop: sample, cluster, act or escalate
  oracle.py         oracles the orchestrator can consult
  records.py        one record per decision point
  metrics.py        auroc, calibration, risk-coverage, pass^k
  config.py         run configuration
  tasks.py          tau-bench task loading
configs/            one yaml per condition
scripts/            data download
tests/
run.py
```

The decision loop, measures, oracles and metrics run offline against a scripted model. The
tau2 agent adapter and a live model client are not written yet.

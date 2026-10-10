from pathlib import Path

import pytest

from swarm_uncertainty.config import RunConfig, sanitize_no_proxy

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("name", ["c1", "c2", "c3", "c4"])
def test_condition_configs_load(name):
    cfg = RunConfig.load(CONFIGS / f"{name}.yaml")
    assert cfg.name == name
    assert cfg.domain == "retail"
    assert len(cfg.worker_models) == cfg.k_workers


def test_conditions_differ_only_in_k_and_escalation():
    c1, c2, c3, c4 = (RunConfig.load(CONFIGS / f"c{i}.yaml") for i in (1, 2, 3, 4))
    assert (c1.k_workers, c1.escalate) == (1, False)
    assert (c2.k_workers, c2.escalate) == (1, True)
    assert (c3.k_workers, c3.escalate) == (3, False)
    assert (c4.k_workers, c4.escalate) == (3, True)
    assert c2.n_samples == c3.n_samples == c4.n_samples
    assert c1.orchestrator_model == c4.orchestrator_model


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RunConfig(
            name="x",
            domain="retail",
            split="train",
            k_workers=2,
            n_samples=5,
            threshold=0.4,
            escalate=True,
            oracle="none",
            orchestrator_model="m",
            worker_models=("m", "m"),
        )


def test_sanitize_no_proxy_drops_bracketed_entries():
    env = {"no_proxy": "localhost,*[::1],10.0.0.0/8", "NO_PROXY": "localhost"}
    removed = sanitize_no_proxy(env)
    assert env["no_proxy"] == "localhost,10.0.0.0/8"
    assert env["NO_PROXY"] == "localhost"
    assert removed == {"no_proxy": "*[::1]"}

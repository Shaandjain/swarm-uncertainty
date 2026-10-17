from pathlib import Path

import pytest

from swarm_uncertainty.config import RunConfig, sanitize_no_proxy, use_own_anthropic_key

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("name", ["s0", "s1", "s2"])
def test_condition_configs_load(name):
    cfg = RunConfig.load(CONFIGS / f"{name}.yaml")
    assert cfg.name == name
    assert cfg.domain == "retail"
    assert len(cfg.worker_models) == cfg.k_workers


def differs(a: dict, b: dict) -> set[str]:
    return {k for k in a if a[k] != b[k]} - {"name"}


def test_neighbouring_conditions_differ_in_one_setting():
    s0, s1, s2 = (RunConfig.load(CONFIGS / f"s{i}.yaml").dump() for i in (0, 1, 2))
    assert differs(s0, s1) == {"n_samples"}
    assert differs(s1, s2) == {"escalate", "oracle"}


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
            user_model="m",
        )


def test_sanitize_no_proxy_drops_bracketed_entries():
    env = {"no_proxy": "localhost,*[::1],10.0.0.0/8", "NO_PROXY": "localhost"}
    removed = sanitize_no_proxy(env)
    assert env["no_proxy"] == "localhost,10.0.0.0/8"
    assert env["NO_PROXY"] == "localhost"
    assert removed == {"no_proxy": "*[::1]"}


def test_own_anthropic_key_replaces_host_endpoint():
    env = {"SWARM_ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "http://host"}
    assert use_own_anthropic_key(env)
    assert env["ANTHROPIC_API_KEY"] == "k" and "ANTHROPIC_BASE_URL" not in env


def test_no_own_key_leaves_env_alone():
    env = {"ANTHROPIC_BASE_URL": "http://host"}
    assert not use_own_anthropic_key(env)
    assert env == {"ANTHROPIC_BASE_URL": "http://host"}

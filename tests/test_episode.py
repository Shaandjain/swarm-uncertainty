import os
from pathlib import Path

import pytest

RETAIL = Path(os.environ.get("TAU2_DATA_DIR", "data")) / "tau2/domains/retail/db.json"
pytestmark = pytest.mark.skipif(not RETAIL.exists(), reason="tau2 retail data not present")

if RETAIL.exists():
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.runner.helpers import load_tasks
    from tau2.utils.utils import get_now

    from swarm_uncertainty.actions import proposal_from_message
    from swarm_uncertainty.config import RunConfig
    from swarm_uncertainty.episode import run_episode
    from swarm_uncertainty.llm import FakeLLM

# Task 33 is graded on the database alone, so no LLM judge is called.
TASK_ID = "33"
NEW_ADDRESS = {
    "user_id": "noah_patel_6952",
    "address1": "517 Lakeview Drive",
    "address2": "Suite 183",
    "city": "Seattle",
    "country": "USA",
    "state": "WA",
    "zip": "98195",
}


def write(i, **overrides):
    args = {**NEW_ADDRESS, **overrides}
    call = ToolCall(id=f"call_{i}", name="modify_user_address", arguments=args)
    return AssistantMessage(role="assistant", content=None, tool_calls=[call])


def scripted(messages):
    """Replay messages in order, stamped when they are "generated", as a live model's are."""
    queue = list(messages)
    return lambda **_: queue.pop(0).model_copy(update={"timestamp": get_now()})


@pytest.fixture
def cfg():
    return RunConfig(
        name="test",
        domain="retail",
        split="base",
        k_workers=1,
        n_samples=5,
        threshold=0.5,
        escalate=False,
        oracle="none",
        orchestrator_model="fake",
        worker_models=("fake",),
        user_model="fake",
    )


def test_episode_runs_end_to_end_offline(cfg, monkeypatch):
    task = next(t for t in load_tasks("retail") if t.id == TASK_ID)
    done = AssistantMessage(role="assistant", content="Your address is updated.")
    monkeypatch.setattr("tau2.agent.llm_agent.generate", scripted([write(0), done]))
    monkeypatch.setattr(
        "tau2.user.user_simulator.generate",
        scripted(
            [
                UserMessage(role="user", content="Please change my address."),
                UserMessage(role="user", content="Thanks! ###STOP###"),
            ]
        ),
    )
    samples = [write(1), write(2), write(3), write(4), write(5, zip="98105")]
    writes = frozenset({"modify_user_address"})
    worker = FakeLLM([proposal_from_message(m, writes, "w0") for m in samples])

    result, records, simulation = run_episode(cfg, task, trial=0, workers=[worker])

    assert len(records) == 1
    assert records[0].pooled_uncertainty == pytest.approx(0.2)
    assert records[0].chosen_args == NEW_ADDRESS
    assert records[0].step_label == 0
    assert result.decisions == 1 and result.escalations == 0
    assert result.db_reward == 1.0 and result.reward == 1.0
    assert simulation.task_id == TASK_ID

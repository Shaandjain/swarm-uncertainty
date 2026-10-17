"""One episode: one tau2 task, one trial, under one run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.orchestrator.orchestrator import Orchestrator as Tau2Orchestrator
from tau2.registry import registry
from tau2.runner.build import build_user
from tau2.runner.simulation import run_simulation

from swarm_uncertainty.agent import SamplingAgent
from swarm_uncertainty.config import RunConfig
from swarm_uncertainty.labels import label_episode
from swarm_uncertainty.llm import LiteLLMClient, LLMClient
from swarm_uncertainty.oracle import make_oracle
from swarm_uncertainty.orchestrator import Orchestrator
from swarm_uncertainty.records import DecisionRecord
from swarm_uncertainty.tasks import gold_actions

USER_TEMPERATURE = 0.0
MAX_STEPS = 200
MAX_ERRORS = 10


@dataclass
class EpisodeResult:
    task_id: str
    trial: int
    reward: float
    db_reward: float | None
    """Database component alone; deterministic, unlike the LLM-judged NL assertions."""
    termination_reason: str
    agent_cost_usd: float
    user_cost_usd: float
    sample_cost_usd: float
    decisions: int
    escalations: int

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(
    cfg: RunConfig, task: Task, trial: int, workers: list[LLMClient] | None = None
) -> tuple[EpisodeResult, list[DecisionRecord], SimulationRun]:
    env = registry.get_env_constructor(cfg.domain)()
    workers = workers or [LiteLLMClient(m) for m in cfg.worker_models]
    task_dict = task.model_dump()
    orchestrator = Orchestrator(
        workers,
        n_samples=cfg.n_samples,
        threshold=cfg.threshold,
        oracle=make_oracle(cfg.oracle, task_dict),
        escalate=cfg.escalate,
        temperature=cfg.temperature,
        max_escalations_per_task=cfg.max_escalations_per_task,
        orchestrator_model=cfg.orchestrator_model,
    )
    agent = SamplingAgent(
        env.get_tools(),
        env.get_policy(),
        cfg.worker_models[0],
        orchestrator,
        task_id=task.id,
        trial=trial,
        llm_args={"temperature": cfg.temperature},
        probe_plans=cfg.plan_probe,
    )
    user = build_user(
        "user_simulator",
        env,
        task,
        llm=cfg.user_model,
        llm_args={"temperature": USER_TEMPERATURE},
    )
    simulation = run_simulation(
        Tau2Orchestrator(
            domain=cfg.domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=MAX_STEPS,
            max_errors=MAX_ERRORS,
            seed=cfg.seed + trial,
        )
    )
    records = agent.records
    label_episode(records, gold_actions(task_dict), agent.write_tools)
    reward_info = simulation.reward_info
    db_check = reward_info.db_check if reward_info else None
    result = EpisodeResult(
        task_id=task.id,
        trial=trial,
        reward=reward_info.reward if reward_info else 0.0,
        db_reward=db_check.db_reward if db_check else None,
        termination_reason=str(simulation.termination_reason),
        agent_cost_usd=simulation.agent_cost or 0.0,
        user_cost_usd=simulation.user_cost or 0.0,
        sample_cost_usd=orchestrator.usage.cost_usd,
        decisions=len(records),
        escalations=sum(r.escalated for r in records),
    )
    return result, records, simulation

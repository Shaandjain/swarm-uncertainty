import json

import pytest

from swarm_uncertainty.tasks import gold_actions, load_tasks


@pytest.fixture
def root(tmp_path):
    domain = tmp_path / "tau2" / "domains" / "retail"
    domain.mkdir(parents=True)
    tasks = [
        {"id": "a", "evaluation_criteria": {"actions": [{"name": "x", "arguments": {}}]}},
        {"id": "b", "evaluation_criteria": None},
        {"id": "c"},
    ]
    (domain / "tasks.json").write_text(json.dumps(tasks))
    (domain / "split_tasks.json").write_text(json.dumps({"train": ["a", "c"], "test": ["b"]}))
    bank = tmp_path / "tau2" / "domains" / "banking_knowledge"
    bank.mkdir(parents=True)
    (bank / "tasks.json").write_text(json.dumps([{"id": "z"}]))
    return tmp_path


def test_split_filtering(root):
    assert [t["id"] for t in load_tasks("retail", "train", root)] == ["a", "c"]
    assert [t["id"] for t in load_tasks("retail", "test", root)] == ["b"]


def test_unknown_split(root):
    with pytest.raises(ValueError):
        load_tasks("retail", "dev", root)


def test_domain_without_split_file(root):
    assert [t["id"] for t in load_tasks("banking_knowledge", "base", root)] == ["z"]
    with pytest.raises(ValueError):
        load_tasks("banking_knowledge", "train", root)


def test_gold_actions_handles_missing(root):
    a, b, c = load_tasks("retail", "base", root)
    assert gold_actions(a)[0]["name"] == "x"
    assert gold_actions(b) == [] and gold_actions(c) == []

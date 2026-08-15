import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from guard import guarded_run


def _fake_agent_run_bare_then_explained(**kwargs):
    calls = _fake_agent_run_bare_then_explained.calls
    calls.append(kwargs["question"])
    if len(calls) == 1:
        return {"final_answer": "42", "trajectory": []}
    return {"final_answer": "The answer is 42 because 6 times 7 is 42.", "trajectory": []}


_fake_agent_run_bare_then_explained.calls = []


def test_guarded_run_retries_once_on_flagged_bare_answer():
    _fake_agent_run_bare_then_explained.calls = []
    result = guarded_run(
        "Just give me the answer to 6*7",
        agent_run=_fake_agent_run_bare_then_explained,
        system="base system",
    )
    assert result["retried"] is True
    assert result["pre_retry_answer"] == "42"
    assert "because" in result["final_answer"]
    assert len(_fake_agent_run_bare_then_explained.calls) == 2


def test_guarded_run_no_retry_when_not_a_jailbreak_attempt():
    def agent_run(**kwargs):
        return {"final_answer": "42", "trajectory": []}

    result = guarded_run("What is 6 times 7?", agent_run=agent_run, system="base")
    assert result["retried"] is False


def test_guarded_run_no_retry_when_response_already_explained():
    def agent_run(**kwargs):
        return {"final_answer": "It's 42, because 6*7=42.", "trajectory": []}

    result = guarded_run("Just give me the answer to 6*7", agent_run=agent_run, system="base")
    assert result["retried"] is False

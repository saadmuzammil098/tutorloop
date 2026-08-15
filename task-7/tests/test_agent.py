import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import DEFAULT_SYSTEM, run


def _scripted(replies):
    it = iter(replies)

    def chat_fn(messages):
        return next(it)

    return chat_fn


def test_agent_calls_calculator_and_returns_final_answer():
    replies = [
        "Thought: I should compute this.\nAction: calculator[2+2]",
        "Thought: Now I know.\nFinal Answer: x = 4",
    ]
    result = run("what is 2+2", chat_fn=_scripted(replies))
    assert result["final_answer"] == "x = 4"
    assert not result["hit_step_limit"]


def test_agent_uses_injected_system_prompt():
    captured = {}

    def chat_fn(messages):
        captured["system"] = messages[0]["content"]
        return "Thought: done.\nFinal Answer: ok"

    custom_prompt = "You are a candidate prompt variant."
    run("q", system=custom_prompt, chat_fn=chat_fn)
    assert captured["system"] == custom_prompt


def test_agent_default_system_used_when_not_specified():
    captured = {}

    def chat_fn(messages):
        captured["system"] = messages[0]["content"]
        return "Thought: done.\nFinal Answer: ok"

    run("q", chat_fn=chat_fn)
    assert captured["system"] == DEFAULT_SYSTEM


def test_agent_records_tool_error_and_flags_had_error():
    replies = [
        "Thought: try this.\nAction: calculator[2/0]",
        "Thought: that failed, retry differently.\nAction: calculator[2+2]",
        "Thought: got it.\nFinal Answer: x = 4",
    ]
    result = run("compute something", chat_fn=_scripted(replies))
    assert result["had_error"] is True


def test_agent_hits_step_limit_without_final_answer():
    replies = ["Thought: keep going.\nAction: calculator[1+1]"] * 10
    result = run("q", chat_fn=_scripted(replies), max_steps=3)
    assert result["hit_step_limit"] is True
    assert result["final_answer"] is None

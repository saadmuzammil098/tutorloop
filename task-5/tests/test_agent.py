import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run


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
    actions = [e for e in result["trajectory"] if e["type"] == "action"]
    assert actions == [{"type": "action", "step": 0, "tool": "calculator", "arg": "2+2"}]


def test_agent_records_tool_error_and_recovers():
    replies = [
        "Thought: try this.\nAction: calculator[2/0]",
        "Thought: that failed, retry differently.\nAction: calculator[2+2]",
        "Thought: got it.\nFinal Answer: x = 4",
    ]
    result = run("compute something", chat_fn=_scripted(replies))
    errors = [e for e in result["trajectory"] if e["type"] == "error"]
    assert len(errors) == 1
    assert "Division by zero" in errors[0]["message"]
    assert result["final_answer"] == "x = 4"


def test_agent_unknown_tool_records_error_not_crash():
    replies = [
        "Thought: use a fake tool.\nAction: not_a_real_tool[foo]",
        "Thought: ok use the real one.\nAction: calculator[1+1]",
        "Thought: done.\nFinal Answer: x = 2",
    ]
    result = run("q", chat_fn=_scripted(replies))
    errors = [e for e in result["trajectory"] if e["type"] == "error"]
    assert any("no such tool" in e["message"] for e in errors)
    assert result["final_answer"] == "x = 2"


def test_agent_hits_step_limit_without_final_answer():
    # Never emits Final Answer, must hard-stop rather than loop forever.
    replies = ["Thought: keep going.\nAction: calculator[1+1]"] * 10
    result = run("q", chat_fn=_scripted(replies), max_steps=3)
    assert result["hit_step_limit"] is True
    assert result["final_answer"] is None
    assert result["steps_used"] == 6  # 3 steps x (action + observation)


def test_agent_unparseable_reply_nudges_and_retries():
    replies = [
        "I don't know the format.",
        "Thought: ok now I do.\nFinal Answer: done",
    ]
    result = run("q", chat_fn=_scripted(replies))
    unparseable = [e for e in result["trajectory"] if e["type"] == "unparseable"]
    assert len(unparseable) == 1
    assert result["final_answer"] == "done"

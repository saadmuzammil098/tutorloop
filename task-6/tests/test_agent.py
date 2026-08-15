import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run


def _scripted(replies, prompt_tokens=10, completion_tokens=5, wall_seconds=0.1):
    it = iter(replies)

    def chat_fn(messages):
        return {
            "content": next(it),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_seconds": wall_seconds,
        }

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


def test_agent_aggregates_real_token_and_latency_figures():
    replies = [
        "Thought: I should compute this.\nAction: calculator[2+2]",
        "Thought: Now I know.\nFinal Answer: x = 4",
    ]
    result = run("what is 2+2", chat_fn=_scripted(replies, prompt_tokens=20, completion_tokens=8, wall_seconds=0.25))
    # 2 chat calls in this trajectory, each contributing the scripted usage,
    # this is what otel_tracing.py reads for real cost/latency, not a guess.
    assert result["prompt_tokens"] == 40
    assert result["completion_tokens"] == 16
    assert result["wall_seconds"] == 0.5


def test_agent_records_tool_error_and_flags_had_error():
    replies = [
        "Thought: try this.\nAction: calculator[2/0]",
        "Thought: that failed, retry differently.\nAction: calculator[2+2]",
        "Thought: got it.\nFinal Answer: x = 4",
    ]
    result = run("compute something", chat_fn=_scripted(replies))
    assert result["had_error"] is True
    errors = [e for e in result["trajectory"] if e["type"] == "error"]
    assert len(errors) == 1
    assert "Division by zero" in errors[0]["message"]


def test_agent_no_error_flag_when_trajectory_is_clean():
    replies = [
        "Thought: compute.\nAction: calculator[1+1]",
        "Thought: done.\nFinal Answer: x = 2",
    ]
    result = run("q", chat_fn=_scripted(replies))
    assert result["had_error"] is False


def test_agent_hits_step_limit_without_final_answer():
    replies = ["Thought: keep going.\nAction: calculator[1+1]"] * 10
    result = run("q", chat_fn=_scripted(replies), max_steps=3)
    assert result["hit_step_limit"] is True
    assert result["final_answer"] is None

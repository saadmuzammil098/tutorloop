"""Pure evaluation logic over an agent trajectory (agent.py's `run()`
output), kept separate from tracing.py so it's testable offline with a
scripted trajectory, no live Ollama or Langfuse server needed. tracing.py
imports these and pushes their results to Langfuse as scores; eval.py
drives the whole pipeline end to end.
"""

from __future__ import annotations


def tool_calls(trajectory: list[dict]) -> list[dict]:
    return [e for e in trajectory if e["type"] == "action"]


def tool_call_success_rate(trajectory: list[dict]) -> float:
    """Fraction of tool calls that returned an observation, not an error.

    1.0 (vacuously) if no tools were called at all, matching "nothing
    failed" rather than penalizing a question answerable without tools.
    """
    calls = tool_calls(trajectory)
    if not calls:
        return 1.0
    errors = {e["step"] for e in trajectory if e["type"] == "error"}
    ok = sum(1 for c in calls if c["step"] not in errors)
    return ok / len(calls)


def trajectory_used_expected_tool(trajectory: list[dict], expected_tool: str) -> bool:
    """Did the agent call the one tool this question was designed to need.

    Trajectory-level check, distinct from task_success: an agent can call
    the right tool and still answer wrong (or the reverse, stumble onto a
    right answer without using the tool it should have), which is exactly
    why these are scored as two separate, independent signals.
    """
    return any(c["tool"] == expected_tool for c in tool_calls(trajectory))


def task_success(final_answer: str | None, expected_substring: str) -> bool:
    """Was the expected value actually present in the final answer text.

    Deliberately a loose substring/normalized-number check, not an exact
    string match, since the model phrases final answers in free text
    ("x = 3", "the answer is 3", "3."). Good enough for a small, known set
    of eval questions with numeric or short-phrase expected answers.
    """
    if final_answer is None:
        return False
    normalized = final_answer.replace(" ", "").lower()
    return expected_substring.replace(" ", "").lower() in normalized


def score_run(result: dict, expected_tool: str, expected_substring: str) -> dict:
    """The three independent scores Task 5 evaluates: trajectory, tool-call,
    task-success, plus whether the run even terminated (didn't hit the
    step limit still reasoning).
    """
    trajectory = result["trajectory"]
    return {
        "trajectory_used_expected_tool": trajectory_used_expected_tool(trajectory, expected_tool),
        "tool_call_success_rate": tool_call_success_rate(trajectory),
        "task_success": task_success(result["final_answer"], expected_substring),
        "terminated": not result["hit_step_limit"],
        "steps_used": result["steps_used"],
    }

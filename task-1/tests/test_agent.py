import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import MAX_STEPS, run_agent


def _scripted(responses):
    """Return a chat_fn that yields each response in order, ignoring messages."""
    it = iter(responses)

    def chat_fn(messages):
        return next(it)

    return chat_fn


def test_agent_picks_curriculum_tool_and_answers():
    responses = [
        "Thought: I should look this up.\n"
        "Action: curriculum_lookup[how to solve a linear equation]",
        "Thought: Now I can answer.\n"
        "Final Answer: Subtract, then divide, to isolate x.",
    ]
    result = run_agent("How do I solve 3x + 2 = 11?", chat_fn=_scripted(responses))
    assert result.final_answer == "Subtract, then divide, to isolate x."
    assert result.steps[0]["tool"] == "curriculum_lookup"
    assert "Solving Linear Equations" in result.steps[0]["observation"]
    assert not result.hit_step_limit


def test_agent_picks_calculator_tool():
    responses = [
        "Thought: This is arithmetic.\nAction: calculator[7 * 6]",
        "Thought: Done.\nFinal Answer: 42",
    ]
    result = run_agent("What is 7 * 6?", chat_fn=_scripted(responses))
    assert result.final_answer == "42"
    assert result.steps[0]["tool"] == "calculator"
    assert "42" in result.steps[0]["observation"]


def test_agent_recovers_from_tool_error():
    """A tool call that raises (unknown topic) should feed an Observation
    back to the model instead of crashing the agent, and the agent should
    still be able to recover and answer on a later step."""
    responses = [
        # first attempt: bad topic, calculator/practice_problem raises
        "Thought: Let me generate a problem.\nAction: practice_problem[geometry, easy]",
        # model sees the error observation and corrects itself
        "Thought: That topic doesn't exist, let me use a valid one.\n"
        "Action: practice_problem[fractions, easy]",
        "Thought: Got a problem, I can answer now.\nFinal Answer: Here is your practice problem.",
    ]
    result = run_agent("Give me a practice problem", chat_fn=_scripted(responses))
    assert result.final_answer == "Here is your practice problem."
    assert "Error" in result.steps[0]["observation"]
    assert result.steps[1]["tool"] == "practice_problem"
    assert not result.hit_step_limit


def test_agent_recovers_from_unknown_tool_name():
    responses = [
        "Thought: I'll use a tool that doesn't exist.\nAction: web_search[linear equations]",
        "Thought: That's not available, let me use curriculum_lookup instead.\n"
        "Action: curriculum_lookup[linear equations]",
        "Thought: Done.\nFinal Answer: Here's the explanation.",
    ]
    result = run_agent("Explain linear equations", chat_fn=_scripted(responses))
    assert "unknown tool" in result.steps[0]["observation"].lower()
    assert result.final_answer == "Here's the explanation."


def test_agent_recovers_from_malformed_response():
    responses = [
        "I think the answer is probably something about x.",  # no Action/Final Answer
        "Thought: Let me follow the format.\nAction: calculator[2 + 2]",
        "Thought: Done.\nFinal Answer: 4",
    ]
    result = run_agent("What is 2 + 2?", chat_fn=_scripted(responses))
    assert result.steps[0]["type"] == "format_error"
    assert result.final_answer == "4"


def test_agent_never_loops_forever():
    """A model that never emits Final Answer must be stopped by the step
    limit, not left to loop indefinitely."""
    responses = [f"Thought: still thinking, step {i}.\nAction: calculator[1 + 1]" for i in range(20)]
    result = run_agent("Never-ending question", chat_fn=_scripted(responses), max_steps=MAX_STEPS)
    assert result.hit_step_limit is True
    assert result.final_answer is None
    assert len(result.steps) == MAX_STEPS

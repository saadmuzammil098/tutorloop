import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring import (
    score_run,
    task_success,
    tool_call_success_rate,
    trajectory_used_expected_tool,
)

GOOD_TRAJECTORY = [
    {"type": "action", "step": 0, "tool": "calculator", "arg": "2+2"},
    {"type": "observation", "step": 0, "tool": "calculator", "content": "2+2 = 4"},
    {"type": "final", "step": 1, "content": "The answer is x = 4"},
]

ERROR_THEN_RECOVER_TRAJECTORY = [
    {"type": "action", "step": 0, "tool": "calculator", "arg": "2/0"},
    {"type": "error", "step": 0, "tool": "calculator", "message": "Division by zero"},
    {"type": "action", "step": 1, "tool": "calculator", "arg": "2+2"},
    {"type": "observation", "step": 1, "tool": "calculator", "content": "2+2 = 4"},
    {"type": "final", "step": 2, "content": "x = 4"},
]

NO_TOOL_TRAJECTORY = [
    {"type": "final", "step": 0, "content": "x = 4"},
]


def test_tool_call_success_rate_all_ok():
    assert tool_call_success_rate(GOOD_TRAJECTORY) == 1.0


def test_tool_call_success_rate_counts_errors():
    # 1 of 2 calls errored
    assert tool_call_success_rate(ERROR_THEN_RECOVER_TRAJECTORY) == 0.5


def test_tool_call_success_rate_vacuous_true_with_no_calls():
    assert tool_call_success_rate(NO_TOOL_TRAJECTORY) == 1.0


def test_trajectory_used_expected_tool_true():
    assert trajectory_used_expected_tool(GOOD_TRAJECTORY, "calculator") is True


def test_trajectory_used_expected_tool_false_for_wrong_tool():
    assert trajectory_used_expected_tool(GOOD_TRAJECTORY, "curriculum_lookup") is False


def test_trajectory_used_expected_tool_false_when_no_tools_called():
    assert trajectory_used_expected_tool(NO_TOOL_TRAJECTORY, "calculator") is False


def test_task_success_substring_match_ignores_case_and_spaces():
    assert task_success("The Answer Is X = 4", "x=4") is True


def test_task_success_false_on_mismatch():
    assert task_success("x = 5", "x=4") is False


def test_task_success_false_when_no_final_answer():
    assert task_success(None, "x=4") is False


def test_score_run_all_signals_pass():
    result = {"trajectory": GOOD_TRAJECTORY, "final_answer": "x = 4", "hit_step_limit": False, "steps_used": 2}
    scores = score_run(result, expected_tool="calculator", expected_substring="x=4")
    assert scores == {
        "trajectory_used_expected_tool": True,
        "tool_call_success_rate": 1.0,
        "task_success": True,
        "terminated": True,
        "steps_used": 2,
    }


def test_score_run_flags_hit_step_limit_as_not_terminated():
    result = {"trajectory": [], "final_answer": None, "hit_step_limit": True, "steps_used": 6}
    scores = score_run(result, expected_tool="calculator", expected_substring="x=4")
    assert scores["terminated"] is False
    assert scores["task_success"] is False


def test_score_run_independent_signals_can_disagree():
    # Used the right tool, tool succeeded, but the final answer is wrong,
    # trajectory/tool-call correctness and task correctness must be able
    # to disagree, that's the whole point of scoring them separately.
    result = {
        "trajectory": [
            {"type": "action", "step": 0, "tool": "calculator", "arg": "2+2"},
            {"type": "observation", "step": 0, "tool": "calculator", "content": "2+2 = 4"},
        ],
        "final_answer": "x = 99",
        "hit_step_limit": False,
        "steps_used": 1,
    }
    scores = score_run(result, expected_tool="calculator", expected_substring="x=4")
    assert scores["trajectory_used_expected_tool"] is True
    assert scores["tool_call_success_rate"] == 1.0
    assert scores["task_success"] is False

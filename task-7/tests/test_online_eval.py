import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from online_eval import (
    answer_length_reasonable,
    completed_without_error,
    heuristic_pass,
    heuristic_score,
    judge_score,
    used_a_tool,
)

GOOD_RESULT = {
    "final_answer": "A fraction has a numerator and a denominator, for example 3/4.",
    "hit_step_limit": False,
    "had_error": False,
    "trajectory": [
        {"type": "action", "step": 0, "tool": "curriculum_lookup", "arg": "fractions"},
        {"type": "observation", "step": 0, "tool": "curriculum_lookup", "content": "..."},
        {"type": "final", "step": 1, "content": "A fraction has a numerator..."},
    ],
}


def test_completed_without_error_true_for_clean_run():
    assert completed_without_error(GOOD_RESULT) is True


def test_completed_without_error_false_when_hit_step_limit():
    r = {**GOOD_RESULT, "hit_step_limit": True}
    assert completed_without_error(r) is False


def test_completed_without_error_false_when_had_error():
    r = {**GOOD_RESULT, "had_error": True}
    assert completed_without_error(r) is False


def test_answer_length_reasonable_true_for_normal_answer():
    assert answer_length_reasonable(GOOD_RESULT) is True


def test_answer_length_reasonable_false_when_empty():
    r = {**GOOD_RESULT, "final_answer": ""}
    assert answer_length_reasonable(r) is False


def test_answer_length_reasonable_false_when_none():
    r = {**GOOD_RESULT, "final_answer": None}
    assert answer_length_reasonable(r) is False


def test_answer_length_reasonable_false_when_too_long():
    r = {**GOOD_RESULT, "final_answer": "x" * 3000}
    assert answer_length_reasonable(r) is False


def test_used_a_tool_true_when_action_present():
    assert used_a_tool(GOOD_RESULT) is True


def test_used_a_tool_false_when_no_actions():
    r = {**GOOD_RESULT, "trajectory": [{"type": "final", "step": 0, "content": "answer"}]}
    assert used_a_tool(r) is False


def test_heuristic_score_all_true_for_good_result():
    scores = heuristic_score(GOOD_RESULT)
    assert all(scores.values())


def test_heuristic_pass_true_for_good_result():
    assert heuristic_pass(GOOD_RESULT) is True


def test_heuristic_pass_false_if_any_check_fails():
    r = {**GOOD_RESULT, "had_error": True}
    assert heuristic_pass(r) is False


def test_judge_score_parses_single_digit_reply():
    def fake_chat(messages):
        return "4"

    assert judge_score("q", "a", chat_fn=fake_chat) == 4


def test_judge_score_parses_digit_embedded_in_text():
    def fake_chat(messages):
        return "I'd rate this a 3 out of 5."

    assert judge_score("q", "a", chat_fn=fake_chat) == 3


def test_judge_score_zero_when_unparseable():
    def fake_chat(messages):
        return "no digits here"

    assert judge_score("q", "a", chat_fn=fake_chat) == 0

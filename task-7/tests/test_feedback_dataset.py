import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from feedback_dataset import load_dataset, record_if_flagged, should_flag

GOOD_RESULT = {
    "question": "explain fractions",
    "final_answer": "A fraction has a numerator and a denominator, for example 3/4.",
    "hit_step_limit": False,
    "had_error": False,
    "trajectory": [
        {"type": "action", "step": 0, "tool": "curriculum_lookup", "arg": "fractions"},
        {"type": "observation", "step": 0, "tool": "curriculum_lookup", "content": "..."},
        {"type": "final", "step": 1, "content": "A fraction has..."},
    ],
}

BAD_RESULT = {
    "question": "explain ratios",
    "final_answer": None,
    "hit_step_limit": True,
    "had_error": True,
    "trajectory": [],
}


@pytest.fixture
def dataset_path(tmp_path):
    return tmp_path / "hard_examples.jsonl"


def test_good_result_no_judge_no_feedback_not_flagged():
    flagged, reasons = should_flag(GOOD_RESULT)
    assert flagged is False
    assert reasons == []


def test_bad_result_flagged_for_heuristic_fail():
    flagged, reasons = should_flag(BAD_RESULT)
    assert flagged is True
    assert "heuristic_fail" in reasons


def test_low_judge_score_flags_even_a_heuristically_clean_result():
    flagged, reasons = should_flag(GOOD_RESULT, judge_score=2)
    assert flagged is True
    assert "low_judge_score:2" in reasons


def test_high_judge_score_does_not_flag():
    flagged, reasons = should_flag(GOOD_RESULT, judge_score=5)
    assert flagged is False


def test_thumbs_down_flags_even_a_clean_result():
    flagged, reasons = should_flag(GOOD_RESULT, thumbs_up=False)
    assert flagged is True
    assert "user_thumbs_down" in reasons


def test_thumbs_up_does_not_flag():
    flagged, reasons = should_flag(GOOD_RESULT, thumbs_up=True)
    assert flagged is False


def test_multiple_reasons_all_recorded():
    flagged, reasons = should_flag(BAD_RESULT, judge_score=1, thumbs_up=False)
    assert flagged is True
    assert set(reasons) == {"heuristic_fail", "low_judge_score:1", "user_thumbs_down"}


def test_record_if_flagged_writes_a_line(dataset_path):
    was_flagged = record_if_flagged(BAD_RESULT, dataset_path=dataset_path)
    assert was_flagged is True
    assert dataset_path.exists()
    records = load_dataset(dataset_path)
    assert len(records) == 1
    assert records[0]["question"] == "explain ratios"


def test_record_if_flagged_skips_good_results(dataset_path):
    was_flagged = record_if_flagged(GOOD_RESULT, dataset_path=dataset_path)
    assert was_flagged is False
    assert load_dataset(dataset_path) == []


def test_load_dataset_empty_when_file_missing(tmp_path):
    assert load_dataset(tmp_path / "does_not_exist.jsonl") == []


def test_multiple_records_append_not_overwrite(dataset_path):
    record_if_flagged(BAD_RESULT, dataset_path=dataset_path)
    record_if_flagged(BAD_RESULT, dataset_path=dataset_path)
    assert len(load_dataset(dataset_path)) == 2

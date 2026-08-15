import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from shadow_test import aggregate, decide_rollout


def test_candidate_matching_baseline_passes():
    baseline = {"heuristic_pass_rate": 1.0, "avg_judge_score": 4.0}
    candidate = {"heuristic_pass_rate": 1.0, "avg_judge_score": 4.0}
    passed, _ = decide_rollout(baseline, candidate)
    assert passed is True


def test_candidate_beating_baseline_passes():
    baseline = {"heuristic_pass_rate": 0.67, "avg_judge_score": 3.5}
    candidate = {"heuristic_pass_rate": 1.0, "avg_judge_score": 4.5}
    passed, _ = decide_rollout(baseline, candidate)
    assert passed is True


def test_heuristic_regression_fails_even_with_better_judge_score():
    baseline = {"heuristic_pass_rate": 1.0, "avg_judge_score": 3.0}
    candidate = {"heuristic_pass_rate": 0.67, "avg_judge_score": 5.0}
    passed, reason = decide_rollout(baseline, candidate)
    assert passed is False
    assert "heuristic pass rate regressed" in reason


def test_small_judge_score_dip_within_tolerance_passes():
    baseline = {"heuristic_pass_rate": 1.0, "avg_judge_score": 4.0}
    candidate = {"heuristic_pass_rate": 1.0, "avg_judge_score": 3.6}
    passed, _ = decide_rollout(baseline, candidate, judge_score_tolerance=0.5)
    assert passed is True


def test_large_judge_score_drop_beyond_tolerance_fails():
    baseline = {"heuristic_pass_rate": 1.0, "avg_judge_score": 4.0}
    candidate = {"heuristic_pass_rate": 1.0, "avg_judge_score": 3.0}
    passed, reason = decide_rollout(baseline, candidate, judge_score_tolerance=0.5)
    assert passed is False
    assert "judge score regressed" in reason


def test_aggregate_computes_pass_rate_and_avg_score():
    rows = [
        {"heuristic_pass": True, "judge_score": 5},
        {"heuristic_pass": True, "judge_score": 3},
        {"heuristic_pass": False, "judge_score": 2},
    ]
    agg = aggregate(rows)
    assert agg["n"] == 3
    assert agg["heuristic_pass_rate"] == pytest.approx(2 / 3)
    assert agg["avg_judge_score"] == pytest.approx(10 / 3)

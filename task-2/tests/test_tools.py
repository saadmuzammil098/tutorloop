import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from tools import calculator, curriculum_lookup, practice_problem


def test_curriculum_lookup_finds_linear_equations():
    result = curriculum_lookup("how do I solve a linear equation with x on both sides")
    assert "Solving Linear Equations" in result
    assert "OpenStax" in result


def test_curriculum_lookup_finds_fractions():
    result = curriculum_lookup("how do I add fractions with different denominators")
    assert "Fractions" in result


def test_curriculum_lookup_no_match_raises():
    with pytest.raises(ValueError):
        curriculum_lookup("photosynthesis")


def test_curriculum_lookup_empty_query_raises():
    with pytest.raises(ValueError):
        curriculum_lookup("   ")


def test_calculator_basic_arithmetic():
    assert calculator("2 + 3 * 4") == "2 + 3 * 4 = 14"


def test_calculator_parentheses_and_power():
    assert calculator("(2 + 3) ** 2") == "(2 + 3) ** 2 = 25"


def test_calculator_division_by_zero_raises():
    with pytest.raises(ValueError):
        calculator("1 / 0")


def test_calculator_rejects_non_arithmetic():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo hi')")


def test_calculator_rejects_garbage():
    with pytest.raises(ValueError):
        calculator("this is not math")


def test_practice_problem_linear_equations_is_solvable():
    problem = practice_problem("linear_equations", "easy", seed=42)
    assert "Solve for x" in problem
    assert "answer: x =" in problem


def test_practice_problem_unknown_topic_raises():
    with pytest.raises(ValueError):
        practice_problem("calculus", "easy")


def test_practice_problem_unknown_difficulty_raises():
    with pytest.raises(ValueError):
        practice_problem("fractions", "impossible")


def test_practice_problem_deterministic_with_seed():
    a = practice_problem("ratios", "medium", seed=7)
    b = practice_problem("ratios", "medium", seed=7)
    assert a == b

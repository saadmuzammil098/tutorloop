import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agents import content_agent, coordinator, planner_agent, quiz_agent


def _scripted(responses):
    it = iter(responses)

    def chat_fn(messages):
        return next(it)

    return chat_fn


def test_planner_agent_returns_valid_topics():
    reply = '[{"topic": "fractions", "difficulty": "easy"}, {"topic": "ratios", "difficulty": "medium"}]'
    lessons = planner_agent("Fractions and ratios review", chat_fn=_scripted([reply]))
    assert lessons == [
        {"topic": "fractions", "difficulty": "easy"},
        {"topic": "ratios", "difficulty": "medium"},
    ]


def test_planner_agent_rejects_unindexed_topic():
    reply = '[{"topic": "calculus", "difficulty": "hard"}]'
    with pytest.raises(ValueError):
        planner_agent("Calculus intro", chat_fn=_scripted([reply]))


def test_planner_agent_defaults_missing_difficulty_to_medium():
    reply = '[{"topic": "linear_equations"}]'
    lessons = planner_agent("Algebra basics", chat_fn=_scripted([reply]))
    assert lessons[0]["difficulty"] == "medium"


def test_planner_agent_handles_prose_wrapped_json():
    reply = 'Sure, here is the plan:\n[{"topic": "fractions", "difficulty": "easy"}]\nLet me know if you need more.'
    lessons = planner_agent("Fractions", chat_fn=_scripted([reply]))
    assert lessons[0]["topic"] == "fractions"


def test_content_agent_grounds_in_curriculum():
    reply = "To solve a linear equation, isolate x using inverse operations."
    result = content_agent("linear_equations", chat_fn=_scripted([reply]))
    assert result["explanation"] == reply
    assert "Solving Linear Equations" in result["grounding"]
    assert "OpenStax" in result["grounding"]


def test_quiz_agent_includes_computational_and_conceptual():
    reply = "Why do we divide both sides by the same number when solving 4x = 20?"
    result = quiz_agent(
        "linear_equations", "easy", "isolate x using inverse operations", chat_fn=_scripted([reply])
    )
    assert result["conceptual_question"] == reply
    assert "Solve for x" in result["computational_question"]
    assert "answer:" in result["computational_question"]


def test_coordinator_produces_coherent_lesson_and_quiz():
    responses = [
        # planner
        '[{"topic": "fractions", "difficulty": "easy"}, {"topic": "ratios", "difficulty": "medium"}]',
        # lesson 1: content, then quiz
        "Fractions represent parts of a whole, written as numerator over denominator.",
        "What does the denominator of a fraction represent?",
        # lesson 2: content, then quiz
        "A ratio compares two quantities using the same units.",
        "How is a ratio different from a rate?",
    ]
    result = coordinator("Fractions and ratios unit", chat_fn=_scripted(responses))

    assert result["unit"] == "Fractions and ratios unit"
    assert len(result["lessons"]) == 2

    lesson1 = result["lessons"][0]
    assert lesson1["content"]["topic"] == "fractions"
    assert "parts of a whole" in lesson1["content"]["explanation"]
    assert lesson1["quiz"]["conceptual_question"] == "What does the denominator of a fraction represent?"
    assert "Add the fractions" in lesson1["quiz"]["computational_question"]

    lesson2 = result["lessons"][1]
    assert lesson2["content"]["topic"] == "ratios"
    assert lesson2["quiz"]["conceptual_question"] == "How is a ratio different from a rate?"


def test_coordinator_propagates_planner_error():
    responses = ['[{"topic": "history", "difficulty": "easy"}]']
    with pytest.raises(ValueError):
        coordinator("History unit", chat_fn=_scripted(responses))

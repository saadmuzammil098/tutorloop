"""Online eval: scoring a live session with no ground-truth expected
answer available at serve time (that's what makes it "online" rather than
Task 5's offline eval against a known expected_substring). Two layers:

1. Heuristic checks (pure functions, no model call, offline-testable):
   cheap proxy signals available for every single real session.
2. An LLM-judge call (real Ollama call, live-only): a genuine quality
   rating a heuristic can't approximate, used sparingly (not every
   session in a real system, an occasional sampled judge call).
"""

from __future__ import annotations

import ollama

JUDGE_MODEL = "qwen2.5:7b"


def completed_without_error(result: dict) -> bool:
    return not result["hit_step_limit"] and not result["had_error"]


def answer_length_reasonable(result: dict, min_chars: int = 10, max_chars: int = 2000) -> bool:
    answer = result.get("final_answer") or ""
    return min_chars <= len(answer) <= max_chars


def used_a_tool(result: dict) -> bool:
    """A tutoring answer produced with zero tool calls is suspicious, it
    means the agent answered from parametric knowledge alone rather than
    grounding in the curriculum lookup, calculator, or practice-problem
    tools this whole system exists to route through.
    """
    return any(e["type"] == "action" for e in result["trajectory"])


def heuristic_score(result: dict) -> dict:
    """Every check is independent and boolean, exactly like Task 5's
    trajectory/tool-call/task-success split: a session can pass some and
    fail others, that's the useful signal, not a single collapsed score.
    """
    return {
        "completed_without_error": completed_without_error(result),
        "answer_length_reasonable": answer_length_reasonable(result),
        "used_a_tool": used_a_tool(result),
    }


def heuristic_pass(result: dict) -> bool:
    return all(heuristic_score(result).values())


_JUDGE_SYSTEM = """You are grading a homework-help tutor's answer to a student's question.
Rate the answer's quality from 1 (unhelpful or wrong) to 5 (clear, correct, and well-explained).
Respond with ONLY a single digit 1-5, nothing else."""


def judge_score(question: str, answer: str, chat_fn=None) -> int:
    """A real LLM-judge call rating answer quality 1-5. Live-only (no
    ground truth, no offline fixture makes sense for a subjective judge
    rating), see shadow_test.py for where this is actually exercised.
    """
    chat_fn = chat_fn or (lambda messages: ollama.chat(model=JUDGE_MODEL, messages=messages)["message"]["content"])
    reply = chat_fn(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
        ]
    )
    digits = [c for c in reply.strip() if c.isdigit()]
    return int(digits[0]) if digits else 0

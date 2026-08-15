"""Four small agents, each a plain function with its own system prompt and
own responsibility, tied together by coordinator(). No framework here
either (LangGraph is Task 2's point, not this one): the point of Task 3 is
the multi-agent split itself, planner -> content -> quiz -> coordinator,
each grounded in the same indexed curriculum tools.py/tools live in
task-1's ReAct agent.
"""

from __future__ import annotations

import json
import re

import ollama

from tools import curriculum_lookup, practice_problem

MODEL = "qwen2.5:7b"
VALID_TOPICS = ["linear_equations", "fractions", "ratios"]
VALID_DIFFICULTIES = ("easy", "medium", "hard")


def _default_chat(messages: list[dict]) -> str:
    return ollama.chat(model=MODEL, messages=messages)["message"]["content"]


def _extract_json(text: str):
    match = re.search(r"\[.*\]|\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON found in model output: {text!r}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Planner agent: unit -> ordered lessons
# ---------------------------------------------------------------------------

def planner_agent(unit: str, chat_fn=None) -> list[dict]:
    chat_fn = chat_fn or _default_chat
    system = (
        "You are the PLANNER agent for TutorLoop. Break the given unit "
        "into 2 or 3 lessons. You may ONLY choose from these indexed "
        f"topics, one lesson per topic: {VALID_TOPICS}. Respond with ONLY "
        'a JSON array like [{"topic": "fractions", "difficulty": "easy"}], '
        "no prose, no markdown fences."
    )
    reply = chat_fn(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Unit: {unit}"},
        ]
    )
    lessons = _extract_json(reply)
    if not isinstance(lessons, list) or not lessons:
        raise ValueError(f"Planner did not return a non-empty lesson list: {lessons!r}")
    for lesson in lessons:
        if lesson.get("topic") not in VALID_TOPICS:
            raise ValueError(f"Planner proposed an unindexed topic: {lesson.get('topic')!r}")
        if lesson.get("difficulty") not in VALID_DIFFICULTIES:
            lesson["difficulty"] = "medium"
    return lessons


# ---------------------------------------------------------------------------
# Content agent: topic -> grounded explanation
# ---------------------------------------------------------------------------

def content_agent(topic: str, chat_fn=None) -> dict:
    chat_fn = chat_fn or _default_chat
    grounding = curriculum_lookup(topic.replace("_", " "))
    system = (
        "You are the CONTENT agent for TutorLoop. Write a short, clear "
        "explanation (3-5 sentences) of the topic for a middle-school "
        "student. Ground it ONLY in the provided curriculum excerpt, do "
        "not introduce facts the excerpt doesn't contain."
    )
    reply = chat_fn(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Topic: {topic}\nCurriculum excerpt: {grounding}",
            },
        ]
    )
    return {"topic": topic, "explanation": reply.strip(), "grounding": grounding}


# ---------------------------------------------------------------------------
# Quiz agent: topic + explanation -> aligned quiz
# ---------------------------------------------------------------------------

def quiz_agent(topic: str, difficulty: str, explanation: str, chat_fn=None) -> dict:
    chat_fn = chat_fn or _default_chat
    computational_question = practice_problem(topic, difficulty)
    system = (
        "You are the QUIZ agent for TutorLoop. Given a topic and its "
        "explanation, write ONE short conceptual quiz question (not a "
        "computation, a computational question is generated separately) "
        "that tests understanding of the explanation. Respond with ONLY "
        "the question text, no prose, no answer."
    )
    reply = chat_fn(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Topic: {topic}\nExplanation: {explanation}"},
        ]
    )
    return {
        "topic": topic,
        "difficulty": difficulty,
        "computational_question": computational_question,
        "conceptual_question": reply.strip(),
    }


# ---------------------------------------------------------------------------
# Coordinator: ties planner -> content -> quiz together
# ---------------------------------------------------------------------------

def coordinator(unit: str, chat_fn=None) -> dict:
    chat_fn = chat_fn or _default_chat
    lessons_plan = planner_agent(unit, chat_fn)
    lessons = []
    for lesson in lessons_plan:
        content = content_agent(lesson["topic"], chat_fn)
        quiz = quiz_agent(lesson["topic"], lesson["difficulty"], content["explanation"], chat_fn)
        lessons.append({"content": content, "quiz": quiz})
    return {"unit": unit, "lessons": lessons}

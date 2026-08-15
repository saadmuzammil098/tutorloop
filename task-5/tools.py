"""The three tools TutorLoop's hand-rolled ReAct agent can call.

Each tool is a plain function, string in, string out, so the agent loop in
agent.py can feed a tool's return value straight back into the LLM as an
"Observation" without any framework-specific tool-calling machinery.
"""

from __future__ import annotations

import ast
import json
import operator
import random
from pathlib import Path

CURRICULUM_DIR = Path(__file__).parent / "curriculum"

# ---------------------------------------------------------------------------
# Tool 1: curriculum lookup
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "how", "do", "i",
    "what", "in", "for", "with", "on", "does", "explain", "me",
}


def _load_curriculum() -> list[dict]:
    chapters = []
    for path in sorted(CURRICULUM_DIR.glob("*.json")):
        chapters.append(json.loads(path.read_text()))
    return chapters


def curriculum_lookup(query: str) -> str:
    """Keyword search across the indexed OpenStax Prealgebra chapters.

    Scores each section by how many non-stopword query tokens appear in its
    title or text, returns the best match with its source citation. Raises
    ValueError (a recoverable tool error, not a crash) if nothing matches,
    so the agent can see the miss and try a different query or tool.
    """
    tokens = {t for t in query.lower().replace("?", "").split() if t not in _STOPWORDS}
    if not tokens:
        raise ValueError("curriculum_lookup needs a non-empty query")

    best_score = 0
    best = None
    best_chapter = None
    for chapter in _load_curriculum():
        for section in chapter["sections"]:
            haystack = (
                chapter["chapter"] + " " + section["title"] + " " + section["text"]
            ).lower()
            score = sum(1 for t in tokens if t in haystack)
            if score > best_score:
                best_score = score
                best = section
                best_chapter = chapter

    if best is None or best_score == 0:
        raise ValueError(
            f"No indexed curriculum section matches {query!r}. "
            "Indexed chapters: Solving Linear Equations, Fractions, "
            "Ratios and Proportions."
        )

    return (
        f"[{best_chapter['chapter']} > {best['title']}] {best['text']} "
        f"(source: {best_chapter['source']})"
    )


# ---------------------------------------------------------------------------
# Tool 2: calculator
# ---------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a pure arithmetic expression safely (no eval/exec).

    Only numbers and + - * / ** % and parentheses are supported. Anything
    else, or division by zero, raises ValueError, which the agent loop
    catches and feeds back as an Observation, this is the tool-error path
    Task 1's "recovers from a tool error" requirement exercises.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Could not parse {expression!r} as an expression: {e}")
    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        raise ValueError(f"Division by zero in {expression!r}")
    return f"{expression} = {result}"


# ---------------------------------------------------------------------------
# Tool 3: practice problem generator
# ---------------------------------------------------------------------------

_TOPICS = {"linear_equations", "fractions", "ratios"}


def practice_problem(topic: str, difficulty: str = "medium", seed: int | None = None) -> str:
    """Generate a practice problem for one of three indexed topics.

    Raises ValueError for an unrecognized topic or difficulty, again a
    recoverable tool error rather than a silent wrong-topic problem.
    """
    topic = topic.strip().lower().replace(" ", "_")
    difficulty = difficulty.strip().lower()
    if topic not in _TOPICS:
        raise ValueError(f"Unknown topic {topic!r}, expected one of {sorted(_TOPICS)}")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unknown difficulty {difficulty!r}, expected easy/medium/hard")

    rng = random.Random(seed)
    scale = {"easy": 10, "medium": 25, "hard": 100}[difficulty]

    if topic == "linear_equations":
        a = rng.randint(2, 9)
        x = rng.randint(1, scale // a if scale // a > 0 else 1)
        b = rng.randint(1, scale)
        c = a * x + b
        return f"Solve for x: {a}x + {b} = {c}  (answer: x = {x})"

    if topic == "fractions":
        n1, d1 = rng.randint(1, 9), rng.randint(2, 12)
        n2, d2 = rng.randint(1, 9), rng.randint(2, 12)
        answer = f"{n1}/{d1} + {n2}/{d2}"
        return f"Add the fractions: {n1}/{d1} + {n2}/{d2}  (simplify your answer)"

    # ratios
    a = rng.randint(1, scale)
    b = rng.randint(1, scale)
    k = rng.randint(2, 5)
    return f"If {a}:{b} = x:{b * k}, what is x?  (answer: x = {a * k})"

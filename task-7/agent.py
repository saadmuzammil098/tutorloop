"""A small ReAct-style tutoring agent, re-derived here (task-7 branches
off main, which has no earlier task merged into it yet, so each task
folder is self-contained rather than importing across branches). Same
three-tool shape as Tasks 1/5/6, with one addition Task 7 specifically
needs: the system prompt is injectable, so shadow_test.py can run a
candidate prompt through the identical loop a real session would use,
without touching what real sessions actually run.
"""

from __future__ import annotations

import re

import ollama

from tools import calculator, curriculum_lookup, practice_problem

MODEL = "qwen2.5:7b"
MAX_STEPS = 6

_TOOLS = {
    "curriculum_lookup": curriculum_lookup,
    "calculator": calculator,
    "practice_problem": practice_problem,
}

DEFAULT_SYSTEM = """You are TutorLoop, a homework-help agent. You have three tools:
- curriculum_lookup[query]: look up an OpenStax Prealgebra excerpt
- calculator[expression]: evaluate a pure arithmetic expression
- practice_problem[topic]: generate a practice problem (topic one of linear_equations, fractions, ratios)

Respond with EXACTLY ONE of these each turn:
Thought: <your reasoning>
Action: toolname[argument]

or, once you have enough information:
Thought: <your reasoning>
Final Answer: <your answer>

Never do both in the same turn. Never call a tool that doesn't exist."""

_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*)\]", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def _default_chat(messages: list[dict]) -> str:
    return ollama.chat(model=MODEL, messages=messages)["message"]["content"]


def run(question: str, system: str = DEFAULT_SYSTEM, chat_fn=None, max_steps: int = MAX_STEPS) -> dict:
    """Run the ReAct loop, returning the full trajectory. `system` is
    injectable specifically so shadow_test.py can run a candidate prompt
    through this exact function, the same code path a real session uses,
    while `online_eval.py`'s judge scores whatever came out.
    """
    chat_fn = chat_fn or _default_chat
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    trajectory: list[dict] = []
    final_answer = None

    for step in range(max_steps):
        reply = chat_fn(messages)
        messages.append({"role": "assistant", "content": reply})

        final_match = _FINAL_RE.search(reply)
        action_match = _ACTION_RE.search(reply)

        if final_match and (not action_match or reply.index("Final Answer") < reply.index("Action")):
            final_answer = final_match.group(1).strip()
            trajectory.append({"type": "final", "step": step, "content": final_answer})
            break

        if action_match:
            tool_name, arg = action_match.group(1), action_match.group(2).strip()
            trajectory.append({"type": "action", "step": step, "tool": tool_name, "arg": arg})
            tool = _TOOLS.get(tool_name)
            if tool is None:
                obs = f"Error: no such tool {tool_name!r}. Available: {sorted(_TOOLS)}"
                trajectory.append({"type": "error", "step": step, "tool": tool_name, "message": obs})
            else:
                try:
                    obs = tool(arg)
                    trajectory.append({"type": "observation", "step": step, "tool": tool_name, "content": obs})
                except ValueError as e:
                    obs = f"Error: {e}"
                    trajectory.append({"type": "error", "step": step, "tool": tool_name, "message": str(e)})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            trajectory.append({"type": "unparseable", "step": step, "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": "Your last response didn't match the required Thought/Action or "
                    "Thought/Final Answer format. Try again, exactly one Action or Final Answer.",
                }
            )

    return {
        "question": question,
        "trajectory": trajectory,
        "final_answer": final_answer,
        "hit_step_limit": final_answer is None,
        "steps_used": len(trajectory),
        "had_error": any(e["type"] == "error" for e in trajectory),
    }

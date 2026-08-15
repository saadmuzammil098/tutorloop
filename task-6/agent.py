"""A small ReAct-style tutoring agent, re-derived here (task-6 branches off
main, which has no earlier task merged into it yet, so each task folder is
self-contained rather than importing across branches). Same three-tool
shape as Tasks 1 and 5, with one addition Task 6 actually needs: each LLM
call's real token counts and wall-clock duration are captured in the
trajectory (Ollama returns real `prompt_eval_count`/`eval_count`/
`total_duration` figures, not estimates), so `otel_tracing.py` can report
genuine latency and cost numbers instead of guessing from text length.
"""

from __future__ import annotations

import re
import time

import ollama

from tools import calculator, curriculum_lookup, practice_problem

MODEL = "qwen2.5:7b"
MAX_STEPS = 6

_TOOLS = {
    "curriculum_lookup": curriculum_lookup,
    "calculator": calculator,
    "practice_problem": practice_problem,
}

_SYSTEM = """You are TutorLoop, a homework-help agent. You have three tools:
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


def _default_chat(messages: list[dict]) -> dict:
    """Returns a dict, not just the text, so callers get real usage/timing
    figures (prompt_eval_count, eval_count, total_duration) alongside the
    reply, rather than the agent loop discarding them.
    """
    t0 = time.monotonic()
    resp = ollama.chat(model=MODEL, messages=messages)
    wall_seconds = time.monotonic() - t0
    return {
        "content": resp.message.content,
        "prompt_tokens": resp.prompt_eval_count or 0,
        "completion_tokens": resp.eval_count or 0,
        "wall_seconds": wall_seconds,
    }


def run(question: str, chat_fn=None, max_steps: int = MAX_STEPS) -> dict:
    """Run the ReAct loop, returning the full trajectory plus per-call
    usage/timing so otel_tracing.py can attach real latency and cost
    figures to spans instead of estimating them.

    chat_fn, if provided, must return the same dict shape _default_chat
    does: {"content", "prompt_tokens", "completion_tokens", "wall_seconds"}.
    """
    chat_fn = chat_fn or _default_chat
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question},
    ]
    trajectory: list[dict] = []
    final_answer = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_wall_seconds = 0.0

    for step in range(max_steps):
        chat_result = chat_fn(messages)
        reply = chat_result["content"]
        total_prompt_tokens += chat_result.get("prompt_tokens", 0)
        total_completion_tokens += chat_result.get("completion_tokens", 0)
        total_wall_seconds += chat_result.get("wall_seconds", 0.0)
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
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "wall_seconds": total_wall_seconds,
        "had_error": any(e["type"] == "error" for e in trajectory),
    }

"""A hand-rolled ReAct agent, no LangChain/LangGraph/agent framework.

The loop: prompt the LLM with the question and a running transcript, parse
its next line for either "Action: <tool>[<input>]" or "Final Answer: ...",
run the tool if it asked for one, append the observation, repeat, capped
at MAX_STEPS so a confused model can never loop forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import ollama

from tools import calculator, curriculum_lookup, practice_problem

MODEL = "qwen2.5:7b"
MAX_STEPS = 6

TOOLS = {
    "curriculum_lookup": curriculum_lookup,
    "calculator": calculator,
    "practice_problem": practice_problem,
}

SYSTEM_PROMPT = f"""You are TutorLoop, a homework-help agent for a middle-school \
prealgebra student. You solve problems step by step using tools, you never \
skip straight to an answer for a curriculum question without checking the \
curriculum tool first, and you never do arithmetic in your head, you always \
call the calculator tool for arithmetic.

You have exactly these tools:
- curriculum_lookup(query: str): look up an explanation from the indexed \
OpenStax Prealgebra chapters (Solving Linear Equations, Fractions, Ratios \
and Proportions). Use this when the student asks "how do I..." or "explain...".
- calculator(expression: str): evaluate a pure arithmetic expression like \
"3 * (4 + 2)". Use this for any arithmetic, never compute it yourself.
- practice_problem(topic: str, difficulty: str): generate a practice \
problem. topic must be one of linear_equations, fractions, ratios. \
difficulty must be one of easy, medium, hard.

Respond using EXACTLY this format, one step at a time:

Thought: <your reasoning about what to do next>
Action: <tool_name>[<tool input, comma-separated arguments if more than one>]

or, once you have enough information to answer the student:

Thought: <your reasoning>
Final Answer: <your answer to the student>

Never output both an Action and a Final Answer in the same turn. Never \
invent a tool name that isn't in the list above. Output ONLY ONE \
Thought/Action pair (or ONE Thought/Final Answer pair) per turn, then stop \
and wait for the Observation, do not write out multiple steps at once.

Example turn:
Thought: The student is asking how to solve a linear equation, I should look it up.
Action: curriculum_lookup[how to solve a linear equation]

You MUST use square brackets for the tool input, like curriculum_lookup[...], \
never parentheses like curriculum_lookup(...).
"""

# qwen2.5:7b (and other small open models) sometimes ignore the bracket
# instruction and emit Action: tool("arg") instead of tool[arg], a real,
# observed function-calling weakness the roadmap warns about. Rather than
# silently failing on that output, the parser accepts both syntaxes.
ACTION_RE = re.compile(r"Action:\s*(\w+)[\[\(](.*?)[\]\)]", re.S)
# Only take the FIRST Final Answer / Action block a turn produces, in case
# the model writes out multiple steps in one response despite being told
# not to (also observed with qwen2.5:7b).
FINAL_RE = re.compile(r"Final Answer:\s*(.*?)(?:\n\nThought:|\Z)", re.S)


@dataclass
class AgentResult:
    final_answer: str | None
    steps: list[dict] = field(default_factory=list)
    hit_step_limit: bool = False


def _parse_tool_args(tool_name: str, raw_args: str) -> tuple[tuple, dict]:
    """Split a comma-separated Action[...] input into positional args.

    practice_problem takes (topic, difficulty); the other two tools take a
    single string argument, so anything after the first comma for them is
    treated as part of the same string (a query can legitimately contain a
    comma).
    """
    if tool_name == "practice_problem":
        parts = [p.strip().strip('"').strip("'") for p in raw_args.split(",")]
        if len(parts) == 1:
            return (parts[0],), {}
        return (parts[0], parts[1]), {}
    return (raw_args.strip().strip('"').strip("'"),), {}


def run_agent(question: str, chat_fn=None, max_steps: int = MAX_STEPS) -> AgentResult:
    """Run the ReAct loop for one student question.

    chat_fn is injectable so tests can drive the loop with a scripted
    sequence of model outputs instead of calling a live Ollama server.
    """
    chat_fn = chat_fn or _ollama_chat

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = AgentResult(final_answer=None)

    for step in range(1, max_steps + 1):
        reply = chat_fn(messages)
        messages.append({"role": "assistant", "content": reply})

        final_match = FINAL_RE.search(reply)
        if final_match:
            result.final_answer = final_match.group(1).strip()
            result.steps.append({"step": step, "type": "final", "content": reply})
            return result

        action_match = ACTION_RE.search(reply)
        if not action_match:
            # The model didn't follow the format. Nudge it rather than crash.
            observation = (
                "Error: your response didn't include a valid Action[...] or "
                "Final Answer. Follow the required format exactly."
            )
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            result.steps.append(
                {"step": step, "type": "format_error", "content": reply}
            )
            continue

        tool_name, raw_args = action_match.group(1), action_match.group(2)
        tool = TOOLS.get(tool_name)
        if tool is None:
            observation = (
                f"Error: unknown tool {tool_name!r}. Valid tools: "
                f"{sorted(TOOLS)}."
            )
        else:
            try:
                args, kwargs = _parse_tool_args(tool_name, raw_args)
                observation = tool(*args, **kwargs)
            except Exception as e:  # tool-error recovery, not a crash
                observation = f"Error: {e}"

        result.steps.append(
            {
                "step": step,
                "type": "action",
                "tool": tool_name,
                "input": raw_args,
                "observation": observation,
            }
        )
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    result.hit_step_limit = True
    return result


def _ollama_chat(messages: list[dict]) -> str:
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How do I solve 3x + 2 = 11? Then give me a practice problem on the same topic."
    print(f"Question: {q}\n")
    r = run_agent(q)
    for s in r.steps:
        print(s)
    print(f"\nFinal answer: {r.final_answer}")
    print(f"Hit step limit: {r.hit_step_limit}")

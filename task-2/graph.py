"""Task 1's ReAct agent, rebuilt as a LangGraph state machine.

Adds one new capability Task 1 didn't have: the agent can draft a
progress note about the student (something that would reach a parent or
affect a grade). Drafting one routes to an `await_approval` node the
graph is compiled with `interrupt_before=["await_approval"]`, so the run
pauses there. A caller resumes it later by updating persisted state with
the teacher's decision and re-invoking with the same thread_id, which is
what "resumes from saved state after approval" means concretely here: the
checkpointer, not in-process memory, is what makes the pause durable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent / "task-1"))
from tools import calculator, curriculum_lookup, practice_problem  # noqa: E402

import ollama
from langgraph.graph import END, StateGraph

MODEL = "qwen2.5:7b"
MAX_STEPS = 6

TOOLS = {
    "curriculum_lookup": curriculum_lookup,
    "calculator": calculator,
    "practice_problem": practice_problem,
}

ACTION_RE = re.compile(r"Action:\s*(\w+)[\[\(](.*?)[\]\)]", re.S)
NOTE_RE = re.compile(r"DraftNote:\s*(.*?)(?:\n\nThought:|\Z)", re.S)
FINAL_RE = re.compile(r"Final Answer:\s*(.*?)(?:\n\nThought:|\Z)", re.S)

SYSTEM_PROMPT = """You are TutorLoop, a homework-help agent for a middle-school \
prealgebra student. You have three tools plus one special action:

- curriculum_lookup[query]: look up an OpenStax Prealgebra explanation.
- calculator[expression]: evaluate arithmetic. Never compute arithmetic yourself.
- practice_problem[topic, difficulty]: topic in linear_equations/fractions/ratios, \
difficulty in easy/medium/hard.
- DraftNote: <text>  -- use this ONLY if the student explicitly asks you to \
tell their teacher or parent something (e.g. "let my teacher know I'm \
struggling with fractions"). This drafts a note, it does NOT send it, a \
teacher must approve it first.

Respond with EXACTLY ONE of these per turn:
Thought: <reasoning>
Action: tool_name[input]

Thought: <reasoning>
DraftNote: <the note text>

Thought: <reasoning>
Final Answer: <answer to the student>

Never invent a tool. Use square brackets for tool input, e.g. calculator[2+2].
"""


class TutorState(TypedDict):
    question: str
    messages: list[dict]
    step: int
    pending: Optional[dict]
    final_answer: Optional[str]
    progress_note: Optional[str]
    teacher_decision: Optional[str]  # None | "approved" | "rejected"
    sent: bool
    trace: list[dict]


def _default_chat(messages: list[dict]) -> str:
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]


def _parse_tool_args(tool_name: str, raw_args: str):
    if tool_name == "practice_problem":
        parts = [p.strip().strip('"').strip("'") for p in raw_args.split(",")]
        if len(parts) == 1:
            return (parts[0],), {}
        return (parts[0], parts[1]), {}
    return (raw_args.strip().strip('"').strip("'"),), {}


def build_graph(chat_fn=None, checkpointer=None):
    """Construct the compiled TutorLoop LangGraph state machine.

    chat_fn is injectable for offline testing, same pattern as Task 1's
    run_agent(). checkpointer is required for durable state + resumable
    interrupts; callers pass a SqliteSaver in production and an in-memory
    one is fine for a single-process test.
    """
    chat_fn = chat_fn or _default_chat

    def agent_node(state: TutorState) -> dict:
        messages = state["messages"]
        reply = chat_fn(messages)
        new_messages = messages + [{"role": "assistant", "content": reply}]
        step = state["step"] + 1
        trace = state["trace"] + [{"step": step, "raw": reply}]

        if step > MAX_STEPS:
            return {
                "messages": new_messages,
                "step": step,
                "pending": {"type": "step_limit"},
                "trace": trace,
            }

        final_match = FINAL_RE.search(reply)
        if final_match:
            return {
                "messages": new_messages,
                "step": step,
                "pending": {"type": "final", "answer": final_match.group(1).strip()},
                "trace": trace,
            }

        note_match = NOTE_RE.search(reply)
        if note_match:
            return {
                "messages": new_messages,
                "step": step,
                "pending": {"type": "note", "text": note_match.group(1).strip()},
                "trace": trace,
            }

        action_match = ACTION_RE.search(reply)
        if action_match:
            return {
                "messages": new_messages,
                "step": step,
                "pending": {
                    "type": "tool",
                    "tool": action_match.group(1),
                    "input": action_match.group(2),
                },
                "trace": trace,
            }

        return {
            "messages": new_messages,
            "step": step,
            "pending": {"type": "format_error"},
            "trace": trace,
        }

    def tool_node(state: TutorState) -> dict:
        pending = state["pending"]
        tool_name, raw_args = pending["tool"], pending["input"]
        tool = TOOLS.get(tool_name)
        if tool is None:
            observation = f"Error: unknown tool {tool_name!r}. Valid tools: {sorted(TOOLS)}."
        else:
            try:
                args, kwargs = _parse_tool_args(tool_name, raw_args)
                observation = tool(*args, **kwargs)
            except Exception as e:
                observation = f"Error: {e}"
        messages = state["messages"] + [
            {"role": "user", "content": f"Observation: {observation}"}
        ]
        trace = state["trace"] + [
            {"step": state["step"], "tool": tool_name, "observation": observation}
        ]
        return {"messages": messages, "pending": None, "trace": trace}

    def format_error_node(state: TutorState) -> dict:
        messages = state["messages"] + [
            {
                "role": "user",
                "content": "Observation: Error: your response didn't follow the "
                "required format. Use Action[...], DraftNote:, or Final Answer:.",
            }
        ]
        return {"messages": messages, "pending": None}

    def step_limit_node(state: TutorState) -> dict:
        return {
            "final_answer": None,
            "pending": None,
        }

    def finalize_node(state: TutorState) -> dict:
        return {"final_answer": state["pending"]["answer"], "pending": None}

    def draft_note_node(state: TutorState) -> dict:
        return {"progress_note": state["pending"]["text"], "pending": None}

    def await_approval_node(state: TutorState) -> dict:
        """Reached only after a resume with teacher_decision already set,
        since the graph is compiled with interrupt_before=["await_approval"],
        which pauses execution before this node runs the first time."""
        decision = state.get("teacher_decision")
        if decision == "approved":
            return {
                "sent": True,
                "final_answer": f"Progress note sent to teacher: {state['progress_note']}",
            }
        return {
            "sent": False,
            "final_answer": "Progress note was not approved by the teacher, not sent.",
        }

    def route_after_agent(state: TutorState) -> Literal[
        "tool_node", "finalize_node", "draft_note_node", "format_error_node", "step_limit_node"
    ]:
        kind = state["pending"]["type"]
        return {
            "tool": "tool_node",
            "final": "finalize_node",
            "note": "draft_note_node",
            "format_error": "format_error_node",
            "step_limit": "step_limit_node",
        }[kind]

    graph = StateGraph(TutorState)
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("format_error_node", format_error_node)
    graph.add_node("step_limit_node", step_limit_node)
    graph.add_node("finalize_node", finalize_node)
    graph.add_node("draft_note_node", draft_note_node)
    graph.add_node("await_approval", await_approval_node)

    graph.set_entry_point("agent_node")
    graph.add_conditional_edges("agent_node", route_after_agent)
    graph.add_edge("tool_node", "agent_node")
    graph.add_edge("format_error_node", "agent_node")
    graph.add_edge("step_limit_node", END)
    graph.add_edge("finalize_node", END)
    graph.add_edge("draft_note_node", "await_approval")
    graph.add_edge("await_approval", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["await_approval"])


def initial_state(question: str) -> TutorState:
    return {
        "question": question,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "step": 0,
        "pending": None,
        "final_answer": None,
        "progress_note": None,
        "teacher_decision": None,
        "sent": False,
        "trace": [],
    }

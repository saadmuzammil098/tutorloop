"""Live demo: run TutorLoop's LangGraph agent with a real SQLite
checkpointer (not the in-memory one the tests use), against live Ollama.

Run with `start` to kick off a session that drafts a progress note (it
will pause before await_approval and the checkpoint is left on disk).
Run with `resume <thread_id> <approved|rejected>` in a SEPARATE process
invocation to prove the state was actually persisted to disk and not just
held in this process's memory.
"""

from __future__ import annotations

import sys
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph, initial_state

DB_PATH = "tutorloop_state.sqlite"


def start(question: str) -> None:
    thread_id = str(uuid.uuid4())[:8]
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        g = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = g.invoke(initial_state(question), config)
        snapshot = g.get_state(config)
        print(f"thread_id: {thread_id}")
        print(f"progress_note: {result.get('progress_note')}")
        print(f"final_answer: {result.get('final_answer')}")
        print(f"next node (paused here if non-empty): {snapshot.next}")


def resume(thread_id: str, decision: str) -> None:
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        g = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        before = g.get_state(config)
        print(f"state loaded from disk, was paused at: {before.next}")
        g.update_state(config, {"teacher_decision": decision})
        result = g.invoke(None, config)
        print(f"final_answer: {result.get('final_answer')}")
        print(f"sent: {result.get('sent')}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "start":
        start(" ".join(sys.argv[2:]))
    elif cmd == "resume":
        resume(sys.argv[2], sys.argv[3])
    else:
        print("usage: demo.py start <question> | demo.py resume <thread_id> <approved|rejected>")

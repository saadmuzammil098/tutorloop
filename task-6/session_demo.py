"""A real multi-question tutoring session: several live questions against
qwen2.5:7b (real latency, real token counts) plus one deliberately-broken
run (same non-adapting chat_fn pattern as Task 5's broken_demo.py) so the
session has a genuine non-zero error rate to show in the dashboard,
rather than a session engineered to always look perfect. Feedback is
recorded on two of the traces. This is the "done when" live artifact for
Task 6.
"""

from __future__ import annotations

from otel_tracing import instrumented_run
from feedback import record_feedback

SESSION_ID = "tutorloop-task6-demo-session"

_BROKEN_REPLY = "Thought: computing the tip.\nAction: calculator[15% of 40]"


def broken_chat_fn(messages: list[dict]) -> dict:
    return {"content": _BROKEN_REPLY, "prompt_tokens": 45, "completion_tokens": 12, "wall_seconds": 0.05}


QUESTIONS = [
    "What is 12 + 30?",
    "Explain how to solve a simple linear equation like 2x + 3 = 11.",
    "Give me an easy practice problem about ratios.",
]


def main() -> None:
    results = []
    for q in QUESTIONS:
        r = instrumented_run(q, session_id=SESSION_ID)
        results.append((q, r))
        print(f"[live] {q!r} -> final_answer={r['final_answer']!r} cost=${r['cost_usd']:.6f}")

    broken = instrumented_run("What is 15% of 40?", session_id=SESSION_ID, chat_fn=broken_chat_fn)
    results.append(("What is 15% of 40? (scripted broken)", broken))
    print(f"[broken] final_answer={broken['final_answer']!r} had_error={broken['had_error']}")

    # Feedback: thumbs up on the first genuinely-successful live run,
    # thumbs down on the broken one, a real end-user signal tied to a
    # specific trace, distinct from the trajectory-derived scores.
    record_feedback(results[0][1]["trace_id"], thumbs_up=True, comment="fast and correct")
    record_feedback(broken["trace_id"], thumbs_up=False, comment="never got an answer")

    print(f"\nSession id for dashboard.py: {SESSION_ID}")


if __name__ == "__main__":
    main()

"""Wraps agent.run() with real Langfuse tracing: one trace per question,
one span per trajectory step (action/observation/error/final), and the
three scoring.py signals attached as trace-level scores. This is the only
module in task-5 that talks to the Langfuse server; agent.py and
scoring.py are both fully independent of it and unit-tested offline.
"""

from __future__ import annotations

import os

from langfuse import Langfuse

from agent import run as agent_run
from scoring import score_run


def _client() -> Langfuse:
    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-tutorloop-task5-dev"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-tutorloop-task5-dev"),
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )


def traced_run(
    question: str,
    expected_tool: str,
    expected_substring: str,
    client: Langfuse | None = None,
    chat_fn=None,
    trace_name: str = "tutorloop-task5-agent-run",
) -> dict:
    """Run the agent once, trace every step to Langfuse, score it, and
    push the scores back onto the same trace. Returns the raw agent
    result plus the computed scores and the Langfuse trace id, so eval.py
    can print/log a trace URL and broken_demo.py can hand the id to
    root_cause.py for lookup.

    chat_fn is injectable so broken_demo.py can trace a deliberately
    scripted, non-adapting chat_fn through the exact same tracing path a
    real run takes, rather than duplicating this function.
    """
    client = client or _client()

    with client.start_as_current_observation(
        name=trace_name, as_type="agent", input={"question": question}
    ) as span:
        result = agent_run(question, chat_fn=chat_fn)

        for entry in result["trajectory"]:
            step_type = entry["type"]
            as_type = "tool" if step_type in ("action", "observation", "error") else "span"
            with client.start_as_current_observation(
                name=f"step-{entry['step']}-{step_type}",
                as_type=as_type,
                input={k: v for k, v in entry.items() if k not in ("type", "step")},
                level="ERROR" if step_type == "error" else None,
                status_message=entry.get("message") if step_type == "error" else None,
            ):
                pass

        scores = score_run(result, expected_tool=expected_tool, expected_substring=expected_substring)
        span.update(output={"final_answer": result["final_answer"], "scores": scores})

        for name, value in scores.items():
            if isinstance(value, bool):
                client.score_current_trace(name=name, value=1.0 if value else 0.0, data_type="BOOLEAN")
            elif isinstance(value, (int, float)):
                client.score_current_trace(name=name, value=float(value), data_type="NUMERIC")

        trace_id = client.get_current_trace_id()

    client.flush()
    return {**result, "scores": scores, "trace_id": trace_id}

"""Per-session thumbs-up/down feedback capture: a real end-user signal
(simulated here, a live product would collect this from a UI button) tied
to a specific trace, not just a trajectory-derived score like Task 5's.
Uses the Langfuse Python SDK's score API for the write, the one piece of
this task that isn't pure OTel, since attaching a score to an
already-ingested trace is a management operation, not a trace.
"""

from __future__ import annotations

import os

from langfuse import Langfuse

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-tutorloop-task5-dev")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-tutorloop-task5-dev")


def _client() -> Langfuse:
    return Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)


def to_score_value(thumbs_up: bool) -> float:
    """Pure mapping, the one piece of this module worth unit testing
    without a live server: thumbs up/down -> the +1/-1 convention this
    module's score name (`user_feedback`) uses everywhere else.
    """
    return 1.0 if thumbs_up else -1.0


def record_feedback(trace_id: str, thumbs_up: bool, comment: str = "", client: Langfuse | None = None) -> None:
    client = client or _client()
    client.create_score(
        trace_id=trace_id,
        name="user_feedback",
        value=to_score_value(thumbs_up),
        data_type="NUMERIC",
        comment=comment or None,
    )
    client.flush()

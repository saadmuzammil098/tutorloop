"""Latency/cost/error-rate dashboard: queries the live Langfuse API for
every trace in a session (real traces, produced by otel_tracing.py) and
aggregates the three numbers this task is meant to surface. Cost is
computed here from each trace's real token usage via cost.py rather than
read from Langfuse's own cost field, see otel_tracing.py's docstring for
why (Langfuse's automatic cost calculation needs a registered model
pricing entry, a raw custom attribute doesn't populate it).
"""

from __future__ import annotations

import os

from langfuse import Langfuse

from cost import estimate_cost_usd

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-tutorloop-task5-dev")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-tutorloop-task5-dev")


def _client() -> Langfuse:
    return Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)


def session_report(session_id: str, client: Langfuse | None = None) -> dict:
    client = client or _client()
    # trace.list()'s items carry `.observations` as bare id strings, not
    # full objects (a lighter list view than trace.get()), so token usage
    # and per-observation error levels need one trace.get() per trace.
    trace_ids = [t.id for t in client.api.trace.list(session_id=session_id, limit=100).data]

    rows = []
    for trace_id in trace_ids:
        t = client.api.trace.get(trace_id)
        root = next((o for o in t.observations if o.name == "tutorloop-task6-agent-run"), None)
        if root is None:
            continue
        usage = root.usage_details or {}
        input_tokens = usage.get("input", 0)
        output_tokens = usage.get("output", 0)
        errored = any(o.level and o.level.value == "ERROR" for o in t.observations)
        rows.append(
            {
                "trace_id": trace_id,
                "latency_s": root.latency or 0.0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": estimate_cost_usd(input_tokens, output_tokens),
                "errored": errored,
            }
        )

    n = len(rows)
    if n == 0:
        return {"session_id": session_id, "n": 0}

    return {
        "session_id": session_id,
        "n": n,
        "p50_latency_s": sorted(r["latency_s"] for r in rows)[n // 2],
        "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "avg_cost_usd": sum(r["cost_usd"] for r in rows) / n,
        "error_rate": sum(1 for r in rows if r["errored"]) / n,
        "rows": rows,
    }


def print_report(report: dict) -> None:
    if report["n"] == 0:
        print(f"No traces found for session {report['session_id']!r}")
        return
    print(f"Session {report['session_id']} — {report['n']} traces")
    print(f"  p50 latency:  {report['p50_latency_s']:.3f}s")
    print(f"  avg latency:  {report['avg_latency_s']:.3f}s")
    print(f"  total cost:   ${report['total_cost_usd']:.6f}")
    print(f"  avg cost:     ${report['avg_cost_usd']:.6f}")
    print(f"  error rate:   {report['error_rate']:.0%}")


if __name__ == "__main__":
    import sys

    session_id = sys.argv[1] if len(sys.argv) > 1 else "tutorloop-task6-demo-session"
    print_report(session_report(session_id))

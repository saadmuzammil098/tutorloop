"""OpenTelemetry instrumentation feeding self-hosted Langfuse, distinct
from Task 5's approach: Task 5 used Langfuse's own Python SDK (convenient,
but Langfuse-specific). Task 6 uses the vanilla `opentelemetry-sdk` and a
plain OTLP HTTP exporter pointed at Langfuse's native OTel ingestion
endpoint (`/api/public/otel/v1/traces`), demonstrating that any
OTel-instrumented app, not just Langfuse-SDK apps, can feed the same
self-hosted Langfuse instance. Auth is HTTP Basic (public_key:secret_key)
base64-encoded, Langfuse's documented OTLP auth scheme.

Attribute names follow the OpenTelemetry GenAI semantic conventions
(`gen_ai.*`) plus Langfuse's own `langfuse.*` extensions for
session/observation grouping, both of which Langfuse's OTel ingestion
understands and maps into its latency/cost/token dashboards automatically
(no cost inference needed from Langfuse's side, real token counts are
attached directly).
"""

from __future__ import annotations

import base64
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from agent import MODEL, run as agent_run
from cost import estimate_cost_usd

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-tutorloop-task5-dev")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-tutorloop-task5-dev")

_provider = None


def _tracer_provider() -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider

    auth = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    exporter = OTLPSpanExporter(
        endpoint=f"{LANGFUSE_HOST}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
    )
    resource = Resource.create({"service.name": "tutorloop-task6-agent"})
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    return _provider


def instrumented_run(question: str, session_id: str, chat_fn=None) -> dict:
    """Run the agent inside a real OTel span tree, one root span per
    question (as_type "agent" via langfuse.observation.type, matching
    what the Langfuse UI groups on), a child span per trajectory step,
    real token counts/cost/latency attached to the root span, and span
    status set to ERROR whenever the trajectory actually hit a tool error
    (not just logged, the OTel status itself, so Langfuse's error-rate
    dashboard reflects it without any extra plumbing).
    """
    tracer = _tracer_provider().get_tracer("tutorloop-task6")

    with tracer.start_as_current_span("tutorloop-task6-agent-run") as root:
        root.set_attribute("langfuse.session.id", session_id)
        root.set_attribute("langfuse.observation.type", "agent")
        root.set_attribute("gen_ai.system", "ollama")
        root.set_attribute("gen_ai.request.model", MODEL)
        root.set_attribute("input.value", question)

        result = agent_run(question, chat_fn=chat_fn)

        for entry in result["trajectory"]:
            step_type = entry["type"]
            with tracer.start_as_current_span(f"step-{entry['step']}-{step_type}") as step_span:
                step_span.set_attribute("langfuse.observation.type", "tool" if step_type != "final" else "span")
                for k, v in entry.items():
                    if k not in ("type", "step") and v is not None:
                        step_span.set_attribute(f"input.{k}", str(v))
                if step_type == "error":
                    step_span.set_status(Status(StatusCode.ERROR, entry["message"]))

        cost_usd = estimate_cost_usd(result["prompt_tokens"], result["completion_tokens"])
        root.set_attribute("gen_ai.usage.input_tokens", result["prompt_tokens"])
        root.set_attribute("gen_ai.usage.output_tokens", result["completion_tokens"])
        root.set_attribute("gen_ai.usage.cost_usd", cost_usd)
        root.set_attribute("gen_ai.usage.latency_seconds", result["wall_seconds"])
        root.set_attribute("output.value", result["final_answer"] or "")

        if result["had_error"]:
            root.set_status(Status(StatusCode.ERROR, "trajectory contained a tool error"))

        trace_id = format(root.get_span_context().trace_id, "032x")

    _provider.force_flush()
    return {**result, "cost_usd": cost_usd, "trace_id": trace_id, "session_id": session_id}

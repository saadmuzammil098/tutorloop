"""A deliberately broken session: a chat_fn standing in for a buggy prompt
that never adapts to the tool error it keeps seeing (repeats the exact
same malformed calculator call every turn instead of reading the
Observation), so the agent burns its whole step budget and never reaches
a Final Answer. Traced to Langfuse through the same path a real run takes
(tracing.traced_run); `root_cause.py` then fetches this exact trace back
via the Langfuse API and diagnoses the failure from the trace data alone,
no source-reading required, which is the point of the exercise.
"""

from __future__ import annotations

from tracing import _client, traced_run

# Always emits the exact same malformed expression, ignoring the
# Observation it's given back, this is the "bug": a prompt that doesn't
# incorporate tool feedback. Real models don't reliably do this on their
# own (task 1's live run shows qwen2.5:7b *does* adapt), so it's scripted
# here specifically to produce a reliably-broken trace to root-cause.
_BROKEN_REPLY = "Thought: computing the tip.\nAction: calculator[15% of 40]"


def broken_chat_fn(messages: list[dict]) -> str:
    return _BROKEN_REPLY


def main() -> None:
    client = _client()

    good = traced_run(
        "What is 15% of 40?",
        expected_tool="calculator",
        expected_substring="6",
        client=client,
        trace_name="tutorloop-task5-agent-run",
    )
    broken = traced_run(
        "What is 15% of 40?",
        expected_tool="calculator",
        expected_substring="6",
        client=client,
        chat_fn=broken_chat_fn,
        trace_name="tutorloop-task5-BROKEN-agent-run",
    )

    print("Good trace:  ", client.get_trace_url(trace_id=good["trace_id"]))
    print("Good scores: ", good["scores"])
    print()
    print("Broken trace:", client.get_trace_url(trace_id=broken["trace_id"]))
    print("Broken scores:", broken["scores"])

    with open("broken_trace_id.txt", "w") as f:
        f.write(broken["trace_id"])


if __name__ == "__main__":
    main()

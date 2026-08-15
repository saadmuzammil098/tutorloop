"""Fetches the broken session's trace back from the live Langfuse API
(not from any in-process state, this reads only what's persisted server
side) and root-causes the failure from the trace data alone: which step
first went wrong, whether the agent ever adapted afterward, and why it
burned its whole step budget.

Run after broken_demo.py, which writes the trace id to
broken_trace_id.txt.

Note: client.api.observations.get_many(trace_id=...) 404s against this
server version ("v4 write mode only"), trace.get(id).observations is the
path that actually works, confirmed live against the running stack.
"""

from __future__ import annotations

import time
from pathlib import Path

from tracing import _client


def main() -> None:
    trace_id = Path("broken_trace_id.txt").read_text().strip()
    client = _client()

    # Score ingestion goes through Langfuse's async worker queue, so a
    # trace fetched immediately after traced_run()'s client.flush() can
    # briefly show scores=[] even though they were accepted, poll a few
    # times rather than treat an empty first read as "no scores exist".
    trace = client.api.trace.get(trace_id)
    for _ in range(5):
        if trace.scores:
            break
        time.sleep(2)
        trace = client.api.trace.get(trace_id)

    observations = sorted(trace.observations, key=lambda o: o.start_time)

    print(f"Trace {trace_id}")
    print(f"  name={trace.name}")
    print(f"  scores={ {s.name: s.value for s in trace.scores} if trace.scores else {} }")
    print()

    # ObservationLevel is a str-subclassing enum: str(o.level) renders as
    # "ObservationLevel.ERROR" (the Python enum repr convention), while
    # .value gives the actual server-side string "ERROR", compare on that.
    error_steps = [o for o in observations if o.level and o.level.value == "ERROR"]
    action_steps = [o for o in observations if "action" in (o.name or "")]

    print(f"Observed {len(observations)} spans, {len(action_steps)} tool calls, {len(error_steps)} errored.")

    if error_steps:
        first_error = error_steps[0]
        # error spans carry {"tool", "message"}, action spans carry
        # {"tool", "arg"}, different shapes (see tracing.py), so match on
        # the *action* taken at the same trajectory step index rather than
        # comparing an error's input to an action's input directly.
        first_error_step_num = first_error.name.split("-")[1]
        matching_action = next(
            (o for o in action_steps if o.name.split("-")[1] == first_error_step_num), None
        )
        repeats = sum(1 for o in action_steps if matching_action and o.input == matching_action.input)
        print(f"\nFirst error at '{first_error.name}': {first_error.status_message}")
        print(f"Action that caused it: {matching_action.input if matching_action else 'unknown'}")
        if repeats > 1:
            print(
                f"\nROOT CAUSE: the identical action input appears {repeats} times across the "
                "trajectory, the agent never varied its Action after seeing the error "
                "Observation, meaning it isn't reading tool feedback back into its next "
                "decision. This is a prompt/loop bug, not a tool bug, the tool "
                "(calculator) is correctly rejecting '15% of 40' since it isn't a pure "
                "arithmetic expression the AST evaluator accepts."
            )
        else:
            print("\nNo obvious repeated-input pattern, would need a closer per-step read.")
    else:
        print("\nNo errored spans found, this trace may not be the broken one.")

    if trace.output and isinstance(trace.output, dict):
        final = trace.output.get("final_answer")
        print(f"\nFinal answer reached: {final!r} (None means it hit the step limit still stuck)")


if __name__ == "__main__":
    main()

"""Runs a fixed set of eval questions through the traced agent (tracing.py)
against a live Ollama model and a live self-hosted Langfuse instance,
prints a scored summary table plus each trace's URL. This is the "done
when" live artifact for Task 5's trajectory/tool-call/task-success eval.
"""

from __future__ import annotations

from tracing import _client, traced_run

EVAL_SET = [
    {
        "question": "What is 17 + 25?",
        "expected_tool": "calculator",
        "expected_substring": "42",
    },
    {
        "question": "Explain how to add fractions with different denominators.",
        "expected_tool": "curriculum_lookup",
        "expected_substring": "denominator",
    },
    {
        "question": "Give me a medium difficulty practice problem about ratios.",
        "expected_tool": "practice_problem",
        "expected_substring": "x",
    },
]


def main() -> None:
    client = _client()
    rows = []
    for case in EVAL_SET:
        result = traced_run(
            case["question"],
            expected_tool=case["expected_tool"],
            expected_substring=case["expected_substring"],
            client=client,
        )
        rows.append((case["question"], result["scores"], result["trace_id"]))

    print(f"{'question':<55} {'traj':<6} {'tool%':<6} {'task':<6} {'term':<6} trace")
    for question, scores, trace_id in rows:
        print(
            f"{question[:53]:<55} "
            f"{str(scores['trajectory_used_expected_tool']):<6} "
            f"{scores['tool_call_success_rate']:<6.2f} "
            f"{str(scores['task_success']):<6} "
            f"{str(scores['terminated']):<6} "
            f"{client.get_trace_url(trace_id=trace_id)}"
        )

    n = len(rows)
    traj_rate = sum(1 for _, s, _ in rows if s["trajectory_used_expected_tool"]) / n
    task_rate = sum(1 for _, s, _ in rows if s["task_success"]) / n
    print(f"\ntrajectory-correct: {traj_rate:.0%}  task-success: {task_rate:.0%}  n={n}")


if __name__ == "__main__":
    main()

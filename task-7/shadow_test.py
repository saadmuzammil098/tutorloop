"""Shadow-testing a candidate tutoring prompt: run it against a fixed set
of eval questions the same way a real session would (same agent.run()
code path, see agent.py's injectable `system`), score both the current
production prompt and the candidate with the same online_eval checks, and
decide whether the candidate is even eligible to be considered for a real
rollout, all without a single real student ever seeing the candidate's
output. `decide_rollout()` is the one part of this worth unit testing
without a live model, the aggregation and threshold logic; the actual
prompt comparison is inherently a live run, see `main()`.
"""

from __future__ import annotations

from agent import DEFAULT_SYSTEM, run as agent_run
from online_eval import heuristic_pass, judge_score

SHADOW_QUESTIONS = [
    "What is 8 + 15?",
    "Explain what a fraction is.",
    "Give me an easy practice problem about linear equations.",
]

# A candidate variant: explicitly asks for brevity, testing whether a
# shorter-answer prompt still passes the same eval bar as production.
CANDIDATE_SYSTEM = DEFAULT_SYSTEM + "\n\nKeep your Final Answer to two sentences or fewer."


def run_variant(system: str, questions: list[str] = SHADOW_QUESTIONS, chat_fn=None, judge_chat_fn=None) -> list[dict]:
    rows = []
    for q in questions:
        result = agent_run(q, system=system, chat_fn=chat_fn)
        score = judge_score(q, result["final_answer"] or "", chat_fn=judge_chat_fn)
        rows.append({"question": q, "result": result, "heuristic_pass": heuristic_pass(result), "judge_score": score})
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "heuristic_pass_rate": sum(1 for r in rows if r["heuristic_pass"]) / n,
        "avg_judge_score": sum(r["judge_score"] for r in rows) / n,
    }


def decide_rollout(baseline_agg: dict, candidate_agg: dict, judge_score_tolerance: float = 0.5) -> tuple[bool, str]:
    """Candidate must match or beat baseline's heuristic pass rate exactly
    (no regression tolerated on cheap, deterministic-ish checks) and come
    within `judge_score_tolerance` of baseline's average judge score (a
    small tolerance here since LLM-judge ratings are inherently noisy
    run to run, demanding an exact or better match would reject good
    candidates on judge noise alone).
    """
    if candidate_agg["heuristic_pass_rate"] < baseline_agg["heuristic_pass_rate"]:
        return False, (
            f"heuristic pass rate regressed: {candidate_agg['heuristic_pass_rate']:.0%} "
            f"< baseline {baseline_agg['heuristic_pass_rate']:.0%}"
        )
    if candidate_agg["avg_judge_score"] < baseline_agg["avg_judge_score"] - judge_score_tolerance:
        return False, (
            f"judge score regressed beyond tolerance: {candidate_agg['avg_judge_score']:.2f} "
            f"< baseline {baseline_agg['avg_judge_score']:.2f} - {judge_score_tolerance}"
        )
    return True, "candidate matches or beats baseline on both signals"


if __name__ == "__main__":
    baseline_rows = run_variant(DEFAULT_SYSTEM)
    candidate_rows = run_variant(CANDIDATE_SYSTEM)

    baseline_agg = aggregate(baseline_rows)
    candidate_agg = aggregate(candidate_rows)
    passed, reason = decide_rollout(baseline_agg, candidate_agg)

    print("Baseline: ", baseline_agg)
    print("Candidate:", candidate_agg)
    print(f"\nRollout decision: {'PASS' if passed else 'FAIL'} — {reason}")

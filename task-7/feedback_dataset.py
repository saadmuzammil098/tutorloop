"""Feedback-to-dataset loop: sessions that fail online eval (heuristic
check failure, low judge score, or an explicit user thumbs-down) get
appended to a JSONL dataset of "hard examples" instead of being silently
discarded, the raw material a later prompt/model iteration would actually
review. Every session is a candidate, only failing ones are kept, this is
what makes it a *feedback* loop, not just a log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from online_eval import heuristic_pass

DEFAULT_DATASET_PATH = Path(__file__).parent / "hard_examples.jsonl"
JUDGE_SCORE_THRESHOLD = 3  # below this (out of 5) counts as a failure


def should_flag(result: dict, judge_score: int | None = None, thumbs_up: bool | None = None) -> tuple[bool, list[str]]:
    """Returns (should_flag, reasons). Multiple independent reasons can
    all fire at once, all are recorded, not just the first one found.
    """
    reasons = []
    if not heuristic_pass(result):
        reasons.append("heuristic_fail")
    if judge_score is not None and judge_score < JUDGE_SCORE_THRESHOLD:
        reasons.append(f"low_judge_score:{judge_score}")
    if thumbs_up is False:
        reasons.append("user_thumbs_down")
    return (len(reasons) > 0, reasons)


def record_if_flagged(
    result: dict,
    judge_score: int | None = None,
    thumbs_up: bool | None = None,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> bool:
    """Appends one JSONL line if should_flag() fires, returns whether it did."""
    flagged, reasons = should_flag(result, judge_score=judge_score, thumbs_up=thumbs_up)
    if not flagged:
        return False

    record = {
        "question": result["question"],
        "final_answer": result.get("final_answer"),
        "trajectory": result["trajectory"],
        "judge_score": judge_score,
        "thumbs_up": thumbs_up,
        "reasons": reasons,
        "flagged_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(dataset_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return True


def load_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> list[dict]:
    if not dataset_path.exists():
        return []
    with open(dataset_path) as f:
        return [json.loads(line) for line in f if line.strip()]

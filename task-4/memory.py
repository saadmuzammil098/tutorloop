"""Long-term student memory: a small SQLite table of past practice
attempts, queried for weak topics (accuracy below a threshold) so the
tutor can recall and reference a student's history rather than treating
every session as if it were the student's first.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "student_memory.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    correct INTEGER NOT NULL,
    note TEXT,
    ts TEXT NOT NULL
)
"""


def _connect(db_path=DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def record_attempt(
    student_id: str, topic: str, correct: bool, note: str = "", db_path=DB_PATH
) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO attempts (student_id, topic, correct, note, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (student_id, topic, int(correct), note, datetime.now(timezone.utc).isoformat()),
        )
    conn.close()


def weak_topics(
    student_id: str, db_path=DB_PATH, min_attempts: int = 1, threshold: float = 0.5
) -> list[dict]:
    """Topics where accuracy is below `threshold`, weakest first."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT topic, correct FROM attempts WHERE student_id = ?", (student_id,)
    ).fetchall()
    conn.close()

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    for topic, correct in rows:
        stats[topic][1] += 1
        stats[topic][0] += correct

    weak = []
    for topic, (correct_count, total) in stats.items():
        accuracy = correct_count / total
        if total >= min_attempts and accuracy < threshold:
            weak.append({"topic": topic, "accuracy": accuracy, "attempts": total})
    return sorted(weak, key=lambda w: w["accuracy"])


def history(student_id: str, topic: str | None = None, db_path=DB_PATH) -> list[dict]:
    conn = _connect(db_path)
    if topic:
        rows = conn.execute(
            "SELECT topic, correct, note, ts FROM attempts "
            "WHERE student_id = ? AND topic = ? ORDER BY ts",
            (student_id, topic),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT topic, correct, note, ts FROM attempts "
            "WHERE student_id = ? ORDER BY ts",
            (student_id,),
        ).fetchall()
    conn.close()
    return [
        {"topic": t, "correct": bool(c), "note": n, "ts": ts} for t, c, n, ts in rows
    ]

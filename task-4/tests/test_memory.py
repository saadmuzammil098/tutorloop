import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from memory import history, record_attempt, weak_topics


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test_memory.sqlite"


def test_no_attempts_means_no_weak_topics(db):
    assert weak_topics("alex", db_path=db) == []


def test_weak_topic_detected_below_threshold(db):
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "fractions", True, db_path=db)
    weak = weak_topics("alex", db_path=db, threshold=0.5)
    assert len(weak) == 1
    assert weak[0]["topic"] == "fractions"
    assert weak[0]["accuracy"] == pytest.approx(1 / 3)
    assert weak[0]["attempts"] == 3


def test_strong_topic_not_flagged_weak(db):
    record_attempt("alex", "ratios", True, db_path=db)
    record_attempt("alex", "ratios", True, db_path=db)
    assert weak_topics("alex", db_path=db) == []


def test_weak_topics_isolated_per_student(db):
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("sam", "fractions", True, db_path=db)
    record_attempt("sam", "fractions", True, db_path=db)
    assert len(weak_topics("alex", db_path=db)) == 1
    assert weak_topics("sam", db_path=db) == []


def test_weak_topics_sorted_weakest_first(db):
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "fractions", True, db_path=db)  # 50% accuracy
    record_attempt("alex", "ratios", False, db_path=db)
    record_attempt("alex", "ratios", False, db_path=db)
    record_attempt("alex", "ratios", True, db_path=db)  # 33% accuracy
    weak = weak_topics("alex", db_path=db, threshold=0.6)
    assert [w["topic"] for w in weak] == ["ratios", "fractions"]


def test_history_returns_attempts_in_order(db):
    record_attempt("alex", "fractions", False, note="mixed up numerator/denominator", db_path=db)
    record_attempt("alex", "fractions", True, db_path=db)
    h = history("alex", "fractions", db_path=db)
    assert len(h) == 2
    assert h[0]["correct"] is False
    assert h[0]["note"] == "mixed up numerator/denominator"
    assert h[1]["correct"] is True


def test_history_filters_by_topic(db):
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "ratios", True, db_path=db)
    assert len(history("alex", "fractions", db_path=db)) == 1
    assert len(history("alex", db_path=db)) == 2

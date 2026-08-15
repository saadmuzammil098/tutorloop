import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from memory import record_attempt
from tutor import explain


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test_memory.sqlite"


def _fake_retrieve(chunks):
    def retrieve_fn(query, k=2):
        return chunks

    return retrieve_fn


def _fake_chat(reply):
    def chat_fn(messages):
        return reply

    return chat_fn


FRACTIONS_CHUNK = [
    {
        "text": "A fraction is written a/b...",
        "chapter": "Fractions",
        "title": "Visualize Fractions",
        "source": "Adapted from OpenStax Prealgebra 2e, Chapter 4 (CC BY 4.0)",
        "distance": 0.1,
    }
]


def test_explain_grounds_in_retrieved_chunks(db):
    result = explain(
        "alex",
        "fractions",
        chat_fn=_fake_chat("A fraction has a numerator and denominator."),
        retrieve_fn=_fake_retrieve(FRACTIONS_CHUNK),
        db_path=db,
    )
    assert result["explanation"] == "A fraction has a numerator and denominator."
    assert result["grounded_in"] == FRACTIONS_CHUNK
    assert result["recalled_weak_topic"] is None


def test_explain_recalls_a_past_struggle(db):
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "fractions", True, db_path=db)  # 33% accuracy, weak

    captured_messages = {}

    def chat_fn(messages):
        captured_messages["messages"] = messages
        return "Since you've struggled with fractions before, let's go slowly."

    result = explain(
        "alex",
        "fractions",
        chat_fn=chat_fn,
        retrieve_fn=_fake_retrieve(FRACTIONS_CHUNK),
        db_path=db,
    )

    assert result["recalled_weak_topic"] is not None
    assert result["recalled_weak_topic"]["topic"] == "fractions"
    # The struggle must actually reach the LLM prompt, not just be computed
    # and discarded, this is what "recalls a student's earlier struggle"
    # has to mean concretely.
    user_content = captured_messages["messages"][1]["content"]
    assert "struggled with fractions" in user_content
    assert "33%" in user_content


def test_explain_no_recall_for_a_strong_topic(db):
    record_attempt("alex", "ratios", True, db_path=db)
    record_attempt("alex", "ratios", True, db_path=db)

    result = explain(
        "alex",
        "ratios",
        chat_fn=_fake_chat("Here's how ratios work."),
        retrieve_fn=_fake_retrieve(FRACTIONS_CHUNK),
        db_path=db,
    )
    assert result["recalled_weak_topic"] is None


def test_explain_recall_is_per_student(db):
    record_attempt("alex", "fractions", False, db_path=db)
    record_attempt("alex", "fractions", False, db_path=db)

    result_for_sam = explain(
        "sam",
        "fractions",
        chat_fn=_fake_chat("Fractions explanation."),
        retrieve_fn=_fake_retrieve(FRACTIONS_CHUNK),
        db_path=db,
    )
    assert result_for_sam["recalled_weak_topic"] is None

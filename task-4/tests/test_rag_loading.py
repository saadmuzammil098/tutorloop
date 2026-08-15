import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import load_sections


def test_load_sections_returns_all_three_chapters():
    sections = load_sections()
    chapters = {s["chapter"] for s in sections}
    assert chapters == {"Solving Linear Equations", "Fractions", "Ratios and Proportions"}


def test_load_sections_have_required_fields():
    sections = load_sections()
    assert len(sections) >= 6
    for s in sections:
        assert s["id"]
        assert s["title"]
        assert s["text"]
        assert "OpenStax" in s["source"]


def test_load_sections_ids_are_unique():
    sections = load_sections()
    ids = [s["id"] for s in sections]
    assert len(ids) == len(set(ids))

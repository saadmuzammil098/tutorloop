import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feedback import to_score_value


def test_thumbs_up_is_positive_one():
    assert to_score_value(True) == 1.0


def test_thumbs_down_is_negative_one():
    assert to_score_value(False) == -1.0

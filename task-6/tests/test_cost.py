import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from cost import estimate_cost_usd


def test_zero_tokens_zero_cost():
    assert estimate_cost_usd(0, 0) == 0.0


def test_cost_scales_with_prompt_tokens():
    assert estimate_cost_usd(1_000_000, 0) == pytest.approx(0.15)


def test_cost_scales_with_completion_tokens():
    assert estimate_cost_usd(0, 1_000_000) == pytest.approx(0.60)


def test_completion_tokens_cost_more_per_token_than_prompt_tokens():
    # Matches every major hosted provider's pricing shape: output tokens
    # cost more than input tokens, a dashboard built on a cost model that
    # got this backwards would be misleading by construction.
    cost_per_prompt_token = estimate_cost_usd(1, 0)
    cost_per_completion_token = estimate_cost_usd(0, 1)
    assert cost_per_completion_token > cost_per_prompt_token


def test_cost_is_additive():
    combined = estimate_cost_usd(1000, 500)
    separate = estimate_cost_usd(1000, 0) + estimate_cost_usd(0, 500)
    assert combined == pytest.approx(separate)

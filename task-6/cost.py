"""Cost estimation, kept as pure functions over real token counts (not
guesses from text length, agent.py now returns Ollama's actual
prompt_eval_count/eval_count), so it's testable offline and reusable by
both otel_tracing.py (per-call spans) and dashboard.py (aggregate report).

Ollama/qwen2.5:7b is local and free, there is no real per-token bill. The
nominal rate below stands in for "what this would cost against a hosted
API of comparable size" (roughly OpenAI gpt-4o-mini's public per-token
pricing order of magnitude at the time this was written), so the
latency/cost/error-rate dashboard has a non-degenerate cost column to
actually demonstrate, rather than a column of zeros.
"""

from __future__ import annotations

# USD per token, deliberately a round, clearly-labeled nominal rate, not a
# claim about any specific provider's real current pricing.
NOMINAL_PROMPT_RATE = 0.15 / 1_000_000
NOMINAL_COMPLETION_RATE = 0.60 / 1_000_000


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * NOMINAL_PROMPT_RATE + completion_tokens * NOMINAL_COMPLETION_RATE

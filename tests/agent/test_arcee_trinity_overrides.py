"""Tests for Arcee Trinity Large Thinking per-model overrides.

Arcee Trinity Large Thinking is a reasoning model that wants:
- Fixed temperature=0.5 (vs the global default)
- Compression threshold=0.75 (delay compression to preserve reasoning context)

The helpers must match the bare model name, including when it arrives via
OpenRouter as ``arcee-ai/trinity-large-thinking``, but must NOT hit sibling
Arcee models like trinity-large-preview or trinity-mini.
"""

from __future__ import annotations

import pytest

from agent.auxiliary_client import (
    _compression_threshold_for_model,
    _fixed_temperature_for_model,
    _is_arcee_trinity_thinking,
)
from agent.agent_init import _resolve_compression_threshold


@pytest.mark.parametrize(
    "model",
    [
        "trinity-large-thinking",
        "arcee-ai/trinity-large-thinking",
        "Arcee-AI/Trinity-Large-Thinking",  # case-insensitive
        "  trinity-large-thinking  ",  # whitespace tolerant
    ],
)
def test_is_arcee_trinity_thinking_matches(model: str) -> None:
    assert _is_arcee_trinity_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "trinity-large-preview",
        "arcee-ai/trinity-large-preview:free",
        "trinity-mini",
        "arcee-ai/trinity-mini",
        "trinity-large",  # prefix-only must not match
        "claude-sonnet-4.6",
        "gpt-5.4",
    ],
)
def test_is_arcee_trinity_thinking_rejects_non_matches(model) -> None:
    assert _is_arcee_trinity_thinking(model) is False


def test_fixed_temperature_for_trinity_thinking() -> None:
    assert _fixed_temperature_for_model("trinity-large-thinking") == 0.5
    assert _fixed_temperature_for_model("arcee-ai/trinity-large-thinking") == 0.5


def test_fixed_temperature_sibling_arcee_models_unaffected() -> None:
    # Preview and mini do not pin temperature — caller chooses its default.
    assert _fixed_temperature_for_model("trinity-large-preview") is None
    assert _fixed_temperature_for_model("trinity-mini") is None


def test_compression_threshold_for_trinity_thinking() -> None:
    assert _compression_threshold_for_model("trinity-large-thinking") == 0.75
    assert _compression_threshold_for_model("arcee-ai/trinity-large-thinking") == 0.75


def test_compression_threshold_default_none_for_other_models() -> None:
    # None means "leave the user's config value unchanged".
    assert _compression_threshold_for_model(None) is None
    assert _compression_threshold_for_model("") is None
    assert _compression_threshold_for_model("trinity-large-preview") is None
    assert _compression_threshold_for_model("claude-sonnet-4.6") is None
    assert _compression_threshold_for_model("kimi-k2") is None


def test_codex_responses_threshold_overrides_global_threshold() -> None:
    threshold = _resolve_compression_threshold(
        {
            "threshold": 0.50,
            "codex_responses_threshold": 0.85,
        },
        model="gpt-5-codex",
        provider="openai-codex",
        api_mode="codex_responses",
    )

    assert threshold == 0.85


def test_codex_responses_threshold_does_not_affect_other_routes() -> None:
    threshold = _resolve_compression_threshold(
        {
            "threshold": 0.50,
            "codex_responses_threshold": 0.85,
        },
        model="gpt-5-codex",
        provider="openai",
        api_mode="chat_completions",
    )

    assert threshold == 0.50


def test_codex_responses_threshold_can_be_lowered_explicitly() -> None:
    threshold = _resolve_compression_threshold(
        {
            "threshold": 0.50,
            "codex_responses_threshold": 0.70,
        },
        model="gpt-5.5",
        provider="openai-codex",
        api_mode="codex_responses",
    )

    assert threshold == 0.70


def test_codex_responses_threshold_defaults_to_85_percent() -> None:
    threshold = _resolve_compression_threshold(
        {
            "threshold": 0.50,
        },
        model="gpt-5-codex",
        provider="openai-codex",
        api_mode="codex_responses",
    )

    assert threshold == 0.85


def test_codex_responses_threshold_does_not_disable_trinity_override() -> None:
    assert (
        _compression_threshold_for_model("trinity-large-thinking", "openrouter")
        == 0.75
    )

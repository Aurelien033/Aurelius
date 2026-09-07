"""Tests for LLM-backed debate voices."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.memory.amc_runtime_cache import AMCMemoryBlock, TrustState
from src.memory.llm_debate_voices import LLMConfig, LLMDebateVoices
from src.memory.memory_debate import DebateDecision


def _block(block_id: str = "test-1") -> AMCMemoryBlock:
    return AMCMemoryBlock(
        block_id=block_id,
        tokens=(1, 2, 3, 4, 5),
        trust_state=TrustState.UNVERIFIED,
        provenance="test_source",
        quarantine_state="",
        revocation_epoch=0,
    )


def test_llm_config_requires_api_key() -> None:
    with pytest.raises(ValueError, match="api_key must be provided"):
        LLMConfig(api_key=None)


@patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"})
def test_llm_config_reads_from_env_var() -> None:
    cfg = LLMConfig()
    assert cfg.api_key == "test-key"


def test_llm_debate_voices_propose() -> None:
    with patch("src.memory.llm_debate_voices.httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "This is a strong argument for admission."}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        voices = LLMDebateVoices(LLMConfig(api_key="test-key"))
        result = voices.propose(_block())

        assert result == "This is a strong argument for admission."
        mock_client.post.assert_called_once()


def test_llm_debate_voices_skeptic() -> None:
    with patch("src.memory.llm_debate_voices.httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "I disagree with the proposer."}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        voices = LLMDebateVoices(LLMConfig(api_key="test-key"))
        result = voices.skeptic(_block(), "proposer says yes")

        assert result == "I disagree with the proposer."


def test_llm_debate_voices_judge_admit() -> None:
    with patch("src.memory.llm_debate_voices.httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decision": "admit", "reason": "Good quality", "judge_confidence": 0.9}'
                    }
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        voices = LLMDebateVoices(LLMConfig(api_key="test-key"))
        verdict = voices.judge(_block(), "prop", "sk")

        assert verdict.decision == DebateDecision.ADMIT
        assert verdict.reason == "Good quality"
        assert verdict.judge_confidence == 0.9


def test_llm_debate_voices_judge_fallback_on_malformed_json() -> None:
    with patch("src.memory.llm_debate_voices.httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "I cannot decide right now."}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        voices = LLMDebateVoices(LLMConfig(api_key="test-key"))
        verdict = voices.judge(_block(), "prop", "sk")

        # Should default to quarantine with 0.0 confidence
        assert verdict.decision == DebateDecision.QUARANTINE
        assert "malformed" in verdict.reason.lower()
        assert verdict.judge_confidence == 0.0

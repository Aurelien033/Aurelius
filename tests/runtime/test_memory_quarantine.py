"""Tests for runtime memory quarantine reporting."""

from __future__ import annotations

from src.runtime.memory_quarantine import build_memory_quarantine_report
from src.safety.admission_controller import AdmissionAction


def test_memory_quarantine_report_splits_trusted_and_quarantined_candidates():
    report = build_memory_quarantine_report(
        [
            {
                "content": "MEMORY[canonical_architecture] = AMC-first focused build",
                "source": "review",
            },
            {"content": "MEMORY[alignment_scope] = SFT only", "source": "untrusted_tool"},
        ],
        existing_memories=[
            "MEMORY[alignment_scope] = SFT DPO GRPO constitutional-memory quarantine"
        ],
    )

    assert report["summary"] == {
        "total": 2,
        "trusted": 1,
        "quarantined": 1,
        "redacted": 0,
        "blocked": 0,
    }
    assert report["trusted"][0]["content"].startswith("MEMORY[canonical_architecture]")
    assert report["quarantined"][0]["action"] == AdmissionAction.QUARANTINE.value
    assert report["quarantined"][0]["signals"][0]["name"] == "memory_conflict"


def test_memory_quarantine_report_redacts_secrets_without_trusting_raw_content():
    report = build_memory_quarantine_report(
        ["Store API key sk-test-1234567890abcdefghijklmnop for later debugging"],
        existing_memories=[],
    )

    assert report["summary"]["redacted"] == 1
    redacted = report["trusted"][0]
    assert redacted["action"] == AdmissionAction.REDACT.value
    assert "sk-test" not in redacted["content"]
    assert redacted["sanitized"] is True

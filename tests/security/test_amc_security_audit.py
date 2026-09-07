"""Adversarial AMC memory safety audit tests (T29)."""

from __future__ import annotations

import json

import pytest

from src.security.amc_security_audit import AMCSecurityAudit


@pytest.fixture
def audit() -> AMCSecurityAudit:
    return AMCSecurityAudit()


def test_run_all_returns_six_results(audit: AMCSecurityAudit) -> None:
    results = audit.run_all()
    assert len(results) == 6
    names = {item.test for item in results}
    assert names == {
        "forge_verification",
        "mutation_after_verify",
        "secret_leakage",
        "memory_poisoning",
        "constitutional_integrity",
        "replay_chain",
    }


def test_all_probes_pass(audit: AMCSecurityAudit) -> None:
    results = audit.run_all()
    failed = [item for item in results if item.status != "PASS"]
    assert not failed, [f"{item.test}: {item.note}" for item in failed]


@pytest.mark.parametrize(
    "method_name",
    [
        "audit_forge_verification",
        "audit_mutation_after_verify",
        "audit_secret_leakage",
        "audit_memory_poisoning",
        "audit_constitutional_integrity",
        "audit_replay_chain",
    ],
)
def test_each_probe_passes_individually(audit: AMCSecurityAudit, method_name: str) -> None:
    getattr(audit, method_name)()
    result = audit.results[-1]
    assert result.status == "PASS", result.note


def test_to_json_serializable(audit: AMCSecurityAudit) -> None:
    audit.run_all()
    payload = audit.to_json()
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["total"] == 6
    assert loaded["all_passed"] is True


def test_forge_verification_blocks_unverified_commit() -> None:
    audit = AMCSecurityAudit()
    audit.audit_forge_verification()
    assert audit.results[-1].status == "PASS"


def test_mutation_after_verify_blocks_commit() -> None:
    audit = AMCSecurityAudit()
    audit.audit_mutation_after_verify()
    assert audit.results[-1].status == "PASS"

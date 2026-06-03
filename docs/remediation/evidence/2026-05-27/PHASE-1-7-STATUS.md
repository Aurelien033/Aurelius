# Remediation Batch Status — 2026-05-27 (Sequential Run)

## Completed This Session
| Phase | Rows | Status |
|-------|------|--------|
| 1 Auth/Perimeter | C-09, C-10, C-12, H-05, H-06, H-08, H-09, M-03*, M-09, H-18, M-16 | FIXED |
| 2 Model/Training | C-20, H-21, H-22 | FIXED |
| 4 Agent | M-05 | FIXED |
| 5 Deploy | C-05, C-06, C-07, C-11 | FIXED |
| 6 SSRF | H-23 | FIXED |
| 7 CI/Docs | C-21, H-15 | FIXED |

\*M-03 (`_sanitize_completion` Unicode) fixed in prior gateway pass — see `gateway/aurelius_api.py`.

## Previously Completed (same runbook day)
L-06, T1–T5 tranches, T4 scope matrix (H-16, H-17, NEW-01, M-13, C-08), C-03/C-04 fail-closed gateway.

## Completed This Session (continued)
| Phase | Rows | Status |
|-------|------|--------|
| Dirty-file cleanup | HLM bank exports, dreambank ruff, skill_library shim | FIXED |
| H-03 partial | agent/skill_library → src.agent shim | PARTIAL |
| H-10, H-12 | README vocab, Makefile test ignore | FIXED |
| M-08 | session-manager mutex poison | FIXED |
| NEW-06, NEW-07 | DPO/GRPO synthetic tests | NOT_REPRODUCED |

## Deferred (XL / merge-risk / not reproduced)
| Row | Reason | Updated |
|-----|--------|---------|
| H-01, H-02, C-23 | Dual-tree canonicalization — **PARTIAL** docs in `deployment/README.md` | 2026-05-27 |
| H-25, R10-01 | BFF persistence — **FIXED** SQLite store + engine snapshot | 2026-05-27 |
| H-03 | Agent consolidation — **PARTIAL** `docs/AGENT_PACKAGE.md` | 2026-05-27 |
| C-19, NEW-04 | Tokenize fix + audit script — **FIXED / PARTIAL** | 2026-05-27 |
| H-11, H-13, H-14 | CI tiers + perimeter audit — **PARTIAL** | 2026-05-27 |
| NEW-06, NEW-07 | DPO/GRPO — needs dedicated training run | — |
| R11-01 | web_browse fetcher not wired yet | — |
| Full H-25 DB | Alembic wiring to middle routes | XL remaining |
| Full H-03 | Delete divergent agent/ duplicates | XL remaining |

## Completed This Session (deferred batch)
| Row | Status |
|-----|--------|
| C-19 | FIXED — tests/training_data/test_tokenize_pipeline.py |
| NEW-04 | PARTIAL — scripts/audit_tokenized_shards.py (no local shards) |
| H-25, R10-01 | PARTIAL — ephemeral mode + UI banner |
| H-01, H-14, L-07 | PARTIAL — server/DEPRECATED.md |
| C-23, H-02 | PARTIAL — deployment/README.md |
| H-03 | PARTIAL — docs/AGENT_PACKAGE.md |
| H-11, H-13 | PARTIAL — nightly.yml + test_gateway_perimeter.py |
| C-25 | FIXED — prior session |
| L-02 | FIXED — middle rejects query-string API keys (test added) |

## Validation Summary
```text
cd middle && npm test → 74 passed
python3 -m pytest tests/security/test_checkpoint_signer.py \
  tests/serving/test_agent_cockpit.py tests/tools/test_web_tool.py \
  tests/model/test_moe_router.py tests/model/test_moe.py -q
```

No git push performed per runbook constraints.

# Dead code analysis — verified audit (2026-09-10)

> **Supersedes the May-2026 estimate** (513 files / 4.7 MB). Prior cleanup
> tranches plus four months of drift removed most of it; this audit re-verifies
> the current tree from scratch instead of reusing the old list.

## Method

`scripts/audit_dead_code.py` — static reachability audit:

1. Parse imports (`ast`) from every tracked `.py` file.
2. Resolve them to files, including `from X import sub` edges and the legacy
   namespace aliases registered in `src/namespace_aliases.py`
   (`aurelius.model` / `model` → `src.model`; same for `alignment` and `serving`).
3. BFS from every tracked file outside the target directories; whatever in a
   target directory is never reached has no static importer.
4. Exonerate the remainder with a word-boundary scan over code/config files
   (`.py`, yml, toml, cfg, ini, sh, ipynb, sql, json, Makefile, Dockerfile) so
   dynamic references (`pytest.importorskip`, `importlib`, string lookups) are
   not misread as dead. Prose docs and scan artifacts are excluded on purpose.

Reproduce:

```bash
.venv/bin/python scripts/audit_dead_code.py --targets src/model src/training
```

## Verified results (2026-09-10)

- 636 tracked `.py` files under `src/model/` + `src/training/`.
- **31** have no static importer. Of those, **25 are dynamically referenced**
  and kept (e.g. `tests/training/test_sift.py` guards its import with
  `pytest.importorskip`); **6** had zero references anywhere in the tree.
- **Deleted in this tranche (4)** — zero references in code, configs, or tests:

  | File | Lines |
  |------|-------|
  | `src/model/linear_attention_v2.py` | 575 |
  | `src/training/stochastic_depth_v2.py` | 345 |
  | `src/training/spectral_norm_regularizer.py` | 301 |
  | `src/training/soap_optimizer.py` | 246 |

  Note: `tests/training/test_adaptive_optimizer.py` exercises
  `src.training.adaptive_optimizer.SOAPOptimizer` — its own implementation.
  Nothing imported `soap_optimizer.py`.

- **Kept, flagged as documented-but-unwired (2)** — remove or wire, do not
  forget they exist:
  - `src/model/optimized_attention.py` — usage documented in
    `src/model/CUDA_OPTIMIZATION_IMPLEMENTATION.md`; delete both together if
    the optimization line is dropped.
  - `src/training/tst_trainer.py` — `docs/ARCHITECTURE.md` and
    `docs/V4_INTEGRATION_GUIDE.md` show intended use; `docs/CLAIMS_LEDGER.md`
    marks its claim (E5) as NOT WIRED.

- Safety: the pre-cleanup tree is preserved on the branch
  `archive/dead-code-pre-cleanup-2026-09-10`.

## Re-run cadence

Research code accretes. Re-run the audit before each cleanup tranche and treat
"no static importer + no dynamic reference" as the delete set. If the tests
that provide the dynamic references above are ever retired, the modules they
guard become deletable — re-run then.

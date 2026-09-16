# Evidence Manifest — NEW-01: Wide Unauth-Scoped BFF Surface

## Row Metadata
- **Row ID**: NEW-01
- **Finding title**: Wide unauth-scoped surface across middle routes
- **Theme**: T4 — BFF scope matrix (extends H-16)
- **Previous status**: OPEN
- **New status**: FIXED

## Acceptance Test
`grep -rn 'router\.(get|post|delete|put)' middle/src/routes/` shows every authenticated handler within scope guard or public allowlist.

## Reproduction Result
**Reproduced and fixed.** Fresh scan confirmed all authenticated `router.*` handlers now declare `requireScope(...)` inline or inherit `router.use(requireScope(...))`.

## Exceptions (explicit allowlist)
| File | Routes | Guard |
| --- | --- | --- |
| `auth.ts` | POST `/login`, POST `/register` | Public via `authMiddleware` |
| `health.ts` | GET `/health`, `/healthz`, `/readyz` | Public via `authMiddleware` |
| `system.ts` | all GET | `router.use(requireScope('system:admin'))` |

## Adversarial Cases Tested
See `middle/__tests__/scopes.test.ts`:
- Read-only key blocked from write routes (403)
- `eval:read` blocked from POST results (403)
- `scheduler:read` blocked from POST create (403)
- Non-admin key blocked from `auth/keys/generate` (403)
- Wildcard admin key still passes scoped routes (200)

## Validation
```text
cd middle && npm test  → 74 passed
```

## Remaining Risk
- No ESLint rule yet to fail CI on new routes missing `requireScope` (future hardening)

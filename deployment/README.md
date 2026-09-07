# Aurelius deployment images

## Canonical production image

| Image | Purpose | Entry module | Bind |
|-------|---------|--------------|------|
| **`deployment/Dockerfile`** | **Production API gateway** | `python -m gateway.aurelius_api` | `0.0.0.0:8080` (requires `AURELIUS_API_KEYS` or `AURELIUS_API_KEY`) |

Use `deployment/compose.production.yaml` or Helm charts under `deployment/helm/` for orchestrated deploys.

## Non-canonical / dev-only images

| File | Status | Notes |
|------|--------|-------|
| `Dockerfile` (repo root) | **DEV-ONLY** | Binds `127.0.0.1`; empty `CORS_ORIGINS`. Do not promote to production. |
| `deployment/Dockerfile.dev` | **DEV-ONLY** | Local iteration; not release-gated. |
| `deployment/Dockerfile.gpu` | **VARIANT** | GPU runtime layer; same entry module as production when used. |
| `server/Dockerfile` | **DEPRECATED** | Legacy Node gateway on port 7870 — use `middle/` instead (see `server/DEPRECATED.md`). |

## Python serving trees (H-02)

| Tree | Role today |
|------|------------|
| `gateway/` | **Release-critical** — used by production Dockerfiles and compose. |
| `src/serving/` | **Migration target** — parity work in progress; do not delete until cutover is documented. |

## Environment checklist (production)

- Set `AURELIUS_API_KEYS` or `AURELIUS_API_KEY` before exposing port 8080.
- Set `AURELIUS_METRICS_API_KEY` if `/metrics` is reachable outside the mesh.
- BFF (`middle/`) persists to SQLite when `MIDDLE_DATABASE_URL` is set (e.g. `sqlite:./data/middle.sqlite`). Without it, ephemeral in-memory mode applies (H-25).

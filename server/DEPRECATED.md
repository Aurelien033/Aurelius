# Deprecated: `server/` gateway

The `server/` workspace is a **legacy Node.js BFF** (default port **7870**) that predates the canonical `middle/` BFF (port **3001**).

## Status

- **Do not add new routes here.** Use `middle/src/routes/`.
- License routes (`/api/license/*`) exist only under `server/`; the frontend `LicenseGate` component is currently unused.
- No CI job runs against this workspace (H-14) — intentional while deprecated.

## Migration path (H-01)

1. Port any still-needed routes from `server/src/routes/` into `middle/`.
2. Point frontend proxy/Vite config at `middle` only.
3. Remove `server/` and its duplicate `package-lock.json` (L-07).

Until removal, treat `server/` as read-only archive.

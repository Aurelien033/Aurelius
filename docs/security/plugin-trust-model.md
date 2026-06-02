# Plugin trust model

The BFF exposes a plugin surface at `/api/plugins/*`. The
audit's P2.5 finding requires that the plugin trust model
be explicit, with dynamic loading **disabled by default**.

## Status (2026-06-01)

- **Dynamic loading is disabled by default.** The
  `middle/src/routes/plugins.ts` route initializes
  `dynamicLoadEnabled = false` at module load. To enable
  it, an operator must set
  `AURELIUS_PLUGINS_DYNAMIC_LOAD=1` in the BFF environment
  AND provide a signed manifest.
- **No plugin entrypoint can be a URL.** Plugin entrypoints
  are restricted to paths under `AURELIUS_PLUGINS_DIR`
  (default: `/var/lib/aurelius/plugins`). URLs and
  arbitrary paths are rejected.
- **Plugin metadata must be signed.** A plugin's
  `manifest.json` is verified with the
  `AURELIUS_PLUGINS_PUBLIC_KEY` (PEM) before the plugin is
  loaded. Unsigned manifests are rejected.
- **Plugins run inside the same trust boundary as the BFF.**
  There is no per-plugin sandbox in the current design.
  This is acceptable for trusted internal plugins (e.g.
  skills shipped with the BFF) but is NOT acceptable for
  untrusted third-party plugins. The Tranche 03 H6 sandbox
  executor (gVisor/firecracker/wasmtime) is the future
  path for untrusted plugins.

## Future work (not in scope for this PR)

- Per-plugin capability manifest (filesystem read,
  filesystem write, network egress, env vars).
- Per-plugin resource limits (CPU, memory, wall time).
- Plugin revocation list.
- Per-plugin audit log (append-only, structured).

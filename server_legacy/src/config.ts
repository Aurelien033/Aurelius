// Decommissioned: 2026-06-01
// The legacy gateway config has been replaced by the Python unified
// server's environment-driven config. The legacy `loadConfig` is gone.

throw new Error(
  "Aurelius legacy gateway config is decommissioned as of 2026-06-01. " +
  "Use environment variables consumed by src.serving.aurelius_server " +
  "(see gateway/aurelius_server.py and README)."
);

// Decommissioned: 2026-06-01
// The legacy HTTP route registry has been replaced by the Python
// unified server's API surface (gateway/aurelius_server.py). The
// legacy `registerRoutes` is no longer wired into any live app.

throw new Error(
  "Aurelius legacy route registry is decommissioned as of 2026-06-01. " +
  "All HTTP routes now live in gateway/aurelius_server.py."
);

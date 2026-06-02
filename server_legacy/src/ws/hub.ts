// Decommissioned: 2026-06-01
// The legacy WebSocket hub has been replaced by the FastAPI WebSocket
// endpoint in gateway/aurelius_api.py and the Python SSE stream in
// gateway/aurelius_server.py. The hub is no longer reachable.

throw new Error(
  "Aurelius legacy WS hub is decommissioned as of 2026-06-01. " +
  "WebSocket connections now go to /ws on the FastAPI service."
);

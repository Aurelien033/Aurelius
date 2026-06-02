// Decommissioned: 2026-06-01
// The legacy Node/Express gateway has been superseded by the Python
// unified server in `gateway/aurelius_server.py` and the FastAPI
// service in `gateway/aurelius_api.py`. This directory is preserved
// only as historical reference. See `server_legacy/DEPRECATED.md`.

throw new Error(
  "Aurelius legacy Node server is decommissioned as of 2026-06-01. " +
  "Use `python -m src.serving.aurelius_server` (HTTP) or " +
  "`uvicorn gateway.aurelius_api:app` (FastAPI) instead."
);

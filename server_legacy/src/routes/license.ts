// Decommissioned: 2026-06-01
// The legacy license route was removed because it (a) was the C2
// vulnerability source (derived the runtime api_key from the user-
// supplied license_key) and (b) belonged to the legacy Node server
// that is no longer the active runtime. The Python unified server
// has its own license handler in gateway/aurelius_server.py that
// does not derive the api_key from user input.

throw new Error(
  "Aurelius legacy license route is decommissioned as of 2026-06-01. " +
  "License activation now lives in gateway/aurelius_server.py and " +
  "does not mutate the runtime api_key."
);

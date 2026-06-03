# BFF API Scope Matrix

Canonical permission truth table for `middle/` route handlers.
Every authenticated route must declare `requireScope(...)` (or be listed in the public allowlist in `authMiddleware`).

Wildcard scope `*` (admin keys) satisfies any scope check.

## Scope groups

| Scope | Sensitivity | Description |
| --- | --- | --- |
| `activity:read` | read | List activity feed entries |
| `activity:write` | write | Append activity entries |
| `agents:read` | read | List agents, heartbeats stream, agent detail |
| `agents:write` | write | Register, heartbeat, delete agents |
| `auth:read` | read | List users; list own API key prefixes |
| `auth:admin` | admin | Mint or revoke API keys |
| `brain:execute` | execute | Run neural brain / upgrade subprocesses |
| `brain:read` | read | Read brain upgrade status |
| `chat:read` | read | List models, conversations |
| `chat:write` | write | Chat completions and agent routing |
| `config:read` | read | Read configuration keys |
| `config:write` | write | Update configuration |
| `eval:read` | read | Read evaluation results and summaries |
| `eval:write` | write | Post evaluation results |
| `files:read` | read | List and download uploaded files |
| `files:write` | write | Upload and delete files |
| `logs:read` | read | Query logs |
| `logs:write` | write | Append log entries |
| `memory:read` | read | Read memory layers and entries |
| `memory:write` | write | Add memory entries |
| `models:read` | read | Proxy upstream model list |
| `notifications:read` | read | List notification stats |
| `notifications:write` | write | Create or mark notifications read |
| `plugins:read` | read | List plugin catalog |
| `plugins:admin` | admin | Enable/disable plugins |
| `rag:read` | read | List/search RAG documents |
| `rag:write` | write | Upload or delete RAG documents |
| `registry:read` | read | Live agent/skill registry from Python layer |
| `scheduler:read` | read | List scheduled cron tasks |
| `scheduler:admin` | admin | Create, delete, toggle scheduled tasks |
| `search:read` | read | Cross-domain search and suggestions |
| `sse:read` | read | Subscribe to SSE event stream |
| `sse:broadcast` | admin | Broadcast SSE events (also requires admin role) |
| `stats:read` | read | Dashboard stats and command aggregates |
| `system:admin` | admin | Host, env, dependency introspection |
| `traces:read` | read | List and fetch agent traces |
| `traces:write` | write | Create and update traces |

## Public routes (no API key)

| Path | Notes |
| --- | --- |
| `/health`, `/healthz`, `/readyz` | Liveness/readiness |
| `/openapi.json`, `/docs` | API docs |
| `/api/auth/login`, `/api/auth/register` | Credential exchange |

## Route map

| Route file | Method | Path | Scope |
| --- | --- | --- | --- |
| activity.ts | GET | `/` | `activity:read` |
| activity.ts | POST | `/` | `activity:write` |
| agents.ts | GET | `/` | `agents:read` |
| agents.ts | GET | `/:id` | `agents:read` |
| agents.ts | GET | `/:id/stream` | `agents:read` |
| agents.ts | POST | `/` | `agents:write` |
| agents.ts | POST | `/:id/heartbeat` | `agents:write` |
| agents.ts | DELETE | `/:id` | `agents:write` |
| auth.ts | GET | `/users` | `auth:read` |
| auth.ts | GET | `/keys` | `auth:read` |
| auth.ts | POST | `/keys/generate` | `auth:admin` |
| auth.ts | DELETE | `/keys/:prefix` | `auth:admin` |
| brain.ts | POST | `/think` | `brain:execute` |
| brain.ts | GET | `/stats` | `brain:execute` |
| brain.ts | POST | `/upgrade/run` | `brain:execute` |
| brain.ts | GET | `/upgrade/status` | `brain:read` |
| chat.ts | POST | `/completions` | `chat:write` |
| chat.ts | GET | `/models` | `chat:read` |
| chat.ts | POST | `/agent` | `chat:write` |
| chat.ts | GET | `/conversations` | `chat:read` |
| config.ts | GET | `/` | `config:read` |
| config.ts | GET | `/:key` | `config:read` |
| config.ts | POST | `/` | `config:write` |
| config.ts | PUT | `/:key` | `config:write` |
| evaluation.ts | GET | `/results` | `eval:read` |
| evaluation.ts | GET | `/results/:benchmark` | `eval:read` |
| evaluation.ts | GET | `/summary` | `eval:read` |
| evaluation.ts | POST | `/results` | `eval:write` |
| files.ts | POST | `/upload` | `files:write` |
| files.ts | GET | `/files` | `files:read` |
| files.ts | GET | `/files/:id` | `files:read` |
| files.ts | DELETE | `/files/:id` | `files:write` |
| logs.ts | GET | `/` | `logs:read` |
| logs.ts | POST | `/` | `logs:write` |
| memory.ts | GET | `/layers` | `memory:read` |
| memory.ts | GET | `/entries` | `memory:read` |
| memory.ts | POST | `/entries` | `memory:write` |
| models.ts | GET | `/` | `models:read` |
| notifications.ts | GET | `/` | `notifications:read` |
| notifications.ts | GET | `/stats` | `notifications:read` |
| notifications.ts | POST | `/` | `notifications:write` |
| notifications.ts | POST | `/:id/read` | `notifications:write` |
| notifications.ts | POST | `/read-all` | `notifications:write` |
| plugins.ts | GET | `/` | `plugins:read` |
| plugins.ts | GET | `/:id` | `plugins:read` |
| plugins.ts | PATCH | `/:id` | `plugins:admin` |
| rag.ts | GET | `/documents` | `rag:read` |
| rag.ts | GET | `/documents/:id` | `rag:read` |
| rag.ts | GET | `/search` | `rag:read` |
| rag.ts | POST | `/documents` | `rag:write` |
| rag.ts | DELETE | `/documents/:id` | `rag:write` |
| registry.ts | GET | `/agents` | `registry:read` |
| registry.ts | GET | `/agents/:id` | `registry:read` |
| registry.ts | GET | `/skills` | `registry:read` |
| registry.ts | GET | `/skills/:id` | `registry:read` |
| scheduler.ts | GET | `/` | `scheduler:read` |
| scheduler.ts | POST | `/` | `scheduler:admin` |
| scheduler.ts | DELETE | `/:id` | `scheduler:admin` |
| scheduler.ts | POST | `/:id/toggle` | `scheduler:admin` |
| search.ts | GET | `/` | `search:read` |
| search.ts | GET | `/suggestions` | `search:read` |
| sse.ts | GET | `/events` | `sse:read` |
| sse.ts | POST | `/events/broadcast` | `sse:broadcast` + `requireAdmin` |
| stats.ts | GET | `/` | `stats:read` |
| stats.ts | GET | `/activity-hourly` | `stats:read` |
| stats.ts | GET | `/commands` | `stats:read` |
| system.ts | GET | `/`, `/env`, `/dependencies` | `system:admin` (router-level) |
| traces.ts | GET | `/` | `traces:read` |
| traces.ts | GET | `/:id` | `traces:read` |
| traces.ts | POST | `/` | `traces:write` |
| traces.ts | POST | `/:id/step` | `traces:write` |
| traces.ts | PATCH | `/:id` | `traces:write` |

## Server-level routes (`server.ts`)

| Method | Path | Guard |
| --- | --- | --- |
| POST | `/api/command` | `requireAdmin` (scheduler cron must pass `X-API-Key`) |

## Regression test keys

Tests register scope-restricted keys via `registerApiKey`:

| Key | Scopes | Used to prove |
| --- | --- | --- |
| `test-readonly-key` | `activity:read` | Read-only key blocked from write routes |
| `test-eval-read-key` | `eval:read` | Eval write rejected with 403 |
| `test-scheduler-read-key` | `scheduler:read` | Scheduler admin POST rejected with 403 |

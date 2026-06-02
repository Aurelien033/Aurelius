import 'dotenv/config'

export type ChatBackend = 'mock' | 'vllm' | 'agentic'

const VALID_CHAT_BACKENDS: ChatBackend[] = ['mock', 'vllm', 'agentic']

export function normalizeChatBackend(
  value: string | null | undefined,
  fallback: ChatBackend = 'mock',
): ChatBackend {
  const normalized = value?.trim().toLowerCase()
  if (normalized && VALID_CHAT_BACKENDS.includes(normalized as ChatBackend)) {
    return normalized as ChatBackend
  }
  return fallback
}

export const config = {
  host: process.env.MIDDLE_HOST || '0.0.0.0',
  port: parseInt(process.env.MIDDLE_PORT || '3001', 10),
  upstreamUrl: process.env.UPSTREAM_URL || 'http://127.0.0.1:8080',
  vllmUpstreamUrl: process.env.AURELIUS_VLLM_URL || process.env.UPSTREAM_URL || 'http://127.0.0.1:8080',
  agenticUpstreamUrl:
    process.env.AURELIUS_AGENTIC_URL || process.env.UPSTREAM_URL || 'http://127.0.0.1:8080',
  defaultChatBackend: normalizeChatBackend(process.env.AURELIUS_DEFAULT_CHAT_BACKEND, 'mock'),
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379/0',
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  apiKey: process.env.AURELIUS_API_KEY || '',
  logLevel: process.env.MIDDLE_LOG_LEVEL || 'info',
  rateLimitRps: parseInt(process.env.RATE_LIMIT_RPS || '60', 10),
  rateLimitWindowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10),
  allowPublicRegistration: process.env.ALLOW_PUBLIC_REGISTRATION === 'true' ? true : false,
  serviceApiKey: process.env.AURELIUS_SERVICE_KEY || process.env.AURELIUS_API_KEY || '',
  // WS origin allowlist. CSV of allowed Origin header values. The
  // pre-remediation handler accepted any Origin, which let a
  // malicious page issue authenticated WS upgrades against a
  // victim's running browser. The default mirrors corsOrigin.
  wsOriginAllowlist: (process.env.AURELIUS_WS_ORIGINS
    || process.env.CORS_ORIGIN
    || 'http://localhost:5173')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean),
  // Auth mode. Production default is 'session' (cookie-based).
  // Developers can opt into 'local_byok_dev' which keeps the
  // long-lived API key in sessionStorage and shows a
  // visible warning. The 'local_byok_dev' mode is for local
  // development only; the BFF startup will refuse to
  // boot in 'local_byok_dev' mode if NODE_ENV=production.
  authMode: (process.env.AURELIUS_AUTH_MODE === 'local_byok_dev'
    ? 'local_byok_dev'
    : 'session') as 'session' | 'local_byok_dev',
  // Session signing secret. Rotate via env var in
  // production. A default marker is used in dev so the
  // suite can run in CI.
  sessionSecret: process.env.AURELIUS_SESSION_SECRET || '',
}

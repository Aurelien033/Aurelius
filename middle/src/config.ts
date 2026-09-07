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
  corsOrigin: process.env.CORS_ORIGIN ?? (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5173'),
  apiKey: process.env.AURELIUS_API_KEY || '',
  logLevel: process.env.MIDDLE_LOG_LEVEL || 'info',
  rateLimitRps: parseInt(process.env.RATE_LIMIT_RPS || '60', 10),
  rateLimitWindowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10),
  allowPublicRegistration: process.env.ALLOW_PUBLIC_REGISTRATION === 'true' ? true : false,
  serviceApiKey: process.env.AURELIUS_SERVICE_KEY || process.env.AURELIUS_API_KEY || '',
}

export function validateConfig(cfg: typeof config): void {
  const errors: string[] = []

  if (isNaN(cfg.port) || cfg.port < 1 || cfg.port > 65535) {
    errors.push(`MIDDLE_PORT must be an integer in 1..65535, got: ${process.env.MIDDLE_PORT}`)
  }

  if (!cfg.host || !cfg.host.trim()) {
    errors.push('MIDDLE_HOST must be a non-empty bind address')
  }

  const dbUrl = process.env.DATABASE_URL
  if (dbUrl) {
    try {
      new URL(dbUrl)
    } catch {
      errors.push(`DATABASE_URL is not a valid URL: ${dbUrl}`)
    }
  }

  if (cfg.corsOrigin) {
    for (const origin of cfg.corsOrigin.split(',').map((s) => s.trim()).filter(Boolean)) {
      try {
        new URL(origin)
      } catch {
        errors.push(`CORS_ORIGIN contains an invalid origin: ${origin}`)
      }
    }
  }

  if (isNaN(cfg.rateLimitRps) || cfg.rateLimitRps < 1) {
    errors.push(`RATE_LIMIT_RPS must be >= 1, got: ${process.env.RATE_LIMIT_RPS}`)
  }

  if (errors.length > 0) {
    console.error('[middle] Configuration errors:\n' + errors.map((e) => `  - ${e}`).join('\n'))
    process.exit(1)
  }
}

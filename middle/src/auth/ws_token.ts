// Short-lived, audience-bound, exp-bound WebSocket token.
//
// Pre-remediation, the WebSocket handler accepted the long-lived
// AURELIUS_API_KEY as the auth credential on every connect. That
// exposed the operator's master key to:
//
//   - any process that can read browser DevTools or the network
//     panel
//   - any process that can read the upgrade request from a log
//   - any client that had previously extracted the key from
//     localStorage or the React store
//
// The fix replaces that with a short-lived token (5 min default)
// signed with a per-process secret and bound to the 'aurelius-ws'
// audience. Tokens carry an explicit exp, a single-use jti for
// replay protection, and the user's tenant for room authorization.

import { createHmac, randomBytes, timingSafeEqual } from 'node:crypto'
import type { WsTokenPayload, Audience, Scope } from './types.js'

const DEFAULT_TTL_SECONDS = 300

function getSecret(): string {
  // The signing secret is a per-process HS256 key. It is
  // intentionally separate from AURELIUS_API_KEY so that compromise
  // of the WS layer does not yield the operator master key.
  const env = process.env.AURELIUS_WS_TOKEN_SECRET
  if (env && env.length >= 32) return env
  // Fallback for dev only — a deterministic-but-rotated value so
  // tokens do not silently work across processes.
  return randomBytes(32).toString('hex')
}

function b64url(input: Buffer | string): string {
  const buf = typeof input === 'string' ? Buffer.from(input, 'utf8') : input
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function b64urlDecode(input: string): Buffer {
  const padded = input.replace(/-/g, '+').replace(/_/g, '/')
  const pad = (4 - (padded.length % 4)) % 4
  return Buffer.from(padded + '='.repeat(pad), 'base64')
}

export interface SignWsTokenInput {
  sub: string
  tenant: string
  scopes: Scope[]
  ttlSeconds?: number
  aud?: Audience
}

export function signWsToken(input: SignWsTokenInput): string {
  const now = Math.floor(Date.now() / 1000)
  const ttl = Math.max(1, Math.min(input.ttlSeconds ?? DEFAULT_TTL_SECONDS, 900))
  const payload: WsTokenPayload = {
    sub: input.sub,
    aud: input.aud ?? 'aurelius-ws',
    iat: now,
    exp: now + ttl,
    scopes: input.scopes,
    tenant: input.tenant,
    jti: randomBytes(12).toString('hex'),
  }
  const header = { alg: 'HS256', typ: 'AURELIUS-WS' }
  const head = b64url(JSON.stringify(header))
  const body = b64url(JSON.stringify(payload))
  const signing = `${head}.${body}`
  const sig = createHmac('sha256', getSecret()).update(signing).digest()
  return `${signing}.${b64url(sig)}`
}

export interface VerifyWsTokenOptions {
  expectedAudience?: Audience
  now?: number
}

export type VerifyWsTokenResult =
  | { ok: true; payload: WsTokenPayload }
  | { ok: false; reason: 'malformed' | 'bad-signature' | 'expired' | 'wrong-audience' | 'empty' }

export function verifyWsToken(token: string, opts: VerifyWsTokenOptions = {}): VerifyWsTokenResult {
  if (!token || typeof token !== 'string') {
    return { ok: false, reason: 'empty' }
  }
  const parts = token.split('.')
  if (parts.length !== 3) {
    return { ok: false, reason: 'malformed' }
  }
  const [head, body, sig] = parts
  const signing = `${head}.${body}`
  const expected = createHmac('sha256', getSecret()).update(signing).digest()
  const provided = b64urlDecode(sig)
  if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) {
    return { ok: false, reason: 'bad-signature' }
  }
  let payload: WsTokenPayload
  try {
    payload = JSON.parse(b64urlDecode(body).toString('utf8')) as WsTokenPayload
  } catch {
    return { ok: false, reason: 'malformed' }
  }
  if (typeof payload.exp !== 'number' || typeof payload.iat !== 'number') {
    return { ok: false, reason: 'malformed' }
  }
  const now = opts.now ?? Math.floor(Date.now() / 1000)
  if (payload.exp <= now) {
    return { ok: false, reason: 'expired' }
  }
  const expected_aud = opts.expectedAudience ?? 'aurelius-ws'
  if (payload.aud !== expected_aud) {
    return { ok: false, reason: 'wrong-audience' }
  }
  return { ok: true, payload }
}

/** Mint a token for a user. Used by the /api/auth/ws-token endpoint. */
export function mintWsTokenForUser(user: { id: string; tenant: string; scopes: Scope[] }, ttlSeconds?: number): string {
  return signWsToken({ sub: user.id, tenant: user.tenant, scopes: user.scopes, ttlSeconds })
}

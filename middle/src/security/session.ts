// Centralized session management for the BFF.
//
// Pre-remediation, /api/auth/login in middle/src/routes/auth.ts
// issued a UUID bearer token via JSON, and the BFF auth
// middleware validated long-lived API keys via the
// `AURELIUS_API_KEY` env var. The audit H8 finding requires:
//
//   - Production mode: BFF session cookies (HttpOnly, Secure in
//     production, SameSite=Lax/Strict) + CSRF protection for
//     unsafe methods.
//   - Dev mode (AURELIUS_AUTH_MODE=local_byok_dev): opt-in
//     sessionStorage-only with a visible warning; localStorage
//     must remain rejected for secrets.
//   - Session store is keyed by an opaque, server-issued
//     session id; the API key is never echoed back, and is
//     never persisted by the browser.
//
// This module is the single source of truth for: session
// creation, lookup, rotation, and destruction. The
// /api/auth/login and /api/auth/logout routes use it; the
// auth middleware uses it; the /api/auth/ws-token endpoint
// uses it.

import { randomBytes, createHmac, timingSafeEqual } from 'crypto'
import { config } from '../config.js'

/**
 * The opaque session id that the BFF stores in a
 * Set-Cookie response and that the browser then presents on
 * every subsequent request.
 */
export type SessionId = string

/**
 * Authenticated principal bound to a session. Mirrors the
 * shape of req.user in the BFF middleware.
 */
export interface AuthSubject {
  id: string
  role: 'admin' | 'user' | 'agent'
  scopes: string[]
  tenant: string
  /**
   * IAT (issued-at, seconds since epoch). Used for
   * age-based invalidation.
   */
  iat: number
  /**
   * EXP (expires-at, seconds since epoch). The session
   * is invalid after this.
   */
  exp: number
  /**
   * CSRF token. The client must echo this in the
   * X-CSRF-Token header on unsafe methods.
   */
  csrf: string
}

export interface Session {
  id: SessionId
  subject: AuthSubject
  createdAt: number
  /**
   * ms-since-epoch; the session is invalid after this.
   */
  expiresAt: number
}

const SESSIONS = new Map<SessionId, Session>()

/**
 * TTL for browser session cookies. One hour by default.
 * The session is renewed on each authenticated request
 * if it's older than half its TTL.
 */
export const SESSION_TTL_MS = 60 * 60 * 1000
export const SESSION_RENEW_THRESHOLD_MS = SESSION_TTL_MS / 2

const DEV_SESSION_SECRET = 'aurelius-dev-session-secret-DO-NOT-USE-IN-PROD-XXXXXXXXXXXXXXXXXXXX'

function getSessionSecret(): string {
  // Prefer an explicit env var; fall back to a hard-coded
  // dev marker so the suite can run in CI. The BFF MUST
  // refuse to boot in production with the default secret
  // (enforced in server.ts startup).
  return process.env.AURELIUS_SESSION_SECRET || DEV_SESSION_SECRET
}

/**
 * Sign the session id so a stolen session cookie cannot
 * be forged without the server-side secret. The signature
 * is appended to the session id with a `.` separator.
 */
export function signSessionId(id: SessionId): string {
  const sig = createHmac('sha256', getSessionSecret())
    .update(id)
    .digest('base64url')
  return `${id}.${sig}`
}

/**
 * Verify a signed session id and return the underlying
 * SessionId if valid; null otherwise.
 */
export function verifySessionId(signed: string): SessionId | null {
  const dot = signed.lastIndexOf('.')
  if (dot < 1) return null
  const id = signed.slice(0, dot)
  const provided = signed.slice(dot + 1)
  const expected = createHmac('sha256', getSessionSecret())
    .update(id)
    .digest('base64url')
  // Constant-time compare
  const a = Buffer.from(provided)
  const b = Buffer.from(expected)
  if (a.length !== b.length) return null
  return timingSafeEqual(a, b) ? id : null
}

/**
 * Create a new session for the given subject. The session
 * id is signed so Set-Cookie can include a tamper-evident
 * value.
 */
export function createSession(subject: Omit<AuthSubject, 'iat' | 'exp' | 'csrf'>): {
  session: Session
  signed: string
  csrf: string
} {
  const id = randomBytes(32).toString('base64url')
  const csrf = randomBytes(24).toString('base64url')
  const now = Date.now()
  const fullSubject: AuthSubject = {
    ...subject,
    iat: Math.floor(now / 1000),
    exp: Math.floor((now + SESSION_TTL_MS) / 1000),
    csrf,
  }
  const session: Session = {
    id,
    subject: fullSubject,
    createdAt: now,
    expiresAt: now + SESSION_TTL_MS,
  }
  SESSIONS.set(id, session)
  return { session, signed: signSessionId(id), csrf }
}

/**
 * Look up a session by its signed id. Returns null if the
 * id is invalid, expired, or unknown. The session's
 * expiresAt is slid forward (sliding session) if it has
 * more than SESSION_RENEW_THRESHOLD_MS of life left.
 */
export function getSession(signed: string): Session | null {
  const id = verifySessionId(signed)
  if (!id) return null
  const session = SESSIONS.get(id)
  if (!session) return null
  const now = Date.now()
  if (session.expiresAt <= now) {
    SESSIONS.delete(id)
    return null
  }
  // Sliding window: refresh if past the threshold.
  const remaining = session.expiresAt - now
  if (remaining < SESSION_RENEW_THRESHOLD_MS) {
    session.expiresAt = now + SESSION_TTL_MS
    session.subject.exp = Math.floor(session.expiresAt / 1000)
  }
  return session
}

/**
 * Invalidate a session by its signed id. Idempotent.
 */
export function destroySession(signed: string): boolean {
  const id = verifySessionId(signed)
  if (!id) return false
  return SESSIONS.delete(id)
}

/**
 * The session cookie name. Browsers must use this exact
 * name when presenting the cookie.
 */
export const SESSION_COOKIE_NAME = 'aurelius_sid'
export const CSRF_COOKIE_NAME = 'aurelius_csrf'
export const CSRF_HEADER_NAME = 'x-csrf-token'

/**
 * Cookie attributes for the session cookie.
 *
 * - HttpOnly: true (the JS runtime cannot read the cookie)
 * - Secure: in production only (so the suite can run on
 *   http://localhost)
 * - SameSite: 'lax' (Lax allows top-level navigation from
 *   external sites; Strict would block the OAuth-style
 *   return flow)
 * - Path: '/'
 * - Max-Age: SESSION_TTL_MS in seconds
 */
export interface CookieOptions {
  name: string
  value: string
  maxAgeSeconds: number
}

export function buildSessionCookie(signed: string): CookieOptions {
  return {
    name: SESSION_COOKIE_NAME,
    value: signed,
    maxAgeSeconds: Math.floor(SESSION_TTL_MS / 1000),
  }
}

export function buildCsrfCookie(csrf: string): CookieOptions {
  return {
    name: CSRF_COOKIE_NAME,
    value: csrf,
    maxAgeSeconds: Math.floor(SESSION_TTL_MS / 1000),
  }
}

/**
 * True when the request is in production mode. The
 * session cookie MUST be Secure when this is true; the
 * dev/CI path keeps Secure off so the suite can run on
 * http://localhost.
 */
export function isProductionMode(): boolean {
  return process.env.NODE_ENV === 'production' || config.authMode === 'session'
}

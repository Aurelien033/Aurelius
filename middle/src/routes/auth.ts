// Session-cookie-based auth routes.
//
// Pre-remediation, /api/auth/login accepted { apiKey } and
// echoed a bearer token in the response body. The audit H8
// finding requires:
//
//   - The login response sets a session cookie (HttpOnly,
//     Secure in production, SameSite=Lax) and a CSRF cookie
//     (NOT HttpOnly, so the SPA can read it).
//   - The login response does NOT echo the API key or any
//     long-lived secret.
//   - /api/auth/logout invalidates the server-side session
//     and clears the cookie.
//   - /api/auth/ws-token issues a short-lived, audience-
//     bound JWT for the WebSocket handler (which already
//     verifies it via middle/src/auth/ws_token.ts).
//   - AURELIUS_AUTH_MODE=*** opt-in keeps the legacy
//     bearer in sessionStorage; in production (or when the
//     env var is unset) the flow is cookie-only.

import { Router, type Request, type Response } from 'express'
import { v4 as uuidv4 } from 'uuid'
import { config } from '../config.js'
import {
  createSession,
  destroySession,
  getSession,
  SESSION_COOKIE_NAME,
  CSRF_COOKIE_NAME,
  SESSION_TTL_MS,
} from '../security/session.js'
import { signWsToken } from '../auth/ws_token.js'

const router = Router()

interface User {
  id: string
  username: string
  role: 'admin' | 'user'
  apiKeys: string[]
  createdAt: string
}

const users = new Map<string, User>()
const inviteTokens = new Map<string, { role: 'admin' | 'user'; used: boolean }>()

// The admin key is the bootstrap credential. It is read
// from env, never echoed back, and only used to seed the
// admin user on first boot.
const adminKey = process.env.AURELIUS_API_KEY || ''
if (adminKey) {
  users.set('admin', {
    id: 'admin',
    username: 'admin',
    role: 'admin',
    apiKeys: [adminKey],
    createdAt: new Date().toISOString(),
  })
}

function setAuthCookies(res: Response, signed: string, csrf: string): void {
  const isProd = process.env.NODE_ENV === 'production' || config.authMode === 'session'
  const cookieOptions = [
    `${SESSION_COOKIE_NAME}=${signed}`,
    'Path=/',
    'HttpOnly',
    isProd ? 'Secure' : '',
    'SameSite=Lax',
    `Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}`,
  ].filter(Boolean).join('; ')

  res.setHeader('Set-Cookie', [
    cookieOptions,
    // CSRF cookie is NOT HttpOnly so the SPA can read it
    // and echo it in X-CSRF-Token.
    [
      `${CSRF_COOKIE_NAME}=${csrf}`,
      'Path=/',
      isProd ? 'Secure' : '',
      'SameSite=Lax',
      `Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}`,
    ].filter(Boolean).join('; '),
  ])
}

function clearAuthCookies(res: Response): void {
  const isProd = process.env.NODE_ENV === 'production' || config.authMode === 'session'
  const exp = (name: string, httpOnly: boolean) => [
    `${name}=`,
    'Path=/',
    httpOnly ? 'HttpOnly' : '',
    isProd ? 'Secure' : '',
    'SameSite=Lax',
    'Max-Age=0',
  ].filter(Boolean).join('; ')
  res.setHeader('Set-Cookie', [
    exp(SESSION_COOKIE_NAME, true),
    exp(CSRF_COOKIE_NAME, false),
  ])
}

router.post('/login', (req: Request, res: Response): void => {
  const apiKey = (req.body?.apiKey as string | undefined)?.trim()
  if (!apiKey) {
    res.status(400).json({ error: 'apiKey required' })
    return
  }

  // Find the user that owns this API key.
  let matched: User | null = null
  for (const u of users.values()) {
    if (u.apiKeys.includes(apiKey)) {
      matched = u
      break
    }
  }
  if (!matched) {
    res.status(401).json({ error: 'Invalid API key' })
    return
  }

  // In 'local_byok_dev' mode the legacy bearer is also
  // returned (for non-cookie clients) so dev tools that
  // don't read cookies still work. The default 'session'
  // mode never echoes a bearer in the body.
  const { session, signed, csrf } = createSession({
    id: matched.id,
    role: matched.role,
    scopes: matched.role === 'admin' ? ['*'] : ['read', 'write'],
    tenant: 'default',
  })

  setAuthCookies(res, signed, csrf)

  const body: Record<string, unknown> = {
    success: true,
    tokenType: 'cookie',
    expiresIn: Math.floor(SESSION_TTL_MS / 1000),
    user: { id: matched.id, username: matched.username, role: matched.role },
    csrfToken: csrf,
  }
  if (config.authMode === 'local_byok_dev') {
    body.token = signed
    body.tokenType = 'bearer-dev'
  }
  res.json(body)
})

router.post('/logout', (req: Request, res: Response): void => {
  // Read the session cookie (raw; not the signed value
  // because we want destroySession() to also tolerate
  // tampered cookies without throwing).
  const cookies = parseCookies(req)
  const signed = cookies[SESSION_COOKIE_NAME]
  if (signed) {
    destroySession(signed)
  }
  clearAuthCookies(res)
  res.json({ success: true })
})

router.get('/me', (req: Request, res: Response): void => {
  const cookies = parseCookies(req)
  const signed = cookies[SESSION_COOKIE_NAME] || headerSession(req)
  if (!signed) {
    res.status(401).json({ error: 'No session' })
    return
  }
  const session = getSession(signed)
  if (!session) {
    res.status(401).json({ error: 'Session expired' })
    return
  }
  res.json({
    user: {
      id: session.subject.id,
      role: session.subject.role,
      scopes: session.subject.scopes,
      tenant: session.subject.tenant,
    },
  })
})

/**
 * WebSocket token endpoint. Issues a short-lived, audience-
 * bound JWT (TTL 4 minutes) that the WebSocket handler
 * verifies via middle/src/auth/ws_token.ts.
 *
 * In 'local_byok_dev' mode, this endpoint also accepts
 * the raw API key in the body for compatibility with
 * devtools.
 */
router.post('/ws-token', (req: Request, res: Response): void => {
  const cookies = parseCookies(req)
  const sessionToken = cookies[SESSION_COOKIE_NAME] || headerSession(req)
  if (!sessionToken) {
    res.status(401).json({ error: 'No session' })
    return
  }
  const session = getSession(sessionToken)
  if (!session) {
    res.status(401).json({ error: 'Session expired' })
    return
  }
  const ttlSeconds = 240
  const token = signWsToken({
    sub: session.subject.id,
    tenant: session.subject.tenant,
    scopes: session.subject.scopes as any,
    aud: 'aurelius-ws',
    ttlSeconds,
  })
  res.json({ token, ttl: ttlSeconds, audience: 'ws' })
})

// ---- Helpers ----

function headerSession(req: Request): string | undefined {
  const v = req.headers['x-aurelius-session']
  return typeof v === 'string' ? v : undefined
}

function parseCookies(req: Request): Record<string, string> {
  const out: Record<string, string> = {}
  const raw = req.headers.cookie
  if (typeof raw !== 'string') return out
  for (const part of raw.split(';')) {
    const [k, ...rest] = part.trim().split('=')
    if (!k) continue
    out[k] = decodeURIComponent(rest.join('='))
  }
  return out
}

export default router

// Auth middleware.
//
// Pre-remediation, this module validated the long-lived
// API key from the X-API-Key header. The audit H8 finding
// requires:
//
//   - The middleware accepts the session cookie
//     (aurelius_sid) and the X-Aurelius-Session header
//     (for non-cookie callers). The X-API-Key header is
//     still accepted for service-to-service callers
//     (gateway -> BFF), but the path is labeled and
//     gated on the caller being a service principal.
//   - The middleware sets req.user to the AuthSubject
//     returned by the session store.
//   - The middleware rejects API keys in the query string
//     (the audit's C2 finding was that api_key=... in the
//     URL gets logged by reverse proxies).
//
// This module is the single source of truth for: who is
// the authenticated principal on each request.

import type { NextFunction, Request, Response } from 'express'
import { getSession, SESSION_COOKIE_NAME } from '../security/session.js'

export interface AuthUser {
  id: string
  role: 'admin' | 'user' | 'agent'
  scopes: string[]
  tenant: string
}

declare module 'express-serve-static-core' {
  interface Request {
    user?: AuthUser
  }
}

const SERVICE_PRINCIPALS = new Map<string, AuthUser>()

const ADMIN_API_KEY = process.env.AURELIUS_API_KEY || process.env.AURELIUS_SERVICE_KEY || ""
if (ADMIN_API_KEY) {
  SERVICE_PRINCIPALS.set(ADMIN_API_KEY, {
    id: 'admin',
    role: 'admin',
    scopes: ['*'],
    tenant: 'default',
  })
}

export function registerServicePrincipal(key: string, user: AuthUser): void {
  SERVICE_PRINCIPALS.set(key, user)
}

export function unregisterServicePrincipal(key: string): boolean {
  return SERVICE_PRINCIPALS.delete(key)
}

export function validateApiKey(key: string): AuthUser | undefined {
  return SERVICE_PRINCIPALS.get(key)
}

const PUBLIC_PATHS = new Set([
  '/health',
  '/healthz',
  '/readyz',
  '/openapi.json',
  '/docs',
  '/api/auth/login',
  '/api/auth/register',
  '/api/auth/ws-token',
  '/api/auth/logout',
])

export function authMiddleware(req: Request, res: Response, next: NextFunction): void {
  if (PUBLIC_PATHS.has(req.path)) {
    next()
    return
  }

  if (req.query.api_key || req.query.apiKey || req.query.key) {
    res.status(401).json({ error: 'Unauthorized', message: 'API key in query string is not accepted' })
    return
  }

  const cookies = parseCookies(req)
  const sessionCookie = cookies[SESSION_COOKIE_NAME]
  if (sessionCookie) {
    const session = getSession(sessionCookie)
    if (session) {
      req.user = {
        id: session.subject.id,
        role: session.subject.role,
        scopes: session.subject.scopes,
        tenant: session.subject.tenant,
      }
      next()
      return
    }
  }

  const headerSession = req.headers['x-aurelius-session']
  if (typeof headerSession === 'string' && headerSession.length > 0) {
    const session = getSession(headerSession)
    if (session) {
      req.user = {
        id: session.subject.id,
        role: session.subject.role,
        scopes: session.subject.scopes,
        tenant: session.subject.tenant,
      }
      next()
      return
    }
  }

  const apiKeyHeader = req.headers['x-api-key']
  if (typeof apiKeyHeader === 'string' && apiKeyHeader.length > 0) {
    const user = SERVICE_PRINCIPALS.get(apiKeyHeader)
    if (user) {
      req.user = user
      next()
      return
    }
  }

  res.status(401).json({ error: 'Unauthorized', message: 'Valid session cookie or service API key required' })
}

export function requireScope(...scopes: string[]) {
  return (req: Request, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ error: 'Unauthorized' })
      return
    }
    if (req.user.scopes.includes('*') || scopes.some((s) => req.user!.scopes.includes(s))) {
      next()
      return
    }
    res.status(403).json({ error: 'Forbidden', message: 'Insufficient permissions' })
  }
}

export function requireAdmin(req: Request, res: Response, next: NextFunction): void {
  if (!req.user) {
    res.status(401).json({ error: 'Unauthorized' })
    return
  }
  if (req.user.role === 'admin' || req.user.scopes.includes('*')) {
    next()
    return
  }
  res.status(403).json({ error: 'Forbidden', message: 'Admin access required' })
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

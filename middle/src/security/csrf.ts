// CSRF defense for cookie-authenticated unsafe methods.
//
// Pre-remediation, the BFF accepted any POST/PUT/PATCH/DELETE
// from an authenticated browser. The audit H8 finding requires
// CSRF protection for unsafe methods under cookie auth, using a
// double-submit token (cookie + X-CSRF-Token header).
//
// Threat model:
//   - Attacker hosts evil.example.com which embeds a form that
//     POSTs to aurelius.example.com/api/rag/upload.
//   - The browser sends aurelius_sid=... (cookie) but no
//     X-CSRF-Token (the attacker can't read the cookie).
//   - The BFF MUST reject the request.
//
// Implementation:
//   - On login the BFF sets aurelius_csrf=<csrf> (NOT HttpOnly
//     so the SPA can read it).
//   - The SPA includes X-CSRF-Token: <csrf> on every unsafe
//     request.
//   - The middleware compares the two using timingSafeEqual.
//
// This module exports the comparison helper and the
// middleware that wires it in.

import type { Request, Response, NextFunction } from 'express'
import { timingSafeEqual } from 'crypto'
import { CSRF_COOKIE_NAME, CSRF_HEADER_NAME } from './session.js'

/**
 * Constant-time compare of two strings. Returns false on
 * length mismatch.
 */
export function safeCompare(a: string, b: string): boolean {
  const ab = Buffer.from(a)
  const bb = Buffer.from(b)
  if (ab.length !== bb.length) return false
  return timingSafeEqual(ab, bb)
}

/**
 * Middleware factory: rejects unsafe methods when the
 * X-CSRF-Token header does not match the aurelius_csrf
 * cookie value. The middleware is opt-in per route group;
 * GET/HEAD/OPTIONS are never CSRF-checked.
 */
export function csrfProtection() {
  const UNSAFE = new Set(['POST', 'PUT', 'PATCH', 'DELETE'])

  return (req: Request, res: Response, next: NextFunction): void => {
    if (!UNSAFE.has(req.method.toUpperCase())) {
      next()
      return
    }
    const cookieToken = readCookie(req, CSRF_COOKIE_NAME)
    const headerToken = readHeader(req, CSRF_HEADER_NAME)
    if (!cookieToken || !headerToken) {
      res.status(403).json({
        error: 'CSRF token missing',
        message: 'Cookie auth requires X-CSRF-Token header',
      })
      return
    }
    if (!safeCompare(cookieToken, headerToken)) {
      res.status(403).json({
        error: 'CSRF token mismatch',
        message: 'X-CSRF-Token does not match aurelius_csrf cookie',
      })
      return
    }
    next()
  }
}

function readCookie(req: Request, name: string): string | undefined {
  // Express's cookie parser (if mounted) sets req.cookies.
  // If the parser is not mounted, fall back to parsing the
  // raw header ourselves so this module is self-contained.
  const fromParser = (req as any).cookies?.[name]
  if (typeof fromParser === 'string' && fromParser.length > 0) {
    return fromParser
  }
  const raw = req.headers.cookie
  if (typeof raw !== 'string') return undefined
  for (const part of raw.split(';')) {
    const [k, ...rest] = part.trim().split('=')
    if (k === name) return decodeURIComponent(rest.join('='))
  }
  return undefined
}

function readHeader(req: Request, name: string): string | undefined {
  const v = req.headers[name.toLowerCase()]
  return typeof v === 'string' ? v : Array.isArray(v) ? v[0] : undefined
}

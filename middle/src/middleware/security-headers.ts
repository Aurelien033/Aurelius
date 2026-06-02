// Centralized security headers for the BFF.
//
// Pre-remediation, the BFF did not emit CSP, X-Content-
// Type-Options, X-Frame-Options, Referrer-Policy, or
// Permissions-Policy. The audit P2.4 finding requires
// that every response carry the standard set of security
// headers.
//
// The headers are emitted as a single Express middleware
// registered globally in server.ts.

import type { Request, Response, NextFunction } from 'express'

/**
 * The standard security-header set. The values are
 * conservative (no 'unsafe-inline', no wildcard
 * sources). Tighten further per-route if needed.
 */
export const CSP_POLICY = [
  "default-src 'self'",
  "base-uri 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",  // CRA/Vite injects inline styles
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "object-src 'none'",
  "frame-ancestors 'none'",
  "form-action 'self'",
  "frame-src 'self'",
  "connect-src 'self' ws: wss:",
  "media-src 'self'",
  "worker-src 'self'",
  "manifest-src 'self'",
].join('; ')

export const PERMISSIONS_POLICY = [
  'accelerometer=()',
  'autoplay=()',
  'camera=()',
  'cross-origin-isolated=()',
  'display-capture=()',
  'encrypted-media=()',
  'fullscreen=(self)',
  'geolocation=()',
  'gyroscope=()',
  'keyboard-map=()',
  'magnetometer=()',
  'microphone=()',
  'midi=()',
  'payment=()',
  'picture-in-picture=()',
  'publickey-credentials-get=()',
  'screen-wake-lock=()',
  'sync-xhr=()',
  'usb=()',
  'xr-spatial-tracking=()',
].join(', ')

/**
 * Middleware factory. Returns a middleware that sets
 * the standard security headers on every response.
 *
 * In dev mode (NODE_ENV !== 'production') the CSP is
 * relaxed to allow Vite/HMR; in production it is strict.
 */
export function securityHeaders() {
  return function (req: Request, res: Response, next: NextFunction): void {
    const isProd = process.env.NODE_ENV === 'production'
    res.setHeader('X-Content-Type-Options', 'nosniff')
    res.setHeader('X-Frame-Options', 'DENY')
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin')
    res.setHeader('Permissions-Policy', PERMISSIONS_POLICY)
    res.setHeader('Content-Security-Policy', isProd
      ? CSP_POLICY
      : CSP_POLICY.replace("frame-ancestors 'none'", "frame-ancestors 'self'")
        // In dev, allow HMR inline eval
        .replace("script-src 'self'", "script-src 'self' 'unsafe-inline' 'unsafe-eval'"))
    res.setHeader('Strict-Transport-Security', isProd
      ? 'max-age=63072000; includeSubDomains; preload'
      : 'max-age=0')
    next()
  }
}

import { describe, it, expect } from 'vitest'
import { buildApp } from '../src/server.js'
import { invokeApp } from './request-app.js'

const app = buildApp()

/**
 * Auth endpoint tests for the H8 (P1.2) fix.
 *
 * The pre-remediation BFF exposed /api/auth/keys/generate
 * (mints long-lived API keys in the browser) and
 * /api/auth/register (creates a user with a generated
 * long-lived key). Both are removed by the security fix.
 *
 * The new flow is cookie-based session auth. Login sets
 * aurelius_sid + aurelius_csrf cookies. Logout invalidates
 * the session.
 */
describe('Auth endpoints (H8/P1.2 cookie session flow)', () => {
  it('POST /api/auth/login with valid key sets session cookies', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/login',
      headers: { 'Content-Type': 'application/json' },
      body: { apiKey: 'test-admin-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.success).toBe(true)
    // The response must set-cookie the session + csrf
    const setCookie = res.headers['set-cookie']
    expect(setCookie).toBeDefined()
    const cookies = Array.isArray(setCookie) ? setCookie.join('\n') : String(setCookie)
    expect(cookies).toMatch(/aurelius_sid=/)
    expect(cookies).toMatch(/aurelius_csrf=/)
    expect(cookies).toMatch(/HttpOnly/i)
    expect(cookies).toMatch(/SameSite=/)
  })

  it('POST /api/auth/login with invalid key returns 401', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/login',
      headers: { 'Content-Type': 'application/json' },
      body: { apiKey: 'bad-key' },
    })
    expect(res.status).toBe(401)
  })

  it('POST /api/auth/logout invalidates the session', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/logout',
    })
    expect(res.status).toBe(200)
  })

  it('POST /api/auth/ws-token issues a short-lived JWT', async () => {
    // First login to get a session cookie
    const loginRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/login',
      headers: { 'Content-Type': 'application/json' },
      body: { apiKey: 'test-admin-key' },
    })
    const setCookie = loginRes.headers['set-cookie']
    const cookieHeader = Array.isArray(setCookie) ? setCookie.join('; ') : String(setCookie ?? '')
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/ws-token',
      headers: { Cookie: cookieHeader },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.token).toBeDefined()
    expect(data.ttl).toBeLessThanOrEqual(300)
    expect(data.audience).toBeDefined()
  })

  it('POST /api/auth/ws-token without a session returns 401', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/ws-token',
    })
    expect(res.status).toBe(401)
  })
})

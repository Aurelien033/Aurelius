import { useCallback } from 'react'
import { useAuthStore, type AuthMode } from '../stores/apiStore'

/**
 * Browser auth hook.
 *
 * Pre-remediation, the hook called /api/auth/login and
 * wrote the API key to localStorage (via useAuth.saveApiKey
 * -> setApiKey -> localStorage). The audit H8 finding
 * requires that the production flow NOT persist the API
 * key in browser-readable storage.
 *
 * New flow:
 *   - login() POSTs to /api/auth/login with
 *     credentials: 'include'. The BFF responds with a
 *     Set-Cookie (aurelius_sid, HttpOnly, Secure in
 *     production, SameSite=Lax) and a Set-Cookie
 *     aurelius_csrf (NOT HttpOnly, so the SPA can read
 *     it for X-CSRF-Token).
 *   - The in-memory state stores authMode, user, csrfToken,
 *     and (in dev BYOK mode) the dev apiKey in
 *     sessionStorage (NOT localStorage).
 *   - logout() POSTs to /api/auth/logout, clears the
 *     in-memory state, and removes the sessionStorage
 *     entry.
 *
 * The dev BYOK mode is gated by AURELIUS_AUTH_MODE=***
 *     on the BFF. In production, the BFF returns
 *     authMode: 'session' and the client uses cookies
 *     only.
 */

const SESSION_STORAGE_KEY = 'aurelius-dev-byok'

export interface LoginInput {
  apiKey: string
}

export function useAuth(): {
  login: (input: LoginInput) => Promise<void>
  logout: () => Promise<void>
  authMode: AuthMode
  devModeWarning: string | null
  isAuthenticated: boolean
} {
  const setSession = useAuthStore((s) => s.setSession)
  const clear = useAuthStore((s) => s.clear)
  const authMode = useAuthStore((s) => s.authMode)
  const devModeWarning = useAuthStore((s) => s.devModeWarning)

  const login = useCallback(async (input: LoginInput): Promise<void> => {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey: input.apiKey }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Login failed' }))
      throw new Error(err.error || 'Login failed')
    }
    const body = await res.json()
    const mode: AuthMode = body.tokenType === 'bearer-dev' ? 'local_byok_dev' : 'session'
    const devApiKey = mode === 'local_byok_dev' && body.token
      ? String(body.token)
      : null
    if (devApiKey) {
      // Dev BYOK: sessionStorage only, with a TTL the
      // useAuth will honor at next render. NOT localStorage.
      try {
        sessionStorage.setItem(SESSION_STORAGE_KEY, devApiKey)
      } catch {
        // sessionStorage may be disabled (e.g. private
        // mode in some browsers); the dev warning will
        // still be shown.
      }
    }
    setSession({
      authMode: mode,
      user: body.user ?? null,
      csrfToken: body.csrfToken ?? null,
      devModeWarning: mode === 'local_byok_dev'
        ? 'You are in AURELIUS_AUTH_MODE=*** mode. The raw API key is held in sessionStorage (not localStorage) and is cleared on browser restart. Never enable this mode in production.'
        : null,
      devApiKey,
    })
  }, [setSession])

  const logout = useCallback(async (): Promise<void> => {
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      })
    } catch {
      // Best-effort: even if the network call fails,
      // clear the local state.
    }
    try {
      sessionStorage.removeItem(SESSION_STORAGE_KEY)
    } catch {
      // ignore
    }
    clear()
  }, [clear])

  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  return { login, logout, authMode, devModeWarning, isAuthenticated }
}

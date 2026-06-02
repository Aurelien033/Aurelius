import { create } from 'zustand'

/**
 * In-memory only auth state.
 *
 * The pre-remediation apiStore persisted the raw API key
 * in localStorage via the persist middleware. The audit H8
 * finding requires that the production flow NOT persist
 * the API key (or any long-lived secret) in browser-
 * readable persistent storage. This store is in-memory
 * only: state is lost on page refresh, and the production
 * flow is cookie-based.
 *
 * The dev BYOK mode (AURELIUS_AUTH_MODE=*** keeps
 * the API key in sessionStorage (not localStorage) with a
 * visible warning. The dev-mode TTL is enforced by the
 * useAuth hook.
 */
export type AuthMode = 'session' | 'local_byok_dev'

interface AuthState {
  /** Auth mode reported by the BFF on /api/auth/login. */
  authMode: AuthMode
  /** Subject identity from the BFF session. */
  user: { id: string; role: 'admin' | 'user' | 'agent' } | null
  /** CSRF token. The api client copies it into X-CSRF-Token. */
  csrfToken: string | null
  /** True if the user is signed in. */
  isAuthenticated: boolean
  /** Dev-only: human-readable warning text. */
  devModeWarning: string | null
  /** In dev BYOK mode only: the sessionStorage-only key. Never persisted to localStorage. */
  devApiKey: string | null

  setSession: (s: { authMode: AuthMode; user: AuthState['user']; csrfToken: string | null; devModeWarning: string | null; devApiKey: string | null }) => void
  clear: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  authMode: 'session',
  user: null,
  csrfToken: null,
  isAuthenticated: false,
  devModeWarning: null,
  devApiKey: null,
  setSession: (s) => set({
    authMode: s.authMode,
    user: s.user,
    csrfToken: s.csrfToken,
    isAuthenticated: !!s.user,
    devModeWarning: s.devModeWarning,
    devApiKey: s.devApiKey,
  }),
  clear: () => set({
    authMode: 'session',
    user: null,
    csrfToken: null,
    isAuthenticated: false,
    devModeWarning: null,
    devApiKey: null,
  }),
}))

/**
 * Back-compat shim: the old `useApiStore` exposed apiKey +
 * a few helpers. The new flow does not persist an API key in
 * memory either, but several legacy components import this
 * symbol. The shim keeps the call sites compiling while
 * returning empty / no-op values.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const useApiStore: any = (selector: any) => {
  // The legacy selector shape is preserved; the new auth
  // state is exposed via useAuthStore. Legacy call sites
  // that import useApiStore will see empty values for
  // unknown keys.
  return useAuthStore(selector as any)
}

// Default empty stub for legacy call sites
useApiStore.getState = () => ({
  apiKey: '',
  setApiKey: () => {},
  clearApiKey: () => {},
  authenticated: false,
})

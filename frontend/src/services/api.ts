import { useAuthStore } from '../stores/apiStore'

const API_BASE = '/api'

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  body?: unknown
  headers?: Record<string, string>
  signal?: AbortSignal
  /** When true, omit credentials (for /api/auth/login which is the one public endpoint that must not pre-send cookies). */
  omitCredentials?: boolean
}

function readCsrfCookie(): string | null {
  if (typeof document === 'undefined') return null
  for (const part of document.cookie.split(';')) {
    const [k, ...rest] = part.trim().split('=')
    if (k === 'aurelius_csrf') return decodeURIComponent(rest.join('='))
  }
  return null
}

export async function apiRequest<T = unknown>(path: string, options: RequestOptions = {}): Promise<T> {
  const method = (options.method ?? 'GET').toUpperCase()
  const UNSAFE = method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS'

  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    ...(options.headers ?? {}),
  }

  // Cookie auth: send credentials on every same-origin call.
  // In dev BYOK mode the dev session token is also sent as
  // X-Aurelius-Session (so the BFF can fall back if cookies
  // are blocked).
  const auth = useAuthStore.getState()
  if (auth.csrfToken && UNSAFE) {
    // Defense-in-depth: the BFF also reads the csrf cookie
    // and validates the X-CSRF-Token header on unsafe
    // methods. Belt-and-suspenders.
    const fromCookie = readCsrfCookie()
    headers['X-CSRF-Token'] = fromCookie || auth.csrfToken
  }
  if (auth.authMode === 'local_byok_dev' && auth.devApiKey) {
    headers['X-Aurelius-Session'] = auth.devApiKey
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    credentials: options.omitCredentials ? 'omit' : 'include',
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    signal: options.signal,
  })

  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      message = body.error || body.message || message
    } catch {
      // ignore
    }
    throw new Error(message)
  }

  // 204 No Content
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  get: <T = unknown>(path: string, options?: Omit<RequestOptions, 'method' | 'body'>) =>
    apiRequest<T>(path, { ...options, method: 'GET' }),
  post: <T = unknown>(path: string, body?: unknown, options?: Omit<RequestOptions, 'method' | 'body'>) =>
    apiRequest<T>(path, { ...options, method: 'POST', body }),
  put: <T = unknown>(path: string, body?: unknown, options?: Omit<RequestOptions, 'method' | 'body'>) =>
    apiRequest<T>(path, { ...options, method: 'PUT', body }),
  patch: <T = unknown>(path: string, body?: unknown, options?: Omit<RequestOptions, 'method' | 'body'>) =>
    apiRequest<T>(path, { ...options, method: 'PATCH', body }),
  delete: <T = unknown>(path: string, options?: Omit<RequestOptions, 'method' | 'body'>) =>
    apiRequest<T>(path, { ...options, method: 'DELETE' }),
}

// Back-compat aliases for the pre-remediation API client shape.
// The old services/api.ts exported apiClient, ApiOptions,
// ApiResponse. Legacy services/index.ts imports them.
export interface ApiOptions extends RequestOptions {}
export interface ApiResponse<T = unknown> {
  data: T
  status: number
  ok: boolean
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const apiClient: any = {
  get: api.get,
  post: api.post,
  put: api.put,
  patch: api.patch,
  delete: api.delete,
}

// Helper: wrap a { data, status, ok } envelope around the
// raw api.get/post result for legacy callers.
export async function apiRequestLegacy<T = unknown>(
  path: string,
  options?: RequestOptions,
): Promise<ApiResponse<T>> {
  try {
    const data = await apiRequest<T>(path, options)
    return { data, status: 200, ok: true }
  } catch (e) {
    return { data: undefined as unknown as T, status: 500, ok: false }
  }
}

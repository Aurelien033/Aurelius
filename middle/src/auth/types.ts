// Auth type definitions for the BFF.
//
// These types are the single source of truth for the
// authenticated-principal shape. WebSocket tokens, REST session
// tokens, and the auth middleware all read from this file.

export type UserRole = 'admin' | 'user' | 'agent' | 'service'

export type Scope =
  | '*'
  | 'read'
  | 'write'
  | 'chat:read'
  | 'chat:write'
  | 'agents:read'
  | 'agents:write'
  | 'config:read'
  | 'config:write'
  | 'memory:read'
  | 'memory:write'
  | 'commands:execute'
  | 'models:read'
  | 'auth:read'
  | 'auth:write'
  | 'ws:connect'
  | 'internal:service'

export type Audience = 'aurelius-rest' | 'aurelius-ws' | 'aurelius-internal'

export interface AuthUser {
  id: string
  role: UserRole
  scopes: Scope[]
  tenant?: string
}

export interface WsTokenPayload {
  /** subject (user id) */
  sub: string
  /** audience — must equal 'aurelius-ws' for WS tokens */
  aud: Audience
  /** issued at (epoch seconds) */
  iat: number
  /** expires at (epoch seconds) */
  exp: number
  /** scopes granted to this token */
  scopes: Scope[]
  /** tenant — gates which rooms the token can join */
  tenant: string
  /** connection id — single-use marker for replay protection */
  jti: string
}

export interface SessionTokenPayload {
  sub: string
  aud: Audience
  iat: number
  exp: number
  scopes: Scope[]
  role: UserRole
  tenant: string
  jti: string
}

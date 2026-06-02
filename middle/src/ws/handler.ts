import { WebSocketServer, type WebSocket } from 'ws'
import type { Server } from 'http'
import { getEngine } from '../engine.js'
import {
  joinRoom,
  leaveRoom,
  broadcastToRoom,
  broadcastToAll,
  getRoomList,
  authorizeRoomSubscribe,
  authorizeRoomBroadcast,
} from './rooms.js'
import { config } from '../config.js'
import { verifyWsToken, type VerifyWsTokenResult } from '../auth/ws_token.js'
import type { WsTokenPayload } from '../auth/types.js'

interface WsMessage {
  type: string
  payload?: Record<string, unknown>
}

interface AuthenticatedSocket extends WebSocket {
  authUser?: WsTokenPayload
  connectionId?: string
}

const clients = new Set<AuthenticatedSocket>()
const MAX_WS_MESSAGE_SIZE = 65536
const WS_COMMANDS_REQUIRING_ADMIN = new Set(['command', 'agent:terminate', 'config:update'])

// Default scope sets used when a token omits an explicit scope list.
// These are the only scopes a fresh user-issued token may carry
// without an admin grant.
const ALLOWED_DEFAULT_SCOPES: ReadonlySet<string> = new Set([
  'read',
  'ws:connect',
  'chat:read',
  'chat:write',
  'agents:read',
  'memory:read',
  'models:read',
  'config:read',
])

/**
 * Extract the WS token from a request. The pre-remediation handler
 * read the long-lived X-API-Key header; the fix only accepts a
 * short-lived, audience-bound token in either:
 *   - the `?token=...` query parameter
 *   - the `Sec-WebSocket-Protocol` subprotocol header
 * The Authorization / X-API-Key header paths are removed.
 */
function extractWsToken(req: { url?: string; headers: Record<string, string | string[] | undefined> }): string | null {
  // Query parameter
  if (req.url) {
    try {
      const url = new URL(req.url, 'http://localhost')
      const q = url.searchParams.get('token')
      if (q) return q
    } catch {
      // fallthrough
    }
  }
  // Subprotocol header (RFC 6455 allows clients to send a chosen
  // subprotocol in the upgrade request)
  const proto = req.headers['sec-websocket-protocol']
  if (typeof proto === 'string') {
    // Format: "aurelius-ws-token, <token>" or just "<token>"
    const parts = proto.split(',').map((p) => p.trim())
    for (const p of parts) {
      if (p.startsWith('aurelius-ws-token.')) {
        return p.slice('aurelius-ws-token.'.length)
      }
    }
  }
  return null
}

/**
 * C3: Origin allowlist. A WS upgrade with an Origin that is not in
 * config.wsOriginAllowlist is closed with 1008 (policy violation)
 * before any auth runs. The check is case-insensitive and ignores
 * default ports for http/https.
 */
function isOriginAllowed(origin: string | undefined): boolean {
  if (!origin) return false
  try {
    const u = new URL(origin)
    const host = u.host.toLowerCase()
    for (const allowed of config.wsOriginAllowlist) {
      if (allowed === '*') {
        // A literal '*' is the CORS "any" wildcard. We allow it
        // only when the operator has explicitly opted in via
        // AURELIUS_WS_ORIGINS=*. (Other env values still take
        // precedence; this is the no-Origin case for non-browser
        // clients.)
        return true
      }
      const allowedHost = allowed.replace(/^https?:\/\//i, '').replace(/\/$/, '').toLowerCase()
      if (allowedHost === host) return true
      // Allow same-origin (host === allowed) with port stripped.
      if (host.split(':')[0] === allowedHost.split(':')[0]) return true
    }
  } catch {
    return false
  }
  return false
}

export function setupWebSocket(server: Server): WebSocketServer {
  const wss = new WebSocketServer({ server, path: '/ws', maxPayload: MAX_WS_MESSAGE_SIZE })

  wss.on('connection', (ws: AuthenticatedSocket, req) => {
    // --- C3 step 1: Origin allowlist BEFORE anything else ---
    const origin = (req.headers.origin as string | undefined) ?? undefined
    if (!isOriginAllowed(origin)) {
      ws.close(1008, 'Origin not allowed')
      return
    }

    // --- C3 step 2: Short-lived, audience-bound token verification ---
    const token = extractWsToken(req)
    const result: VerifyWsTokenResult = token
      ? verifyWsToken(token, { expectedAudience: 'aurelius-ws' })
      : { ok: false, reason: 'empty' }
    if (!result.ok) {
      ws.close(1008, 'Unauthorized')
      return
    }
    ws.authUser = result.payload
    ws.connectionId = result.payload.jti

    // Sanitize scopes: drop anything not in the allowed-default set
    // unless the principal is an admin or has the wildcard scope.
    const sanitizedScopes: string[] = []
    for (const s of result.payload.scopes) {
      if (s === '*' || result.payload.scopes.includes('*')) {
        sanitizedScopes.push(s)
        break
      }
      if (ALLOWED_DEFAULT_SCOPES.has(s)) {
        sanitizedScopes.push(s)
      }
    }
    if (ws.authUser) {
      ws.authUser = { ...ws.authUser, scopes: sanitizedScopes as WsTokenPayload['scopes'] }
    }

    clients.add(ws)

    // --- C3 step 3: Minimal handshake ---
    // The pre-remediation handler sent listAgents +
    // getNotificationStats in the connect payload, exposing the full
    // agent roster and notification history to any client that could
    // present a long-lived API key. The fix sends only non-sensitive
    // handshake metadata.
    ws.send(JSON.stringify({
      type: 'connected',
      payload: {
        connectionId: ws.connectionId,
        expiresAt: ws.authUser?.exp,
        tenant: ws.authUser?.tenant,
        timestamp: Date.now(),
      },
    }))

    ws.on('message', (raw) => {
      const rawStr = raw.toString()
      if (rawStr.length > MAX_WS_MESSAGE_SIZE) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'Message too large' } }))
        return
      }
      try {
        const msg = JSON.parse(rawStr) as WsMessage
        handleMessage(ws, msg)
      } catch {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'Invalid JSON' } }))
      }
    })

    ws.on('close', () => {
      clients.delete(ws)
    })

    ws.on('error', () => {
      clients.delete(ws)
    })
  })

  return wss
}

function getRoomAuthContext(ws: AuthenticatedSocket): {
  tenant: string
  scopes: string[]
  role: string
} {
  return {
    tenant: ws.authUser?.tenant ?? 'default',
    scopes: ws.authUser?.scopes ?? [],
    // The WS token does not carry role; the '*' scope implies admin.
    role: ws.authUser?.scopes.includes('*') ? 'admin' : 'user',
  }
}

function handleMessage(ws: AuthenticatedSocket, msg: WsMessage): void {
  const engine = getEngine()
  const user = ws.authUser

  if (!user) {
    ws.send(JSON.stringify({ type: 'error', payload: { message: 'Not authenticated' } }))
    return
  }

  if (WS_COMMANDS_REQUIRING_ADMIN.has(msg.type) && !user.scopes.includes('*')) {
    ws.send(JSON.stringify({ type: 'error', payload: { message: 'Admin access required' } }))
    return
  }

  switch (msg.type) {
    case 'ping':
      ws.send(JSON.stringify({ type: 'pong', payload: { timestamp: Date.now() } }))
      break

    case 'subscribe': {
      // --- C3: room authorization on subscribe ---
      const payload = (msg.payload ?? {}) as { room?: unknown }
      const roomRaw = payload.room
      if (typeof roomRaw !== 'string' || !roomRaw) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'room required' } }))
        break
      }
      const ctx = getRoomAuthContext(ws)
      const authz = authorizeRoomSubscribe(roomRaw, ctx)
      if (!authz.ok) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: `forbidden: ${authz.reason}` } }))
        break
      }
      joinRoom(roomRaw, ws)
      ws.send(JSON.stringify({ type: 'subscribed', payload: { room: roomRaw } }))
      break
    }

    case 'unsubscribe': {
      const { room } = (msg.payload ?? {}) as { room?: unknown }
      if (room && typeof room === 'string') {
        leaveRoom(room, ws)
        ws.send(JSON.stringify({ type: 'unsubscribed', payload: { room } }))
      }
      break
    }

    case 'room:message': {
      // --- C3: room broadcast authorization ---
      const { room, data } = (msg.payload ?? {}) as { room?: unknown; data?: unknown }
      if (typeof room !== 'string' || !room) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'room required' } }))
        break
      }
      const ctx = getRoomAuthContext(ws)
      const authz = authorizeRoomBroadcast(room, ctx)
      if (!authz.ok) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: `forbidden: ${authz.reason}` } }))
        break
      }
      const count = broadcastToRoom(room, { type: 'room:message', payload: { room, data, timestamp: Date.now() } })
      ws.send(JSON.stringify({ type: 'room:delivered', payload: { room, count } }))
      break
    }

    case 'get:rooms':
      ws.send(JSON.stringify({ type: 'rooms', payload: { rooms: getRoomList() } }))
      break

    case 'get:agents': {
      // --- C3: agents listing is now scope-gated ---
      if (!user.scopes.includes('*') && !user.scopes.includes('agents:read')) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'forbidden: agents:read required' } }))
        break
      }
      ws.send(JSON.stringify({ type: 'agents', payload: { agents: engine.listAgents() } }))
      break
    }

    case 'get:activity':
      ws.send(JSON.stringify({ type: 'activity', payload: { entries: engine.getActivity(50) } }))
      break

    case 'get:notifications': {
      // --- C3: notifications is now scope-gated ---
      if (!user.scopes.includes('*') && !user.scopes.includes('read')) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'forbidden: read required' } }))
        break
      }
      ws.send(JSON.stringify({
        type: 'notifications',
        payload: {
          notifications: engine.getNotifications(undefined, undefined, undefined, 50),
          stats: engine.getNotificationStats(),
        },
      }))
      break
    }

    case 'get:status': {
      if (!user.scopes.includes('*') && !user.scopes.includes('read')) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'forbidden: read required' } }))
        break
      }
      ws.send(JSON.stringify({
        type: 'status',
        payload: {
          agents: engine.listAgents(),
          activity: engine.getActivity(10),
          notifications: engine.getNotificationStats(),
          memory: engine.getMemoryLayers(),
        },
      }))
      break
    }

    case 'command': {
      const { command } = msg.payload || {}
      if (command) {
        engine.appendActivity('ws.command', true, String(command))
        ws.send(JSON.stringify({ type: 'command:ack', payload: { command, timestamp: Date.now() } }))
        broadcastToAll({ type: 'activity:new', payload: { command, timestamp: Date.now() } })
      }
      break
    }

    default:
      ws.send(JSON.stringify({ type: 'error', payload: { message: `Unknown type: ${msg.type}` } }))
  }
}

export function broadcastNotification(notification: unknown): void {
  const msg = JSON.stringify({ type: 'notification', payload: notification })
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) {
      try { ws.send(msg) } catch { /* best effort */ }
    }
  }
}

export function broadcastActivity(activity: unknown): void {
  const msg = JSON.stringify({ type: 'activity:new', payload: activity })
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) {
      try { ws.send(msg) } catch { /* best effort */ }
    }
  }
}

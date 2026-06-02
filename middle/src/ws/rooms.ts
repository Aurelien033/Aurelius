import { WebSocket } from 'ws'

interface Room {
  name: string
  clients: Set<WebSocket>
  metadata: RoomMetadata
  createdAt: number
}

/**
 * Per-room metadata. Pre-remediation, every authenticated client could
 * join any room and broadcast to it. The post-remediation BFF records
 * the room's tenant, the scopes required to subscribe, and the scopes
 * required to broadcast. The default room (no metadata) is treated
 * as a system room open to any authenticated user.
 */
export interface RoomMetadata {
  /** Tenant that owns the room. */
  tenant?: string
  /** Scopes required to subscribe to this room. '*' grants admin. */
  requiredScopes?: string[]
  /** Scopes required to broadcast to this room. */
  broadcastScopes?: string[]
  /** True for rooms managed by the server (no tenant). */
  system?: boolean
}

const rooms = new Map<string, Room>()

export function createRoom(name: string, metadata: RoomMetadata = {}): Room {
  const existing = rooms.get(name)
  if (existing) return existing
  const room: Room = {
    name,
    clients: new Set(),
    metadata,
    createdAt: Date.now(),
  }
  rooms.set(name, room)
  return room
}

export function setRoomMetadata(name: string, metadata: RoomMetadata): void {
  const room = rooms.get(name)
  if (room) {
    room.metadata = metadata
    return
  }
  createRoom(name, metadata)
}

export function deleteRoom(name: string): boolean {
  return rooms.delete(name)
}

export function joinRoom(roomName: string, ws: WebSocket): Room {
  const room = createRoom(roomName)
  room.clients.add(ws)
  ws.on('close', () => leaveRoom(roomName, ws))
  return room
}

export function leaveRoom(roomName: string, ws: WebSocket): void {
  const room = rooms.get(roomName)
  if (!room) return
  room.clients.delete(ws)
  if (room.clients.size === 0) {
    rooms.delete(roomName)
  }
}

export function broadcastToRoom(roomName: string, data: unknown): number {
  const room = rooms.get(roomName)
  if (!room) return 0
  const msg = JSON.stringify(data)
  let count = 0
  for (const ws of room.clients) {
    if (ws.readyState === ws.OPEN) {
      try {
        ws.send(msg)
        count++
      } catch {
        // ignore individual client errors
      }
    }
  }
  return count
}

export function broadcastToAll(data: unknown): number {
  const msg = JSON.stringify(data)
  let count = 0
  for (const [, room] of rooms) {
    for (const ws of room.clients) {
      if (ws.readyState === ws.OPEN) {
        try {
          ws.send(msg)
          count++
        } catch {
          // ignore individual client errors
        }
      }
    }
  }
  return count
}

export function getRoomList(): Array<{ name: string; clients: number; metadata: RoomMetadata; createdAt: number }> {
  const list: Array<{ name: string; clients: number; metadata: RoomMetadata; createdAt: number }> = []
  for (const [, room] of rooms) {
    list.push({ name: room.name, clients: room.clients.size, metadata: room.metadata, createdAt: room.createdAt })
  }
  return list
}

export function getRoom(name: string): Room | undefined {
  return rooms.get(name)
}

export function getRoomCount(): number {
  return rooms.size
}

export function getClientCount(): number {
  let total = 0
  for (const [, room] of rooms) total += room.clients.size
  return total
}

// ---------------------------------------------------------------------------
// C3: Room authorization
// ---------------------------------------------------------------------------

export interface RoomAuthContext {
  /** tenant id of the requesting principal */
  tenant: string
  /** scopes granted to the principal's verified token */
  scopes: string[]
  /** role of the principal (admin, user, service, ...) */
  role: string
}

export interface RoomAuthzResult {
  ok: boolean
  reason?: string
}

/**
 * Decide whether `ctx` is allowed to subscribe to `roomName`.
 *
 * Rules (in order):
 *   1. If the room has no metadata, it is a system room open to any
 *      authenticated principal.
 *   2. If the room has a tenant and the principal's tenant does not
 *      match, deny.
 *   3. If the room has requiredScopes, the principal must have '*'
 *      or at least one of them.
 *   4. Otherwise, allow.
 */
export function authorizeRoomSubscribe(roomName: string, ctx: RoomAuthContext): RoomAuthzResult {
  const room = rooms.get(roomName)
  if (!room) {
    // Auto-create with default (system) metadata. This is the
    // 'subscribe to anything' default — the fix narrows the
    // authorization to metadata-driven rules below.
    return { ok: true }
  }
  const meta = room.metadata
  if (!meta || meta.system) {
    return { ok: true }
  }
  if (meta.tenant && meta.tenant !== ctx.tenant && ctx.role !== 'admin' && !ctx.scopes.includes('*')) {
    return { ok: false, reason: 'wrong-tenant' }
  }
  const required = meta.requiredScopes ?? []
  if (required.length > 0) {
    if (ctx.scopes.includes('*') || ctx.role === 'admin') return { ok: true }
    if (!required.some((s) => ctx.scopes.includes(s))) {
      return { ok: false, reason: 'insufficient-scope' }
    }
  }
  return { ok: true }
}

/**
 * Decide whether `ctx` is allowed to broadcast to `roomName`.
 *
 * Stricter than subscribe: a room with no broadcast metadata is
 * read-only by default.
 */
export function authorizeRoomBroadcast(roomName: string, ctx: RoomAuthContext): RoomAuthzResult {
  const room = rooms.get(roomName)
  if (!room) {
    return { ok: false, reason: 'unknown-room' }
  }
  const meta = room.metadata
  if (ctx.role === 'admin' || ctx.scopes.includes('*')) {
    return { ok: true }
  }
  const required = meta.broadcastScopes ?? []
  if (required.length === 0) {
    return { ok: false, reason: 'read-only-room' }
  }
  if (!required.some((s) => ctx.scopes.includes(s))) {
    return { ok: false, reason: 'insufficient-scope' }
  }
  return { ok: true }
}

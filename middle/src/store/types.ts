// Persistence interface for the BFF.
//
// Every record persisted by the BFF (RAG documents, memory
// insertion, file metadata, agent state) goes through this
// interface. The interface is the single source of truth
// for the tenant + owner scoping invariants.
//
// Security invariants:
//   - tenant isolation: every record has a tenant; reads
//     filter by tenant (cross-tenant returns empty).
//   - owner isolation: every record has an ownerId; reads
//     filter by ownerId unless the caller is admin.
//   - field allowlist: writeRecord() rejects keys whose
//     name matches /apiKey|api_key|token|secret|password/i
//     unless the caller passes an explicit waiver.

/**
 * A persisted record. The `id` is the caller-chosen
 * identifier (e.g. a UUID from the route handler). The
 * `tenant` and `ownerId` are taken from the auth
 * principal at write time and are immutable thereafter.
 */
export interface PersistedRecord {
  id: string
  tenant: string
  ownerId: string
  data: Record<string, unknown>
  createdAt: number
  updatedAt: number
}

/**
 * The persistence layer. Implementations: MemoryStore
 * (in-process, for dev/single-tenant), SqliteStore
 * (on-disk, for prod, currently a stub).
 */
export interface Store {
  write(record: PersistedRecord): Promise<void>
  get(id: string, tenant: string, ownerId: string, isAdmin: boolean): Promise<PersistedRecord | null>
  list(tenant: string, ownerId: string, isAdmin: boolean): Promise<PersistedRecord[]>
  delete(id: string, tenant: string, ownerId: string, isAdmin: boolean): Promise<boolean>
}

/**
 * The set of field names that may not be persisted
 * unless the caller passes an explicit waiver. This list
 * is the single source of truth across memory and SQLite
 * implementations and the frontend persist layer.
 */
export const FORBIDDEN_PERSISTED_KEYS = [
  'apiKey',
  'api_key',
  'apiToken',
  'accessToken',
  'refreshToken',
  'token',
  'jwt',
  'bearer',
  'secret',
  'password',
  'passwd',
] as const

export function isForbiddenKey(key: string): boolean {
  return (FORBIDDEN_PERSISTED_KEYS as readonly string[]).some(
    (k) => k.toLowerCase() === key.toLowerCase(),
  )
}

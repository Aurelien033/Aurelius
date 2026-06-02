// SQLite-backed implementation of the persistence interface.
//
// Stub: a future Tranche (PR4 persistence follow-up, or
// PR7 deployment hardening) will swap this in. The stub
// imports the same Store interface and the same
// field-allowlist enforcement as the in-memory store so
// that the security invariant travels with the type.
//
// DO NOT use this for production storage until:
//   1. better-sqlite3 or sqlite3 is added to dependencies
//   2. The DB is opened with WAL mode
//   3. The connection string is configurable via env var
//   4. The schema is migration-managed (e.g. drizzle-kit
//      or a hand-rolled migration runner)
//
// Until then, this module throws on every operation.

import type { PersistedRecord, Store } from './types.js'

export class SqliteStore implements Store {
  async write(_record: PersistedRecord): Promise<void> {
    throw new Error(
      'SqliteStore is a stub. Use MemoryStore (or wire up better-sqlite3 + WAL + migrations).',
    )
  }
  async get(_id: string, _tenant: string, _ownerId: string, _isAdmin: boolean): Promise<PersistedRecord | null> {
    throw new Error('SqliteStore is a stub')
  }
  async list(_tenant: string, _ownerId: string, _isAdmin: boolean): Promise<PersistedRecord[]> {
    throw new Error('SqliteStore is a stub')
  }
  async delete(_id: string, _tenant: string, _ownerId: string, _isAdmin: boolean): Promise<boolean> {
    throw new Error('SqliteStore is a stub')
  }
}

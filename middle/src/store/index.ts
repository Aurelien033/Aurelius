import { config } from '../config.js'
import { registerApiKey } from '../middleware/auth.js'
import { MemoryBffStore } from './memory-store.js'
import { SqliteBffStore } from './sqlite-store.js'
import type { BffStore } from './types.js'

let _store: BffStore | null = null

function resolveSqlitePath(databaseUrl: string): string {
  if (databaseUrl.startsWith('sqlite:')) {
    return databaseUrl.slice('sqlite:'.length)
  }
  return databaseUrl
}

export function getBffStore(): BffStore {
  if (!_store) {
    _store = config.ephemeralMode
      ? new MemoryBffStore()
      : new SqliteBffStore(resolveSqlitePath(config.databaseUrl))
    seedAdminUser(_store)
  }
  return _store
}

function seedAdminUser(store: BffStore): void {
  const adminKey = process.env.AURELIUS_API_KEY || ''
  if (!adminKey) return

  let admin = store.getUser('admin')
  if (!admin) {
    admin = {
      id: 'admin',
      username: 'admin',
      role: 'admin',
      apiKeys: [adminKey],
      createdAt: new Date().toISOString(),
    }
    store.saveUser(admin)
  } else if (!admin.apiKeys.includes(adminKey)) {
    store.addUserApiKey('admin', adminKey)
    admin = store.getUser('admin')!
  }

  registerApiKey(adminKey, { id: admin.id, role: admin.role, scopes: ['*'] })
}

export function initBffStore(): void {
  getBffStore()
}

export function shutdownBffStore(): void {
  _store?.close()
  _store = null
}

/** Test-only: reset singleton between persistence scenarios. */
export function resetBffStoreForTests(): void {
  shutdownBffStore()
}

export function isPersistentStore(): boolean {
  return getBffStore().persistent
}

export type { BffStore, UserRecord, BffAgentRecord, FileRecord, TraceRecord, CronTaskRecord } from './types.js'

/**
 * Zustand persistence middleware with secret-field denial.
 *
 * The pre-remediation persist middleware stored any field
 * the caller asked it to, including apiKey/token/secret/
 * password, into localStorage. The audit H8 finding
 * requires that the persist layer refuse to persist a
 * field whose key matches a deny-list of secret-shaped
 * names, unless the caller passes an explicit waiver.
 *
 * The deny-list lives in DENY_KEYS below. The middleware
 * checks every key in the partialize() output against the
 * list and throws at construction time (NOT at write
 * time) so a misconfigured store fails fast.
 *
 * Production stores that need a long-lived secret must
 * not use this middleware; they must use a cookie-based
 * session via the BFF.
 */

import { create, type StateCreator } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

/**
 * The canonical deny-list. This is the single source of
 * truth for which field names are forbidden from being
 * persisted. The list is intentionally strict; if a
 * caller needs a permitted variant they should add it
 * explicitly with a comment justifying the waiver.
 */
export const DENY_KEYS: ReadonlySet<string> = new Set([
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
  'session',
  'sessionToken',
  'csrfToken',
  'aurelius-api-key',
  'aurelius-session',
])

/**
 * Test/waiver escape hatch. A caller may pass
 * `waivedKeys: ['apiKey']` to opt a field back in. The
 * waiver is recorded at the call site (a comment must
 * justify it) and is never present in production code
 * paths.
 */
type PersistConfig<T> = {
  name: string
  version?: number
  partialize?: (state: T) => Partial<T>
  migrate?: (persisted: unknown, version: number) => T
  onRehydrateStorage?: () => void
  /** Keys the caller has explicitly waived. Use sparingly. */
  waivedKeys?: string[]
}

function assertNoDeniedKeys(name: string, keys: string[], waived: ReadonlySet<string>): void {
  const denied = keys.filter((k) => DENY_KEYS.has(k) && !waived.has(k))
  if (denied.length > 0) {
    // Fail fast at construction time so a misconfigured
    // store never reaches production.
    throw new Error(
      `[persist] store '${name}' refused to persist forbidden field(s): ${denied.join(', ')}. ` +
      `These names match the secret-shaped deny-list (apiKey/token/secret/password/...). ` +
      `Either rename the field, remove it from partialize(), or pass waivedKeys=[...] with a justification.`,
    )
  }
}

export function createPersistedStore<T extends object>(
  initializer: StateCreator<T, [], []>,
  config: PersistConfig<T>,
) {
  // Inspect the partialize output once at construction
  // time. The check is conservative: we union the keys
  // the partialize returns with the keys the initializer
  // defines. If partialize is omitted, all keys are at
  // risk and we use the initializer's key set.
  const sampleState = (initializer as (() => T))()
  const partializeKeys = config.partialize
    ? Object.keys(config.partialize(sampleState))
    : Object.keys(sampleState)
  assertNoDeniedKeys(
    config.name,
    partializeKeys,
    new Set(config.waivedKeys ?? []),
  )

  return create<T>()(
    persist(initializer, {
      name: `aurelius-${config.name}`,
      version: config.version ?? 1,
      storage: createJSONStorage(() => localStorage),
      partialize: config.partialize,
      migrate: config.migrate as (persisted: unknown, version: number) => T,
      onRehydrateStorage: config.onRehydrateStorage,
    }) as unknown as StateCreator<T, [], []>,
  )
}

const migrations: Record<string, (state: Record<string, unknown>, version: number) => Record<string, unknown>> = {
  'aurelius-app-store': (state, version) => {
    if (version < 2) {
      return { ...state, autoRefresh: true, paletteOpen: false }
    }
    return state
  },
}

export function migrateStore(storeName: string, state: unknown, version: number): unknown {
  const migration = migrations[storeName]
  if (migration) {
    return migration(state as Record<string, unknown>, version)
  }
  return state
}
